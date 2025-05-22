import numpy as np
import math
import itertools
from scipy.optimize import minimize
import scipy
from itertools import combinations
from gpthandler import get_preference_relationship
import os
import openai
import time
import json
import ast
import re

def query_with_counteroffer(offer, A, b, items, int_constrained=True, max_trade_value=5):
    """
        Query an agent with utility function x^TAx + 2bx to determine if they will accept an offer. The agent may also respond with a counteroffer if the current offer is not feasible.

        Args:
            offer (np.array): n-dimensional vector representing an offer from the receiving agent's perspective.
            A (np.array): nxn matrix.
            b (list): n-dimensional vector.
            items (list): Current State of agent.
            int_constrained (bool, optional): Whether the counteroffer should be integer constrained

        Returns:
            flag_n_items (bool): Response value which states if the agent has enough items to complete the trade
            flag_utility_improvement (bool): Response value which states if the agent's utility improves with the offer
            counteroffer (np.array): If the offer is not feasible, return a scaled down offer that is feasible for the agent.
    """
    next_step = offer + items
    flag_utility_improvement = utility_improvement(offer, A, b, items)

    if all(i >= 0 for i in next_step):
        flag_n_items = True
    else:
        flag_n_items = False
    if flag_utility_improvement == False:
        agent_grad_val = n_dim_quad_grad(A, b, items)
        counteroffer, max_scaling_factor, improvement = find_scaling_offering(agent_grad_val, A, b, items,
                                                                              items, max_trade_value=max_trade_value)
        if not improvement:
            return flag_n_items and flag_utility_improvement, None
        if int_constrained:
            counteroffer = branch_and_bound_agent(counteroffer, agent_grad_val)
        return flag_n_items and flag_utility_improvement, counteroffer
    else:
        if flag_n_items == False:
            agent_grad_val = n_dim_quad_grad(A, b, items)
            counteroffer, max_scaling_factor, improvement = find_scaling_offering(offer, A, b, items, items, max_trade_value=max_trade_value)
            if not improvement:
                return flag_n_items and flag_utility_improvement, None
            if int_constrained:
                counteroffer = branch_and_bound_agent(counteroffer, agent_grad_val)
            return flag_n_items and flag_utility_improvement, counteroffer
        else:
            return flag_n_items and flag_utility_improvement, None

def state_string_deterministic(state, item_list):
    state_string = ''
    for item_idx in range(0, len(item_list) - 1):
        state_string += f'{state[item_idx]} {item_list[item_idx]}, '
    state_string += f'and {state[-1]} {item_list[-1]}.'
    return state_string

def query_gpt_with_counteroffer(offer, item_list, ui, current_human_state, prev_trade_accepted=False, prev_counteroffer_flag=False, prev_counteroffer=[]):
    """
        Query the responding with utility function via gpt to see if they will accept an offer.

        Args:
            offer (np.array): n-dimensional vector representing an offer from the receiving agent's perspective.
            item_list (np.array): List of item names.
            ui (list): Directory where the program will place the offer.

        Returns:
            accepted (bool): Response value which states if the agent has accepted the offer.
            counteroffer (np.array): Feedback from the responding agent, parsed into a counteroffer.
            stopped_trading (bool): Whether the responding agent has decided to stop trading
    """
    # Turn the offer into a string
    offer_string = offer_string_deterministic(offer, item_list) + "\n"
    offer_string_sent = offer_string.replace("Alice", "User").replace("Bob", "ST-CR")

    # Turn Current State into a String
    accepted_string = "Trade has Been Accepted! \n"

    current_state_string = "Your Current State: " + state_string_deterministic(current_human_state, item_list) + "\n \n"
    next_state_string = "Your Next State After This Trade: " + state_string_deterministic(current_human_state + offer, item_list) + "\n"
    query = "Do you accept this trade? \n"
    full_sent_string = current_state_string + offer_string_sent + next_state_string + query
    if prev_trade_accepted:
        full_sent_string = accepted_string + full_sent_string
    if prev_counteroffer_flag:
        starting_string = "Computer: I understand that you would prefer the following offer: "
        counteroffer_string = offer_string_deterministic(prev_counteroffer, item_list)
        # print("Counteroffer String: ", counteroffer_string)
        counteroffer_string = counteroffer_string.replace("Bob's Trade Offer:", "")
        counteroffer_string = counteroffer_string.replace("Bob receives", "Alice gives")
        counteroffer_string = counteroffer_string.replace("Alice", "User")
        ending_string = "This trade is not beneficial for me. I will keep your preferences in mind for future trades. For now, please consider the following offer: "
        full_sent_string = starting_string + "\n" + counteroffer_string + "\n" + ending_string + "\n" + full_sent_string
    # Add the offer to the log
    # print("Full Sent String: ", full_sent_string)
    log_file = os.path.join(ui, 'log.txt')
    log_opened = open(log_file, "a")
    log_opened.write(full_sent_string)
    log_opened.close()

    # Send the offer to the receiving agent
    prompt_file = os.path.join(ui, 'offer.txt')
    with open(prompt_file, 'w') as f:
        f.write(full_sent_string)

    # Parse the receiving agent's response
    response, counteroffer, stopped_trading, offer_time, response_time = parse_response(offer_string, item_list, ui)

    # Return if the agent has stopped trading.
    if stopped_trading:
        return False, [], True, offer_time, response_time

    # If rejected, check to see if the responding agent has provided a counteroffer
    if response == 0:
        if len(counteroffer) != 0:
            return False, tuple(counteroffer), False, offer_time, response_time
        else:
            return False, [], False, offer_time, response_time
    else:
        return True, counteroffer, False, offer_time, response_time


def parse_response(prior_trade_offer, item_list, ui):
    """
        Give a natural language response from the responding agent, determine if they are accepting or rejecting the offer, or if they are giving any feedback

        Args:
            prior_trade_offer (string): string representing the last offer from to the receiving agent.
            item_list (np.array): List of item names.
            ui (list): Directory where the user will respond to the offer.

        Returns:
            response (bool): Response value which states if the agent has accepted the offer.
            counteroffer (np.array): Feedback from the responding agent, parsed into a counteroffer.
            stopped_trading (bool): Whether the responding agent has decided to stop trading
    """
    offer_time = time.time()
    # Wait until the user responds
    response_file = os.path.join(ui, 'response.txt')
    while not os.path.exists(response_file):
        # print(f"Waiting for response")
        time.sleep(1)
    response_time = time.time()
    # Read the response
    # print("Response Obtained")
    with open(response_file, 'r') as f:
        user_input = f.read().strip()
    os.remove(response_file)

    # Log the response
    if user_input == "stop":
        log_file = os.path.join(ui, 'log.txt')
        log_opened = open(log_file, "a")
        log_opened.write("\n User: Stopped Trading \n")
        log_opened.close()
        return False, [], True, offer_time, response_time
    log_file = os.path.join(ui, 'log.txt')
    log_opened = open(log_file, "a")
    log_opened.write("User: " + user_input + "\n")
    log_opened.close()

    # Perform sentiment analysis to determine if the responding agent has accepted the offer
    user_input = "Alice: " + user_input
    convo_history = prior_trade_offer + user_input
    response_val = sentiment_analysis(convo_history)
    if response_val == 1:
        return response_val, [], False, offer_time, response_time
    else:
        # If not, parse the counteroffer
        counteroffer, counteroffer_string = obtain_counteroffer_text(convo_history, item_list)

    # Log the parsed counteroffer
    counteroffer_string = counteroffer_string.replace("Alice", "User").replace("Bob", "ST-CR")
    # print(counteroffer_string)
    # lines = counteroffer_string.split('\n')
    # modified_lines = ['\t' + line for line in lines[2:]]
    # counteroffer_string = '\n'.join(modified_lines)
    parsed_counteroffer_string = "Parsed User Counteroffer: \n" + counteroffer_string + "\n"
    # print(parsed_counteroffer_string)
    # print(parsed_counteroffer_string, counteroffer)
    log_file = os.path.join(ui, 'log.txt')
    log_opened = open(log_file, "a")
    log_opened.write(parsed_counteroffer_string)
    log_opened.close()
    return response_val, counteroffer, False, offer_time, response_time


def branch_and_bound_agent(offer, agent_grad):
    """
        Given a fractional offer, return an integer offer that is most aligned with a given gradient
        Args:
            offer (np.array): fractional offer
            agent_grad (np.array): Gradient that we want to align the offer with

        Returns:
            output_offer (np.array): Rounded offer that is most closely aligned with the gradient
    """
    rounded_list = generate_int_vectors(offer)
    theta_list = []
    for int_vector in rounded_list:
        norm_int_offer = int_vector / np.linalg.norm(int_vector)
        theta_list.append(np.dot(norm_int_offer, agent_grad))
    output_offer = rounded_list[np.argmax(theta_list)]
    return output_offer


def utility_improvement(offer, A, b, items, reduction_idx=[]):
    """
        Determine if an offer leads to a utility improvement for an agent with utility function x^TAx +x^Tb

        Args:
            offer (np.array): n-dimensional vector representing an offer from the agent's perspective.
            A (np.array): nxn matrix.
            b (np.array): n-dimensional vector.
            items (list): Current State of agent.
            reduction_idx (np.array, optional): Set of item categories that are not being considered for trading.

        Returns:
            (bool): Response value which states if the agent's utility improves with the offer
    """
    if utility_value(offer, A, b, items, reduction_idx=reduction_idx) > 0:
        return True
    else:
        return False


def utility_value(offer, A, b, items, reduction_idx=[]):
    """
        Return the value of an offer given a utility function of the form x^TAx + bx, an offer, and the current set of items.
        Args:
            offer (np.array): n-dimensional vector representing an offer from the agent's perspective.
            A (np.array): nxn matrix.
            b (np.array): n-dimensional vector.
            items (list): Current State of agent.
            reduction_idx (np.array, optional): Set of item categories that are not being considered for trading.

        Returns:
            (bool): Response value which states if the agent's utility improves with the offer
    """
    idx_counter = 0
    true_offer = np.zeros(len(b))
    true_items = items
    for element in range(0, len(b)):
        if element not in reduction_idx:
            true_offer[element] = offer[idx_counter]
            idx_counter += 1
    next_step = true_items + true_offer
    prev_state_value = true_items.transpose() @ A @ true_items + b @ true_items
    next_state_value = next_step.transpose() @ A @ next_step + b @ next_step
    return next_state_value - prev_state_value


def query_init(offer, responding_grad):
    """
        Query that uses the responding's gradient to determine if it will accept or reject an offer. This is used to set up the initial quadrant when trading.
        Args:
            offer (np.array): n-dimensional vector representing an offer from the agent's perspective.
            responding_grad (np.array): Approximation of the responding's gradient

        Returns:
            (bool): Response value which states if the agent's utility gradient is aligned with the offer
    """
    dot_product = np.dot(offer, responding_grad)
    if dot_product > 0:
        return True
    else:
        return False

def sentiment_analysis(convo_history):
    """
        Given a conversation history comprised of an offer and a response, determine if the user is accepting or rejecting the offer
        Args:
            convo_history (string): A conversation history comprised of an offer and a response

        Returns:
            (int): Response value which states if the user is accepting or rejecting the offer (1 for accept, 0 for reject)
    """
    test_string = convo_history.replace("Bob receives", "Alice gives")
    lines = test_string.splitlines()
    for line in lines:
        if line.strip().startswith("Alice:"):
            alice_response = line.strip()
        if line.strip().startswith("Alice gives"):
            alice_gives = line.strip()
        if line.strip().startswith("Alice receives"):
            alice_receives = line.strip()
    user_message = (
        f"In a trading scenario, Alice has been given the following trade offer {alice_gives}, {alice_receives}, Alice has given the following response: {alice_response}. Given Alice's response, is she accepting or rejecting the offer. Please answer with 1 for accept, otherwise answer with 0. If the text doesn't provide information on whether Alice accepting or rejecting Bob's trade offer, respond with 0."
    )
    # user_message = f"Given the following conversation history: \"{convo_history}\", is Alice accepting or rejecting Bob's trade offer? Please answer with 1 for accept, otherwise answer with 0. If the text doesn't provide information on whether Alice accepting or rejecting Bob's trade offer, respond with 0."
    messages = [{"role": "user", "content": user_message}]
    max_retries = 10
    num_retries = 0
    while num_retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Ensure you specify the correct model name here
                messages=messages,
                max_tokens=10
            )
            break  # Exit the loop if the request was successful
        except openai.error.RateLimitError as e:
            num_retries += 1
            wait_time = 2 ** num_retries  # Exponential backoff (increase wait time after each retry)
            print(f"RateLimitError encountered. Retrying in {wait_time} seconds...")
            time.sleep(min(wait_time, 30))
    response_val = int(response.choices[0].message.content)
    return response_val


def is_dict_string(s):
    """
        Debugging function used to determine if a string response represents a dictionary object
        Args:
            s (string): A string

        Returns:
            (bool): Whether the string represents a dictionary
    """
    try:
        result = ast.literal_eval(s)
        return isinstance(result, dict)
    except (ValueError, SyntaxError):
        return False


def obtain_counteroffer_text(convo_history, item_list):
    """
        Given an offer and a responding response, determine if the responding is providing feedback, and parse the feedback into a counteroffer.

        Args:
            convo_history (string): A conversation history comprised of an offer and a response
            item_list (np.array): List of item names.

        Returns:
            counteroffer (np.array): Feedback from the responding agent, parsed into a counteroffer.
            counteroffer_string (string): Counteroffer from the responding agent parsed into a standardized format
    """

    # user_message = f"In a negotiation context, responses can be classified as either a 'counteroffer' or a 'preference.' A counteroffer is when one party indicates a conditional acceptance that suggests specific changes to the original offer, including requests for additional items or quantities. A preference, on the other hand, expresses a general desire without specific terms. Given the following conversation history: \"{convo_history}\", Is Alice providing a counteroffer or adjustment to the previous offer, a general preference, or neither? Respond with \"counteroffer\" if she is indicating a conditional acceptance that suggests specific or general changes to the offer, \"preference\" if she expresses a general desire without specific terms, and \"neither\" if neither option is applicable. Only respond with \"counteroffer\", \"preference\", or \"neither\""
    user_message = f"In the context of negotiation, responses can be classified as either a 'counteroffer', an 'adjusted preference', or a 'general preference.' A counteroffer occurs when one party indicates a conditional acceptance that specifies changes to the original offer, such as requesting additional items or quantities. An adjusted preference suggests a desire for changes to the original offer but does not fully reject it. A general preference expresses a general desire without indicating any adjustments to the offer. Given the following conversation history: \"{convo_history}\", please determine if Alice is providing a counteroffer, an adjusted preference, a general preference, or neither. Respond with \"counteroffer\" if she indicates a conditional acceptance that specifies changes to the offer, including requests for more or fewer items. Respond with \"adjusted preference\" if she expresses a desire for changes without fully rejecting the offer. Respond with \"preference\" if she expresses a general desire. If neither option applies, respond with \"neither.\" Only use the responses \"counteroffer\", \"adjusted preference\", \"preference\", or \"neither.\""

    messages = [{"role": "user", "content": user_message}]
    max_retries = 10
    num_retries = 0
    while num_retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Ensure you specify the correct model name here
                messages=messages,
                max_tokens=10
            )
            break  # Exit the loop if the request was successful
        except openai.error.RateLimitError as e:
            num_retries += 1
            wait_time = 2 ** num_retries  # Exponential backoff (increase wait time after each retry)
            print(f"RateLimitError encountered. Retrying in {wait_time} seconds...")
            time.sleep(min(wait_time, 30))
    response_val = response.choices[0].message.content
    # print("Feedback Type: ", user_message, response_val)
    response_val = response_val.lower()
    # Case 1: User is givin an Explicit Counteroffer
    if response_val == 'counteroffer' or response_val == 'adjusted preference' :
        correct_counteroffer = False
        loop_counter = 0
        while not correct_counteroffer:  # or loop_counter < 3:
            dict_string = parse_counteroffer(convo_history, item_list)
            correct_counteroffer = check_if_counteroffer_correct(dict_string, convo_history)
            loop_counter += 1
        actual_dict = parse_trade_offer(dict_string, item_list)
        # print("Actual Dict: ", actual_dict)
        counteroffer = np.zeros(len(item_list))
        for item_idx in range(len(item_list)):
            if item_list[item_idx] in actual_dict.keys():
                counteroffer[item_idx] = actual_dict[item_list[item_idx]]
        # print("Checking Dict String: ", counteroffer, dict_string, actual_dict)
        return counteroffer, dict_string

    # Case 2: User is giving a General Preference
    if response_val == 'preference':
        item_list_string = ', '.join(item_list)
        index = convo_history.find("Alice: ")
        substring = convo_history[index + len("Alice: "):]
        correct_counteroffer = False
        while not correct_counteroffer:
            reltationship_string = get_preference_relationship(substring, item_list_string)
            if 'function_call' in reltationship_string.keys():
                if is_dict_string(reltationship_string["function_call"]["arguments"]):
                    parsed_dict = json.loads(reltationship_string['function_call']['arguments'])
                else:
                    parsed_dict = {}
            else:
                if is_dict_string(reltationship_string["content"]):
                    parsed_dict = json.loads(reltationship_string['content'])
                else:
                    parsed_dict = {}
            if len(parsed_dict) == 0:
                return [], ""
            w_1 = parsed_dict['w_1']
            w_2 = parsed_dict['w_2']
            out_items = parsed_dict['items']
            if parsed_dict['rel'] == '>=' or parsed_dict['rel'] == '=>':
                counteroffer = np.array(w_1) - np.array(w_2)
            if parsed_dict['rel'] == '<=' or parsed_dict['rel'] == '=<':
                counteroffer = np.array(w_2) - np.array(w_1)
            full_counteroffer = np.zeros(len(item_list))
            for index in range(len(counteroffer)):
                item = out_items[index].lower()
                counteroffer_item_index = item_list.index(item)
                full_counteroffer[counteroffer_item_index] = counteroffer[index]
            counteroffer_string = offer_string_deterministic(full_counteroffer, item_list, pov="Alice")
            correct_counteroffer = check_if_preference_correct(counteroffer_string, substring)
        return full_counteroffer, counteroffer_string
    if response_val == 'neither':
        return [], ""


def check_if_counteroffer_correct(dict_string, convo_history):
    """
        Given a parsed counteroffer and a conversation history, determine if the counteroffer is an accurate interpretation of the responding's feedback

        Args:
            dict_string (string): String representing a counteroffer parsed from the responding's response
            convo_history (string): A conversation history comprised of an offer and a response

        Returns:
            response_val (bool): Whether the parsed counteroffer is an accurate interpretation of the responding's feedback
    """
    test_string = convo_history.replace("Bob receives", "Alice gives")
    lines = test_string.splitlines()
    for line in lines:
        if line.strip().startswith("Alice:"):
            alice_response = line.strip()
        if line.strip().startswith("Alice gives"):
            alice_gives = line.strip()
        if line.strip().startswith("Alice receives"):
            alice_receives = line.strip()
    user_message = (
        f"In a trading scenario, Alice has been given the following trade offer {alice_gives}, {alice_receives}, Alice has given the following response: {alice_response}. Given Alice's feedback, does the following counteroffer {dict_string} accurately reflect her feedback? Please respond with only yes or no. If the counteroffer includes ambiguous or non-numerical values for the items (e.g. 'some apples'), please respond with no."
    )
    print("User Message: ", user_message, "\n")
    messages = [{"role": "user", "content": user_message}]
    max_retries = 10
    num_retries = 0
    while num_retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Ensure you specify the correct model name here
                messages=messages,
                max_tokens=10
            )
            break  # Exit the loop if the request was successful
        except openai.error.RateLimitError as e:
            num_retries += 1
            wait_time = 2 ** num_retries  # Exponential backoff (increase wait time after each retry)
            print(f"RateLimitError encountered. Retrying in {wait_time} seconds...")
            time.sleep(min(wait_time, 30))


    response_val = response.choices[0].message.content

    if response_val == "Yes" or response_val == "yes":
        return True
    else:
        return False


def check_if_preference_correct(dict_string, convo_history):
    """
        Given a parsed general preference and a conversation history, determine if the counteroffer is an accurate interpretation of the responding's feedback

        Args:
            dict_string (string): String representing a general preference parsed from the responding's response
            convo_history (string): A conversation history comprised of an offer and a response

        Returns:
            response_val (bool): Whether the parsed counteroffer is an accurate interpretation of the responding's feedback
    """
    user_message = f"Given the following conversation history: \"{convo_history}\", is the following counteroffer a correct interpretation of Alice's preference? Counteroffer: \"{dict_string}\" Please respond 1 for yes, 0 for no. (only answer with 0 and 1)"
    messages = [{"role": "user", "content": user_message}]
    max_retries = 10
    num_retries = 0
    while num_retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Ensure you specify the correct model name here
                messages=messages,
                max_tokens=10
            )
            break  # Exit the loop if the request was successful
        except openai.error.RateLimitError as e:
            num_retries += 1
            wait_time = 2 ** num_retries  # Exponential backoff (increase wait time after each retry)
            print(f"RateLimitError encountered. Retrying in {wait_time} seconds...")
            time.sleep(min(wait_time, 30))
    response_val = int(response.choices[0].message.content)
    if response_val == 1:
        return True
    else:
        return False

def parse_trade_offer(trade_string, item_list):
    """
    Given a string representing a trade offer, return a dictionary corresponding to the offer.

    Args:
        trade_string (string): String representing a trade offer.
        item_list (list): List of item names (can be plural or singular).

    Returns:
        trade_dict (dictionary): Dictionary representing the offer.
    """
    print("Parsing Counteroffer: ", trade_string, item_list)
    # Prepare singular and plural mappings
    item_singular = {item.rstrip('s').lower(): item for item in item_list}  # Singular to original mapping

    # Initialize trade dictionary with zero quantities
    trade_dict = {item: 0.0 for item in item_list}

    # Remove trailing punctuation and split by 'Alice'
    trade_string = trade_string.strip().rstrip('.')
    alice_parts = trade_string.split('Alice')

    for part in alice_parts:
        part = part.strip()
        if not part:  # Skip empty strings
            continue

        # Handle 'gives' part
        if 'gives:' in part:
            items = part.split('gives:')[1].strip()

            # If Alice gives nothing, skip this part
            if items.lower().rstrip(',') == 'nothing':
                continue

            # Process the items Alice gives
            for item in items.split(','):
                if 'nothing' in item.lower().rstrip(','):
                    continue
                words = item.strip().split()
                if len(words) != 0:
                    quantity = float(words[0])
                    item_name = words[1].strip('.').lower()  # Remove punctuation
                    singular_item_name = item_name.rstrip('s')  # Convert plural to singular if necessary
                    if singular_item_name in item_singular:
                        trade_dict[item_singular[singular_item_name]] -= quantity  # Alice loses items

        # Handle 'receives' part
        elif 'receives:' in part:
            items = part.split('receives:')[1].strip()

            # If Alice receives nothing, skip this part
            if items.lower().rstrip(',') == 'nothing':
                continue

            # Process the items Alice receives
            for item in items.split(','):
                if 'nothing' in item.lower().rstrip(','):
                    continue
                words = item.strip().split()
                if len(words) != 0:
                    quantity = float(words[0])
                    item_name = words[1].strip('.').lower()  # Remove punctuation
                    singular_item_name = item_name.rstrip('s')  # Convert plural to singular if necessary
                    if singular_item_name in item_singular:
                        trade_dict[item_singular[singular_item_name]] += quantity  # Alice gains items
    # time.sleep(10000)
    return trade_dict

def parse_counteroffer(convo_history, item_list):
    """
        Given a conversation history comprised of an offer and a response, find a counteroffer string that represents the feedback.

        Args:
            convo_history (string): A conversation history comprised of an offer and a response
            item_list (np.array): List of item names.

        Returns:
            offer (string): String representing the parsed counteroffer
    """
    test_string = convo_history.replace("Bob receives", "Alice gives")
    lines = test_string.splitlines()
    for line in lines:
        if line.strip().startswith("Alice:"):
            alice_response = line.strip()
        if line.strip().startswith("Alice gives"):
            alice_gives = line.strip()
        if line.strip().startswith("Alice receives"):
            alice_receives = line.strip()
    # user_message = (
    #     f"In a trading scenario, Alice has been given the following trade offer {alice_gives}, {alice_receives}, Alice has given the following response: {alice_response}. Given Alice's feedback, please provide a counteroffer that reflects her feedback. The counteroffer should be in the following format 'Alice gives: a apples, b oranges, c bananas, Alice receives: d apples, e oranges, f bananas', where a, b, c, d, e, and f are replaced by numerical values. Do not respond with any other information. If Alice provides general feedback or non-explicit values (e.g., 'some apples,' 'some oranges'), infer and substitute reasonable numerical values based on typical exchange patterns in the conversation. If Alice's feedback includes 'all' of a given fruit, just add a some more of that fruit to the trade. If one of the two categories is empty (i.e. Alice is not giving or receiving any items), please write 'Nothing' instead of the item list. Do not use placeholders like '?' or ambiguous words like 'some', 'all', or 'more'; provide a specific numerical value for each item even if the feedback does not include specific values. Also, if Alice is asking to receive an amount of a fruit, and the previous trade offer had her giving a fruit, the amount of the fruit she is giving in the new trade offer should be 0. The same should hold true in the reverse case. Ensure the offer includes 'Alice gives' as the first entry and is formatted using the provided item list: {item_list}."
    # )

    user_message = (f"""
In a trading scenario, Alice has been given the following trade offer: {alice_gives}, {alice_receives}. Alice has given the following response: {alice_response}. Based on Alice's feedback, provide a counteroffer that accurately reflects her request. The counteroffer should be formatted as:

Alice gives: a apples, b oranges, c bananas, Alice receives: d apples, e oranges, f bananas.

where 'a, b, c, d, e, f' are specific numerical values.

Guidelines:
- If Alice asks to give an item (e.g., "Can I give you X bananas?"), set the number of that item she gives to that amount."
- If Alice asks to receive an item, set the number of that item she receives to the requested amount. If the previous offer had her giving that, ensure that her new offer reflects the current request accurately.
- If the previous offer has Alice giving an item, and Alice's response asks to receive the same item, then set the number of that item she gives to '0'. 
- If the previous offer has Alice receiving an item, and Alice's response asks to give the same item, then set the number of that item she receives to '0'.
- If Alice is not giving any items in the new trade, replace the 'Alice gives' list with 'Nothing.' If Alice is not receiving any items in the new trade, replace the 'Alice receives' list with 'Nothing.'
- Do not respond with placeholders or ambiguous terms like 'some', 'more', 'fewer', etc. I want you to infer specific numerical values, even if they are inferred (ex. '1 apple' instead of 'some apples') 
- Ensure the trade offer is specific and formatted as requested.
"""
    )
    messages = [{"role": "system", "content": user_message}]
    max_retries = 10
    num_retries = 0
    while num_retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Ensure you specify the correct model name here
                messages=messages,
                max_tokens=30
            )
            break  # Exit the loop if the request was successful
        except openai.error.RateLimitError as e:
            num_retries += 1
            wait_time = 2 ** num_retries  # Exponential backoff (increase wait time after each retry)
            print(f"RateLimitError encountered. Retrying in {wait_time} seconds...")
            time.sleep(min(wait_time, 30))
    print("Checking Counteroffer: ", response.choices[0].message.content)
    return response.choices[0].message.content


def offer_string_deterministic(offer, item_list, pov='Bob'):
    """
    Given a vector representing an offer, generate a standardized string offer.
    Args:
        offer (np.array): Offer
        item_list (np.array): List of item names.
        pov (string): Name of the agent making the trade.

    Returns:
        trade_offer_string (string): String representing the offer
    """
    positive_items = {}
    negative_items = {}
    receive_string = f""
    give_string = f""
    
    for item_idx in range(len(offer)):
        # Clean formatting: avoid -0.0 and remove unnecessary .0
        formatted_offer = "{:.1f}".format(abs(offer[item_idx])).rstrip('0').rstrip('.')
        
        if offer[item_idx] > 0:
            positive_items[item_list[item_idx]] = offer[item_idx]
            give_string += f"{formatted_offer} {item_list[item_idx]}, "
        elif offer[item_idx] < 0:
            negative_items[item_list[item_idx]] = offer[item_idx]
            receive_string += f"{formatted_offer} {item_list[item_idx]}, "
    
    if len(positive_items) == 0:
        give_string = "Nothing"
    if len(negative_items) == 0:
        receive_string = "Nothing"

    # Create trade offer string
    trade_offer_string = f"""{pov}'s Trade Offer: 

    Alice gives: {receive_string.strip(', ')}
    Alice receives: {give_string.strip(', ')}
    """
    
    # print("Checking Offer: ", offer, trade_offer_string)
    return trade_offer_string


def branch_and_bound(offer, center_of_cone, offering_grad):
    """
        Given an offer, return an integer offer that is within 90 degrees of the cone's center and closest to the offering_gradient
        Args:
            offer (np.array): n-dimensional vector representing an offer from the agent's perspective.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
            offering_grad (np.array): n-dimensional vector corresponding to the offering agent's gradient.
            
        Returns:
            output_offer (np.array): Rounded Offer
    """
    
    # Generate a set of rounded offers
    rounded_list = generate_int_vectors(offer)
    theta_list = []

    # This for loop ensures that the rounded offers are aligned with the center of the cone
    for int_vector in rounded_list:
        # Represent the offer from the perspective of the responding
        int_list = list(int_vector)
        neg_list = [-1 * x for x in int_list]
        neg_list_norm = neg_list / np.linalg.norm(neg_list)
        center_of_cone_norm = center_of_cone / np.linalg.norm(center_of_cone)
        dot_product = np.dot(neg_list_norm, center_of_cone_norm)

        # If the offer and the center of the cone are not algined, then the responding agent will not accept the offer
        # In this case, we remove the offer from consideration
        if dot_product < 0:
            theta_list.append(-1 * np.inf)
        else:
            theta_list.append(np.dot(int_list/np.linalg.norm(int_list), offering_grad))
    
    # Select the rounded offer that is most closely aligned with the offering gradient direction
    output_offer = list(rounded_list[np.argmax(theta_list)])
    # If the offer is not aligned with the offering gradient, negate it's direction.
    if np.dot(output_offer, offering_grad) < 0:
        output_offer = -1 * np.array(output_offer)
    return output_offer

def generate_int_vectors(float_vector):
    """
        Given a vector of floats, return a set of vectors that represents the integer rounding of the set of floats
        Args:
            float_vector (np.array): n-dimensional float

        Returns:
            integer_combinations (list of np.array): Set of all possible roundings of the float vector
    """
    float_vector = [float(num) for num in float_vector]
    combinations = set(itertools.product(*[range(math.floor(val), math.ceil(val) + 1) for val in float_vector]))
    integer_combinations = [tuple(round(val) if not isinstance(val, int) else val for val in combo) for combo in combinations]
    icc = integer_combinations.copy()
    for combination in icc:
        if all(element == 0 for element in combination):
            integer_combinations.remove(combination)
    return integer_combinations

def find_init_offer_greedy(offering_grad, center_of_cone):
    """
        Determine an offer that is orthogonal and is in the direction of the offering agent's gradient
        Args:
            offering_grad (np.array): n-dimensional vector corresponding to the offering agent's gradient.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.

        Returns:
            offer (np.array): Projection of the offering agent's gradient on the null space of the cone center.
    """
    projection = np.dot(offering_grad, center_of_cone) / np.dot(center_of_cone, center_of_cone) * center_of_cone
    offer = offering_grad - projection
    return offer

def find_init_offer_random(offering_grad, center_of_cone):
    """
        Determine a random initial offer that is orthogonal and that is aligned with the offering agent's gradient
        Args:
            offering_grad (np.array): n-dimensional vector corresponding to the offering agent's gradient.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.

        Returns:
            offer (np.array): Random vector in the null space of the cone center
    """
    # Check if the vector is not a zero vector
    if np.linalg.norm(center_of_cone) == 0:
        raise ValueError("Input vector should not be a zero vector")

    # Create a random vector
    random_vector = np.random.randn(len(center_of_cone))

    # Use the Gram-Schmidt process to get an orthogonal vector
    orthogonal_vector = random_vector - (np.dot(random_vector, center_of_cone) / np.dot(center_of_cone, center_of_cone)) * center_of_cone

    # Normalize the orthogonal vector
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)
    if np.dot(orthogonal_vector, offering_grad) < 0:
        orthogonal_vector *= -1
    return orthogonal_vector

def find_orth_vector(vectors, offering_gradient):
    """
        Given a set of n-dimensional vectors, find a set of vectors that are orthogonal to the given set
        Args:
            vectors (list of np.array): set of vectors
            offering_gradient (np.array): n-dimensional vector corresponding to the offering agent's gradient.

        Returns:
            orthogonal_vectors (list of np.array): Set of vectors orthogonal to all the vectors in the input vector set and the center of the cone.
    """

    # Calculate the null space of the given set of vectors
    null_space = scipy.linalg.null_space(vectors)
    orthogonal_vectors = []

    # Iterate over the basis of the null space to find orthogonal vectors
    for i in range(null_space.shape[1]):
        potential_vector = null_space[:, i]
        is_orthogonal = True

        # Check that the null space vectors are orthogonal to all the vectors in the given set.
        for vector in orthogonal_vectors:
            if abs(np.dot(potential_vector, vector)) > 1e-10:  # Checking if dot product is close to zero
                is_orthogonal = False
                break
        # If the offer is orthogonal, then add it to the set of orthogonal vectors
        if is_orthogonal:
            # Ensure the direction is beneficial for the offering
            if np.dot(potential_vector, offering_gradient) < 0:
                potential_vector = -1 * potential_vector
            orthogonal_vectors.append(potential_vector)
    
    return orthogonal_vectors

def sort_orth_vectors(A, b, items, vectors, reduction_idx = []):
    """
        Sort the set of orthogonal vectors in terms of utility for a function x^TAx + b^Tx.
        Args:
            A (np.array): nxn matrix.
            b (np.array): n-dimensional vector.
            items (list): Current State of agent.
            vectors (list of np.array): vectors to be sorted
            reduction_idx (list, optional): set of item indices that need to be removed from consideration.

        Returns:
            sorted_offers (list of np.array): Set vectors sorted in decreasing order of utility value.
    """
    # This ensures that the most beneifical trades for the offering will be offered first.
    sorted_offers = sorted(vectors, key=lambda vector: utility_value(vector, A, b, items, reduction_idx=reduction_idx))
    return sorted_offers

def angle_between(v1, v2):
    """
        Return the angle between two vectors.
        Args:
            v1 (np.array): n-dimensional vector.
            v2 (np.array): n-dimensional vector.
            
        Returns:
            angle (float): Angle between the two vectors in radians
    """
    dot_product = np.dot(v1, v2)
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    cos_theta = dot_product / (m1 * m2)
    angle = np.arccos(cos_theta)
    return angle

def find_scaling_offering(vector, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, need_improvement = True, reduction_idx = [], int_constrained=True):
    """
        Find a scaled vector for the offering agent that is feasible

        Args:
            vector (np.array): n-dimensional vector representing the current offer.
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            offering_items (np.array): Number of items that the offering agent has in its possession from categories that are being comsidered currently.
            offering_items_original (np.array): Number of items that the offering agent has in its possession from all categories.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            need_improvement (bool, optional): Whether the offering agent needs to improve its utility with this offer Defaults to True.
            int_constrained (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            reduction_idx (list, optional): set of item indices that need to be removed from consideration.

        Returns:
            tuple:
                - scaled_vector (np.array): Scaled version of the original vector
                - scaling_factor (float): Scaling factor used to increase the magnitude of the unit vector
                - improvement (bool): Whether the offering agent is improving its utility with this offer
    """
    # Find a scaling factor that scales a given vector to a given number of items
    
    # If the offer is not aligned with the offering's gradient, reverse its direction
    offering_gradient = n_dim_quad_grad(offering_A, offering_b, offering_items_original)
    full_vector = np.zeros(len(offering_b))
    idx_counter = 0
    for element in range(0, len(offering_b)):
        if element not in reduction_idx:
            full_vector[element] = vector[idx_counter]
            idx_counter += 1
    if np.dot(full_vector, offering_gradient) < 0:
        vector = -1 * np.array(vector)
    
    # If we are looking for a trade that benefits the offering, we may need to scale down the offer to ensure we don't overshoot any optimal points.
    if need_improvement:
        improvement = False
        while not improvement:

            # Scale the offer based on the maximum amount of a given item the offering can trade (max_trade_value)
            scaled_vector = vector.copy()
            max_scaling_factor = find_scaling_factor(vector, max_trade_value, offering_items)       
            scaled_vector = [scaled_vector[i] * max_scaling_factor for i in range(len(scaled_vector))]
            max_index, max_value = max(enumerate(scaled_vector), key=lambda x: abs(x[1]))
            if int_constrained:
                scaled_vector[max_index] = round(scaled_vector[max_index])

            # Check if the offer improves the offering's utility
            improvement = utility_improvement(scaled_vector, offering_A, offering_b, offering_items_original, reduction_idx=reduction_idx)
            # If the maximum trade value is one, we cannot scale the offer down an further
            if max_trade_value == 1:
                break
            # If the offering does not benefit from the offer, but it is aligned with its gradent, then the trade mangiute is too large
            max_trade_value  = math.ceil(max_trade_value/2)
        
        return scaled_vector, max_scaling_factor, improvement
    else:
        # If improvement is not required, then scale the offer to the max_trade_value
        scaled_vector = vector.copy()
        max_scaling_factor = find_scaling_factor(vector, max_trade_value, offering_items)
        scaled_vector = [scaled_vector[i] * max_scaling_factor for i in range(len(scaled_vector))]
        improvement = utility_improvement(scaled_vector, offering_A, offering_b, offering_items_original, reduction_idx=reduction_idx)
        return scaled_vector, max_scaling_factor, improvement

def find_scaling_factor(vector, max_trade_value, offering_items):
    """
        Given a trade vector and a maximum amount of a given item that can be traded, find a scaling factor for the trade

        Args:
            vector (np.array): n-dimensional vector representing the current offer.
            offering_items (np.array): Number of items that the offering agent has in its possession from categories that are being comsidered currently.
            max_trade_value (int): Maximum number of items that can be traded from any item category.
            
        Returns:
            max_scaling_factor (float): Scaling factor for the offer to trade the maximum item amount.
    """
    abs_vector = np.abs(vector)
    # Determine the maximum scaling factor given the maximum trade value
    max_scaling_factor = max_trade_value / max(abs_vector)
    for i in range(len(vector)):
        # Account for cases that lead to negative item values
        if offering_items[i] > 0 and vector[i] != 0:
            scaling_factor = max(0, -1 * offering_items[i] / vector[i])
            if scaling_factor != 0:
                max_scaling_factor = min(max_scaling_factor, scaling_factor)
    return max_scaling_factor

def find_scaling_responding(vector, item_val, item_index):
    """
        Find a scaling factor that scales a given vector to a given number of items
        Args:
            vector (np.array): n-dimensional vector representing the current offer.
            item_val (float): Target number of items to trade
            item_index (int): Item category we want to scale up to item_val.

        Returns:
            vector_mod (np.array): Vector scaled such that it is trading item_val from item_index
    """
    vector_mod = vector.copy()
    scaling_factor = item_val / vector[item_index]
    vector_mod = [num*scaling_factor for num in vector_mod]
    return vector_mod

def n_dim_quad_grad(A, b, x):
    """
        Find the gradient vector of an n-dimensional quadratic function of the form x^TAx + x^Tb

        Args:
            A (np.array): nxn matrix.
            b (np.array): n-dimensional vector.
            x (list): Current State.

        Returns:
            gradient (np.array): n-dimensional vector representing the gradient of the function at state x.
    """
    gradient = 2 * np.dot(A, x) + b
    return gradient

def intersection_between_hyperplanes(hyperplanes):
    """
        Given a set of hyperplanes, find the intersection point beteween the hyperplanes.
        Args:
            hyperplanes (list of tuples): Set of hyperplanes of the form ax = b

        Returns:
            intersection (np.array): point representing the intersection of the hyperplanes
    """
    num_hyperplanes = len(hyperplanes)
    dim = len(hyperplanes[0][0])

    # Initialize coefficient matrix and constant vector
    A = np.zeros((num_hyperplanes, dim))
    B = np.zeros(num_hyperplanes)

    # Populate coefficient matrix and constant vector ax = b
    for i, (normal, constant) in enumerate(hyperplanes):
        A[i] = normal
        B[i] = constant
    intersection = scipy.linalg.solve(A, B)

    return intersection

def is_in_intersection(point, halfspaces):
    """
        Check if a point is in the intersection of the given set of halfspaces
        Args:
            point (np.array): point
            halfspaces (list of tuples): Set of halfspaces of the form ax >= b

        Returns:
            (bool): Whether the point is in the intersection of the halfspaces
    """
    tolerance = 1e-10
    for a, b in halfspaces:
        if not np.dot(a, point) - b > -1 * tolerance:
            return False
    return True

def cross_prod_check(vector1, vector2):
    """
        Use the cross product to check if two vectors are parallel
        Args:
            vector1 (np.array): vector
            vector2 (np.array): vector

        Returns:
            (bool): Whether the vectors are parallel
    """

    if len(vector1) != len(vector2):
        return False 
    
    if all(x == 0 for x in vector1) or all(x == 0 for x in vector2):
        return True

    ratio = None
    for i in range(len(vector1)):
        if vector1[i] != 0:
            if ratio is None:
                ratio = vector2[i] / vector1[i]
            elif vector2[i] / vector1[i] != ratio:
                return False
        elif vector2[i] != 0:
            return False

    return True

def parallel_check(offer_set):
    """
        Check if any two vectors in the given set are parallel
        Args:
            offer_set (list of np.array): set of vectors

        Returns:
            (bool): Any two of the vectors are parallel
    """
    if not all(len(vector) == len(offer_set[0]) for vector in offer_set):
        return False
    for i, vector1 in enumerate(offer_set):
        for vector2 in offer_set[i + 1:]:
            if cross_prod_check(vector1, vector2):
                return True
    return False

def generate_corner_points(halfspaces, num_dimensions):
    """
        Given a set of halfspaces, generate corner points of the polytope defined by the halfspaces
        Args:
            halfspaces (list of tuples): set of halfspaces of the form ax >= b
            num_dimensions (int): dimensionality of the halfspaces

        Returns:
            point_set (list of np.array): Set of corner points for the polytope
    """
    # Generate Corner Points for a set of halfspaces
    point_set = []
    # Iterate over combinations of n-1 halfspaces
    for halfspace_combination in list(combinations(range(len(halfspaces)), num_dimensions)):
        halfspace_set = [halfspaces[h] for h in halfspace_combination]
        a_set = [h[0] for h in halfspace_set]
        A = np.zeros((len(halfspace_set), len(halfspace_set[0][0])))
        for i, (normal, constant) in enumerate(halfspace_set):
            A[i] = normal

        # Check if any two halfspaces in the set are parallel (In such cases, corner points do not exist)
        if not parallel_check(a_set) and not np.isclose(np.linalg.det(A), 0):

            # Calculate the intersection point
            intersection_point = intersection_between_hyperplanes(halfspace_set)

            # Check that the point is within the given set of halfspaces
            if is_in_intersection(intersection_point, halfspaces):
                intersection_tuple = tuple(intersection_point)
                if intersection_tuple not in map(tuple, point_set):
                    point_set.append(intersection_point)
    return point_set

def generate_hypercube(radius_val, basis_set):
    """
        Given a radius of a circle centered at the origin and a basis set, create a hypercube that encloses the circle
        Args:
            radius_val (float): radius of the circle
            basis_set (list of np.array): Set of basis vectors for the space
        Returns:
            hypercube_halfspace_set (list of tuples): Set of halfspaces of the form ax >= b that define the hypercube
    """
    hypercube_halfspace_set = []
    len_val = len(basis_set)

    # Iterate over dimensions
    for i in range(0, len_val):
        output_vals = np.zeros(len_val)
        output_vals[i] = 1

        # Format of halfspace contrants is ax >= b
        for sign in [-1, 1]:
            a = sign * output_vals
            b = -1 * radius_val
            hypercube_halfspace_set.append((a, b))
    return hypercube_halfspace_set

def qr_decomposition(normal_vector):
    """
        Use QR decomposition to obtain a basis set of the orthogonal space of a given vector
        Args:
            normal_vector (np.array): normal vector to the basis set.
        Returns:
            basis_vectors (np.array): Set of basis vectors that map from n-dimensional space to the n-1 dimensional orthogonal space of the normal_vector.
    """
    Q, _ = np.linalg.qr(np.column_stack([normal_vector] + [np.random.randn(len(normal_vector)) for _ in range(len(normal_vector) - 1)]))
    basis_vectors = np.array(Q[:, 1:])

    return basis_vectors.T

def vector_projection(v, basis_vectors):
    """
        Project a vector into the space defined by a set of basis vectors
        Args:
            v (np.array): n-dimensional vector to be projected onto the n-1 dimensional space
            basis_vectors (np.array): Set of basis vectors that map from n-dimensional space to the n-1 dimensional orthogonal space of the normal_vector.
        Returns:
            projection (np.array): n-1 dimensional projection of v onto the space defined by the basis vectors.
    """
    projection = []
    for basis in basis_vectors:
        norm_b_squared = np.linalg.norm(basis) ** 2
        # print(basis_vectors, v)
        proj_component = np.dot(v, basis) / norm_b_squared
        projection.append(proj_component)
    return projection

def generate_halfspaces(offer_set, basis, center_of_cone):
    """
        Generate halfspaces given a set of offers
        Args:
            offer_set (list of np.array): set of n-dimensional offers
            basis (np.array): Set of basis vectors that map from n-dimensional space to the n-1 dimensional orthogonal space of the normal_vector.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
        Returns:
            halfspace_set (list of tuples): Set of halfspaces corresponding to the set of offers.
    """
    halfspace_set = []
    for offer in offer_set:
        halfspace_set.append(calc_projected_halfspace(offer, basis, center_of_cone))        
    return halfspace_set

def calc_projected_halfspace(offer, basis, center_of_cone):
    """
        Given an n-dimensional offer, generate a corresponding halfspace constraint in the n-1 dimensional null space of the cone center defined by a set of basis vectors.
        Args:
            offer (np.array): n-dimensional offers
            basis (np.array): Set of basis vectors that map from n-dimensional space to the n-1 dimensional orthogonal space of the normal_vector.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
        Returns:
            (projected_vector, b_val) (list of tuples): Halfspace corresponding to the offer in the form projected_vector x >= b_val.
    """
    # Transform an offer into a halfspace contraint. ax >= b

    # The offer's direction is the normal vector the halfspace
    offer_norm = offer / np.linalg.norm(offer)
    # print("Checking Vector Projection: ", offer, offer_norm)
    projected_vector = np.array(vector_projection(offer_norm, basis))

    # Halfspace offset is determined by the angle offset of the offer and the orthogonal plane
    angle_offset = angle_between(offer_norm, center_of_cone)
    angle_minus_90 = angle_offset - (np.pi / 2)
    b_val = np.tan(angle_minus_90) * np.linalg.norm(projected_vector)

    return (projected_vector, b_val)

def rotate_vector(r_vector, d_vector, theta):
    """
        Rotate a vector in a given direction by an angle theta
        Args:
            r_vector (np.array): Vector to be rotated.
            d_vector (np.array): Rotation direction.
            theta (float): Rotation angle in radians.
        Returns:
            (np.array): r_vector rotated in the direction of d_vector by an angle theta
    """
    # Gram-Schmidt orthogonalization
    n1 = r_vector / np.linalg.norm(r_vector)
    v2 = d_vector - np.dot(n1,d_vector) * n1
    n2 = v2 / np.linalg.norm(v2)
    # print("Rotating Vector: ", r_vector, d_vector, n1, np.dot(n1,d_vector), v2, n2)
    # rotation by pi/2
    a = theta
        
    I = np.identity(len(n2))
        
    R = I + (np.outer(n2,n1) - np.outer(n1,n2)) * np.sin(a) + ( np.outer(n1,n1) + np.outer(n2,n2)) * (np.cos(a)-1)

    # check result
    return np.matmul(R,n1)

def is_separating(hyperplane, points):
    """
        Determine if a hyperplane is separating two points
        Args:
            hyperplane (tuple): Hyperplane of the form ax = b
            points (np.array, np.array): The two points to be separated.
        Returns:
            (bool): Whether the hyperplane is separating the two points
    """
    tolerance = 1e-10

    # Check if the two points are on different sides of the hyperplane
    val_1 = np.dot(points[0], hyperplane[0]) - hyperplane[1]
    val_2 = np.dot(points[1], hyperplane[0]) - hyperplane[1]
    if np.abs(val_1) <= tolerance or np.abs(val_2) <= tolerance:
        return False
    if np.sign(val_1) != np.sign(val_2):
        return True
    else:
        return False


def farthest_points(points):
    """
        Given a set of points, return the two points that are farthest apart
        Args:
            points (list of np.array): Set of two points
        Returns:
            tuple:
                - farthest_pair (tuple): The two farthest points
                - max_distance (float): Distance between the two points
    """
    # Given a set of points, return the two points that are farthest apart
    max_distance = 0
    farthest_pair = []

    # Iterate through all pairs of points
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i != j:  # Exclude comparing the same point
                distance = np.linalg.norm(point1 - point2)
                if distance > max_distance:
                    max_distance = distance
                    farthest_pair = (point1, point2)

    return farthest_pair, max_distance

def calculate_new_cone_integer_contstrained(offer_set, hypercube, basis_set, center_of_cone):
    """
        Given a set of integer offers and the hypercube enclosing the current cone, determine a new cone of potential gradients.
        Args:
            offer_set (list of np.array): Set of past integer offers
            hypercube (list of tuples): Halfspace constraints corresponding to the hypercube that encloses the current cone
            basis_set (np.array): Set of basis vectors that map from n-dimensional space to the n-1 dimensional orthogonal space of the normal_vector.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
        Returns:
            tuple:
                - center_of_circle (np.array): n-1 dimensional center of the hypersphere that encloses the space of potential gradients
                - potential_new_theta (float): Angle of the cone that encloses the hypersphere
                - corner_points (list of np.array): Farthest corner points of the polytope of potential gradients
                - (bool): Boolean that signals if the polytope of potential gradients is empty.
    """
    num_dim = len(basis_set)

    # Generate Corner Points of the Polyhedron
    # print("Offer Set: ", offer_set)
    halfspace = generate_halfspaces(offer_set, basis_set, center_of_cone)
    full_halfspace_set = halfspace + hypercube
    point_set = generate_corner_points(full_halfspace_set, num_dim)
    corner_points, point_dist = farthest_points(point_set)

    # If the halfspace contraints do not allow for corner points, return an error case
    if len(corner_points) == 0:
        return None, None, None, True
    # print("Halfspaces and Corner Points: ", full_halfspace_set, corner_points)
    # Calculate new circle parameters
    center_of_circle = np.mean(corner_points, axis=0)
    radius_of_circle = (point_dist)/2 
    radius_of_circle = np.sqrt(3) * radius_of_circle
    # print("Corner Points: ", full_halfspace_set, corner_points)
    if all(c == 0 for c in center_of_circle):
        center_norm = np.zeros(num_dim)
        center_norm[0] = 1
    else:
        center_norm = center_of_circle / np.linalg.norm(center_of_circle)

    # Calculate new angle of opening (Theta)
    point_a = center_of_circle - (radius_of_circle * center_norm)
    point_b = center_of_circle + (radius_of_circle * center_norm)
    d_a = np.sqrt(1 + (np.linalg.norm(point_a)**2))
    d_b = np.sqrt(1 + (np.linalg.norm(point_b)**2))
    diameter = point_dist
    ratio = ((diameter**2) - (d_a**2) - (d_b**2))/(-2*d_a*d_b)
    # print("Point Distance Calculation: ", point_a, point_b, d_a, d_b, diameter)
    potential_new_theta = np.arccos(ratio) / 2
    # print("Potential New Theta: ", center_of_circle, potential_new_theta)
    return center_of_circle, potential_new_theta, corner_points, False


def calculate_new_cone(offer_set, theta, center_of_cone, num_items):
    """
       Given a set of fractional offers and the current cone, determine a new cone of potential gradients.
       Args:
           offer_set (list of np.array): Set of past fractional offers
           theta (float): semi-vertical angle of the current cone
           center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
           num_items (int): Total number of item categories
        Returns:
           tuple:
               - new_center_of_cone (np.array): Direction of the new cone of potential gradients
               - potential_new_theta (float): Angle of the new cone of potential gradients
   """
    # Return New Cone Given the current cuts (vector set)
    w_list = [center_of_cone]
    sum_value = np.zeros(num_items)

    # Use cone update rule to determine the new cone of potential gradients
    for i in range(0, num_items-1):
        w_i = center_of_cone * np.cos(theta) + (offer_set[i]/np.linalg.norm(offer_set[i])) * np.sin(theta)
        w_list.append(np.array(w_i))
        sum_value += (w_i / num_items)
    new_center_of_cone = (sum_value / np.linalg.norm(sum_value))
    scaling_factor_theta = np.sqrt((2 * num_items - 1) / (2 * num_items))
    potential_new_theta = np.arcsin(scaling_factor_theta * np.sin(theta))
    return new_center_of_cone, potential_new_theta

def obtain_full_offer(offer, reduction_idx, full_size):
    """
        Given an offer that may be reduced by removing item categories with zero items, return the full offer

        Args:
            offer (np.array): Reduced offer.
            reduction_idx (list): set of item indices that need to be removed from consideration.
            full_size (int): Total number of item categories

        Returns:
            full_offer (np.array): Vector representing the full trade offer. Item categories that are not considered are filled in with 0 values.
    """
    idx_counter = 0
    full_offer = np.zeros(full_size)
    for i in range(0, full_size):
        if i not in reduction_idx:
            full_offer[i] = offer[idx_counter]
            idx_counter += 1
    # print("Returning from obtain_full_offer: ", offer, reduction_idx, full_size, full_offer)
    return full_offer

def round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value):
    """
        Given a fractional offer, return an integer offer that benefits the offering agent

        Args:
            offer (np.array): Fractional offer
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
            offering_gradient (np.array): n-dimensional vector corresponding to the offering agent's gradient.
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            offering_items (np.array):List of current items with excluded categories for the offering agent
            offering_items_original (np.array):List of current items from all categories for the offering agent
            reduction_idx (np.array): List of item categories to be excluded from trading
            int_constrained (bool): Whether the trade should be restricted to integer values. Defaults to True.
            max_trade_value (int): Maximum number of items that can be traded from any item category

        Returns:
            tuple:
                - offer (np.array): Rounded Offer
                - improvement (boolean): Whether the offering agent benefits fromt he offer
    """
    # Round the offer to contain only integer values
    offer = branch_and_bound(offer, center_of_cone, offering_gradient)
    full_offer = obtain_full_offer(offer, reduction_idx, len(offering_items_original))

    # Scale the offer in accordance with the offering's items
    while any(element < 0 for element in offering_items_original + full_offer):
        offer, scaling_factor, improvement = find_scaling_offering(offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
        offer = branch_and_bound(offer, center_of_cone, offering_gradient)
        full_offer = obtain_full_offer(offer, reduction_idx, len(offering_items_original))
    improvement = utility_improvement(offer, offering_A, offering_b, offering_items_original, reduction_idx=reduction_idx)

    return offer, improvement


def obtain_responding_offer(offer, responding_items_original, reduction_idx):
    """
        Given an offer that may not be for the responding agent, return a scaled down offer that is feasible

        Args:
            offer (np.array): Offer from the
            responding_items_original (np.array):List of current items from all categories for the responding agent
            reduction_idx (np.array): List of item categories to be excluded from trading
        Returns:
            tuple:
                - neg_offer_q (np.array): Scaled down offer for the responding agent
    """
    neg_offer = [-1 * x for x in offer]
    neg_offer_q = obtain_full_offer(neg_offer, reduction_idx, len(responding_items_original))
    print("Checking After Obtain_full_offer: ", neg_offer, neg_offer_q)
    next_step = neg_offer_q + responding_items_original

    # Scale down the offer to match the responding agent's items
    if not all(i >= 0 for i in next_step):
        flag_n_items = False
        min_index = np.argmin(next_step)
        min_items = responding_items_original[min_index]
        neg_offer_q[min_index] = -1 * min_items
    return neg_offer_q

def offer_search(offering_A, offering_b, responding_A, responding_b, offering_items_original, responding_items_original, num_items,
                 center_of_cone, theta, item_list, ui, theta_closeness=0.00001, int_constrained=True, prev_offer=[],
                 prev_offer_flag=True, max_trade_value=5, first_trade_attempt=False):
    """
        Use ST-CR to find a mutually beneficial offer
        Args:
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items_original (np.array): List of current items for the offering agent across all item categories.
            responding_items_original (np.array): List of current items for the responding agent across all item categories.
            center_of_cone (np.array): n-dimensional vector corresponding to the current center of the cone of potential gradients.
            theta (float): Angle of the cone of potential gradients in radians.
            num_items (int): Total number of item categories.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            theta_closeness (float): Hyperparamter used by ST-CR to stop trading.
            prev_offer (np.array): Previously accepted offer used for heuristic trading. Default: Empty
            prev_offer_flag (bool, optional): Whether ST-CR will use the previously accepted trade heuristic
            int_constrained (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            
        Returns:
            tuple:
                - found_trade (bool): Whether a mutually beneficial offer was found
                - offer (np.array): The mutually beneficial offer (if found)
                - offer_count (int): Number of offers made to the responding agent,
                - iterations (int): Number of cone refinements
                - center_of_cone (np.array): n-dimensional vector corresponding to the center of the cone of potential gradients after searching for a mutually beneficial offer.
                - theta (float): Angle (in radians) of the cone of potential gradients after trading.
                - edge_case_break (bool): Whether ST-CR stopped trading due to an edge case.
    """
    # Initialize Gradient Values (The responding agent's gradient is only used with respect to the query function)
    original_responding_gradient = list(n_dim_quad_grad(responding_A, responding_b, responding_items_original))
    original_offering_gradient = list(n_dim_quad_grad(offering_A, offering_b, offering_items_original))

    # Initialize List of Past Offers:
    offer_list = []
    waiting_times = []
    # Account for cases where the number of items for a given category are zero
    new_grad_h = list(original_responding_gradient.copy())
    new_grad_a = list(original_offering_gradient.copy())
    responding_items_mod = list(responding_items_original)
    offering_items_mod = list(offering_items_original)
    reduction_idx = []
    center_list = list(center_of_cone)


    # Remove Item Categories that have zero items
    for i in range(len(offering_items_original)):
        if offering_items_original[i] == 0 or responding_items_original[i] == 0:
            reduction_idx.append(i)
    reduction_num = len(reduction_idx)
    prev_offer = list(prev_offer)
    for i in sorted(reduction_idx, reverse=True):
        a = new_grad_a.pop(i)
        h = new_grad_h.pop(i)
        prev_trade = prev_offer.pop(i)
        item_a = offering_items_mod.pop(i)
        item_h = responding_items_mod.pop(i)
        c = center_list.pop(i)
    center_of_cone = np.array(center_list)
    offering_items = np.array(offering_items_mod)
    offering_gradient = new_grad_a / np.linalg.norm(new_grad_a)

    # Initialize Feedback Variables
    prev_counteroffer = []
    prev_counteroffer_flag = False
    # Initialize iteration variables
    offer_count = 0
    iterations = 0
    edge_case_break = False
    # If there is only one item left to trade, end the cone refinement.
    if reduction_num >= num_items-1:
        edge_case_string = "The algorithm has ended trading, please press the Stop Trading button to end the scenario"
        prompt_file = os.path.join(ui, 'offer.txt')
        with open(prompt_file, 'w') as f:
            f.write(edge_case_string)
        return False, [], offer_count, iterations, center_of_cone, theta, edge_case_break, True, offer_list, waiting_times
    # If the offering's gradient is zero, end the cone refinement
    if all(grad_entry == 0 for grad_entry in new_grad_a):
        edge_case_string = "The algorithm has ended trading, please press the Stop Trading button to end the scenario"
        prompt_file = os.path.join(ui, 'offer.txt')
        with open(prompt_file, 'w') as f:
            f.write(edge_case_string)
        return False, [], offer_count, iterations, center_of_cone, theta, edge_case_break, True, offer_list, waiting_times
    remaining_items = num_items - reduction_num
    # Make Offers Based on Previously accepted trade offers:
    heuristic_offers = []
    if prev_offer_flag:
        if len(prev_offer) != 0:
            if not all(item == 0 for item in prev_offer):
                prev_info_offer = prev_offer / np.linalg.norm(prev_offer)
                offer, scaling_factor, improvement = find_scaling_offering(prev_info_offer, offering_A, offering_b,
                                                                           offering_items, offering_items_original,
                                                                           reduction_idx=reduction_idx,
                                                                           int_constrained=int_constrained, max_trade_value=max_trade_value)
                if int_constrained:
                    offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                heuristic_offers.append(offer)
                rejected_dot_product = False
                while not rejected_dot_product and improvement == True:
                    neg_offer_q = obtain_responding_offer(offer, responding_items_original, reduction_idx)
                    if offer_count == 0 and not first_trade_attempt:
                        prev_trade_accepted = True
                    else:
                        prev_trade_accepted = False
                    offer_list.append(obtain_full_offer(offer, reduction_idx, num_items))
                    print("Checking Offer: ", neg_offer_q)
                    response_product, counteroffer, stopped_trading, offer_time, response_time = query_gpt_with_counteroffer(neg_offer_q,
                                                                                                  item_list, ui, responding_items_original, prev_trade_accepted=prev_trade_accepted, prev_counteroffer_flag=prev_counteroffer_flag, prev_counteroffer=prev_counteroffer)
                    print("Checking Counteroffer: ", counteroffer)
                    prev_counteroffer_flag = False
                    prev_counteroffer = []
                    waiting_times.append((offer_time, response_time))
                    offer_count += 1
                    if stopped_trading:
                        return False, None, offer_count, iterations, center_of_cone, theta, edge_case_break, stopped_trading, offer_list, waiting_times
                    if response_product:
                        offer = obtain_full_offer(offer, reduction_idx, len(offering_items_original))
                        return True, offer, offer_count, iterations, center_of_cone, theta, edge_case_break, False, offer_list, waiting_times
                    else:
                        if len(counteroffer) != 0:
                            neg_counteroffer = -1 * np.array(counteroffer)
                            reduced_counteroffer = np.array(
                                [x for i, x in enumerate(counteroffer) if i not in reduction_idx])
                            response_product, counteroffer = query_with_counteroffer(neg_counteroffer, offering_A,
                                                                                     offering_b,
                                                                                     offering_items_original, max_trade_value=max_trade_value)
                            if response_product:
                                log_string = f"ST-CR:  Counteroffer Direction is Beneficial \n"
                                log_file = os.path.join(ui, 'log.txt')
                                log_opened = open(log_file, "a")
                                log_opened.write(log_string)
                                log_opened.close()
                                offer = -1 * reduced_counteroffer
                                continue
                            else:
                                log_string = f"ST-CR:  Counteroffer Direction is not Beneficial. Incorporating feedback into Gradient Cone \n"
                                log_file = os.path.join(ui, 'log.txt')
                                log_opened = open(log_file, "a")
                                log_opened.write(log_string)
                                log_opened.close()
                                prev_counteroffer_flag = True
                                prev_counteroffer = -1 * neg_counteroffer
                                rejected_dot_product = True
                        else:
                            rejected_dot_product = True

        # Determine Initial Quadrant of the responding agent's gradient
    if theta >= np.pi / 2:
        quadrant = np.zeros(num_items - len(reduction_idx))
        i = 0
        og_offer = np.zeros(num_items - len(reduction_idx))
        og_offer[i] = 1
        offer, scaling_factor, improvement = find_scaling_offering(og_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
        # Round the offer to integer values
        if int_constrained:
            offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
        while i < num_items - len(reduction_idx):
            # Make the offer to the responding
            neg_offer_q = obtain_responding_offer(offer, responding_items_original, reduction_idx)

            # Make the offer to the responding
            if offer_count == 0 and not first_trade_attempt:
                prev_trade_accepted = True
            else:
                prev_trade_accepted = False
            offer_list.append(obtain_full_offer(offer, reduction_idx, num_items))
            print("Offer: ", neg_offer_q)
            response_product, counteroffer, stopped_trading, offer_time, response_time = query_gpt_with_counteroffer(neg_offer_q,
                                                                                            item_list, ui,
                                                                                            responding_items_original, prev_trade_accepted=prev_trade_accepted, prev_counteroffer_flag=prev_counteroffer_flag, prev_counteroffer=prev_counteroffer)
            print("Counteroffer: ", counteroffer)
            prev_counteroffer_flag = False
            prev_counteroffer = []
            waiting_times.append((offer_time, response_time))
            offer_count += 1
            # Return the accepted offer
            if stopped_trading:
                return False, None, offer_count, iterations, center_of_cone, theta, edge_case_break, stopped_trading, offer_list, waiting_times
            if response_product:
                offer = obtain_full_offer(offer, reduction_idx, len(offering_items_original))
                return True, offer, offer_count, iterations, center_of_cone, theta, edge_case_break, False, offer_list, waiting_times
            else:
                if len(counteroffer) != 0:
                    # print("Counteroffer: ", counteroffer)
                    neg_counteroffer = -1 * np.array(counteroffer)
                    reduced_counteroffer = np.array([x for i, x in enumerate(counteroffer) if i not in reduction_idx])
                    # print("Checking Neg Counteroffer: ", counteroffer, neg_counteroffer)
                    response_product, counteroffer = query_with_counteroffer(neg_counteroffer, offering_A,
                                                                                offering_b,
                                                                                offering_items_original,
                                                                                max_trade_value=max_trade_value)
                    if response_product:
                        # print("Direction Beneficial")
                        log_string = f"ST-CR:  Counteroffer Direction is Beneficial \n"
                        log_file = os.path.join(ui, 'log.txt')
                        log_opened = open(log_file, "a")
                        log_opened.write(log_string)
                        log_opened.close()
                        offer = -1 * reduced_counteroffer
                        continue
                    else:
                        log_string = f"ST-CR:  Counteroffer Direction is not Beneficial. Incorporating feedback into Gradient Cone \n"
                        log_file = os.path.join(ui, 'log.txt')
                        log_opened = open(log_file, "a")
                        log_opened.write(log_string)
                        log_opened.close()
                        prev_counteroffer_flag = True

                        prev_counteroffer = -1 * neg_counteroffer
                        if offer[i] < 0:
                            quadrant[i] = -1
                        else:
                            quadrant[i] = 1
                        i += 1
                        if i < num_items - len(reduction_idx):
                            og_offer = np.zeros(num_items - len(reduction_idx))
                            og_offer[i] = 1
                            offer, scaling_factor, improvement = find_scaling_offering(og_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
                            # Round the offer to integer values
                            if int_constrained:
                                offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                            continue
                else:
                    if offer[i] < 0:
                        quadrant[i] = -1
                    else:
                        quadrant[i] = 1
                    i += 1
                    if i < num_items - len(reduction_idx):
                        og_offer = np.zeros(num_items - len(reduction_idx))
                        og_offer[i] = 1
                        offer, scaling_factor, improvement = find_scaling_offering(og_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
                        # Round the offer to integer values
                        if int_constrained:
                            offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                        continue
                    
            # If all offers were rejected, initialize the center of cone and angle
        center_of_cone = (1 / np.sqrt(num_items - len(reduction_idx))) * quadrant
        theta = np.arccos(1 / np.sqrt(num_items - len(reduction_idx)))

    # Generate basis set and hypercube for Calculating Next Cone Update
    basis_set = qr_decomposition(center_of_cone)
    hypercube = generate_hypercube(np.abs(np.tan(theta)) , basis_set)
    found_trade = False
    while found_trade == False and edge_case_break == False and theta >= theta_closeness:
        # Generate an initial offer
        offering_gradient_norm = offering_gradient/np.linalg.norm(offering_gradient)
        center_of_cone_norm = center_of_cone/np.linalg.norm(center_of_cone)

        # If the offering gradient is exactly aligned with the center of the cone, then we cannot generate beneficial offers for the offering
        # Return as an edge case
        # diff_vector = offering_gradient_norm - center_of_cone_norm
        # if all(entry == 0 for entry in diff_vector):
        #     print("Edge Case 1")
        #     edge_case_string = "The algorithm has ended trading, please press the Stop Trading button to end the scenario"
        #     prompt_file = os.path.join(ui, 'offer.txt')
        #     with open(prompt_file, 'w') as f:
        #         f.write(edge_case_string)
        #     edge_case_break = True
        #     refined_cone = True
        #     break

        # Find an initial offer by determining the orthogonal offer (to the cone center) that is closest to the offering's gradient
        original_offer = find_init_offer_random(offering_gradient, center_of_cone)

        # Determine the other orthogonal vectors that will be used for cone refinement
        orthogonal_vectors = find_orth_vector(np.array([center_of_cone, original_offer]), offering_gradient)
        orthogonal_vectors = sort_orth_vectors(offering_A, offering_b, offering_items_original, orthogonal_vectors, reduction_idx=reduction_idx)

        # Scale up the offer while accounting for the items the offering possesses
        offer, scaling_factor, improvement = find_scaling_offering(original_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
        
        # Round the offer to integer values
        if int_constrained:
            offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)

        # Initialize set of queries used to determine the updated center of the cone
        i = 0
        offer_set = []
        offer_set_og = []
        offer_set_og.append(center_of_cone)
        offer_set.append(center_of_cone)

        # Refine the cone using n-1 requests
        refined_cone = False
        
        while not refined_cone:
            # Present the trade offer to the responding
            neg_offer_q = obtain_responding_offer(offer, responding_items_original, reduction_idx)
            if offer_count == 0 and not first_trade_attempt:
                prev_trade_accepted = True
            else:
                prev_trade_accepted = False
            # print("Checking Offer Parsing: ", neg_offer, reduction_idx, neg_offer_q)
            offer_list.append(obtain_full_offer(offer, reduction_idx, num_items))
            response_product, counteroffer, stopped_trading, offer_time, response_time = query_gpt_with_counteroffer(neg_offer_q, item_list,
                                                                                          ui, responding_items_original, prev_trade_accepted=prev_trade_accepted, prev_counteroffer_flag=prev_counteroffer_flag, prev_counteroffer=prev_counteroffer)
            prev_counteroffer_flag = False
            prev_counteroffer = []
            waiting_times.append((offer_time, response_time))
            # print("Counteroffer After Offer:", counteroffer)
            offer_count += 1
            if stopped_trading:
                log_file = os.path.join(ui, 'log.txt')
                log_opened = open(log_file, "a")
                log_opened.write("Trading Ended By User \n")
                log_opened.close()
                return False, None, offer_count, iterations, center_of_cone, theta, edge_case_break, stopped_trading, offer_list, waiting_times
            # If the responding accepted the offer, but it wasn't beneifical for the offering offering, negate the offer to ensure that the appropriate direction is maintained.
            if response_product == True:
                found_trade = True
                refined_cone = True
                out_offer = offer.copy()
                offer = -1 * np.array(offer)
            if len(counteroffer) != 0:
                original_counteroffer = np.array(counteroffer).copy()
                mod_counteroffer = np.array([x for i, x in enumerate(original_counteroffer) if i not in reduction_idx])
                # print("Counteroffer After Offer:", original_counteroffer, mod_counteroffer, reduction_idx)
                neg_counteroffer = -1 * np.array(counteroffer)
                response_product, counteroffer = query_with_counteroffer(neg_counteroffer, offering_A, offering_b,
                                                                         offering_items_original)
                if response_product:
                    # print("Counteroffer Direction is Beneficial")
                    log_string = f"ST-CR:  Counteroffer Direction is Beneficial \n"
                    log_file = os.path.join(ui, 'log.txt')
                    log_opened = open(log_file, "a")
                    log_opened.write(log_string)
                    log_opened.close()
                    offer = -1 * mod_counteroffer
                    continue
                else:
                    if not any(np.array_equal(mod_counteroffer, v) for v in offer_set) and not np.array_equal(mod_counteroffer, offer):
                        offer_set.append(mod_counteroffer)
                    # print("Counteroffer Direction is not Beneficial")

                    prev_counteroffer = -1 * neg_counteroffer
                    prev_counteroffer_flag = True

                    log_string = f"ST-CR:  Counteroffer Direction is not Beneficial. Incorporating feedback into Gradient Cone \n"
                    log_file = os.path.join(ui, 'log.txt')
                    log_opened = open(log_file, "a")
                    log_opened.write(log_string)
                    log_opened.close()

            if any(np.array_equal(offer, v) for v in offer_set):
                    # print("Break Case 1: ", offer, offer_set)
                    print("Edge Case 2")
                    edge_case_string = "The algorithm has ended trading, please press the Stop Trading button to end the scenario"
                    prompt_file = os.path.join(ui, 'offer.txt')
                    with open(prompt_file, 'w') as f:
                        f.write(edge_case_string)
                    edge_case_break = True
                    refined_cone = True
                    break
                # Add the offer to the set of previous offers used to refine the cone
            else:
                offer_set.append(offer)
                offer_set_og.append(original_offer)
            potential_new_theta = np.inf
            i += 1
            
            # If we have made n-1 offer, we can begin refining the cone
            if i >= remaining_items - 1:
                # Calculate the new cone using the prior offers
                if int_constrained:
                    center_of_circle, potential_new_theta, corner_points, error_case = calculate_new_cone_integer_contstrained(offer_set[1:], hypercube, basis_set, center_of_cone)
                    # print("Checking Theta:", potential_new_theta, theta)
                    # If there are no corner points enclosed by the cone, we have made an incorrect cut in a prior trade and we must increase the angle to account for the lost trade directions.
                    while error_case == True:
                        theta += 0.01
                        hypercube = generate_hypercube(np.abs(np.tan(theta)), basis_set)
                        center_of_circle, potential_new_theta, corner_points, error_case = calculate_new_cone_integer_contstrained(
                            offer_set[1:], hypercube, basis_set, center_of_cone)
                else:
                    # Without integer constraints
                    potential_center_of_cone, potential_new_theta = calculate_new_cone(offer_set[1:], theta, center_of_cone, num_items - len(reduction_idx))
                # If the set of contraints does not contain any corner points, it is likely due to a sign error in a prior refinement.
                # In such cases, we increase the cone's angle of opening and perform the calculation again.

                # If the new cone is smaller than the current cone, we can make a cone refinement
                if potential_new_theta < theta and not all(c == 0 for c in center_of_circle) :
                    if int_constrained:
                        # print("New Theta: ", theta, potential_new_theta)
                        theta = potential_new_theta
                        # Obtain a new center of the cone by rotating the old cone center in the direction of the circle center
                        rotation_direction_n = center_of_circle @ basis_set
                        new_center_of_cone = rotate_vector(center_of_cone, rotation_direction_n, np.arctan(np.linalg.norm(center_of_circle)))
                        center_of_cone = new_center_of_cone
                        # Obtain the basis and hypercube for the new cone
                        basis_set = qr_decomposition(center_of_cone)
                        hypercube = generate_hypercube(np.abs(np.tan(theta)), basis_set)
                        refined_cone = True
                    else:
                        # Make an unconstrained cone refinement
                        theta = potential_new_theta
                        center_of_cone = potential_center_of_cone
                        basis_set = qr_decomposition(center_of_cone)
                        hypercube = generate_hypercube(np.abs(np.tan(theta)), basis_set)
                        refined_cone = True

                
                # If the new cone is not smaller than the prior cone, we must find a new offer to refine the cone.
                else:
                    # The new offer is obtained by finding the hyperplane that separates the two corner points.
                    # Equations found here: https://math.stackexchange.com/questions/933266/compute-the-bisecting-normal-hyperplane-between-two-n-dimensional-points
                    
                    bisecting_hyperplane_a = (corner_points[0] - corner_points[1])/np.linalg.norm(corner_points[0] - corner_points[1]) 
                    bisecting_hyperplane_b = ((np.linalg.norm(corner_points[0])**2) - (np.linalg.norm(corner_points[1])**2))/(2 * (np.linalg.norm(corner_points[0] - corner_points[1])))
                    # print("Check Bisection: ", bisecting_hyperplane_a, bisecting_hyperplane_b)
                    dist_from_origin = np.abs(bisecting_hyperplane_b) / np.linalg.norm(bisecting_hyperplane_a)
                    theta_offset = np.arctan(dist_from_origin)
                    if bisecting_hyperplane_b >= 0:
                        theta_offset = -1 * theta_offset

                    # Once the hyperplane that bisects the two farthest corner points is obtained, we turn it into an n-dimesional offer
                    offer_in_n_dimensions = bisecting_hyperplane_a @ basis_set
                    # Rotate the offer to account for offset from origin in n-1 dimensions
                    original_offer = rotate_vector(offer_in_n_dimensions, center_of_cone, theta_offset)
                    # Scale and round the offer
                    scaled_offer, scaling_factor, improvement = find_scaling_offering(original_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
                    if int_constrained:
                        offer, improvement = round_offer(scaled_offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                    # print("Rounded Original Offer: ", offer, scaled_offer, calc_projected_halfspace(offer, basis_set, center_of_cone))
                    # Due to rounding, the bisecting offer may no longer separate the corner points.
                    # In this case, we must scale up the offer to reduce the impact of roudning on the trade direction.
                    mtv = 5
                    last_offer = offer.copy()
                    while not is_separating(calc_projected_halfspace(offer, basis_set, center_of_cone), corner_points):
                        
                        # Scale up and round the offer
                        mtv *= 2
                        #### CHECK THIS WITH MUSTAFA
                        offer, scaling_factor, improvement = find_scaling_offering(original_offer, offering_A, offering_b, offering_items, offering_items_original, mtv, need_improvement=False, reduction_idx=reduction_idx)
                        if int_constrained:
                            offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                        # print("Rounded Original Offer: ", offer, scaled_offer, calc_projected_halfspace(offer, basis_set, center_of_cone))
                        # If the offer doesn't change after being scaled up, it means that there is insufficiant quantities of each item to be scaled furthur.
                        # In such scenarios, we can no longer refine the cone, and we exit the algorithm
                        if np.array_equal(offer, last_offer) or mtv > 200:
                            print("Edge Case 3")
                            edge_case_string = "The algorithm has ended trading, please press the Stop Trading button to end the scenario"
                            prompt_file = os.path.join(ui, 'offer.txt')
                            with open(prompt_file, 'w') as f:
                                f.write(edge_case_string)
                            edge_case_break = True
                            refined_cone = True
                            break
                        else:
                            last_offer = offer.copy()
                    # print("Next Offer 1: ", offer)
                        
            # If we haven't made n-1 offers yet, we use the orthogonal offers to inform the next trade
            else:
                original_offer = orthogonal_vectors.pop()
                offer, scaling_factor, improvement = find_scaling_offering(original_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
                if int_constrained:
                    offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                    
                    # Ensure that the orthogonal offers are properly scaled after rounding
                    unscaled_offer = offer.copy()
                    for mtv in range(2, 6):
                        max_scaling_factor = find_scaling_factor(unscaled_offer, mtv, offering_items)
                        temp_offer = list(max_scaling_factor * np.array(unscaled_offer))
                        temp_offer_rounded = branch_and_bound(temp_offer, center_of_cone, offering_gradient)
                        after_comp_state = np.array(offering_items) + np.array(temp_offer_rounded)
                        improvement = utility_improvement(temp_offer_rounded, offering_A, offering_b, offering_items_original, reduction_idx=reduction_idx)
                        if improvement and not any(entry < 0 for entry in after_comp_state):
                            offer = temp_offer_rounded.copy()
                        else:
                            break
                # print("Next Offer 2: ", offer)

            # Increment the number cone refinement iterations
            iterations += 1
        if found_trade == True:
            full_offer = obtain_full_offer(out_offer, reduction_idx, len(offering_items_original))
            return True, full_offer, offer_count, iterations, center_of_cone, theta, edge_case_break, False, offer_list, waiting_times
    return False, None, offer_count, iterations, center_of_cone, theta, edge_case_break, False, offer_list, waiting_times


def run_trading_scenario_stcr_gpt(num_items, offering_A, offering_b, responding_A, responding_b, offering_items, responding_items,
                              item_list, ui_folder, integer_constraint=True, theta_closeness= 0.00001, max_trade_value=5):
    """
        Run a trading scenario using ST-CR and GPT
        Args:
            num_items (int): Total number of item categories
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items (np.array): Initial number of items in each category for the offering agent.
            responding_items (np.array): Initial number of items in each category for the responding agent.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            theta_closeness (float): Hyperparamter used by ST-CR to stop trading.
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
        Returns:
            log (list of dicts): Log of trading progression
        """
    center_of_cone = np.zeros(num_items)
    center_of_cone[0] = 1
    theta = np.pi / 2
    trade_num = 0
    beta = 0.01
    log = []
    prev_responding_states = []
    prev_offer = []
    error_flag = False
    while trade_num <= 100:
        if trade_num == 0:
            first_trade_attempt = True
        else:
            first_trade_attempt = False
        # Determine gradients of both parties
        trade_num += 1
        original_responding_gradient = list(n_dim_quad_grad(responding_A, responding_b, responding_items))
        original_offering_gradient = list(n_dim_quad_grad(offering_A, offering_b, offering_items))

        # Remove categories if an agent has all items in a given category
        reduction_idx = []
        for i in range(len(original_offering_gradient)):
            if offering_items[i] == 0 or responding_items[i] == 0:
                reduction_idx.append(i)

        # Find a mutually beneficial trade
        found_trade, out_offer, offer_count, iterations, center_of_cone, theta, edge_case_break, stopped_trading, offer_list, waiting_times = offer_search(
            offering_A, offering_b, responding_A, responding_b, np.array(offering_items), np.array(responding_items), num_items,
            center_of_cone, theta, item_list, ui_folder, int_constrained=integer_constraint, prev_offer=prev_offer, theta_closeness=theta_closeness, max_trade_value=max_trade_value, first_trade_attempt=first_trade_attempt)
        # if stopped_trading:
        #     break
        true_out_offer = out_offer
        true_center_of_cone = np.zeros(num_items)
        idx_counter = 0
        for i in range(0, num_items):
            if i not in reduction_idx:
                true_center_of_cone[i] = center_of_cone[idx_counter]
                idx_counter += 1
        prev_offer = true_out_offer
        center_of_cone = true_center_of_cone

        # Update the item values if a trade has been found
        information_dict = {}
        if found_trade == True and not stopped_trading:
            information_dict["found_trade"] = True

            # Increase theta after an accepted trade
            # Increase the cone's angle of opening to account for changes in the responding agent's gradient
            # FLAG HERE FOR MODIFICATION (TALK WITH MUSTAFA)
            theta = theta + beta * np.linalg.norm(true_out_offer)
            if theta > np.arccos(1 / np.sqrt(num_items)):
                theta = np.pi/2
            # else:
            #     if theta > np.arccos(1 / np.sqrt(num_items)):
            #         theta = np.arccos(1 / np.sqrt(num_items))

            # Save prior state for benefit calculation
            prev_responding_items = responding_items.copy()
            prev_responding_states.append(prev_responding_items.tolist())
            prev_offering_items = offering_items.copy()

            # Transition to a new state depending on the offer
            responding_items -= true_out_offer
            offering_items += true_out_offer

            # Calculate benefit values
            prev_state_value_h = prev_responding_items.transpose() @ responding_A @ prev_responding_items + responding_b @ prev_responding_items
            prev_state_value_a = prev_offering_items.transpose() @ offering_A @ prev_offering_items + offering_b @ prev_offering_items
            next_state_value_h = responding_items.transpose() @ responding_A @ responding_items + responding_b @ responding_items
            next_state_value_a = offering_items.transpose() @ offering_A @ offering_items + offering_b @ offering_items

            # Log trading progression information
            information_dict["responding_items"] = responding_items.tolist()
            information_dict["offering_items"] = offering_items.tolist()
            responding_state_string = f"New User State: "
            offering_state_string = f"New ST-CR State: "
            for item_idx in range(0, len(item_list)):
                responding_state_string += f"{responding_items[item_idx]} {item_list[item_idx]}, "
                offering_state_string += f"{offering_items[item_idx]} {item_list[item_idx]}"
            responding_state_string += "\n"
            offering_state_string += "\n"
            responding_benefit = f"Estimated User Benefit: {next_state_value_h - prev_state_value_h} \n"
            offering_benefit = f"ST-CR's Benefit: {next_state_value_a - prev_state_value_a} \n"
            result_string = f"Offer Accepted! \n"
            log_file = os.path.join(ui_folder, 'log.txt')
            log_opened = open(log_file, "a")
            log_opened.write(result_string)
            log_opened.write(responding_state_string)
            log_opened.write(offering_state_string)
            log_opened.write(responding_benefit)
            log_opened.write(offering_benefit)
            log_opened.close()

            if not (all(entry >= 0 for entry in offering_items)):
                trade_num = 101
                error_flag = True
            information_dict["offer_count"] = offer_count
            information_dict["responding_benefit"] = next_state_value_h - prev_state_value_h
            information_dict["offering_benefit"] = next_state_value_a - prev_state_value_a
            information_dict["edge_case"] = edge_case_break
            information_dict["waiting_times"] = waiting_times
            information_dict["offer_list"] = [list(offer) for offer in offer_list]
            score_dict = {}
            human_target = responding_b / 2
            score_dict["Initial State"] = [50, 50, 50]
            human_final = responding_items
            score_dict["Final State"] = [int(element) for element in human_final]
            score_dict["Target State"] = [int(element / 2) for element in responding_b]
            score = np.sum(np.abs(human_target - human_final)) / np.sum(np.abs(human_target - np.array([50, 50, 50])))
            if score > 1:
                score = 1
            score_dict["Score"] = float(1 - score)
            print("Score: ", score_dict)
            chat_folder = ui_folder.rsplit("/", 1)[0]
            with open(f'{chat_folder}/score.json', 'w') as score_file:
                json.dump(score_dict, score_file, indent=4)
            with open(f'{ui_folder}/score.json', 'w') as score_file:
                json.dump(score_dict, score_file, indent=4)
            # print("\n")
            log.append(information_dict)
            with open(f'{ui_folder}/numerical_log.json', 'w') as json_file:
                json.dump(log, json_file, indent=4)
        else:
            # print("Result: ", found_trade)
            information_dict["offer_count"] = offer_count
            information_dict["responding_items"] = responding_items.tolist()
            information_dict["found_trade"] = False
            information_dict["waiting_times"] = waiting_times
            information_dict["offer_list"] = [list(offer) for offer in offer_list]
            log.append(information_dict)
            with open(f'{ui_folder}/numerical_log.json', 'w') as json_file:
                json.dump(log, json_file, indent=4)
            trade_num = 101
        if not all(i >= 0 for i in responding_items):
            break
        if not all(i >= 0 for i in offering_items):
            break
    return log, error_flag