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
        ending_string = "This trade is not beneficial for me. For now, please consider the following offer: "
        full_sent_string = starting_string + "\n" + counteroffer_string + "\n" + ending_string + "\n" + full_sent_string

    # Add the offer to the log
    log_file = os.path.join(ui, 'log.txt')
    log_opened = open(log_file, "a")
    log_opened.write(full_sent_string)
    log_opened.close()

    # Send the offer to the receiving agent
    prompt_file = os.path.join(ui, 'offer.txt')
    with open(prompt_file, 'w') as f:
        f.write(full_sent_string)

    # Parse the receiving agent's response
    response, counteroffer, stopped_trading, offering_time, responding_time = parse_response(offer_string, item_list, ui)

    # Return if the agent has stopped trading.
    if stopped_trading:
        return False, [], True, offering_time, responding_time

    # If rejected, check to see if the responding agent has provided a counteroffer
    if response == 0:
        if len(counteroffer) != 0:
            return False, tuple(counteroffer), False, offering_time, responding_time
        else:
            return False, [], False, offering_time, responding_time
    else:
        return True, counteroffer, False, offering_time, responding_time


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

    # Wait until the user responds
    offering_time = time.time()
    response_file = os.path.join(ui, 'response.txt')
    while not os.path.exists(response_file):
        # print(f"Waiting for response")
        time.sleep(1)
    responding_time = time.time()
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
        return False, [], True, offering_time, responding_time
    log_file = os.path.join(ui, 'log.txt')
    log_opened = open(log_file, "a")
    log_opened.write("User: " + user_input + "\n")
    log_opened.close()

    # Perform sentiment analysis to determine if the responding agent has accepted the offer
    user_input = "Alice: " + user_input
    convo_history = prior_trade_offer + user_input
    response_val = sentiment_analysis(convo_history)
    if response_val == 1:
        return response_val, [], False, offering_time, responding_time
    else:
        # If not, parse the counteroffer
        counteroffer, counteroffer_string = obtain_counteroffer_text(convo_history, item_list)

    # Log the parsed counteroffer
    counteroffer_string = counteroffer_string.replace("Alice", "User").replace("Bob", "ST-CR")
    # lines = counteroffer_string.split('\n')
    # modified_lines = ['\t' + line for line in lines[2:]]
    # counteroffer_string = '\n'.join(modified_lines)
    parsed_counteroffer_string = "Parsed User Counteroffer: \n" + counteroffer_string + "\n"
    # print("Checking Parsed Counteroffer: ", parsed_counteroffer_string)
    log_file = os.path.join(ui, 'log.txt')
    log_opened = open(log_file, "a")
    log_opened.write(parsed_counteroffer_string)
    log_opened.close()

    return response_val, counteroffer, False, offering_time, responding_time


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
    # print("Checking Value: ", convo_history)
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
            # print("Dict String: ", dict_string)
            correct_counteroffer = check_if_counteroffer_correct(dict_string, convo_history)
            loop_counter += 1
        actual_dict = parse_trade_offer(dict_string, item_list)
        counteroffer = np.zeros(len(item_list))
        for item_idx in range(len(item_list)):
            if item_list[item_idx] in actual_dict.keys():
                counteroffer[item_idx] = actual_dict[item_list[item_idx]]
        # print("Counteroffer:  ", convo_history, dict_string, counteroffer)
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
    # user_message = f"Given the following conversation history: \"{convo_history}\", is the following counteroffer a correct interpretation of Alice's response? Counteroffer: \"{dict_string}\" Please respond 1 for yes, 0 for no. (only answer with 0 and 1)"
    user_message = (
        f"In a trading scenario, Alice has been given the following trade offer {alice_gives}, {alice_receives}, Alice has given the following response: {alice_response}. Given Alice's feedback, does the following counteroffer {dict_string} accurately reflect her feedback? Please respond with only yes or no. If the counteroffer includes ambiguous or non-numerical values for the items (e.g. 'some apples'), please respond with no."
    )
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
- If Alice provides general feedback or non-explicit values (e.g., 'some apples'), infer reasonable numerical values based on typical exchange patterns.
- Do not respond with placeholders or ambiguous terms. Ensure the trade offer is specific and formatted as requested.
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

    Bob receives: {receive_string.strip(', ')}
    Alice receives: {give_string.strip(', ')}
    """
    
    # print("Checking Offer: ", offer, trade_offer_string)
    return trade_offer_string

def query(offer, A, b, responding_items):
    """
        Query an agent with utility function x^TAx + 2bx to determine if they will accept an offer.

        Args:
            offer (np.array): n-dimensional vector representing an offer from the receiving agent's perspective.
            A (np.array): nxn matrix.
            b (list): n-dimensional vector.
            responding_items (list): Current State of responding agent.

        Returns:
            flag_n_items (bool): Response value which states if the responding agent has enough items to complete the trade
            flag_utility_improvement (bool): Response value which states if the responding agent's utility improves with the offer
            min_index (int): Item index corresponding to the smallest number of items in the responding agent's possession
            min_items (int/float): Minimum number of items that the responding agent possess for any give category.
    """
    next_step = offer + responding_items
    flag_dot_product = utility_improvement(offer, A, b, responding_items)
    min_index = -1
    min_items = 0
    if all(i >= -0.00000001 for i in next_step):
        flag_n_items = True
    else:
        flag_n_items = False
        min_index = np.argmin(next_step)
        min_items = responding_items[min_index]
    return flag_n_items, flag_dot_product, min_index, min_items

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

def random_offer_search(offering_A, offering_b, responding_A, responding_b, offering_items, responding_items, max_trade_value, item_list, ui, prev_offer=[], informed = True, integer_constraint=True, offer_budget=1000):
    """
        Find a mutually beneficial trade using a random search approach

        Args:
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            prev_offer (np.array): Previously accepted offer used for heuristic trading. Default: Empty
            informed (bool, optional): Whether the algorithm should use the previously accepted trade heuristic for the first offer. Defaults to True.
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            offer_budget (int, optional): Maximum number of offers allowed to the receiving agent. Defaults to 1000.

        Returns:
            tuple:
                - accepted_trade (bool): True if an acceptable trade was found, False otherwise.
                - offer (list): The trade offer that was generated.
                - offer_counter (int): Number of offers made to the receiving agent during the search.
    """
    accepted_trade = False
    offer_counter = 0
    offer_list = []
    waiting_times = []
    prev_counteroffer_flag = False
    prev_counteroffer = []
    # Remove categories where one agent has all the items
    reduction_idx = []
    for i in range(len(offering_items)):
        if offering_items[i] == 0 or responding_items[i] == 0:
            reduction_idx.append(i)
    if len(reduction_idx) >= len(offering_items) - 1:
        return False, [], 0, offer_list, waiting_times
    
    prev_trade_accepted=False
    if informed and offer_counter == 0:
        # print("Checking Previous Offer: ", informed, prev_offer)
        if len(prev_offer) != 0:
            prev_trade_accepted=True
        offer = generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value=max_trade_value, reduction_idx = reduction_idx, prev_offer = prev_offer, integer_constraint=integer_constraint)
    else:
        offer = generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value=max_trade_value, reduction_idx = reduction_idx, integer_constraint=integer_constraint)

    while not accepted_trade and offer_counter < offer_budget:
        # If no beneficial offer was found for the offering agent, stop trading
        if len(offer) == 0:
            edge_case_string = "The algorithm has ended trading, please press the Stop Trading button to end the scenario"
            prompt_file = os.path.join(ui, 'offer.txt')
            with open(prompt_file, 'w') as f:
                f.write(edge_case_string)
            return False, [], offer_counter, True, offer_list, waiting_times

        # Present the offer to the responding agent
        neg_offer = -1 * offer
        offer_counter += 1
        # print("Making Offer: ", neg_offer)
        offer_list.append(offer)
        response_product, counteroffer, stopped_trading, offering_time, responding_time = query_gpt_with_counteroffer(neg_offer, item_list,
                                                                                          ui, responding_items, prev_trade_accepted=prev_trade_accepted, prev_counteroffer_flag=prev_counteroffer_flag, prev_counteroffer=prev_counteroffer)
        prev_counteroffer_flag = False
        prev_trade_accepted = False
        prev_counteroffer = []
        waiting_times.append((offering_time, responding_time))
        if stopped_trading:
            log_file = os.path.join(ui, 'log.txt')
            log_opened = open(log_file, "a")
            log_opened.write("Trading Ended By User \n")
            log_opened.close()
            return False, None, offer_counter, stopped_trading, offer_list, waiting_times
        # If the responding accepted the offer, but it wasn't beneifical for the offering offering, negate the offer to ensure that the appropriate direction is maintained.
        if response_product == True:
            # print("Trade Accepted: ", offer)
            accepted_trade = True
            continue
        if len(counteroffer) != 0:
            original_counteroffer = np.array(counteroffer).copy()
            # print("Checking Counteroffer: ", original_counteroffer)
            mod_counteroffer = np.array([x for i, x in enumerate(original_counteroffer) if i not in reduction_idx])
            neg_counteroffer = -1 * np.array(counteroffer)
            response_product, counteroffer = query_with_counteroffer(neg_counteroffer, offering_A, offering_b,
                                                                        offering_items)
            if response_product:
                log_string = f"ST-CR:  Counteroffer Direction is Beneficial \n"
                log_file = os.path.join(ui, 'log.txt')
                log_opened = open(log_file, "a")
                log_opened.write(log_string)
                log_opened.close()
                offer = -1 * original_counteroffer
                continue
            else:
                log_string = f"ST-CR:  Counteroffer Direction is not Beneficial.\n"
                log_file = os.path.join(ui, 'log.txt')
                log_opened = open(log_file, "a")
                log_opened.write(log_string)
                log_opened.close()
                prev_counteroffer_flag = True
                prev_counteroffer = -1 * neg_counteroffer
                if informed and offer_counter == 0:
                    offer = generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value=max_trade_value, reduction_idx = reduction_idx, prev_offer = prev_offer, integer_constraint=integer_constraint)
                else:
                    offer = generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value=max_trade_value, reduction_idx = reduction_idx, integer_constraint=integer_constraint)
        else:
            if informed and offer_counter == 0:
                offer = generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value=max_trade_value, reduction_idx = reduction_idx, prev_offer = prev_offer, integer_constraint=integer_constraint)
            else:
                offer = generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value=max_trade_value, reduction_idx = reduction_idx, integer_constraint=integer_constraint)
        prev_trade_accepted=False


    return accepted_trade, offer, offer_counter, False, offer_list, waiting_times

def obtain_scaling(offer, offering_items, responding_items, max_trade_value):
    """
        Find a scaling factor for a potential offer results in a feasible offer that abides by the maximum trade value.

        Args:
            offer (np.array): nxn matrix used for the offering agent's utility function.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding.
            max_trade_value (int): Maximum number of items that can be traded from any item category

        Returns:
            scale (float): scaling factor for the offer that, when multiplied to a normalized offer, will result in a feasible offer for both parties.
    """
    dimensions = len(offering_items)
    scaling_factors = []
    for i in range(dimensions):
        if offer[i] > 0:
            max_scale_offering = (np.inf - offering_items[i]) / offer[i]
            max_scale_responding = responding_items[i] / offer[i]
        else:
            max_scale_offering = offering_items[i] / abs(offer[i])
            max_scale_responding = (np.inf - responding_items[i]) / abs(offer[i])

        scaling_factors.append(min(max_scale_offering, max_scale_responding))

    # Find the maximum scaling factor that keeps the offer in within the bound
    max_scaling_factor = min(scaling_factors)

    # Ensure no entry exceeds the maximum magnitude
    scale = min(max_scaling_factor, max_trade_value / max(abs(offer)))
    return scale

def generate_random_feasible_offer(max_trade_value, offering_items, responding_items, dimensions):
    """
        Find a random feasible offer for both agents

        Args:
            max_trade_value (int): Maximum number of items that can be traded from any item category.
            offer (np.array): nxn matrix used for the offering agent's utility function.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding.


        Returns:
            scaled_vector (np.array): scaled random offer that is feasible for both agents.
    """
    while True:
        # Generate a random unit vector
        random_vector = np.random.randn(dimensions)
        random_vector /= np.linalg.norm(random_vector)

        # Scale the vector so that no entry violates the constraints

        scale = obtain_scaling(random_vector,offering_items,responding_items,max_trade_value)
        # Scale the vector
        scaled_vector = random_vector * scale

        # Check if the resultant vector meets the requirements
        if all(abs(scaled_vector[i]) <= max_trade_value for i in range(dimensions)):
            return scaled_vector

def generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value = 5, reduction_idx = [], prev_offer = [], integer_constraint=True):
    """
        Find a random feasible offer that is beneficial for the offering agent

        Args:
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            reduction_idx (list, optional): set of item indices that need to be removed from consideration.
            prev_offer (np.array): Previously accepted offer used for heuristic trading. Default: Empty
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.


        Returns:
            random_vector_full (np.array): Scaled random offer that is beneficial for the offering agent.
    """

    # If there is no previous offer, generate a random vector of the same dimensionality as the original vector
    if len(prev_offer) == 0:
        if integer_constraint:
            random_vector = np.random.randint(-max_trade_value, max_trade_value+1, size=len(offering_items)-len(reduction_idx))
        else:
            mask = np.ones(len(offering_items), dtype=bool)
            mask[reduction_idx] = False
            offering_red = offering_items[mask]
            responding_red = responding_items[mask]
            random_vector = generate_random_feasible_offer(max_trade_value, offering_red, responding_red, len(offering_items)-len(reduction_idx))
    else:

        # If there is previous offer, ensure it is feasible.
        random_vector = prev_offer
        random_vector = random_vector/np.linalg.norm(random_vector)
        scale = obtain_scaling(random_vector, offering_items, responding_items, max_trade_value)
        random_vector *= scale
        if integer_constraint:
            random_vector = np.round(random_vector)
        # print("Checking Fesibility: ", prev_offer, random_vector)

    # Ensure the randomly generated offer is beneficial to the offering agent. If not, generate a new offer.
    max_generations = 1000
    i_counter = 0
    random_vector_full = obtain_full_offer(random_vector, reduction_idx, len(offering_b))
    flag_n_items, flag_dot_product, min_index, min_items = query(random_vector_full, offering_A, offering_b, offering_items)
    while not (flag_dot_product and flag_n_items) and i_counter <= max_generations:
        # Ensure the offer is feasible
        if (not flag_n_items) and min_items < max_trade_value:
            max_trade_value = min_items
            i_counter -= 1

        # If the offer direction is not beneficial, generate a new offer.
        if integer_constraint:
            random_vector = np.random.randint(-max_trade_value, max_trade_value+1, size=len(offering_items)-len(reduction_idx))
        else:
            mask = np.ones(len(offering_items), dtype=bool)
            mask[reduction_idx] = False
            offering_red = offering_items[mask]
            responding_red = responding_items[mask]
            random_vector = generate_random_feasible_offer(max_trade_value, offering_red, responding_red, len(offering_items)-len(reduction_idx))
        random_vector_full = obtain_full_offer(random_vector, reduction_idx, len(offering_b))
        flag_n_items, flag_dot_product, min_index, min_items = query(random_vector_full, offering_A, offering_b, offering_items)
        i_counter += 1

    # If no beneficial offer was found, stop trading
    if not(flag_dot_product and flag_n_items):
        return []
    else:
        return random_vector_full

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
    return full_offer

def run_trading_scenario_random(num_items, offering_A, offering_b, responding_A, responding_b, offering_items, responding_items, offer_budget, item_list, ui, max_trade_value=5, integer_constraint=True, informed = True, debug=False):
    """
        Run a trading scenario with using the random trading with momentum baseline.

        Args:
            num_items (int): Total number of item categories in the trading scenario
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding agent.
            offer_budget (int): The total number of offers to the responding agent that the negotiating agent is allowed to make.
            max_trade_value (int, optional): maximum number of items that can be traded from any item category
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            informed (bool, optional): Whether the algorithm should use the previous trade heuristic. Defaults to True.
            debug (bool, optional): Boolean that controls if extra debugging information should be printed.
        Returns:
            log (list of dictionaries): List that represents trading progression for the scenario
    """

    # Initialize Trading Loop
    trade_num = 0
    log = []
    prev_offer = []
    cumulative_offer_count = 0

    # Loop for a set number of trades or until the algorithm hits the query limit
    while trade_num <= 100 and cumulative_offer_count < offer_budget:
        trade_num += 1
        original_responding_gradient = list(n_dim_quad_grad(responding_A, responding_b, responding_items))
        original_offering_gradient = list(n_dim_quad_grad(offering_A, offering_b, offering_items))

        # Use the respective algorithm to find a trade
        stopped_trading = False
        if informed:
            # print("The Trade is informed: ", informed, prev_offer)
            found_trade, true_out_offer, query_count, stopped_trading, offer_list, waiting_times = random_offer_search(offering_A, offering_b, responding_A, responding_b, np.array(offering_items), np.array(responding_items),  max_trade_value, item_list, ui, prev_offer=prev_offer, informed=True, integer_constraint=integer_constraint, offer_budget=offer_budget-cumulative_offer_count)
            # print("Checking Output: ", found_trade, true_out_offer)
        else:
            found_trade, true_out_offer, query_count, stopped_trading, offer_list, waiting_times = random_offer_search(offering_A, offering_b, responding_A, responding_b, np.array(offering_items), item_list, np.array(responding_items), max_trade_value, ui, informed=False, integer_constraint=integer_constraint, offer_budget=offer_budget-cumulative_offer_count)
        
        # Update the item values if a trade has been found
        # if stopped_trading:
        #     break
        information_dict = {}
        if found_trade == True and not stopped_trading:
            information_dict["found_trade"] = True
            prev_responding_items = responding_items.copy()
            prev_offering_items = offering_items.copy()
            responding_items -= true_out_offer
            offering_items += true_out_offer

            # Account for cases where items are very close to zero
            for i in range(0, len(responding_items)):
                if responding_items[i] < 0.00000001:
                    responding_items[i] = 0
                if offering_items[i] < 0.00000001:
                    offering_items[i] = 0

            # Update State Values
            neg_offer = [-1 * x for x in true_out_offer]
            prev_state_value_h = prev_responding_items.transpose() @ responding_A @ prev_responding_items + responding_b @ prev_responding_items
            prev_state_value_a = prev_offering_items.transpose() @ offering_A @ prev_offering_items + offering_b @ prev_offering_items

            next_state_value_h = responding_items.transpose() @ responding_A @ responding_items + responding_b @ responding_items
            next_state_value_a = offering_items.transpose() @ offering_A @ offering_items + offering_b @ offering_items
            if debug:
                print("Random Trading with Momentum Result: ", found_trade)
                print("Accepted Offer: ", true_out_offer)
                print("Offering Agent Gradient: ", original_offering_gradient)
                print("Responding Agent Gradient: ", original_responding_gradient)
                print("New Responding Agent Item List: ", responding_items)
                print("New Offering Agent Item List: ", offering_items)
                print("Offering Agent Gradient Original: ",
                      (original_offering_gradient / np.linalg.norm(original_offering_gradient)))
                print("Offering Agent predicted benefit ",
                      np.dot(true_out_offer, (original_offering_gradient / np.linalg.norm(original_offering_gradient))))
                print("Responding Agent predicted benefit ",
                      np.dot(neg_offer, (original_responding_gradient / np.linalg.norm(original_responding_gradient))))

                print("Offering Agent benefit ", next_state_value_a - prev_state_value_a)
                print("Responding Agent benefit ", next_state_value_h, prev_state_value_h, next_state_value_h - prev_state_value_h)
                print("Query Count: ", query_count)
                print("\n")

            # Log trade progression information
            information_dict["offer"] = true_out_offer.tolist()
            information_dict["responding_items"] = responding_items.tolist()
            information_dict["offering_items"] = offering_items.tolist()
            information_dict["responding_benefit"] = next_state_value_h - prev_state_value_h
            information_dict["offering_benefit"] = next_state_value_a - prev_state_value_a
            information_dict["offer_count"] = query_count
            information_dict["waiting_times"] = waiting_times
            information_dict["offer_list"] = [list(offer) for offer in offer_list]

            score_dict = {}
            human_target = responding_b / 2
            score_dict["Initial State"] = [50, 50, 50]
            human_final = responding_items
            score_dict["Final State"] = [int(element) for element in human_final]
            score_dict["Target State"] = [int(element/2) for element in responding_b]
            score = np.sum(np.abs(human_target - human_final)) / np.sum(np.abs(human_target - np.array([50, 50, 50])))
            if score > 1:
                score = 1
            print("Score: ", score)
            score_dict["Score"] = float(1 - score)
            chat_folder = ui.rsplit("/", 1)[0]
            with open(f'{chat_folder}/score.json', 'w') as score_file:
                json.dump(score_dict, score_file, indent=4)
            with open(f'{ui}/score.json', 'w') as score_file:
                json.dump(score_dict, score_file, indent=4)

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
            log_file = os.path.join(ui, 'log.txt')
            log_opened = open(log_file, "a")
            log_opened.write(result_string)
            log_opened.write(responding_state_string)
            log_opened.write(offering_state_string)
            log_opened.write(responding_benefit)
            log_opened.write(offering_benefit)
            log_opened.close()

            # Update Cumulative Query Count
            cumulative_offer_count += query_count
            prev_offer = true_out_offer.copy()
            log.append(information_dict)

            with open(f'{ui}/numerical_log.json', 'w') as json_file:
                json.dump(log, json_file, indent=4)

            # If the trade was not beneficial to both parties, then an error has occurred.
            # if next_state_value_h - prev_state_value_h < 0.000001 and next_state_value_a - prev_state_value_a < 0.000001:
            #     trade_num = 101
        else:
            # print("Stopped Trading")
            information_dict["found_trade"] = False
            information_dict["offer_count"] = query_count
            information_dict["offer_list"] = [list(offer) for offer in offer_list]
            information_dict["responding_items"] = responding_items.tolist()
            information_dict["offering_items"] = offering_items.tolist()
            information_dict["waiting_times"] = waiting_times
            log.append(information_dict)
            with open(f'{ui}/numerical_log.json', 'w') as json_file:
                json.dump(log, json_file, indent=4)
            # If informed trading failed to find a trade, switch to pure random trading
            trade_num = 101
        if not all(i >= 0 for i in responding_items):
            break
        if not all(i >= 0 for i in offering_items):
            break
    return log