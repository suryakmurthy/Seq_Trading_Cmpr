import numpy as np
import os
import openai
import time
import re
import json

def query_gpt_pure(offer, ui):
    """
        Query the responding agent via gpt to see if they will accept an offer.

        Args:
            offer (np.array): n-dimensional vector representing an offer from the receiving agent's perspective.
            ui (string): Directory where the program will place the offer.

        Returns:
            accepted (bool): Response value which states if the agent has accepted the offer.
    """
    offer_string = offer
    log_file = os.path.join(ui, 'log.txt')
    log_opened = open(log_file, "a")
    log_opened.write(offer_string)
    log_opened.close()
    prompt_file = os.path.join(ui, 'offer.txt')
    with open(prompt_file, 'w') as f:
        f.write(offer_string)
    return parse_response(ui)


def parse_response(ui):
    """
        Parse a natural language response from a responding agent

        Args:
            ui (string): Directory where the program will place the offer.

        Returns:
            user_input (string): The responding agent's response to the trade offer
            accepted (bool): Response value which states if the agent has accepted the offer.
    """
    offer_time = time.time()
    response_file = os.path.join(ui, 'response.txt')
    while not os.path.exists(response_file):
        time.sleep(1)
        
    # Read the response
    with open(response_file, 'r') as f:
        user_input = f.read().strip()
    os.remove(response_file)
    response_time = time.time()
    # print("User input: ", user_input)
    if user_input == "stop":
        log_file = os.path.join(ui, 'log.txt')
        log_opened = open(log_file, "a")
        log_opened.write("\n User: Stopped Trading \n")
        log_opened.write("\n" + "Offer Time: " + str(offer_time) + "\n")
        log_opened.write("\n" + "Response Time: " + str(response_time) + "\n")
        log_opened.close()
        return None, True, offer_time, response_time
    log_file = os.path.join(ui, 'log.txt')
    log_opened = open(log_file, "a")
    log_opened.write("\n" + "User: " + user_input + "\n")
    log_opened.write("\n" + "Offer Time: " + str(offer_time) + "\n")
    log_opened.write("\n" + "Response Time: " + str(response_time) + "\n")
    log_opened.close()

    user_input = "User: " + user_input
    return user_input, False, offer_time, response_time


def generate_offer_gpt_pure(convo_history):
    """
        Generate a trade offer using GPT

        Args:
            convo_history (dict): Dictionary with the entire conversation history up to this point

        Returns:
            offer (string): GPT's trade offer
    """
    convo_history.append({"role": "system", "content": "Reminder: You should only make offers that improve your uility function"})
    messages = convo_history
    # print("Generating New Offer: ", convo_history)
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

    return response.choices[0].message['content']


def sentiment_analysis(convo_history):
    """
        Determine if a human user is accepting or rejecting an offer using sentiment analysis

        Args:
            convo_history (dict): Dictionary with the entire conversation history up to this point

        Returns:
            (boolean): Whether the human user is accepting or rejecting the offer
    """
    user_message = f"Given the following conversation history: \"{convo_history}\", is the user accepting or rejecting the offering's trade offer? Please answer with 1 for accept, otherwise answer with 0. If the text doesn't provide information on whether the responding is accepting or rejecting the offering's trade offer, respond with 0."
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


def parse_trade_offer(trade_string, item_list):
    """
        Given a standard string representing a trade offer, return a dictionary corresponding to the offer

        Args:
            trade_string (string): String representing a trade offer
            item_list (np.array): List of item names.

        Returns:
            trade_dict (dictionary): Dictionary representing the offer
    """
        # Prepare singular and plural mappings
    item_singular = {item.rstrip('s').lower(): item for item in item_list}  # Map singular to original
    
    # Initialize trade dictionary with zero quantities
    trade_dict = {item: 0.0 for item in item_list}
    
    # Replace plural and singular forms in the string
    for singular, original in item_singular.items():
        plural = singular + 's'
        trade_string = re.sub(rf'\b{plural}\b', singular, trade_string, flags=re.IGNORECASE)
        trade_string = re.sub(rf'\b{original}\b', singular, trade_string, flags=re.IGNORECASE)

    # Split trade string into lines
    lines = trade_string.split('\n')

    # Process each line in the trade string
    for line in lines:
        if "receives:" in line:
            parts = line.split()

            sign = -1  # Determine the sign based on the receiver

            for i in range(len(parts)):
                for singular, original in item_singular.items():
                    if singular in parts[i]:
                        # Get the quantity before the item
                        quantity = sign * float(parts[i - 1])  # Apply the sign
                        trade_dict[original] += quantity

        if "gives:" in line:
            parts = line.split()

            sign = 1  # Determine the sign based on the receiver

            for i in range(len(parts)):
                for singular, original in item_singular.items():
                    if singular in parts[i]:
                        # Get the quantity before the item
                        quantity = sign * float(parts[i - 1])  # Apply the sign
                        trade_dict[original] += quantity

    return trade_dict
    # trade_dict = {item: 0.0 for item in item_list}
    # lines = trade_string.split('\n')
    # for line in lines:
    #     for item in item_list:
    #         if f"GPT receives:" in line:
    #             if item in line:
    #                 parts = line.split()
    #                 for i in range(len(parts)):
    #                     if item in parts[i]:
    #                         quantity = float(parts[i - 1])
    #                         trade_dict[item] += quantity
    #         elif f"User receives:" in line:
    #             if item in line:
    #                 parts = line.split()
    #                 for i in range(len(parts)):
    #                     if item in parts[i]:
    #                         quantity = -float(parts[i - 1])
    #                         trade_dict[item] += quantity
    # return trade_dict
def simplify_trade_offer(trade_string, item_list):
    # Create regex patterns to handle both singular and plural forms together
    patterns = {item: re.compile(rf'(\d+)\s+({item[:-1]}|{item})') for item in item_list}
    
    # Initialize counters for GPT's and User's items
    gpt_items = {item: 0 for item in item_list}
    user_items = {item: 0 for item in item_list}
    
    # Split the trade string into lines for GPT and User
    lines = trade_string.strip().split('\n')
    
    # Process GPT's received items
    for item, pattern in patterns.items():
        match = pattern.findall(lines[1])
        for quantity, _ in match:
            gpt_items[item] += int(quantity)
    
    # Process User's received items
    for item, pattern in patterns.items():
        match = pattern.findall(lines[2])
        for quantity, _ in match:
            user_items[item] += int(quantity)
    
    # Simplify the trade: calculate net quantities
    gpt_net = {item: gpt_items[item] - user_items[item] for item in item_list}
    user_net = {item: user_items[item] - gpt_items[item] for item in item_list}
    
    # Prepare simplified offer string
    gpt_offer = ', '.join([f'{gpt_net[item]} {item}' for item in item_list if gpt_net[item] > 0])
    user_offer = ', '.join([f'{user_net[item]} {item}' for item in item_list if user_net[item] > 0])
    
    return f"GPT's Trade Offer:\n    User gives: {gpt_offer if gpt_offer else 'Nothing'}\n    User receives: {user_offer if user_offer else 'Nothing'}"

def state_string_deterministic(state, item_list):
    state_string = ''
    for item_idx in range(0, len(item_list) - 1):
        state_string += f'{state[item_idx]} {item_list[item_idx]}, '
    state_string += f'and {state[-1]} {item_list[-1]}.'
    return state_string

def check_if_counteroffer_correct(prev_offer, current_offer ,convo_history):
    """
        Given a parsed counteroffer and a conversation history, determine if the counteroffer is an accurate interpretation of the responding's feedback

        Args:
            dict_string (string): String representing a counteroffer parsed from the responding's response
            convo_history (string): A conversation history comprised of an offer and a response

        Returns:
            response_val (bool): Whether the parsed counteroffer is an accurate interpretation of the responding's feedback
    """
    print("Prev Offer: ", prev_offer)
    print("Convo History: ", convo_history)
    print("Prev Offer: ", current_offer)
    current_offer = current_offer.replace('User', 'Alice')
    test_string = convo_history.replace("User", "Alice")
    lines = prev_offer.replace('User', 'Alice').splitlines()
    for line in lines:
        if line.strip().startswith("Alice gives"):
            alice_gives = line.strip()
        if line.strip().startswith("Alice receives"):
            alice_receives = line.strip()
    # user_message = f"Given the following conversation history: \"{convo_history}\", is the following counteroffer a correct interpretation of Alice's response? Counteroffer: \"{dict_string}\" Please respond 1 for yes, 0 for no. (only answer with 0 and 1)"
    user_message = (
        f"In a trading scenario, Alice has been given the following trade offer {alice_gives}, {alice_receives}, Alice has given the following response: {test_string}. Given Alice's feedback, does the following counteroffer {current_offer} accurately reflect her feedback? Please respond with only yes or no. If the counteroffer includes ambiguous or non-numerical values for the items (e.g. 'some apples'), please respond with no."
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

    print("User Message: ", user_message)
    response_val = response.choices[0].message.content
    print("Response Value: ", response_val)
    if response_val == "Yes" or response_val == "yes":
        return True
    else:
        return False

def determine_correctness(prev_offer, offer, user_response):
    user_message = f"In the context of negotiation, responses can be classified as either a 'counteroffer', an 'adjusted preference', or a 'general preference.' A counteroffer occurs when one party indicates a conditional acceptance that specifies changes to the original offer, such as requesting additional items or quantities. An adjusted preference suggests a desire for changes to the original offer but does not fully reject it. A general preference expresses a general desire without indicating any adjustments to the offer. Given the following offer: \"{prev_offer}\" and the user's response \" {user_response} \", please determine if the user is providing a counteroffer, an adjusted preference, a general preference, or neither. Respond with \"counteroffer\" if she indicates a conditional acceptance that specifies changes to the offer, including requests for more or fewer items. Respond with \"adjusted preference\" if she expresses a desire for changes without fully rejecting the offer. Respond with \"preference\" if she expresses a general desire. If neither option applies, respond with \"neither\" Only use the responses \"counteroffer\", \"adjusted preference\", \"preference\", or \"neither\"."
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
    if 'neither' in response_val:
        return True
    else:
        return check_if_counteroffer_correct(prev_offer, offer, user_response)



def offer_search(item_list, ui, convo_history, offer_budget, offering_items, responding_items, prev_trade_accepted=False):
    """
        Use GPT to find a mutually beneficial offer.

        Args:
            item_list (np.array): List of item names.
            ui (string): Folder containing the chat log and offer exchanges
            convo_history (dictionary): Entire conversation history up to this point.
            offer_budget (int): Total number of offers GPT is allowed to make

        Returns:
            trade_dict (dictionary): Dictionary representing the offer
    """
    query_count = 0
    waiting_times = []
    offer_list = []
    prev_offer = ''
    while query_count < offer_budget:
        offer_feasible = False
        correct_flag = False
        user_feedback = ""
        edge_case_flag = False
        while not offer_feasible:
            # print("Convo History: ", convo_history
            gpt_offer = generate_offer_gpt_pure(convo_history)
            while "GPT's Trade Offer" not in gpt_offer or "User gives" not in gpt_offer or "User receives" not in gpt_offer:
                convo_history.append({"role": "system", "content": "You are not responding with a trade offer in the correct format. Try again."})
                gpt_offer = generate_offer_gpt_pure(convo_history)
            gpt_offer = simplify_trade_offer(gpt_offer, item_list)
            print("convo_history: ", convo_history[-2])

            # Checking if Counteroffer Accurately Reflects the User's Feedback
            # response_string = convo_history[-2]
            # if not edge_case_flag:
            #     if response_string['role'] == 'user' or len(user_feedback) != 0:
            #         if response_string['role'] == 'user':
            #             correct_flag = determine_correctness(prev_offer, gpt_offer, response_string['content'])
            #         else:
            #             correct_flag = determine_correctness(prev_offer, gpt_offer, user_feedback)
            #         if not correct_flag:
            #             if len(user_feedback) == 0:
            #                 user_feedback = response_string['content']
            #             convo_history.append({"role": "system",
            #                                       "content": f"The trade offer your provided ({gpt_offer}) either has ambigous values or does not accuratly reflect the user's feedback ({user_feedback}). Try again."})
            #     else:
            #         correct_flag = True
            # else:
            #     correct_flag = True

            # Checking Feasibility of Offer
            parsed_trade_offer = np.array(list(parse_trade_offer(gpt_offer, item_list).values()))
            print("Parsed Trade Offer: ", parsed_trade_offer)
            if any((responding_items - parsed_trade_offer) < 0) or any((offering_items + parsed_trade_offer) < 0):
                convo_history.append({"role": "system", "content": "The trade offer your provided is not feasible given the current number of items you or the user possess. Try again."})
                offering_state_string = "GPT Current State: " + state_string_deterministic(offering_items, item_list) + "\n"
                receiving_state_string = "User Current State: " + state_string_deterministic(responding_items, item_list) + "\n"
                convo_history.append({"role": "system", "content": offering_state_string})
                convo_history.append({"role": "system", "content": receiving_state_string})
            else:
                offer_feasible = True
            # if correct_flag and not offer_feasible:
            #     edge_case_flag = True

        accepted_string = "Trade has Been Accepted! \n"
        current_state_string = "Your Current State: " + state_string_deterministic(responding_items, item_list) + "\n \n"
        next_state_string = "Your Next State After This Trade: " + state_string_deterministic(responding_items - parsed_trade_offer, item_list) + "\n"
        query_statement = "Do you accept this trade? \n"
        if prev_trade_accepted == True:
            prev_trade_accepted = False
            full_sent_string = accepted_string + current_state_string + gpt_offer + "\n" + next_state_string + query_statement
        else:
            full_sent_string = current_state_string + gpt_offer + "\n" + next_state_string + query_statement
        query_count += 1
        responding_response, stopped_trading, offer_time, response_time = query_gpt_pure(full_sent_string, ui)
        waiting_times.append((offer_time, response_time))
        offer_list.append(parse_trade_offer(gpt_offer, item_list))
        convo_history.append({"role": "assistant", "content": gpt_offer})
        convo_history.append({"role": "user", "content": responding_response.replace('Human', 'User')})
        prev_offer = gpt_offer
        if stopped_trading:
            return False, [], "", query_count, convo_history, stopped_trading, offer_time, waiting_times
        accepted = sentiment_analysis(convo_history)
        if accepted == 1:
            vector_trade = parse_trade_offer(gpt_offer, item_list)
            return True, vector_trade, gpt_offer, query_count, convo_history, False, offer_list, waiting_times
    return False, [], "", query_count, convo_history, True, offer_list, waiting_times


def run_trading_scenario_gpt_pure(num_items, offering_A, offering_b, responding_A, responding_b, offering_items, responding_items,
                              item_list, ui_folder, offer_budget=50):
    """
        Run a trading scenario using GPT to generate trades
        Args:
            num_items (int): Total number of item categories
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items (np.array): Initial number of items in each category for the offering agent.
            responding_items (np.array): Initial number of items in each category for the responding agent.
            item_list (list): List of item categories.
            ui_folder (string): Directory of chat log and trade offers/responses.
        Returns:
            log (file written to ui_folder): Log of trading progression
    """
    # print("Pure GPT Implmentation")
    # Provide GPT a prompt to motivate negotiation
    starting_prompt = f"""System: You are a computer negotiation agent aiming to maximize your utility function by trading apples, oranges, and bananas with a human negotiator. Your utility function is:

    num_apples^2 + 2 * {offering_b[0] / 2} * num_apples - num_bananas^2 + 2 * {offering_b[1] / 2} * num_bananas - num_oranges^2 + 2 * {offering_b[2] / 2} * num_oranges

    where `num_apples`, `num_bananas`, and `num_oranges` represent the number of each fruit you currently possess. You should only make offers that increase your utility.

    Both you and the user start with 50 apples, 50 bananas, and 50 oranges. You will present trade offers to the user, and the user will respond or provide feedback.

    Important:
    - You will only present trade offers to the user. Do not include any additional information in your responses like comments or queries for feedback. Only make offers that improve your utility while accounting for the user's feedback. 
    - Use "GPT" to refer to yourself and "User" to refer to the human negotiator.
    - Include "GPT's Trade Offer: " at the start of each offer.
    - Ensure that every trade offer improves your utility based on the current state.
    - You should vary the fruits and quantities depending on what improves utility and the user's feedback.
    - When an offer is accepted, track the current inventory for both parties and continue trading.
    - The following are an example trade offers whose format you should follow, the number and type of the items should vary based on the trade offer you want to make:
    - If the previous offer has the user giving an item, and the user's response asks to receive the same item, then set the number of that item she gives to '0'. 
    - If the previous offer has the user receiving an item, and the user's response asks to give the same item, then set the number of that item she receives to '0'.
    - If the user is not giving any items in the new trade, replace the 'User gives' list with 'Nothing.' If the user is not receiving any items in the new trade, replace the 'User receives' list with 'Nothing.'
    - Do not respond with placeholders or ambiguous terms like 'some', 'more', 'fewer', etc. I want you to infer specific numerical values, even if they are inferred (ex. '1 apple' instead of 'some apples') 
    - Ensure the trade offer is specific and formatted as requested.
    Offer Format:
    "GPT's Trade Offer: 
        User gives: a apples, b bananas, c oranges
        User receives: d apples, e bananas, r oranges"
    
    Where a, b, c, d, e, f are specific numerical values. If the user provides general feedback or non-explicit values (e.g., 'some apples', 'fewer oranges', 'all bananas'), infer reasonable numerical values based on typical exchange patterns.
    Now, start the negotiation by proposing an initial trade offer that improves your utility."""
    convo_history = [{"role": "system", "content": starting_prompt}]
    trade_num = 0
    log = []
    prev_responding_states = []
    error_flag = False
    prev_trade_accepted = False
    while trade_num <= 100:

        # Search for a trade
        trade_num += 1
        found_trade, out_offer, out_offer_string, query_count, convo_history, stopped_trading, offer_list, waiting_times = offer_search(item_list, ui_folder, convo_history, offer_budget, offering_items, responding_items, prev_trade_accepted=prev_trade_accepted)
        if stopped_trading:
            break
        true_out_offer = np.array(list(out_offer.values()))

        # Update the item values if a trade has been found
        information_dict = {}
        if found_trade == True:
            prev_trade_accepted = True
            # Transition to the next state
            prev_responding_items = responding_items.copy()
            prev_responding_states.append(prev_responding_items.tolist())
            prev_offering_items = offering_items.copy()
            responding_items -= true_out_offer
            offering_items += true_out_offer

            prev_state_value_h = prev_responding_items.transpose() @ responding_A @ prev_responding_items + responding_b @ prev_responding_items
            prev_state_value_a = prev_offering_items.transpose() @ offering_A @ prev_offering_items + offering_b @ prev_offering_items
            next_state_value_h = responding_items.transpose() @ responding_A @ responding_items + responding_b @ responding_items

            # Log trade progression
            next_state_value_a = offering_items.transpose() @ offering_A @ offering_items + offering_b @ offering_items
            responding_state_string = f"New User State: "
            offering_state_string = f"New GPT State: "
            for item_idx in range(0, len(item_list)):
                responding_state_string += f"{responding_items[item_idx]} {item_list[item_idx]}, "
                offering_state_string += f"{offering_items[item_idx]} {item_list[item_idx]}, "
            responding_state_string += "\n"
            offering_state_string += "\n"
            convo_history.append({"role": "system", "content": responding_state_string + offering_state_string})
            responding_benefit = f"Estimated User Benefit: {next_state_value_h - prev_state_value_h} \n"
            offering_benefit = f"GPT's Benefit: {next_state_value_a - prev_state_value_a} \n"
            result_string = f"Offer Accepted! \n"
            log_file = os.path.join(ui_folder, 'log.txt')
            log_opened = open(log_file, "a")
            log_opened.write(result_string)
            log_opened.write(responding_state_string)
            log_opened.write(offering_state_string)
            log_opened.write(responding_benefit)
            log_opened.write(offering_benefit)
            log_opened.close()

            # Check that no negative items have been obtained
            if not (all( entry >= 0 for entry in offering_items)):
                trade_num = 101
                error_flag = True

            # Add trading progression to the log
            information_dict["found_trade"] = found_trade
            information_dict["responding_items"] = responding_items.tolist()
            information_dict["offering_items"] = offering_items.tolist()
            information_dict["responding_benefit"] = next_state_value_h - prev_state_value_h
            information_dict["offering_benefit"] = next_state_value_a - prev_state_value_a
            information_dict["offer_count"] = query_count
            information_dict["waiting_times"] = waiting_times
            information_dict["offer_list"] = offer_list
            log.append(information_dict)
            with open(f'{ui_folder}/numerical_log.json', 'w') as json_file:
                json.dump(log, json_file, indent=4)
        else:
            information_dict["found_trade"] = found_trade
            information_dict["offer_count"] = query_count
            information_dict["waiting_times"] = waiting_times
            information_dict["offer_list"] = offer_list
            log.append(information_dict)
            with open(f'{ui_folder}/numerical_log.json', 'w') as json_file:
                json.dump(log, json_file, indent=4)
            trade_num = 101
        # DEBUG: Check to see if the prior iteration resulted in any negative item values
        if not all(i >= 0 for i in responding_items):
            break
        if not all(i >= 0 for i in offering_items):
            break
    return log, error_flag