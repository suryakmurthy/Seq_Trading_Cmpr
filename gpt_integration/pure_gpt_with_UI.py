import numpy as np
import os
import openai
import time

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
    response_file = os.path.join(ui, 'response.txt')
    while not os.path.exists(response_file):
        print(f"Waiting for response")
        time.sleep(1)
    # Read the response
    print("Response Obtained")
    with open(response_file, 'r') as f:
        user_input = f.read().strip()
    os.remove(response_file)
    if user_input == "stop":
        log_file = os.path.join(ui, 'log.txt')
        log_opened = open(log_file, "a")
        log_opened.write("\n User: Stopped Trading \n")
        log_opened.close()
        return None, True
    log_file = os.path.join(ui, 'log.txt')
    log_opened = open(log_file, "a")
    log_opened.write("\n" + "User: " + user_input + "\n")
    log_opened.close()

    user_input = "Human: " + user_input
    return user_input, False


def generate_offer_gpt_pure(convo_history):
    """
        Generate a trade offer using GPT

        Args:
            convo_history (dict): Dictionary with the entire conversation history up to this point

        Returns:
            offer (string): GPT's trade offer
    """
    messages = [{"role": "system", "content": "You are an AI negotiation assistant."}] + convo_history
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",  # Ensure you specify the correct model name here
        messages=messages,
    )
    return response.choices[0].message['content']


def sentiment_analysis(convo_history):
    """
        Determine if a human user is accepting or rejecting an offer using sentiment analysis

        Args:
            convo_history (dict): Dictionary with the entire conversation history up to this point

        Returns:
            (boolean): Whether the human user is accepting or rejecting the offer
    """
    user_message = f"Given the following conversation history: \"{convo_history}\", is the responding accepting or rejecting the offering's trade offer? Please answer with 1 for accept, otherwise answer with 0. If the text doesn't provide information on whether the responding is accepting or rejecting the offering's trade offer, respond with 0."
    messages = [{"role": "user", "content": user_message}]
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Ensure you specify the correct model name here
        messages=messages,
    )
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
    trade_dict = {item: 0.0 for item in item_list}
    lines = trade_string.split('\n')
    for line in lines:
        for item in item_list:
            if f"GPT receives:" in line:
                if item in line:
                    parts = line.split()
                    for i in range(len(parts)):
                        if item in parts[i]:
                            quantity = float(parts[i - 1])
                            trade_dict[item] += quantity
            elif f"User receives:" in line:
                if item in line:
                    parts = line.split()
                    for i in range(len(parts)):
                        if item in parts[i]:
                            quantity = -float(parts[i - 1])
                            trade_dict[item] += quantity
    return trade_dict


def offer_search(item_list, ui, convo_history, offer_budget):
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
    while query_count < offer_budget:
        gpt_offer = generate_offer_gpt_pure(convo_history)
        query_count += 1
        responding_response, stopped_trading = query_gpt_pure(gpt_offer, ui)
        convo_history.append({"role": "assistant", "content": gpt_offer})
        convo_history.append({"role": "user", "content": responding_response})
        if stopped_trading:
            return False, [], "", query_count, convo_history, stopped_trading
        accepted = sentiment_analysis(convo_history)
        if accepted == 1:
            vector_trade = parse_trade_offer(gpt_offer, item_list)
            return True, vector_trade, gpt_offer, query_count, convo_history, False
    return False, [], "", query_count, convo_history, True


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

    # Provide GPT a prompt to motivate negotiation
    starting_prompt = """System: You are a offering negotiation agent that aims to maximize your utility function by trading apples, oranges, and bananas with a responding negotiator. Your utility function is: num_apples^2 + 2 * 25 * num_apples} - num_bananas}^2 + 2 * 50 * num_bananas} - num_oranges^2 + 2 * 75 * num_oranges$. You should only make offers if they improve your utility. You start with 50 apples, oranges, and bananas. You will present trade offers to the responding, and they will provide responses. Please only respond with trade offers in the same format as the following example (the ordering and number of each fruit can vary depending on the offer). Please use "GPT" to refer to yourself and "User" to refer to the negotiating opponent: 

    GPT's Trade Offer: 
    
        GPT receives: 5 apples, 2 bananas
        User receives: 3 oranges
    
    
    Negotiation will continue from a new state even after a trade offer is accepted by the human user. Please start the negotiation by providing an initial trade offer for the User."""


    convo_history = [{"role": "system", "content": starting_prompt}]
    trade_num = 0
    log = []
    prev_responding_states = []
    error_flag = False

    while trade_num <= 100:

        # Search for a trade
        trade_num += 1
        found_trade, out_offer, out_offer_string, query_count, convo_history, stopped_trading = offer_search(item_list, ui_folder, convo_history, offer_budget)
        if stopped_trading:
            break
        true_out_offer = np.array(list(out_offer.values()))

        # Update the item values if a trade has been found
        information_dict = {}
        if found_trade == True:

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
            information_dict["responding_items"] = responding_items.tolist()
            information_dict["offering_items"] = offering_items.tolist()
            information_dict["responding_benefit"] = next_state_value_h - prev_state_value_h
            information_dict["offering_benefit"] = next_state_value_a - prev_state_value_a
            information_dict["query_count"] = query_count
            log.append(information_dict)
        else:
            trade_num = 101
        # DEBUG: Check to see if the prior iteration resulted in any negative item values
        if not all(i >= 0 for i in responding_items):
            break
        if not all(i >= 0 for i in offering_items):
            break
    return log, error_flag