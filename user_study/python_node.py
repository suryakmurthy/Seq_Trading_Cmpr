import numpy as np
import os
import sys
import json
from algo_STCR import run_trading_scenario_stcr_gpt
from algo_pure_GPT import run_trading_scenario_gpt_pure
from algo_random import run_trading_scenario_random
from algo_GCA import run_trading_scenario_GCA
import openai
import random

"""
    This program is the backend for the GPT testing website. When negotiation starts, this program will be initialized

    Args:
        target_items (np.array): Total number of items the user would like to obtain
        chat_folder (string): Location of chat folder. At initialization, this will hold the user's target items.
        api_key (string): API key for OpenAI. This is necessary to run the tests.
    Returns:
        log: Stores a log of the chat in a folder denoted by a timestamp.
"""

def get_user_input(folder):
    """
        Obtain a user target items from a folder
        Args:
            folder (string): Location of chat folder. This will hold the user's target values.
        Returns:
            target_values (np.array): Amount of each item the user wants to obtain through trading
    """
    with open(os.path.join(folder, 'target_values.txt'), 'r') as f:
        target_values = json.load(f)
    return [int(target_values['apples']), int(target_values['bananas']), int(target_values['oranges'])], [int(target_values['apples_importance']), int(target_values['bananas_importance']), int(target_values['oranges_importance'])]

def generate_positive_semidefinite_matrix(n):
    """
        Generate an nxn random positive semi-definite matrix
        Args:
            n (int): Number of item categories
        Returns:
            A (np.array): nxn random positive semi-definite matrix
    """
    A = np.random.rand(n, n)  # Generate a random matrix
    A = np.dot(A, A.T)  # Make it symmetric
    return A

def update_counter_value(chat_folder, counter_value):
    # Define the path to the counter.json file
    counter_file_path = os.path.join(chat_folder.split("/")[0], 'counter.json')
    # counter_file_path = os.path.join(chat_folder, 'counter.json')

    # Increment the counter and write it back to the counter.json file
    with open(counter_file_path, 'w') as f:
        json.dump({'counter': counter_value + 1}, f)

def get_counter_value(chat_folder):
    # Define the path to the counter.json file
    counter_file_path = os.path.join(chat_folder.split("/")[0], 'counter.json')
    # print("Folder Directory: ", counter_file_path)
    # If the counter file does not exist, create it with an initial counter of 0
    if not os.path.exists(counter_file_path):
        with open(counter_file_path, 'w') as f:
            json.dump({'counter': 0}, f)

    # Read the current counter value from the file
    with open(counter_file_path, 'r') as f:
        counter_data = json.load(f)
        return counter_data.get('counter', 0)

def main(chat_folder, counter_val=0):
    """
        Run a trading scenario with a human responding agent
        Args:
            chat_folder (string): Location of chat folder. At initialization, this will hold the user's target items.
        Returns:
            log: Stores a log of the chat in a folder denoted by a timestamp.
    """
    openai.api_key = None # Insert Your API Key Here
    item_list = ["apples", "bananas", "oranges"]
    num_items = 3
    integer_constraint = True
    num_scenarios = 1
    # Set seed initially
    b_vals = [[66, 33, 33], [33, 66, 33], [33, 33, 66], [66, 66, 66], [33, 33, 33]]
    program_types = ['stcr', 'random', 'gca']
    combinations = []
    for b_val in b_vals:
        for p_type in program_types:
            combinations.append((p_type, b_val))

    program_type = combinations[counter_val % len(combinations)][0]
    b_agent = 2 * np.array(combinations[counter_val % len(combinations)][1])
    print("Counter Check: "
          "", program_type, b_agent)
    # np.random.seed(None)
    chat_folder_full = chat_folder + "/" + program_type
    for scenario_idx in range(0, num_scenarios):
        human_items = np.array([50, 50, 50]).astype(np.float32)
        agent_items = np.array([50, 50, 50]).astype(np.float32)
        A_human = -np.eye(num_items)
        A_agent = -np.eye(num_items)

        human_target, human_importance = get_user_input(chat_folder_full)

        print("Checking human target: ", chat_folder_full, human_target, human_importance)
        b_human = 2 * np.array(human_target)

        A_human_copy = A_human.copy()
        A_agent_copy = A_agent.copy()
        b_human_copy = b_human.copy()
        b_agent_copy = b_agent.copy()

        # Testing Program Type REMOVE LATER
        # program_type = "gca"

        # Select underlying program
        if program_type == "stcr":
            computer_agent = "ST-CR"
        if program_type == "gca":
            computer_agent = "GCA"
        if program_type == "random":
            computer_agent = "Random Agent"

        # Log the initial state
        algo_item_copy_h = human_items.copy()
        algo_item_copy_a = agent_items.copy()
        human_target_string = f""
        for item_idx in range(0, len(human_target)):
            human_target_string += f"{human_target[item_idx]} {item_list[item_idx]}, "
        log_file = os.path.join(chat_folder_full, 'log.txt')
        log_opened = open(log_file, "a")
        log_opened.write("Start of Trading\n")
        log_opened.write("User Target Items: " + human_target_string + "\n")
        log_opened.write(computer_agent + " Utility Function Q: " + np.array2string(A_agent, precision=2, separator=',',
                                                                        suppress_small=True) + "\n")
        log_opened.write(computer_agent + " Utility Function u: " + str(b_agent / 2) + "\n")
        log_opened.close()

        score_dict = {}
        score_dict["Initial State"] = [50, 50, 50]
        human_final = [50,50,50]
        score_dict["Final State"] = list(human_final)
        score_dict["Target State"] = list(human_target)
        score = np.sum(np.abs(np.array(human_target) - np.array(human_final))) / np.sum(np.abs(np.array(human_target) - human_items))
        if score > 1:
            score = 1
        print("Score: ", score)
        score_dict["Score"] = 1 - score
        with open(f'{chat_folder}/score.json', 'w') as score_file:
            json.dump(score_dict, score_file, indent=4)
        # Run the trading scenario with the specified algorithm
        error_flag = False
        if program_type == "stcr":
            log_algo, error_flag = run_trading_scenario_stcr_gpt(num_items, A_agent_copy, b_agent_copy, A_human_copy,
                                                         b_human_copy, algo_item_copy_a, algo_item_copy_h, item_list,
                                                         chat_folder_full,
                                                         integer_constraint=integer_constraint)
        if program_type == "gca":
            log_algo, error_flag = run_trading_scenario_GCA(num_items, A_agent_copy, b_agent_copy, A_human_copy, b_human_copy,
                                         algo_item_copy_a, algo_item_copy_h, chat_folder_full, item_list)

        if program_type == "gpt":
            log_algo, error_flag = run_trading_scenario_gpt_pure(num_items, A_agent_copy, b_agent_copy, A_human_copy,
                                                                 b_human_copy, algo_item_copy_a, algo_item_copy_h,
                                                                 item_list,
                                                                 chat_folder_full)
        if program_type == "random":
            log_algo = run_trading_scenario_random(num_items, A_agent_copy, b_agent_copy, A_human_copy,
                                                                 b_human_copy, algo_item_copy_a, algo_item_copy_h, 1000,
                                                                 item_list,
                                                                 chat_folder_full)
        # Write Score to file:
        score_dict = {}
        score_dict["Initial State"] = [50, 50, 50]
        human_final = np.array(log_algo[-1]['responding_items'])
        score_dict["Final State"] = list(human_final)
        score_dict["Target State"] = list(human_target)
        # L2
        # score = np.linalg.norm(human_target - human_final) / np.linalg.norm(human_target - human_items)
        # L1
        score = np.sum(np.abs(human_target - human_final)) / np.sum(np.abs(human_target - human_items))
        if score > 1:
            score = 1
        print("Score: ", score)
        score_dict["Score"] = 1 - score
        with open(f'{chat_folder}/score.json', 'w') as score_file:
            json.dump(score_dict, score_file, indent=4)
        # End if an error occurs
        with open(f'{chat_folder_full}/numerical_log.json', 'w') as json_file:
            json.dump(log_algo, json_file, indent=4)
        if error_flag:
            break

if __name__ == '__main__':
    chat_folder = sys.argv[1]
    program_type = sys.argv[2]
    counter_val = int(sys.argv[3])
    print("Reached Main: ", chat_folder, counter_val)
    main(chat_folder, counter_val)