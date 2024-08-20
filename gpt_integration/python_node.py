import numpy as np
import os
import sys
import json
from stcr_ver_gpt_with_UI import run_trading_scenario_stcr_gpt
from pure_gpt_with_UI import run_trading_scenario_gpt_pure
import openai

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
    return [int(target_values['apples']), int(target_values['bananas']), int(target_values['oranges'])]

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

def main(program_type, chat_folder, api_key, random_seed):
    """
        Run a trading scenario with a human responding agent
        Args:
            program_type (string): The model that the human will negotiate with either "gpt" or "st-cr"
            chat_folder (string): Location of chat folder. At initialization, this will hold the user's target items.
            api_key (string): API key for OpenAI. This is necessary to run the tests.
        Returns:
            log: Stores a log of the chat in a folder denoted by a timestamp.
    """
    np.random.seed(int(random_seed))  # Convert random_seed to integer
    openai.api_key = api_key
    item_list = ["apples", "bananas", "oranges"]
    num_items = 3
    integer_constraint = True
    num_scenarios = 1
    for scenario_idx in range(0, num_scenarios):
        human_items = np.array([50, 50, 50]).astype(np.float32)
        agent_items = np.array([50, 50, 50]).astype(np.float32)
        A_human = -np.eye(num_items)
        A_agent = -np.eye(num_items)

        human_target = get_user_input(chat_folder)

        b_human = 2 * np.array(human_target)
        if program_type == "gpt":
            b_agent = 2 * np.array([25, 50, 75]) # np.random.randint(1, 51, size=num_items)
        else:
            b_agent = 2 * np.random.randint(1, 100, size=num_items)
        A_human_copy = A_human.copy()
        A_agent_copy = A_agent.copy()
        b_human_copy = b_human.copy()
        b_agent_copy = b_agent.copy()

        # Select underlying program
        if program_type == "stcr":
            computer_agent = "ST-CR"
        else:
            computer_agent = "GPT"

        # Log the initial state
        algo_item_copy_h = human_items.copy()
        algo_item_copy_a = agent_items.copy()
        human_target_string = f""
        for item_idx in range(0, len(human_target)):
            human_target_string += f"{human_target[item_idx]} {item_list[item_idx]}, "
        log_file = os.path.join(chat_folder, 'log.txt')
        log_opened = open(log_file, "a")
        log_opened.write("Start of Trading\n")
        log_opened.write("User Target Items: " + human_target_string + "\n")
        log_opened.write(computer_agent + " Utility Function Q: " + np.array2string(A_agent, precision=2, separator=',',
                                                                        suppress_small=True) + "\n")
        log_opened.write(computer_agent + " Utility Function u: " + str(b_agent) + "\n")
        log_opened.close()

        # Run the trading scenario with the specified algorithm
        if program_type == "stcr":
            log_algo, error_flag = run_trading_scenario_stcr_gpt(num_items, A_agent_copy, b_agent_copy, A_human_copy,
                                                         b_human_copy, algo_item_copy_a, algo_item_copy_h, item_list,
                                                         chat_folder,
                                                         integer_constraint=integer_constraint)
        else:
            log_algo, error_flag = run_trading_scenario_gpt_pure(num_items, A_agent_copy, b_agent_copy, A_human_copy,
                                                                 b_human_copy, algo_item_copy_a, algo_item_copy_h,
                                                                 item_list,
                                                                 chat_folder)
        # End if an error occurs
        if error_flag:
            break

if __name__ == '__main__':
    chat_folder = sys.argv[2]
    program_type = sys.argv[1]  # Fixed typo here
    api_key = sys.argv[3]
    random_seed = sys.argv[4]
    main(program_type, chat_folder, api_key, random_seed)