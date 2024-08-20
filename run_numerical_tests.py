import os
import numpy as np
import json
from tqdm import tqdm
from algo_stcr import run_trading_scenario_stcr
from algo_random_version import run_trading_scenario_random
from algo_random_momentum import run_trading_scenario_random_trading_momentum
import matplotlib.pyplot as plt
import tikzplotlib
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

def pad_lists(lists, max_length):
    """
        Given a set of lists, pad the lists so they reach the maximum length.
        Args:
            lists (list of lists): Set of lists represeting trading progression of different algorithms/scenarios
            max_length (int): Maximum length of any trading progression. All of the lists in the set will be this length at the end of this function.
        Returns:
            padded_lists (list of lists): Set of lists that have been padded to max_length by repeating the final value.
    """
    padded_lists = []
    for lst in lists:
        if len(lst) != 0:
            padded_lst = lst + [lst[-1]] * (max_length - len(lst))  # Pad with zeros
            padded_lists.append(padded_lst)

    return padded_lists

def obtain_plot_data(data):
    """
        Given trading progression data for all algorithms, obtain the normalized average cumulative benefit information for each algorithm
        Args:
            data (dictonary): Trading progression for all scenarios/algorithms for a given test set.
        Returns:
            tuple:

                - average_benefit_list (dict): Dictionary containing lists of cumulative societal benefit for each algorithm.
                - average_benefit_list_responding (dict): Dictionary containing lists of cumulative benefit for the responding agent for each algorithm.
                - average_benefit_list_offering (dict): Dictionary containing lists of cumulative benefit for the offering agent for each algorithm.
                - average_trade_list (dict):  Dictionary containing lists of cumulative offers for the offering agent for each algorithm.
    """
    benefit = {}
    query = {}
    length = {}
    responding_benefit = {}
    offering_benefit = {}
    for key in data.keys():
        benefit[key] = [
            [data[key][i][entry_idx]["responding_benefit"] + data[key][i][entry_idx]["offering_benefit"] for entry_idx in
             range(0, len(data[key][i]))] for i in range(0, len(data[key]))]
        responding_benefit[key] = [[data[key][i][entry_idx]["responding_benefit"] for entry_idx in range(0, len(data[key][i]))]
                              for i in range(0, len(data[key]))]
        offering_benefit[key] = [
            [data[key][i][entry_idx]["offering_benefit"] for entry_idx in range(0, len(data[key][i]))] for i in
            range(0, len(data[key]))]
        query[key] = [[entry["query_count"] for entry in data[key][i]] for i in range(0, len(data[key]))]
        length[key] = np.mean([len(data[key][i]) for i in range(0, len(data[key]))])
    query_benefit_list = {}
    query_responding_benefit_list = {}
    query_offering_benefit_list = {}
    query_trade_list = {}
    for key in data.keys():
        query_benefit_list[key] = []
        query_responding_benefit_list[key] = []
        query_offering_benefit_list[key] = []
        query_trade_list[key] = []
        for i in range(len(benefit[key])):
            sum_queries = 0
            sum_benefit = 0
            sum_benefit_responding = 0
            sum_benefit_offering = 0
            sub_query_list = []
            sub_trade_list = []
            sub_query_list_h = []
            sub_query_list_c = []
            for j in range(len(benefit[key][i])):
                for q in range(query[key][i][j]):
                    sub_query_list.append(sum_benefit)
                    sub_query_list_h.append(sum_benefit_responding)
                    sub_query_list_c.append(sum_benefit_offering)
                    sub_trade_list.append(j)
                sum_benefit += benefit[key][i][j]
                sum_queries += query[key][i][j]
                sum_benefit_offering += offering_benefit[key][i][j]
                sum_benefit_responding += responding_benefit[key][i][j]
            query_benefit_list[key].append(sub_query_list)
            query_responding_benefit_list[key].append(sub_query_list_h)
            query_offering_benefit_list[key].append(sub_query_list_c)
            query_trade_list[key].append(sub_trade_list)
    len_list = {}
    max_len_list = []
    max_benefit_vals = {}
    for key in data.keys():
        len_list[key] = [len(entry) for entry in query_benefit_list[key]]
        max_benefit_vals[key] = np.mean(
            [query_benefit_list[key][i][-1] for i in range(len(query_benefit_list[key])) if query_benefit_list[key][i]])
        max_len_list.append(max(len_list[key]))

    max_len_list_val = max(max_len_list)
    padded_query_benefit_lists = {}
    padded_query_benefit_lists_h = {}
    padded_query_benefit_lists_c = {}
    padded_query_trade_lists = {}
    for key in data.keys():
        padded_query_benefit_lists[key] = pad_lists(query_benefit_list[key], max_len_list_val)
        padded_query_benefit_lists_h[key] = pad_lists(query_responding_benefit_list[key], max_len_list_val)
        padded_query_benefit_lists_c[key] = pad_lists(query_offering_benefit_list[key], max_len_list_val)
        padded_query_trade_lists[key] = pad_lists(query_trade_list[key], max_len_list_val)
    average_benefit_list = {}
    average_benefit_list_responding = {}
    average_benefit_list_offering = {}
    average_trade_list = {}
    for key in data.keys():
        # if "Random" in key:
        average_benefit_list[key] = []
        average_benefit_list_responding[key] = []
        average_benefit_list_offering[key] = []
        average_trade_list[key] = []
        for i in range(max_len_list_val):
            count = 0
            sum_benefit = 0
            sum_benefit_responding = 0
            sum_benefit_offering = 0
            sum_trades = 0
            for j in range(len(padded_query_benefit_lists[key])):
                if len(padded_query_benefit_lists[key][j]) > i:
                    sum_benefit += padded_query_benefit_lists[key][j][i]
                    sum_benefit_responding += padded_query_benefit_lists_h[key][j][i]
                    sum_benefit_offering += padded_query_benefit_lists_c[key][j][i]
                    sum_trades += padded_query_trade_lists[key][j][i]
                    count += 1
            average_benefit_list[key].append(sum_benefit / count)
            average_benefit_list_responding[key].append(sum_benefit_responding / count)
            average_benefit_list_offering[key].append(sum_benefit_offering / count)
            average_trade_list[key].append(sum_trades / count)
    return average_benefit_list, average_benefit_list_responding, average_benefit_list_offering, average_trade_list

def plot_for_combinations(num_items, mixing_constant, log_stcr, log_stcr_prev, log_random_momentum, log_random_prev, log_random_pure, integer_constraint=False, debug=False):
    """
        Plot normalized cumulative benefit for all algorithms
        Args:
            num_items (int): Total number of item categories
            mixing_constant (float): Mixing constant that controls alignment between utility functions
            log_stcr (dict): Log of trading progression for ST-CR without heuristics
            log_stcr_prev (dict): Log of trading progression for ST-CR with previous trade heuristic
            log_random_momentum (dict): Log of trading progression for random trading with momentum
            log_random_prev (dict): Log of trading progression for random trading with previous trade heuristic
            log_random_pure (dict): Log of trading progression for random trading without heuristics
            integer_constraint (bool, optional): Whether the offers were integer-constrained.
            debug (bool, optional): Whether the program should display the plots in real-time.
        Returns:
            Nothing
    """
    # Extract plot data for each algorithm
    mixing_constant_string = str(mixing_constant).replace(".", "_")
    data = {}
    data["ST-CR Pure"] = log_stcr
    data["ST-CR with Previous Trade Heuristic"] = log_stcr_prev
    data["Random Trades without Heuristic"] = log_random_pure
    data["Random Trades with Previous Trade Heuristic"] = log_random_prev
    data["Random Trades with Momentum"] = log_random_momentum

    # Obtain plot data
    average_benefit_list, average_benefit_list_responding, average_benefit_list_offering, average_trade_list = obtain_plot_data(
        data)
    max_value = max(max(average_benefit_list[key][0:1000]) for key in average_benefit_list.keys())

    # Plotting
    for key in average_benefit_list.keys():
        plt.plot([value / max_value for value in average_benefit_list[key][0:1000]], label=key)

    plt.xlabel('Offer Count')
    plt.ylabel('Cumulative Societal Benefit')
    plt.title(f'Comparison of Cumulative Benefit with {mixing_constant} alignment ({num_items} Items)')
    plt.legend()

    # Save the plot as a TikZ file
    if integer_constraint:
        print("Plot Saved as: " + f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int.tex")
        tikzplotlib.save(f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int.tex")
        plt.savefig(f'numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int.png')
    else:
        print("Plot Saved as: " + f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}.tex")
        tikzplotlib.save(f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}.tex")
        plt.savefig(f'numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}.png')

    # Display the plot
    if debug:
        plt.show()
    plt.clf()

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run trading scenarios with specified options.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--integer_constraint', action='store_true',
                        help='Enable integer constraint in the trading scenarios')
    parser.add_argument('--random_seed', type=int, default=10,
                        help='Random Seed for initializing testing scenarios. Default Value: 10')
    parser.add_argument('--max_trade_value', type=int, default=5,
                        help='Maximum number of items from each category that can be exchanged in a single trade. Default Value: 5')
    parser.add_argument('--theta_closeness', type=float, default=0.00001,
                        help='Semi-Vertical Angle Threshold for ST-CR. Default Value: 0.00001')
    parser.add_argument('--deviation_interval', type=float, default=0.05,
                        help='Deviation interval of increase for random trading with momentum. Default Value: 0.05')
    parser.add_argument('--max_deviation_magnitude', type=float, default=5,
                        help='Maximum deviation magnitude for random trading with momentum. Default Value: 5')
    parser.add_argument('--num_scenarios', type=float, default=1000,
                        help='Number of Randomized Testing Scenarios. Default Value: 1000')
    parser.add_argument('--offer_budget', type=float, default=1000,
                        help='Offer Budget for Negotiation Algorithms. Default Value: 1000')
    args = parser.parse_args()

    # Debug flag causes the algorithms to print status messages after each trade
    debug = args.debug

    # Initialize Scenario Parameters
    # Number of Item Categories
    num_items_list = [3, 5]

    # Mixing Constants Control Alignment between the Agent's Utilities.
    mixing_constants = [1.1, 4, 5]

    # The total number of testing scenarios per testing group
    num_scenarios = args.num_scenarios

    # Constrain offers to integer values
    integer_constraint = args.integer_constraint

    # Initialize Algorithm Hyperparameters:
    # The maximum number of items from any category that can be exchanged in a single trade.
    # This parameter corresponds to the maximum trade magnitude d described in the "Sequential Trading with Cone Refinement" Section.
    max_trade_value = args.max_trade_value

    # Theta Closeness is a threshold on the semi-vertical angle for ST-CR. If the cone's angle is less than theta_closeness, then the two agent's gradients are highly aligned and ST-CR stops trading (p.5)
    theta_closeness = args.theta_closeness

    # Random trading with Momentum Hyperparameters
    max_deviation_magnitude = args.max_deviation_magnitude
    deviation_interval = args.deviation_interval

    # The maximum number of offers each algorithm can make per scenario
    query_budget = args.offer_budget

    for num_items in num_items_list:
        for mixing_constant in mixing_constants:
            iteration_logs_algo_pure = []
            iteration_logs_algo_prev_accepted_trade = []
            iteration_logs_random_momentum = []
            iteration_logs_random_informed = []
            iteration_logs_random_pure = []
            # Set random seed for reproducibility
            np.random.seed(args.random_seed)
            for scenario_idx in tqdm(range(num_scenarios), desc=f"Items: {num_items}, Mixing: {mixing_constant}"):
                # Don't randomize starting states for alignment testing
                if mixing_constant != 4:
                    responding_items = np.array([float(100) for i in range(num_items)])
                    offering_items = np.array([float(100) for i in range(num_items)])
                else:
                    responding_items = np.random.randint(1, 201, size=num_items)
                    responding_items = responding_items.astype(np.float32)
                    offering_items = np.random.randint(1, 201, size=num_items)
                    offering_items = offering_items.astype(np.float32)

                A_responding = -generate_positive_semidefinite_matrix(num_items)
                A_offering = -generate_positive_semidefinite_matrix(num_items)
                A_responding, A_offering = ((mixing_constant * A_responding) + A_offering) / (mixing_constant + 1), (
                            A_responding + (mixing_constant * A_offering)) / (mixing_constant + 1)

                b_responding = 2 * np.random.randint(1, 201, size=num_items)
                b_offering = 2 * np.random.randint(1, 201, size=num_items)

                b_responding, b_offering = ((mixing_constant * b_responding) + b_offering) / (mixing_constant + 1), (
                            b_responding + (mixing_constant * b_offering)) / (mixing_constant + 1)

                # ST-CR without Heuristics
                A_responding_copy = A_responding.copy()
                A_offering_copy = A_offering.copy()
                b_responding_copy = b_responding.copy()
                b_offering_copy = b_offering.copy()
                algo_item_copy_h = responding_items.copy()
                algo_item_copy_a = offering_items.copy()
                log_algo_pure, error_flag = run_trading_scenario_stcr(num_items, A_offering_copy, b_offering_copy,
                                                                      A_responding_copy, b_responding_copy, algo_item_copy_a,
                                                                      algo_item_copy_h, query_budget, max_trade_value, theta_closeness,
                                                                      integer_constraint=integer_constraint, debug=debug)
                if error_flag:
                    break
                iteration_logs_algo_pure.append(log_algo_pure)

                # ST-CR with Prior Trade Heuristic
                A_responding_copy = A_responding.copy()
                A_offering_copy = A_offering.copy()
                b_responding_copy = b_responding.copy()
                b_offering_copy = b_offering.copy()
                algo_item_copy_h = responding_items.copy()
                algo_item_copy_a = offering_items.copy()
                log_algo_prev, error_flag = run_trading_scenario_stcr(num_items, A_offering_copy, b_offering_copy,
                                                                      A_responding_copy, b_responding_copy, algo_item_copy_a,
                                                                      algo_item_copy_h, query_budget, max_trade_value, theta_closeness,
                                                                      integer_constraint=integer_constraint,
                                                                      prev_offer_flag=True, debug=debug)  # item_list)
                if error_flag:
                    break
                iteration_logs_algo_prev_accepted_trade.append(log_algo_prev)

                # Random Search Method with Momentum
                A_responding_copy = A_responding.copy()
                A_offering_copy = A_offering.copy()
                b_responding_copy = b_responding.copy()
                b_offering_copy = b_offering.copy()
                algo_item_copy_h = responding_items.copy()
                algo_item_copy_a = offering_items.copy()
                log_random_momentum = run_trading_scenario_random_trading_momentum(num_items, A_offering_copy,
                                                                                   b_offering_copy, A_responding_copy,
                                                                                   b_responding_copy, algo_item_copy_a,
                                                                                   algo_item_copy_h, query_budget, deviation_interval=deviation_interval, max_deviation_magnitude=max_deviation_magnitude, max_trade_value=max_trade_value,
                                                                                   integer_constraint=integer_constraint, debug=debug)

                # Random Search Method with Prior Trade Heuristics
                A_responding_copy = A_responding.copy()
                A_offering_copy = A_offering.copy()
                b_responding_copy = b_responding.copy()
                b_offering_copy = b_offering.copy()
                algo_item_copy_h = responding_items.copy()
                algo_item_copy_a = offering_items.copy()
                log_random_informed = run_trading_scenario_random(num_items, A_offering_copy, b_offering_copy,
                                                                  A_responding_copy, b_responding_copy, algo_item_copy_a,
                                                                  algo_item_copy_h, query_budget, max_trade_value=max_trade_value,
                                                                  integer_constraint=integer_constraint, informed=True, debug = debug)

                # Random Search Method without Heuristics
                A_responding_copy = A_responding.copy()
                A_offering_copy = A_offering.copy()
                b_responding_copy = b_responding.copy()
                b_offering_copy = b_offering.copy()
                algo_item_copy_h = responding_items.copy()
                algo_item_copy_a = offering_items.copy()
                log_random_pure = run_trading_scenario_random(num_items, A_offering_copy, b_offering_copy, A_responding_copy,
                                                              b_responding_copy, algo_item_copy_a, algo_item_copy_h, query_budget, max_trade_value=max_trade_value,
                                                              integer_constraint=integer_constraint, informed=False, debug = debug)

                iteration_logs_random_momentum.append(log_random_momentum)
                iteration_logs_random_informed.append(log_random_informed)
                iteration_logs_random_pure.append(log_random_pure)
                mixing_constant_string = str(mixing_constant).replace(".", "_")
                # Log Trading Progression
                if debug:
                    if scenario_idx % 10 == 0:
                        folder_path = f'testing_results_{num_items}_items_alignment_{mixing_constant_string}'
                        if integer_constraint:
                            folder_path += "_int"
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        with open(f'{folder_path}/st_cr_pure.json',
                                  'w') as file:
                            json.dump(iteration_logs_algo_pure, file, indent=4)
                        with open(
                                f'{folder_path}/st_cr_prev_offer.json',
                                'w') as file:
                            json.dump(iteration_logs_algo_prev_accepted_trade, file, indent=4)
                        with open(
                                f'{folder_path}/random_momentum.json',
                                'w') as file:
                            json.dump(iteration_logs_random_informed, file, indent=4)
                        with open(
                                f'{folder_path}/random_informed.json',
                                'w') as file:
                            json.dump(iteration_logs_random_informed, file, indent=4)
                        with open(f'{folder_path}/random_pure.json',
                                  'w') as file:
                            json.dump(iteration_logs_random_pure, file, indent=4)
            plot_for_combinations(num_items, mixing_constant, iteration_logs_algo_pure,
                                  iteration_logs_algo_prev_accepted_trade, iteration_logs_random_momentum,
                                  iteration_logs_random_informed, iteration_logs_random_pure, integer_constraint=integer_constraint, debug=debug)


if __name__ == "__main__":
    main()