import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tikzplotlib
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

def plot_for_combinations(num_items, mixing_constant, log_stcr, log_stcr_prev, log_random_pure, log_random_informed, log_random_momentum, log_gca, integer_constraint=False, debug=False):
    """
        Plot normalized cumulative benefit for all algorithms
        Args:
            num_items (int): Total number of item categories
            mixing_constant (float): Mixing constant that controls alignment between utility functions
            log_stcr (dict): Log of trading progression for ST-CR without heuristics
            log_stcr_prev (dict): Log of trading progression for ST-CR with previous trade heuristic
            log_random_momentum (dict): Log of trading progression for random trading with momentum
            log_random_prev (dict): Log of trading progression for random trading with previous trade heuristic
            log_random_momentum (dict): Log of trading progression for random trading without heuristics
            integer_constraint (bool, optional): Whether the offers were integer-constrained.
            debug (bool, optional): Whether the program should display the plots in real-time.
        Returns:
            Nothing
    """
    # Extract plot data for each algorithm
    mixing_constant_string = str(mixing_constant).replace(".", "_")
    data = {}
    data["ST-CR without Heuristics"] = log_stcr
    data["ST-CR with Previous Trade Heuristic"] = log_stcr_prev
    data["Random Trades without Heuristics"] = log_random_pure
    data["Random Trades with Previous Trade Heuristic"] = log_random_informed
    data["Random Trades with Momentum"] = log_random_momentum
    if integer_constraint:
        for update_interval in log_gca.keys():
            data[f"GCA Updating After {update_interval} Offers"] = log_gca[update_interval]

    # Obtain plot data
    average_benefit_list, average_benefit_list_responding, average_benefit_list_offering, average_trade_list = obtain_plot_data(
        data)
    max_value = max(max(average_benefit_list_offering[key][0:500]) for key in average_benefit_list_offering.keys())

    for key in average_benefit_list_offering.keys():
        plt.plot([value / max_value for value in average_benefit_list_offering[key][0:500]], label=key)

    plt.xlabel('Offer Count')
    plt.ylabel('Cumulative Offering Benefit')
    plt.title(f'Comparison of Cumulative Offering Benefit ({num_items} Items)')
    plt.legend()
    if integer_constraint:
        print("Plot Saved as: " + f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int_offering.tex")
        tikzplotlib.save(f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int_offering.tex")
        plt.savefig(f'numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int_offering.png')
    else:
        print("Plot Saved as: " + f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_offering.tex")
        tikzplotlib.save(f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_offering.tex")
        plt.savefig(f'numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_offering.png')
    plt.show()

    max_value = max(max(average_benefit_list_responding[key][0:500]) for key in average_benefit_list_responding.keys())
    for key in average_benefit_list_responding.keys():
        plt.plot([value / max_value for value in average_benefit_list_responding[key][0:500]], label=key)

    plt.xlabel('Offer Count')
    plt.ylabel('Cumulative Responding Benefit')
    plt.title(f'Comparison of Cumulative Responding Benefit alignment ({num_items} Items)')
    plt.legend()
    if integer_constraint:
        print("Plot Saved as: " + f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int_responding.tex")
        tikzplotlib.save(f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int_responding.tex")
        plt.savefig(f'numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int_responding.png')
    else:
        print("Plot Saved as: " + f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_responding.tex")
        tikzplotlib.save(f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_responding.tex")
        plt.savefig(f'numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_responding.png')
    
    plt.show()

    max_value = max(max(average_benefit_list[key][0:500]) for key in average_benefit_list.keys())
    # Plotting
    for key in average_benefit_list.keys():
        plt.plot([value / max_value for value in average_benefit_list[key][0:500]], label=key)

    plt.xlabel('Offer Count')
    plt.ylabel('Cumulative Societal Benefit')
    plt.title(f'Comparison of Cumulative Benefit ({num_items} Items)')
    plt.legend()
    if integer_constraint:
        print("Plot Saved as: " + f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int.tex")
        tikzplotlib.save(f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int.tex")
        plt.savefig(f'numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}_int.png')
    else:
        print("Plot Saved as: " + f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}.tex")
        tikzplotlib.save(f"numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}.tex")
        plt.savefig(f'numerical_results/query_benefit_curves_alignment_{mixing_constant_string}_items_{num_items}.png')
    plt.show()

    # Save the plot as a TikZ file

    plt.clf()

num_items=3
mixing_constant_string="10"
iteration_logs_gca = {}
folder_path = f'testing_results_{num_items}_items_alignment_{mixing_constant_string}'
with open(f'{folder_path}/st_cr_pure.json','r') as file:
    iteration_logs_algo_pure = json.load(file)
with open(f'{folder_path}/st_cr_prev_offer.json','r') as file:
    iteration_logs_algo_prev_offer = json.load(file)
with open(f'{folder_path}/gca_update_1.json','r') as file:
    iteration_logs_gca[1] = json.load(file)
with open(f'{folder_path}/gca_update_10.json','r') as file:
    iteration_logs_gca[10] = json.load(file)
with open(f'{folder_path}/gca_update_100.json','r') as file:
    iteration_logs_gca[100] = json.load(file)
with open(f'{folder_path}/random_momentum.json', 'r') as file:
    iteration_logs_random_momentum = json.load(file)
with open(f'{folder_path}/random_pure.json', 'r') as file:
    iteration_logs_random_pure = json.load(file)
with open(f'{folder_path}/random_informed.json', 'r') as file:
    iteration_logs_random_informed = json.load(file)
print(len(iteration_logs_random_informed))
plot_for_combinations(num_items, mixing_constant_string, iteration_logs_algo_pure, iteration_logs_algo_prev_offer, iteration_logs_random_pure, iteration_logs_random_informed, iteration_logs_random_momentum, iteration_logs_gca, integer_constraint=True, debug=True)