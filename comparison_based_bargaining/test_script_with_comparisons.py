import torch
import random
import json
import numpy as np
import concurrent.futures
from helper_functions import sample_from_simplex, sample_random_ranges_and_lambdas, setup_markowitz_environment_cached
from helper_functions import from_simplex_to_subspace, from_subspace_to_simplex
from solution_concepts import solve_markowitz_subspace_barrier, run_our_solution_concept_actual, run_our_solution_concept_comparisons, solve_nbs_first_order_subspace, solve_nbs_zeroth_order, run_our_solution_concept_comparisons_parallel

def single_test_run(num_agents, n, seed_offset=0):
    seed = 42 + seed_offset
    torch.manual_seed(seed)
    random.seed(seed)

    with open('top_100_tickers_2023.json', 'r') as f:
        tickers = json.load(f)[:n]

    start_date_list, end_date_list, lambda_vals = sample_random_ranges_and_lambdas(num_agents)
    Sigma_set = []
    lambda_mu_set = []

    for agent in range(num_agents):
        Sigma, lambda_mu, _ = setup_markowitz_environment_cached(
            tickers, start_date_list[agent], end_date_list[agent], lambda_vals[agent])
        Sigma_set.append(torch.tensor(Sigma, dtype=torch.float64))
        lambda_mu_set.append(torch.tensor(lambda_mu, dtype=torch.float64))

    solution_set = []
    for Sigma, lambda_mu in zip(Sigma_set, lambda_mu_set):
        w_opt = solve_markowitz_subspace_barrier(Sigma, lambda_mu)
        x_opt = from_simplex_to_subspace(w_opt)
        solution_set.append(x_opt)

    starting_state_x = torch.tensor(sample_from_simplex(n), dtype=torch.float64)
    starting_state_projected = from_simplex_to_subspace(starting_state_x)

    final_point_comparisons, query_count_ours = run_our_solution_concept_comparisons_parallel(starting_state_projected, Sigma_set, lambda_mu_set, solution_set)
    final_point = run_our_solution_concept_actual(starting_state_projected, Sigma_set, lambda_mu_set, solution_set)
    nbs_point = solve_nbs_first_order_subspace(Sigma_set, lambda_mu_set, starting_point=starting_state_projected)
    nbs_point_zeroth_order, query_count_nbs = solve_nbs_zeroth_order(Sigma_set, lambda_mu_set, starting_point=starting_state_projected)

    final_simplex = from_subspace_to_simplex(final_point)
    final_simplex_comparison = from_subspace_to_simplex(final_point_comparisons)
    nbs_simplex = nbs_point
    distance = torch.norm(final_simplex - nbs_simplex).item()
    return final_simplex.tolist(), nbs_simplex.tolist(), final_simplex_comparison.tolist(), nbs_point_zeroth_order.tolist(),query_count_ours, query_count_nbs, distance


if __name__ == "__main__":
    seed = 42
    torch.set_default_dtype(torch.float64)
    num_agents_list = [2, 3, 5, 10, 50]
    n_list = [5, 10, 20, 50]
    distance_dict = {}
    num_tests = 10

    for num_agents in num_agents_list:
        distance_dict[num_agents] = {}
        for n in n_list:
            print(f"Running {num_tests} tests for {num_agents} agents and {n} stocks...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(single_test_run, num_agents, n, i) for i in range(num_tests)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            distance_dict[num_agents][n] = results
            distances = [r[-1] for r in results]
            print(f"Average Distance with {num_agents} Agents and {n} Stocks: {np.mean(distances):.6f}")

    with open('solution_concept_nash_comparisons_results.json', 'w') as f:
        json.dump(distance_dict, f)