import torch
import random
import json
import numpy as np
import concurrent.futures
from helper_functions import sample_from_simplex, sample_random_ranges_and_lambdas, setup_markowitz_environment_cached
from solution_concepts import solve_markowitz, run_our_solution_concept_actual, run_our_solution_concept_comparisons, solve_nbs_first_order_simplex, solve_nbs_zeroth_order, run_our_solution_concept_comparisons_parallel, run_our_solution_concept_comparisons_parallel_sign_opt

def single_test_run(num_agents, n, seed_offset=0):
    base_seed = 577
    torch.manual_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)

    with open('top_100_tickers_2023.json', 'r') as f:
        tickers = json.load(f)[:n]

    success = False
    attempt = 0
    solution_set = []
    solution_set_list = []
    Sigma_set_list = []
    lambda_mu_set_list = []
    while not success:
        # Increment seed to ensure variation across retries
        seed = base_seed + attempt
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        start_date_list, end_date_list, lambda_vals = sample_random_ranges_and_lambdas(num_agents)
        Sigma_set = []
        lambda_mu_set = []
        Sigma_set_list = []
        lambda_mu_set_list = []
        for agent in range(num_agents):
            Sigma, lambda_mu, _ = setup_markowitz_environment_cached(
                tickers, start_date_list[agent], end_date_list[agent], lambda_vals[agent])
            Sigma_set_list.append(Sigma.tolist())
            lambda_mu_set_list.append(lambda_mu.tolist())
            Sigma_set.append(torch.tensor(Sigma, dtype=torch.float64))
            lambda_mu_set.append(torch.tensor(lambda_mu, dtype=torch.float64))

        solution_set = []
        solution_set_list = []
        valid = True
        for Sigma, lambda_mu in zip(Sigma_set, lambda_mu_set):
            w_opt = solve_markowitz(Sigma, lambda_mu)
            if w_opt is None:
                valid = False
                break
            solution_set.append(w_opt)
            solution_set_list.append(w_opt.detach().cpu().numpy().tolist())

        if valid:
            success = True
        else:
            print(f"Resampling due to solver failure... (seed: {seed})")
            attempt += 1


    starting_state_w = torch.tensor(sample_from_simplex(n), dtype=torch.float64)
    # print("1: ", starting_state_w)
    final_point_comparisons, query_count_ours = run_our_solution_concept_comparisons_parallel_sign_opt(starting_state_w, Sigma_set, lambda_mu_set, solution_set)
    # print("2: ", starting_state_w)
    final_point = run_our_solution_concept_actual(starting_state_w, Sigma_set, lambda_mu_set, solution_set)
    # print("3: ", starting_state_w)
    nbs_point = solve_nbs_first_order_simplex(Sigma_set, lambda_mu_set, starting_point=starting_state_w)
    # print("4: ", starting_state_w)
    nbs_point_zeroth_order, query_count_nbs = solve_nbs_zeroth_order(Sigma_set, lambda_mu_set, starting_point=starting_state_w)

    final_simplex = final_point
    final_simplex_comparison = final_point_comparisons

    print("Checking final points: ", nbs_point, nbs_point_zeroth_order, seed)
    nbs_simplex = nbs_point
    distance = torch.norm(final_simplex - nbs_simplex).item()
    distance_between_comparison_solutions = torch.norm(final_simplex - final_simplex_comparison).item()
    distance_between_nash_solutions = torch.norm(nbs_point - nbs_point_zeroth_order).item()
    return Sigma_set_list, lambda_mu_set_list, final_simplex.tolist(), nbs_simplex.tolist(), final_simplex_comparison.tolist(), nbs_point_zeroth_order.tolist(), query_count_ours, query_count_nbs, starting_state_w.tolist(), solution_set_list, distance


if __name__ == "__main__":
    seed = 42
    torch.set_default_dtype(torch.float64)
    num_agents_list = [50]
    n_list = [50]
    distance_dict = {}
    num_tests = 1

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