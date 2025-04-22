import torch
import random
import json
import numpy as np
import concurrent.futures
from helper_functions import sample_from_simplex, sample_random_ranges_and_lambdas, setup_markowitz_environment_cached
from solution_concepts import solve_markowitz, run_our_solution_concept_actual, solve_nbs_first_order_simplex

def single_test_run(num_agents, n, seed_offset=0):
    base_seed = 577

    with open('top_100_tickers_2023.json', 'r') as f:
        tickers = json.load(f)[:n]

    success = False
    attempt = 0
    solution_set = []
    solution_set_np = []
    while not success:
        # Increment seed to ensure variation across retries
        seed = base_seed + attempt
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        start_date_list, end_date_list, lambda_vals = sample_random_ranges_and_lambdas(num_agents)
        Sigma_set = []
        lambda_mu_set = []

        for agent in range(num_agents):
            Sigma, lambda_mu, _ = setup_markowitz_environment_cached(
                tickers, start_date_list[agent], end_date_list[agent], lambda_vals[agent])
            Sigma_set.append(torch.tensor(Sigma, dtype=torch.float64))
            lambda_mu_set.append(torch.tensor(lambda_mu, dtype=torch.float64))

        solution_set = []
        solution_set_np = []
        valid = True
        for Sigma, lambda_mu in zip(Sigma_set, lambda_mu_set):
            w_opt = solve_markowitz(Sigma, lambda_mu)
            if w_opt is None:
                valid = False
                break
            solution_set.append(w_opt)
            solution_set_np.append(w_opt.detach().cpu().numpy().tolist())

        if valid:
            success = True
        else:
            print(f"Resampling due to solver failure... (seed: {seed})")
            attempt += 1

    # Rest of your logic (now guaranteed to have valid solution_set)
    starting_state_w = torch.tensor(sample_from_simplex(n), dtype=torch.float64)
    print("Initial Check of Starting Point: ", starting_state_w)
    final_point = run_our_solution_concept_actual(starting_state_w, Sigma_set, lambda_mu_set, solution_set)
    print("Double-Checking Starting Point: ", starting_state_w)
    nbs_point = solve_nbs_first_order_simplex(Sigma_set, lambda_mu_set, starting_point=starting_state_w)

    distance = torch.norm(final_point - nbs_point).item()
    if ((final_point < 0).any()) or ((nbs_point < 0).any()) or torch.abs(torch.sum(final_point) - 1) > 1e-6 or torch.abs(torch.sum(nbs_point) - 1) > 1e-6:
        print("ERROR CASE: ", final_point, nbs_point, torch.sum(final_point), torch.sum(nbs_point), distance, seed)

    return final_point.tolist(), nbs_point.tolist(), starting_state_w.detach().cpu().numpy().tolist(), solution_set_np, distance



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
            with open('solution_concept_nash_results.json', 'w') as f:
                json.dump(distance_dict, f)