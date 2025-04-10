import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
from cone_refinement_functions import estimate_gradient_f_i_comparisons, true_markowitz_gradient, true_markowitz_gradient_w, angle_between
import random
import torch
from itertools import combinations


import torch

def sample_from_simplex(n):
    """Sample a point uniformly at random from the n-dimensional simplex."""
    return np.random.dirichlet(np.ones(n))

def is_pareto_optimal_via_perturbation(x, Sigma_set, lambda_mu_set, epsilon=1e-2, num_directions=50, val_eps=1e-6):
    """
    Check local Pareto optimality by testing if any small perturbation improves all objectives.
    """
    w = new_softmax_transform(x)

    def evaluate(w):
        return torch.stack([
            torch.dot(w, Sigma_set[i] @ w) - torch.dot(lambda_mu_set[i], w)
            for i in range(len(Sigma_set))
        ])

    f_x = evaluate(w)

    for _ in range(num_directions):
        # Sample random perturbation direction
        delta = torch.randn_like(x)
        delta = delta / torch.norm(delta)

        x_perturbed = x + epsilon * delta
        w_perturbed = new_softmax_transform(x_perturbed)
        f_perturbed = evaluate(w_perturbed)

        # If all objectives are improved, it's not Pareto-optimal
        if torch.all(f_x - f_perturbed > val_eps):
            # print("Failure Delta:", delta, "\nObjective Improvement:", f_x - f_perturbed)
            return False

    return True  # No improving direction found ⇒ locally Pareto-optimal


def softmax(x):
    e = torch.exp(x - torch.max(x))  # Subtract max for numerical stability
    return e / torch.sum(e)

def new_softmax_transform(x_space_vector):
    x_n = -torch.sum(x_space_vector)
    x_full = torch.cat([x_space_vector, x_n.unsqueeze(0)])
    return softmax(x_full)

def new_softmax_inverse_transform(w_space_vector, epsilon=1e-12):
    w_clipped = torch.clamp(w_space_vector, min=epsilon, max=1.0)
    n = w_clipped.shape[0]
    log_w = torch.log(w_clipped)
    sum_value = torch.mean(log_w)
    output = log_w - sum_value
    return output[:n-1]

def inverse_softmax(w, epsilon=1e-12):
    w = np.clip(w, epsilon, 1.0)  # avoid log(0)
    w_inv = []
    for index in range(len(w)):
        w_inv.append(np.log(w[index]))
    return np.array(w_inv)

def run_gradient_descent(x0, Sigma, lambda_mu, steps=50, step_size=0.1, verbose=True):
    x = x0.copy()
    for i in range(steps):
        grad = estimate_gradient_f_i_comparisons(x, Sigma, lambda_mu)
        x -= step_size * grad
        if verbose and i % 10 == 0:
            w = softmax(x)
            obj = w.T @ Sigma @ w - lambda_mu @ w
            print(f"Step {i:3d}: Objective = {obj:.6f}, Norm(grad) = {np.linalg.norm(grad):.4f}")
    return softmax(x)

def run_nash_bargaining(x0, Sigma_set, lambda_mu_set, x_i_set, steps=1000, step_size=0.1, verbose=True):
    x = x0.copy()
    print("Check at start: ", len(x0))
    for i in range(steps + 1):
        grad_set = []
        grad_sum = 0
        norm_sum = 0
        w_current = new_softmax_transform(x)
        for j in range(len(lambda_mu_set)):
            grad = true_markowitz_gradient_w(x, Sigma_set[j], lambda_mu_set[j])
            w_value = evaluate_objective(w_current, Sigma_set[j], lambda_mu_set[j])
            grad_sum += grad / w_value

        w_new = w_current - grad_sum
        x_new = new_softmax_inverse_transform(w_new)
        x_post_step = x + (x_new - x) * step_size
        w_new = new_softmax_transform(x_new)
        w = new_softmax_transform(x)
        if verbose and i % 100 == 0:
            print(f"Step {i:3d}: Current State = {np.linalg.norm(w_new - w)}")
        x = x_post_step
    return x


def run_our_solution_concept_comparisons(x0, Sigma_set, lambda_mu_set, x_i_set, steps=1000, step_size=0.1, verbose=True):
    x = x0.clone()
    print("Check at start: ", len(x0))
    for i in range(steps):
        grad_set = []
        grad_sum = 0
        grad_sum_w = 0
        norm_sum = 0
        for j in range(len(lambda_mu_set)):
            grad = estimate_gradient_f_i_comparisons(x, Sigma_set[j], lambda_mu_set[j], theta_threshold=1e-3)
            x_opt = x_i_set[j]
            # print("Checking Gradients Comparisons: ", grad/torch.norm(grad))
            grad_sum += (grad / torch.norm(grad)) * torch.norm(x - x_opt)
            norm_sum += torch.norm(x - x_opt)

        x_new = x - step_size * (grad_sum / norm_sum)
        # print("Checking New x comparisons: ", grad_sum, step_size)
        w_new = new_softmax_transform(x_new)
        w = new_softmax_transform(x)
        if verbose and i % 10 == 0:
            print(f"Step {i:3d}: X change = {torch.norm(x_new - x)}")
        x = x_new.detach().clone().requires_grad_(True)  # Detach to avoid backprop through history
        # print("New X comparisons: ", x)
    return x


import torch


def run_our_solution_concept_actual(x0, Sigma_set, lambda_mu_set, x_i_set, steps=1000, step_size=0.1, verbose=True):
    x = x0.clone().detach().requires_grad_(True)

    for i in range(steps):
        grad_sum = torch.zeros_like(x)
        norm_sum = 0.0

        for j in range(len(lambda_mu_set)):
            # Compute gradient for j-th agent
            grad = true_markowitz_gradient(x, Sigma_set[j], lambda_mu_set[j])


            # Get the optimal x_i for the j-th agent
            x_opt = x_i_set[j]

            # Normalize gradient and weight it by the distance to x_opt
            grad_norm = torch.norm(grad)
            # if verbose:
            #     print("Checking Gradients Actual: ", grad/grad_norm)
            x_diff_norm = torch.norm(x - x_opt)

            if grad_norm > 0:
                grad_sum += (grad / grad_norm) * x_diff_norm
                norm_sum += x_diff_norm

        # Perform gradient step
        # print("Checking New x actual: ", grad_sum, step_size)

        if norm_sum > 0:
            x_new = x - step_size * (grad_sum / norm_sum)
        else:
            x_new = x
        w_new = new_softmax_transform(x_new)
        w = new_softmax_transform(x)

        if verbose and i % 10 == 0:
            print(f"Step {i:3d}: X change = {torch.norm(x_new - x).item()}")

        x = x_new.detach().clone().requires_grad_(True)  # Detach to avoid backprop through history
        # print("New X actual: ", x)

    return x

def setup_markowitz_environment(tickers, start_date, end_date, lambda_ret=0.5):
    """
    Sets up the Markowitz environment by computing:
    - Covariance matrix Σ
    - Scaled expected returns vector λ * μ

    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in 'YYYY-MM-DD'
        end_date (str): End date in 'YYYY-MM-DD'
        lambda_ret (float): Lagrangian multiplier for expected return

    Returns:
        Sigma (np.ndarray): Covariance matrix (n x n)
        lambda_mu (np.ndarray): λ * μ vector (n,)
        tickers (list): Cleaned ticker list (may differ from input)
    """
    # Download adjusted close prices
    n = len(tickers)
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]

    prices.dropna(inplace=True)

    # Compute daily returns
    returns = prices.pct_change().dropna()

    # Compute expected return vector and covariance matrix
    mu = np.ones(n) # returns.mean().values       # shape (n,)
    Sigma = np.eye(n) # returns.cov().values     # shape (n, n)
    r = np.ones(n)
    print(mu)
    # Scale expected return vector
    lambda_mu = lambda_ret * mu

    return Sigma, lambda_mu, returns.columns.tolist()


def solve_markowitz(Sigma, lambda_mu):
    """
    Solves the Lagrangian form of the Markowitz optimization problem.

    Args:
        Sigma (np.ndarray): Covariance matrix (n x n)
        lambda_mu (np.ndarray): λ * μ vector (n,)

    Returns:
        w_opt (np.ndarray): Optimal portfolio weights (n,)
    """
    n = len(lambda_mu)
    w = cp.Variable(n)

    # Objective: minimize risk - lambda * expected return
    objective = cp.Minimize(cp.quad_form(w, Sigma) - lambda_mu @ w)

    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        raise ValueError("Optimization failed.")

    return w.value

def evaluate_objective(w, Sigma, lambda_mu):
    return w.T @ Sigma @ w - lambda_mu @ w

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    start_date = "2018-01-01"
    end_date = "2023-12-31"
    lambda_ret = 0

    print("Setting up Markowitz environment...")

    Sigma_set = []
    lambda_mu_set = []
    for lambda_val in [0, 0]:
        Sigma, lambda_mu, clean_tickers = setup_markowitz_environment(
            tickers, start_date, end_date, lambda_val
        )
        Sigma_set.append(torch.tensor(Sigma, dtype=torch.float32))
        lambda_mu_set.append(torch.tensor(lambda_mu, dtype=torch.float32))

    print("Finding Individual Optimal Solutions...")
    solution_set = []
    n = 5
    for index in range(len(lambda_mu_set)):
        Sigma = Sigma_set[index]
        lambda_mu = lambda_mu_set[index]
        w_opt = solve_markowitz(Sigma, lambda_mu)
        w_opt_tensor = torch.tensor(w_opt, dtype=torch.float32)
        x_star = new_softmax_inverse_transform(w_opt_tensor)
        w_opt_check = new_softmax_transform(x_star)
        # print("Checking Values: ", w_opt, w_opt_check, torch.norm(w_opt_tensor - w_opt_check))
        solution_set.append(x_star)
    #
    po_flag = False
    step_size = 0.01

    # Create random convex combination weights
    num_solutions = len(solution_set)
    alpha = np.random.rand(num_solutions)
    alpha /= np.sum(alpha)  # Normalize to make it a convex combination

    # Compute convex combination of x_star vectors
    generate_flag = True
    while generate_flag:
        w_start_np = sample_from_simplex(n)
        w_start = torch.tensor(w_start_np, dtype=torch.float32, requires_grad=True)  # track gradients

        x_start = new_softmax_inverse_transform(w_start)  # should also produce a tensor w/ grad
        x_start.requires_grad_(True)  # ensure x_start is differentiable

        generate_flag = is_pareto_optimal_via_perturbation(x_start, Sigma_set, lambda_mu_set)
    print("Starting Weights: ", w_start)

    x_copy = x_start.clone()
    while not po_flag and step_size > 1e-6:
        print("Starting Iteration: ", x_start, new_softmax_transform(x_start), step_size)
        #  tensor([0.0034, 0.0013, 0.0023, 0.0023], grad_fn=<DivBackward0>) tensor([0.2007, 0.2003, 0.2005, 0.2005, 0.1981],
        our_solution_comp = run_our_solution_concept_comparisons(x_start, Sigma_set, lambda_mu_set, solution_set, step_size=step_size)
        po_flag = is_pareto_optimal_via_perturbation(our_solution_comp, Sigma_set, lambda_mu_set)
        if not po_flag:
            step_size /= 10
            print("Checking Solution: ", our_solution_comp, new_softmax_transform(our_solution_comp))
            x_start = our_solution_comp
        po_flag = True
    #
    # x_start = x_copy
    step_size = 0.01
    po_flag = False
    while not po_flag and step_size > 1e-6:
        print("Starting Iteration: ", x_start, new_softmax_transform(x_start), step_size)
        our_solution_no_comp = run_our_solution_concept_actual(x_start, Sigma_set, lambda_mu_set, solution_set, step_size=step_size)
        po_flag = is_pareto_optimal_via_perturbation(our_solution_no_comp, Sigma_set, lambda_mu_set)
        if not po_flag:
            step_size /= 10
            print("Checking Solution: ", our_solution_no_comp, new_softmax_transform(our_solution_no_comp))
            x_start = our_solution_no_comp
        po_flag = True

    print("Final Check: ", torch.norm(our_solution_no_comp - our_solution_comp), new_softmax_transform(our_solution_no_comp), new_softmax_transform(our_solution_comp))
