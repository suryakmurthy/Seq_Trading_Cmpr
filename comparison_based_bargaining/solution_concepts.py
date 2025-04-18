import torch
import cvxpy as cp
import numpy as np
from scipy.optimize import approx_fprime
from helper_functions import from_subspace_to_simplex, from_simplex_to_subspace, compute_subspace_gradient
from comparison_based_estimation import estimate_gradient_f_i_comparisons
import concurrent.futures


def solve_markowitz(Sigma: torch.Tensor, lambda_mu: torch.Tensor):
    """
    Solve Markowitz optimization using convex programming with simplex constraints.
    """
    Sigma_np = Sigma.detach().cpu().numpy()
    lambda_mu_np = lambda_mu.detach().cpu().numpy()
    n = len(lambda_mu_np)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, Sigma_np) - lambda_mu_np @ w)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        raise ValueError("Optimization failed.")

    return torch.tensor(w.value, dtype=Sigma.dtype)

def solve_markowitz_subspace_barrier(Sigma: torch.Tensor, lambda_mu: torch.Tensor, barrier_coeff=1e-6):
    """
    Solve Markowitz using subspace log-barrier formulation.
    """
    Sigma_np = Sigma.detach().cpu().numpy()
    lambda_mu_np = lambda_mu.detach().cpu().numpy()
    n = len(lambda_mu_np)
    v = cp.Variable(n - 1)
    last_coord = 1 - cp.sum(v)
    w = cp.hstack([v, last_coord])

    barrier = -cp.sum(cp.log(v)) - cp.log(1 - cp.sum(v))
    objective = cp.Minimize(cp.quad_form(w, Sigma_np) - lambda_mu_np @ w + barrier_coeff * barrier)
    constraints = [v >= 1e-8, cp.sum(v) <= 1 - 1e-8]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if v.value is None:
        raise ValueError("Optimization failed.")

    v_tensor = torch.tensor(v.value, dtype=Sigma.dtype)
    w_tensor = torch.cat([v_tensor, 1.0 - torch.sum(v_tensor).unsqueeze(0)], dim=0)
    return w_tensor

def solve_nbs_first_order_subspace(Sigma_list, lambda_mu_list, starting_point=None,
                                   disagreement=-1.0, steps=1000, lr=0.01, barrier_coeff=1e-6):
    """
    First-order solver for Nash Bargaining in (n-1) space with log-barrier.
    """
    m = len(Sigma_list)
    n = Sigma_list[0].shape[0]
    v = torch.ones(n - 1, dtype=torch.float64) * (1.0 / n) if starting_point is None else starting_point.clone()

    for step in range(steps):
        grad_sum = torch.zeros_like(v)
        for i in range(m):
            w = from_subspace_to_simplex(v)
            u_i = torch.dot(lambda_mu_list[i], w) - torch.dot(w, Sigma_list[i] @ w)
            if u_i <= disagreement:
                continue
            grad_i = -compute_subspace_gradient(v, Sigma_list[i], lambda_mu_list[i], barrier_coeff)
            grad_sum += grad_i / (u_i - disagreement + 1e-8)

        grad_norm = grad_sum.norm()
        v = v + lr * grad_sum / grad_norm

    return from_subspace_to_simplex(v).detach()

def solve_nbs_cvxpy(Sigma_list, lambda_mu_list, disagreement=-1.0):
    """
    Solve Nash Bargaining analytically with log utilities using CVXPY.
    """
    m = len(Sigma_list)
    n = Sigma_list[0].shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]

    exprs = []
    for i in range(m):
        util = lambda_mu_list[i] @ w - cp.quad_form(w, Sigma_list[i])
        exprs.append(cp.log(cp.reshape(util - disagreement, ())))

    objective = cp.Maximize(cp.sum(exprs))
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=cp.SCS)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Problem status:", problem.status)
        return None

    return w.value

def estimate_gradient_fd_subspace(v, Sigma, lambda_mu, disagreement=-1.0, epsilon=1e-5, barrier_coeff=1e-6):
    v_np = v.detach().numpy()
    query_counter = {'count': 0}

    def wrapped_fn(v_input_np):
        query_counter['count'] += 1
        v_input = torch.tensor(v_input_np, dtype=v.dtype)
        w = from_subspace_to_simplex(v_input)

        eps = 1e-6
        quad = torch.dot(w, Sigma @ w)
        linear = torch.dot(lambda_mu, w)
        loss = quad - linear

        barrier = -torch.sum(torch.log(w + eps)) - torch.log(1.0 - torch.sum(v_input) + eps)
        total = loss + barrier_coeff * barrier
        return total.item()

    grad_np = approx_fprime(v_np, wrapped_fn, epsilon)
    grad_est = torch.tensor(grad_np, dtype=v.dtype)
    grad_true = compute_subspace_gradient(v, Sigma, lambda_mu, barrier_coeff)
    grad_error = torch.norm(grad_est - grad_true).item()
    return grad_est, query_counter['count'], grad_error

def solve_nbs_zeroth_order(Sigma_list, lambda_mu_list, starting_point=None,
                           disagreement=-1.0, steps=1000, lr=0.01, barrier_coeff=1e-6):
    m = len(Sigma_list)
    n = Sigma_list[0].shape[0]
    v = torch.ones(n - 1, dtype=torch.float64) * (1.0 / n) if starting_point is None else starting_point.clone()
    total_queries = 0
    grad_errors = []

    for step in range(steps):
        grad_sum = torch.zeros_like(v)
        for i in range(m):
            w = from_subspace_to_simplex(v)
            u_i = torch.dot(lambda_mu_list[i], w) - torch.dot(w, Sigma_list[i] @ w)
            if u_i <= disagreement:
                continue
            grad_i, query_count, grad_error = estimate_gradient_fd_subspace(v, Sigma_list[i], lambda_mu_list[i], disagreement, epsilon=1e-5, barrier_coeff=barrier_coeff)
            grad_i = -1 * grad_i
            grad_sum += grad_i / (u_i - disagreement + 1e-8)
            total_queries += query_count
            grad_errors.append(grad_error)

        grad_norm = grad_sum.norm()
        if grad_norm > 0:
            v = v + lr * grad_sum / grad_norm

    avg_grad_error = np.mean(grad_errors) if grad_errors else 0.0
    return from_subspace_to_simplex(v).detach(), total_queries

def solve_nbs_barrier(Sigma_list, lambda_mu_list, disagreement=-1.0, barrier_weight=1e-6):
    """
    Solve Nash Bargaining using subspace log-barrier directly in CVXPY.
    """
    m = len(Sigma_list)
    n = Sigma_list[0].shape[0]
    v = cp.Variable(n - 1)
    w = cp.hstack([v, 1 - cp.sum(v)])

    exprs = [cp.log(cp.reshape(lambda_mu_list[i] @ w - cp.quad_form(w, Sigma_list[i]) - disagreement, ()))
             for i in range(m)]

    barrier_terms = cp.sum(cp.log(v)) + cp.log(1 - cp.sum(v))
    total_obj = cp.sum(exprs) + barrier_weight * barrier_terms

    objective = cp.Maximize(total_obj)
    problem = cp.Problem(objective)
    result = problem.solve()

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Problem status:", problem.status)
        return None

    return np.append(v.value, 1 - np.sum(v.value))

def run_our_solution_concept_actual(x0, Sigma_set, lambda_mu_set, x_i_set, steps=1000, step_size=0.01):
    """
    Run iterative update minimizing distance to all agents' optima.
    """
    x = x0.clone().detach().requires_grad_(True)

    for _ in range(steps):
        grad_sum = torch.zeros_like(x)
        norm_sum = 0.0
        for j in range(len(lambda_mu_set)):
            grad = compute_subspace_gradient(x, Sigma_set[j], lambda_mu_set[j])
            x_opt = x_i_set[j]
            grad_norm = torch.norm(grad)
            x_diff_norm = torch.norm(x - x_opt)

            if grad_norm > 0:
                grad_sum += (grad / grad_norm) * x_diff_norm
                norm_sum += x_diff_norm

        x_new = x - step_size * (grad_sum / norm_sum) if norm_sum > 0 else x
        x = x_new.detach().clone().requires_grad_(True)

    return x

def run_our_solution_concept_comparisons(x0, Sigma_set, lambda_mu_set, x_i_set, steps=100, step_size=0.1):
    """
    Run iterative update using comparison-based gradient estimation.
    """
    x = x0.clone()
    total_queries = 0
    for _ in range(steps):
        grad_sum = 0
        norm_sum = 0
        for j in range(len(lambda_mu_set)):
            grad, query_count = estimate_gradient_f_i_comparisons(x, Sigma_set[j], lambda_mu_set[j])
            total_queries += query_count
            x_opt = x_i_set[j]
            grad_sum += (grad / torch.norm(grad)) * torch.norm(x - x_opt)
            norm_sum += torch.norm(x - x_opt)

        x_new = x - step_size * (grad_sum / norm_sum)
        x = x_new.detach().clone().requires_grad_(True)
    return x, total_queries

def run_our_solution_concept_comparisons_parallel(x0, Sigma_set, lambda_mu_set, x_i_set, steps=100, step_size=0.1):
    """
    Run iterative update using comparison-based gradient estimation.
    Parallelizes gradient estimation across agents.
    """
    x = x0.clone().detach().requires_grad_(True)
    total_queries = 0

    for _ in range(steps):
        grad_sum = torch.zeros_like(x)
        norm_sum = 0.0

        def estimate_for_agent(j):
            grad, query_count = estimate_gradient_f_i_comparisons(x, Sigma_set[j], lambda_mu_set[j])
            x_opt = x_i_set[j]
            diff_norm = torch.norm(x - x_opt)
            grad_unit = grad / torch.norm(grad)
            return grad_unit * diff_norm, diff_norm, query_count

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(estimate_for_agent, j) for j in range(len(lambda_mu_set))]
            for future in concurrent.futures.as_completed(futures):
                try:
                    grad_contrib, norm_contrib, query_count = future.result()
                    grad_sum += grad_contrib
                    norm_sum += norm_contrib
                    total_queries += query_count
                except Exception as e:
                    print(f"⚠️ Gradient estimation failed for one agent: {e}")

        if norm_sum > 0:
            x = x - step_size * (grad_sum / norm_sum)
        x = x.detach().clone().requires_grad_(True)

    return x, total_queries
