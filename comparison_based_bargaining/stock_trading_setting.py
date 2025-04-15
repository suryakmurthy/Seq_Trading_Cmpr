import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
from cone_refinement_functions import estimate_gradient_f_i_comparisons, true_markowitz_gradient, true_markowitz_gradient_w, angle_between
# from sign_opt import estimate_gradient_sign_opt
import random
from scipy.optimize import minimize
import numpy as np
import json
import time
from itertools import combinations
from scipy.optimize import approx_fprime
import torch

## NASH BARGAINING ZEROTH ORDER:

class UtilityWithCounter:
    def __init__(self, Sigma, lambda_mu):
        self.Sigma = Sigma
        self.lambda_mu = lambda_mu
        self.query_count = 0

    def __call__(self, w):
        self.query_count += 1
        return (self.lambda_mu @ w - w @ self.Sigma @ w)

def estimate_gradient_fd_xspace_tensor(x, utility_fn, epsilon=1e-5):
    grad = torch.zeros_like(x)
    for i in range(len(x)):
        perturb = torch.zeros_like(x)
        perturb[i] = epsilon
        f_plus = utility_fn(new_softmax_transform(x + perturb))
        f_minus = utility_fn(new_softmax_transform(x - perturb))
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    return grad

def estimate_gradient_with_scipy(x, utility_fn, epsilon=1e-5):
    # Convert tensor to NumPy
    x_np = x.detach().numpy()

    # Define a wrapped version of the utility function on NumPy inputs
    def wrapped_fn(x_np_input):
        x_tensor = torch.tensor(x_np_input, dtype=torch.float32)
        w = new_softmax_transform(x_tensor)
        return utility_fn(w).item()  # Make sure this returns a float, not tensor

    grad = approx_fprime(x_np, wrapped_fn, epsilon)
    return -torch.tensor(grad, dtype=torch.float32)

# Zeroth-order NBS solver
def solve_nbs_zeroth_order_simplex(Sigma_list, lambda_mu_list, disagreement=-1.0,
                                    steps=1000, lr=0.1, epsilon=1e-4):
    m = len(Sigma_list)
    n = Sigma_list[0].shape[0]

    # Start with uniform point on the simplex
    w = torch.ones(n, dtype=torch.float32) / n

    utility_wrappers = [UtilityWithCounter(Sigma_list[i], lambda_mu_list[i]) for i in range(m)]

    for step in range(steps):
        grad_sum = torch.zeros_like(w)

        for i in range(m):
            u_i = utility_wrappers[i](w)
            if u_i <= disagreement:
                continue

            grad_i = estimate_gradient_fd_wspace(w, utility_wrappers[i], epsilon)
            grad_sum += grad_i / (u_i - disagreement)

        grad_norm = grad_sum.norm()
        if grad_norm < 1e-6:
            print(f"Step {step}: Zero gradient. Injecting noise.")
            grad_sum = torch.randn_like(w)
            grad_norm = grad_sum.norm()

        # Gradient ascent step
        w = w + lr * grad_sum / grad_norm

        # Project back to simplex
        w = project_to_simplex(w)

        # if step % 100 == 0 or step == steps - 1:
        #     print(f"Step {step}: ||grad|| = {grad_norm:.2e}")

    total_queries = sum(u.query_count for u in utility_wrappers)
    return w.detach(), total_queries

def project_to_simplex(v):
    """Projection of vector v onto the probability simplex."""
    v_np = v.detach().cpu().numpy()
    n = len(v_np)
    u = np.sort(v_np)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u + (1 - cssv) / np.arange(1, n+1) > 0)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w_proj = np.maximum(v_np - theta, 0)
    return torch.tensor(w_proj, dtype=torch.float32)

def estimate_gradient_fd_wspace(w, utility_fn, epsilon=1e-5):
    """
    Use scipy's approx_fprime to estimate gradient of utility_fn at w in w-space.
    The utility_fn should accept a torch tensor and return a scalar.
    """
    # Convert torch tensor to numpy
    w_np = w.detach().numpy()

    # Define a wrapped function that accepts numpy and returns float
    def wrapped_fn(w_np_input):
        w_tensor = torch.tensor(w_np_input, dtype=torch.float32)
        w_projected = project_to_simplex(w_tensor)
        return utility_fn(w_projected).item()  # Must return scalar float

    # Estimate gradient
    grad_np = approx_fprime(w_np, wrapped_fn, epsilon)

    return torch.tensor(grad_np, dtype=torch.float32)


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

def new_softmax_inverse_transform(w_space_vector, epsilon=1e-6):
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

def run_our_solution_concept_comparisons(x0, Sigma_set, lambda_mu_set, x_i_set, steps=1000, step_size=0.1, verbose=True):
    x = x0.clone()
    for i in range(steps):
        grad_set = []
        grad_sum = 0
        grad_sum_w = 0
        norm_sum = 0
        for j in range(len(lambda_mu_set)):
            stamp_1 = time.time()
            grad = estimate_gradient_f_i_comparisons(x, Sigma_set[j], lambda_mu_set[j], theta_threshold=0.1)
            stamp_2 = time.time()
            print("Checking time per gradient estimation: ", stamp_2 - stamp_1)
            x_opt = x_i_set[j]
            # print("Checking Gradients Comparisons: ", grad/torch.norm(grad))
            grad_sum += (grad / torch.norm(grad)) * torch.norm(x - x_opt)
            norm_sum += torch.norm(x - x_opt)

        x_new = x - step_size * (grad_sum / norm_sum)
        # print("Checking New x comparisons: ", grad_sum, step_size)
        if verbose and i % 10 == 0:
            print(f"Step {i:3d}: X change = {torch.norm(x_new - x)}")
        x = x_new.detach().clone().requires_grad_(True)  # Detach to avoid backprop through history
        # print("New X comparisons: ", x)
    return x

def run_our_solution_concept_actual(x0, Sigma_set, lambda_mu_set, x_i_set, steps=10000, step_size=0.01, verbose=True):
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

        # if verbose and i % 10 == 0:
        #     print(f"Step {i:3d}: X change = {torch.norm(x_new - x).item()}")

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
    mu = returns.mean().values       # np.ones(n) # shape (n,)
    Sigma = returns.cov().values     # np.eye(n) # shape (n, n)
    r = np.ones(n)
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

def markowitz_cost_numpy(w, Sigma_np, lambda_mu_np):
    """
    Objective function (negative of Markowitz cost) for scipy.optimize.
    All inputs must be numpy arrays.
    """
    w = np.asarray(w)
    return -(w.T @ Sigma_np @ w - lambda_mu_np @ w)

def solve_markowitz_max_cost(Sigma, lambda_mu, num_restarts=10):
    """
    Solves max_w wᵀ Σ w - λ μᵀ w over the simplex, using scipy.optimize.
    Sigma and lambda_mu are expected to be PyTorch tensors.

    Args:
        Sigma (torch.Tensor): (n x n) covariance matrix
        lambda_mu (torch.Tensor): (n,) vector of λ * μ
        num_restarts (int): Number of random restarts

    Returns:
        max_cost (float)
        best_w (torch.Tensor): optimal weights as torch tensor
    """
    # Convert to NumPy arrays for use with scipy
    Sigma_np = Sigma.detach().cpu().numpy()
    lambda_mu_np = lambda_mu.detach().cpu().numpy()
    n = len(lambda_mu_np)

    bounds = [(0, 1) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    best_val = -np.inf
    best_w = None

    for _ in range(num_restarts):
        w0 = np.random.dirichlet(np.ones(n))

        result = minimize(
            markowitz_cost_numpy,
            w0,
            args=(Sigma_np, lambda_mu_np),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
        )

        if result.success:
            val = -result.fun  # flip sign back
            if val > best_val:
                best_val = val
                best_w = result.x

    if best_w is None:
        raise ValueError("Optimization failed in all restarts")

    return best_val, torch.tensor(best_w, dtype=Sigma.dtype)

def solve_nbs_cvxpy(Sigma_list, lambda_mu_list, disagreement=-1.0):
    m = len(Sigma_list)
    n = Sigma_list[0].shape[0]

    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]

    exprs = []
    for i in range(m):
        Sigma = Sigma_list[i]
        lambda_mu = lambda_mu_list[i]
        util = lambda_mu @ w - cp.quad_form(w, Sigma)
        # Force expression to scalar shape to satisfy cp.log() input requirement
        exprs.append(cp.log(cp.reshape(util - disagreement, ())))

    objective = cp.Maximize(cp.sum(exprs))
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=cp.SCS)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Problem status:", problem.status)
        return None

    return w.value

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    with open('top_100_tickers_2023.json', 'r') as f:
        tickers = json.load(f)
    tickers = tickers[:25]
    start_date = ["2018-01-01", "2022-01-01", "2018-01-01"]
    end_date = ["2023-12-31", "2023-12-31", "2022-01-01"]
    lambda_ret = 0

    print("Setting up Markowitz environment...")

    Sigma_set = []
    lambda_mu_set = []

    Sigma_set_np = []
    lambda_mu_set_np = []

    lambda_values = [0,0.01,0.1]
    for idx in range(len(lambda_values)):
        Sigma, lambda_mu, clean_tickers = setup_markowitz_environment(
            tickers, start_date[idx], end_date[idx], lambda_values[idx]
        )
        Sigma_set.append(torch.tensor(Sigma, dtype=torch.float32))
        lambda_mu_set.append(torch.tensor(lambda_mu, dtype=torch.float32))

        Sigma_set_np.append(Sigma)
        lambda_mu_set_np.append(lambda_mu)

    print("Finding Individual Optimal Solutions...")
    solution_set = []
    max_value_set = []
    n = len(tickers)
    for index in range(len(lambda_mu_set)):
        Sigma = Sigma_set[index]
        lambda_mu = lambda_mu_set[index]
        w_opt = solve_markowitz(Sigma, lambda_mu)
        max_val, w_max = solve_markowitz_max_cost(Sigma, lambda_mu)
        # print("Double Checking Maximization Results: ", max_val, w_max)
        w_max_tensor = torch.tensor(w_opt, dtype=torch.float32)
        w_opt_tensor = torch.tensor(w_opt, dtype=torch.float32)

        x_star = new_softmax_inverse_transform(w_opt_tensor)
        x_max = new_softmax_inverse_transform(w_max_tensor)

        w_opt_check = new_softmax_transform(x_star)
        # print("Checking Values: ", w_opt)
        solution_set.append(x_star)
        max_value_set.append(max_val)
    #
    po_flag = False
    step_size = 0.01

    # Create random convex combination weights
    num_solutions = len(solution_set)
    alpha = np.random.rand(num_solutions)
    alpha /= np.sum(alpha)  # Normalize to make it a convex combination

    print("Generating Starting Point...")
    # Compute convex combination of x_star vectors
    generate_flag = True
    # while generate_flag:
    # print("Looping")
    w_start_np = sample_from_simplex(n)
    # print("Starting Point: ", w_start_np)
    w_start = torch.tensor(w_start_np, dtype=torch.float32, requires_grad=True)  # track gradients

    x_start = new_softmax_inverse_transform(w_start)  # should also produce a tensor w/ grad
    x_start.requires_grad_(True)  # ensure x_start is differentiable

    # generate_flag = is_pareto_optimal_via_perturbation(x_start, Sigma_set, lambda_mu_set)
    x_copy = x_start.clone()

    print("Solving for Our Solution Concept...")
    # No Comparisons
    x_start = x_copy.clone().requires_grad_(True)
    step_size = 0.01
    po_flag = False
    while not po_flag and step_size > 1e-6:
        our_solution_no_comp = run_our_solution_concept_actual(x_start, Sigma_set, lambda_mu_set, solution_set, step_size=step_size)
        po_flag = is_pareto_optimal_via_perturbation(our_solution_no_comp, Sigma_set, lambda_mu_set)
        if not po_flag:
            step_size /= 10
            print("Checking Solution: ", our_solution_no_comp, new_softmax_transform(our_solution_no_comp))
            x_start = our_solution_no_comp
        po_flag = True

    x_start = x_copy.clone().requires_grad_(True)
    step_size = 0.01
    po_flag = False
    while not po_flag and step_size > 1e-6:
        our_solution_comp = run_our_solution_concept_comparisons(x_start, Sigma_set, lambda_mu_set, solution_set,
                                                               step_size=step_size)
        po_flag = is_pareto_optimal_via_perturbation(our_solution_comp, Sigma_set, lambda_mu_set)
        if not po_flag:
            step_size /= 10
            print("Checking Solution: ", our_solution_comp, new_softmax_transform(our_solution_comp))
            x_start = our_solution_comp
        po_flag = True

    # Nash Bargaining
    # print(Sigma_set_np, lambda_mu_set_np)
    print("Solving for Nash Solution Concept...")
    nash_solution = solve_nbs_cvxpy(Sigma_set_np, lambda_mu_set_np)
    nash_solution_zeroth_order, num_queries = solve_nbs_zeroth_order_simplex(Sigma_set, lambda_mu_set)
    nash_solution_tensor = torch.tensor(nash_solution, dtype=torch.float32)
    # print("Final Check: ", nash_solution_tensor, nash_solution_zeroth_order, new_softmax_transform(our_solution_no_comp)) #, new_softmax_transform(our_solution_comp))
    print("Final Check: ", torch.norm(nash_solution_tensor - nash_solution_zeroth_order), torch.norm(nash_solution_tensor - new_softmax_transform(our_solution_no_comp)), torch.norm(new_softmax_transform(our_solution_comp) - new_softmax_transform(our_solution_no_comp))) #, new_softmax_transform(our_solution_comp))

