import numpy as np
from scipy.optimize import minimize
import scipy
import time

def generate_negative_semidefinite_A(dim):
    """Generate a random negative semi-definite matrix A."""
    Q = np.random.randn(dim, dim)
    A = np.dot(Q.T, Q)  # Ensures A is negative semi-definite
    return A

import numpy as np

def generate_constrained_A_b(dim, x_min, x_max):
    """Generate A_i and b_i such that the optimal point x_i_star is within [x_min, x_max]."""
    while True:
        A_i = generate_negative_semidefinite_A(dim)
        b_i = np.random.randn(dim)

        # Compute x_i_star
        if np.linalg.matrix_rank(A_i) == dim:
            x_i_star = -0.5 * np.linalg.pinv(A_i) @ b_i
        else:
            x_i_star = np.zeros_like(b_i)  # Default if A_i is singular

        # Check if x_i_star is within [x_min, x_max] element-wise
        if np.all(x_min <= x_i_star) and np.all(x_i_star <= x_max):
            return A_i, b_i


def angle_between(v1, v2):
    """
        Return the angle between two vectors.
        Args:
            v1 (np.array): n-dimensional vector.
            v2 (np.array): n-dimensional vector.

        Returns:
            angle (float): Angle between the two vectors in radians
    """
    dot_product = np.dot(v1, v2)
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    cos_theta = dot_product / (m1 * m2)
    angle = np.arccos(cos_theta)
    return angle

def is_pareto_optimal(x, A_list, b_list, num_samples=1000):
    """Check if x is Pareto-optimal by verifying if small random perturbations improve all functions."""
    gradients = np.array([gradient_f_i(x, A_list[i], b_list[i]) for i in range(len(A_list))])

    for _ in range(num_samples):
        d = np.random.randn(len(x))  # Generate a random direction
        d /= np.linalg.norm(d)  # Normalize to unit length

        if all(np.dot(gradients[i], d) >= 0 for i in range(len(A_list))):
            return False  # If a random perturbation improves all functions, x is not Pareto-optimal
    return True


# Function to generate random starting point within [-100, 100]
def random_starting_point(n, x_min, x_max):
    return np.random.uniform(x_min, x_max, n)

def scale_utilities(A, b, x_min, x_max, epsilon=1):
    """
    Scales the quadratic utility functions such that they are non-negative
    in the feasible space [x_min, x_max].

    Arguments:
    - A: List of n matrices A_i for each player's utility function.
    - b: List of n vectors b_i for each player's utility function.
    - x_min, x_max: The bounds of the feasible region.

    Returns:
    - c_list: List of constants for each player's utility function after scaling.
    """
    n = len(A)
    c_list = []

    for i in range(n):
        def u(x):
            return x.T @ A[i] @ x + b[i].T @ x  # Player i's utility function

        # Bounds for the optimization
        bounds = [(x_min, x_max)] * len(A[i])  # Assumes A[i] is square, adjust as necessary

        # Minimize the utility function in the feasible space [x_min, x_max]
        result = minimize(u, np.zeros(len(b[i])), bounds=bounds, method='trust-constr')

        # Get the minimum utility found
        min_utility = result.fun
        # print(f"Min utility for player {i}: {min_utility}")

        # Append the scaling constant to make the utility non-negative
        c_list.append(-min_utility + epsilon)  # Invert sign for minimization

    return c_list

def utility(x, A, b, c):
    return [x.T @ A[i] @ x + b[i].T @ x + c[i] for i in range(len(A))]

def gradient_f_i(x, A_i, b_i):
    """Compute the gradient of f_i(x) = x^T A_i x + b_i^T x."""
    return 2 * np.dot(A_i, x) + b_i

def query(A_i, b_i, current_state, offer):
    next_state = offer + current_state
    current_value = current_state.transpose() @ A_i @ current_state + b_i @ current_state
    next_value = next_state.transpose() @ A_i @ next_state + b_i @ next_state
    if next_value - current_value > 0:
        return True
    else:
        return False

def generate_offers(center_of_cone, step_size_orth=0.001):
    center_of_cone = center_of_cone
    # Create a random orthogonal vector
    random_vector = np.random.randn(len(center_of_cone))

    # Use the Gram-Schmidt process to get an orthogonal vector
    orthogonal_vector = random_vector - (
                np.dot(random_vector, center_of_cone) / np.dot(center_of_cone, center_of_cone)) * center_of_cone

    # Normalize the orthogonal vector
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)
    orthogonal_vector *= step_size_orth
    # Generate n-2 additional offers that are orthogonal to the cone center and the existing vectors.
    vectors = [center_of_cone, orthogonal_vector]
    null_space = scipy.linalg.null_space(vectors)
    offer_directions = [orthogonal_vector]

    # Iterate over the basis of the null space to find orthogonal vectors
    for i in range(null_space.shape[1]):
        potential_vector = null_space[:, i]
        is_orthogonal = True
        # Check that the null space vectors are orthogonal to all the vectors in the given set.
        for vector in offer_directions:
            if abs(np.dot(potential_vector, vector)) > 1e-10:  # Checking if dot product is close to zero
                is_orthogonal = False
                break
        # If the offer is orthogonal, then add it to the set of orthogonal vectors
        if is_orthogonal:
            offer_directions.append(step_size_orth * potential_vector / np.linalg.norm(potential_vector))
    return offer_directions

def refine_cone(center_of_cone, theta, offers, offer_responses):

    center_of_cone = center_of_cone
    theta = theta
    w_list = [center_of_cone]
    sum_value = np.zeros(len(center_of_cone))
    sum_value += center_of_cone / np.linalg.norm(center_of_cone)
    # Use cone update rule to determine the new cone of potential gradients
    for i in range(0, len(center_of_cone) - 1):
        if offer_responses[i]:
            w_i = center_of_cone * np.cos(theta) + (offers[i] / np.linalg.norm(offers[i])) * np.sin(theta)
            w_list.append(np.array(w_i))
        else:
            w_i = center_of_cone * np.cos(theta) - (offers[i] / np.linalg.norm(offers[i])) * np.sin(theta)
            w_list.append(np.array(w_i))
        sum_value += (w_i / len(center_of_cone))
    new_center_of_cone = (sum_value / np.linalg.norm(sum_value))
    scaling_factor_theta = np.sqrt((2 * len(center_of_cone) - 1) / (2 * len(center_of_cone)))
    new_theta = np.arcsin(scaling_factor_theta * np.sin(theta))
    # print("Theta Update:", theta, new_theta)

    return new_center_of_cone, new_theta

def estimate_gradient_f_i_comparisons(x, A_i, b_i, theta_threshold = 0.1):
    """Compute the gradient of f_i(x) = x^T A_i x + b_i^T x."""
    ## Initalize Cone:
    # print(f"Theta Threshold Values: {theta_threshold}")
    num_dim = len(x)
    cone_center = np.zeros(num_dim)
    for index in range(num_dim):
        initialization_offer = np.zeros(num_dim)
        initialization_offer[index] = 0.000001
        response = query(A_i, b_i, x, initialization_offer)
        cone_center[index] = 1 if response else -1

    cone_center = np.array(cone_center) / np.linalg.norm(cone_center)
    theta = np.arccos(1 / np.sqrt(num_dim))

    while theta > theta_threshold:
        offers = generate_offers(cone_center)
        responses = []
        neg_responses = []
        for offer in offers:
            response = query(A_i, b_i, x, offer)
            neg_response = query(A_i, b_i, x, -1 * offer)
            scale_down = 0.1

            while response == neg_response and scale_down > 1e-10:
                scaled_offer = scale_down * offer

                response = query(A_i, b_i, x, scaled_offer)
                neg_response = query(A_i, b_i, x, -1 * scaled_offer)
                scale_down *= 0.1
                # print(offer, scaled_offer, scale_down, response, neg_response)
            responses.append(response)
            neg_responses.append(neg_response)


        new_cone_center, new_theta = refine_cone(cone_center, theta, offers, responses)
        true_grad = gradient_f_i(x, A_i, b_i)
        angle = angle_between(new_cone_center, true_grad)
        if angle > new_theta:
            print("Failure: ", offers, responses, neg_responses, angle, new_theta)
            time.sleep(10)
        cone_center = new_cone_center
        theta = new_theta

    return cone_center

def compute_lipschitz_constants(A_list):
    """Compute the Lipschitz constants for the gradients of the functions."""
    return [2 * np.linalg.norm(A_i, ord=2) for A_i in A_list]

def find_x_i_star(A_i, b_i):
    """Find the maximum point x_i^* by solving \nabla f_i(x) = 0."""
    return -0.5 * np.linalg.pinv(A_i) @ b_i if np.linalg.matrix_rank(A_i) == A_i.shape[0] else np.zeros_like(b_i)

def fixed_point_iteration(x_init, A_list, b_list, num_iters=10000, tol=1e-6, step_size = 0.01):
    x = x_init
    num_functions = len(A_list)
    iter_counter = 0

    for _ in range(num_iters):
        # print(f"Iteration {iter_counter}: x = {x}")
        iter_counter += 1
        grad_sum = np.zeros_like(x)
        norm_sum = 0.0

        for i in range(num_functions):
            grad_i = gradient_f_i(x, A_list[i], b_list[i])
            norm_grad_i = np.linalg.norm(grad_i)
            # print(f"Function {i}: grad_i = {grad_i}, norm_grad_i = {norm_grad_i}")
            if norm_grad_i == 0:
                continue
            x_i_star = find_x_i_star(A_list[i], b_list[i])
            # print(f"Function {i}: x_i_star = {x_i_star}")

            grad_sum += (grad_i / norm_grad_i) * np.linalg.norm(x - x_i_star)
            norm_sum += np.linalg.norm(x - x_i_star)

        if norm_sum == 0:
            break

        x_new = x - step_size * (grad_sum / norm_sum)   # Fixed-point update rule
        # print("x_new: ", x_new)
        # time.sleep(10)
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, iter_counter

def fixed_point_iteration_with_comparisons(x_init, A_list, b_list, num_iters=10000, tol=1e-6, step_size = 0.01, theta_threshold = 0.001, debug = False):
    x = x_init
    num_functions = len(A_list)
    iter_counter = 0
    state_progression = []
    state_progression.append(x_init)
    for _ in range(num_iters):
        # print(f"Iteration {iter_counter}: x = {x}")
        iter_counter += 1
        grad_sum = np.zeros_like(x)
        norm_sum = 0.0

        for i in range(num_functions):

            grad_i = estimate_gradient_f_i_comparisons(x, A_list[i], b_list[i], theta_threshold=theta_threshold)
            actual_grad_i = gradient_f_i(x, A_list[i], b_list[i])
            # time.sleep(10)
            norm_grad_i = np.linalg.norm(grad_i)
            if norm_grad_i == 0:
                continue
            x_i_star = find_x_i_star(A_list[i], b_list[i])

            grad_sum += (grad_i / norm_grad_i) * np.linalg.norm(x - x_i_star)
            norm_sum += np.linalg.norm(x - x_i_star)

        if norm_sum == 0:
            if debug:
                print(
                    f"Stopping Comparisons at Iteration {iter_counter} due to the sum og distances to to optimality being equal")
            break

        x_new = x - step_size * (grad_sum / norm_sum)   # Fixed-point update rule

        state_progression.append(x_new)
        # print("x_new: ", x_new)
        # time.sleep(10)
        if np.linalg.norm(x_new - x) < tol:
            if debug:
                print(
                    f"Stopping Comparisons at Iteration {iter_counter} due to lack of progress {x}, {x_new}, {grad_sum}, {step_size * (grad_sum / norm_sum)}")
            break

        x = x_new

    return x, iter_counter, state_progression

def fixed_point_iteration(x_init, A_list, b_list, num_iters=10000, tol=1e-6, step_size = 0.01,debug=False):
    x = x_init
    num_functions = len(A_list)
    iter_counter = 0
    state_progression = [x_init]
    for _ in range(num_iters):
        # print(f"Iteration {iter_counter}: x = {x}")
        iter_counter += 1
        grad_sum = np.zeros_like(x)
        norm_sum = 0.0

        for i in range(num_functions):
            grad_i = gradient_f_i(x, A_list[i], b_list[i])
            norm_grad_i = np.linalg.norm(grad_i)
            # print(f"Function {i}: grad_i = {grad_i}, norm_grad_i = {norm_grad_i}")
            if norm_grad_i == 0:
                continue
            x_i_star = find_x_i_star(A_list[i], b_list[i])
            # print(f"Function {i}: x_i_star = {x_i_star}")

            grad_sum += (grad_i / norm_grad_i) * np.linalg.norm(x - x_i_star)
            norm_sum += np.linalg.norm(x - x_i_star)

        if norm_sum == 0:
            if debug:
                print(f"Stopping Fixed Point at Iteration {iter_counter} due to the sum og distances to to optimality being equal")
            break

        x_new = x - step_size * (grad_sum / norm_sum)   # Fixed-point update rule
        state_progression.append(x_new)
        # print("x_new: ", x_new)
        # time.sleep(10)
        if np.linalg.norm(x_new - x) < tol:
            if debug:
                print(f"Stopping Fixed Point at Iteration {iter_counter} due to lack of progress {x}, {x_new}, {grad_sum}, {norm_sum}, {step_size}")
            break

        x = x_new

    return x, iter_counter, state_progression


def nash_bargaining_solution(A, b, c, starting_state):
    """
    Solves the Nash Bargaining Solution for n quadratic convex utility functions.

    Arguments:
    - A: List of n matrices A_i for each player's utility function.
    - b: List of n vectors b_i for each player's utility function.

    Returns:
    - Optimal decision vector x.
    """
    n = len(A)

    # Define the utility functions
    def utility(x, A, b, c):
        return [x.T @ A[i] @ x + b[i].T @ x + c[i] for i in range(n)]

    # Objective function to minimize the product of utilities (adjusted for convexity)
    def objective(x):
        u = utility(x, A, b, c)
        product = np.prod(u)  # minimize the product of utilities
        return product  # No sign inversion needed as we are minimizing

    # Initial guess for x
    x0 = starting_state  # Assume the initial point as zeros

    # Minimize the product of utilities (Nash Bargaining for convex utility)
    result = minimize(objective, x0, method='trust-constr')
    return result.x, result.fun


def kalai_smorodinsky_solution(A, b, c, current_state):
    """
    Solves the Kalai-Smorodinsky Solution for n quadratic convex utility functions.

    Arguments:
    - A: List of n matrices A_i for each player's utility function.
    - b: List of n vectors b_i for each player's utility function.

    Returns:
    - Optimal decision vector x.
    """
    n = len(A)

    # Define the utility functions
    def utility(x, A, b, c):
        return [x.T @ A[i] @ x + b[i].T @ x + c[i] for i in range(n)]

    # Compute the maximum utility for each player
    def max_utility(A, b, c):
        max_utility_values = []
        for i in range(n):
            # For each player, solve for the x that maximizes their utility
            def u(x): return x.T @ A[i] @ x + b[i].T @ x + c[i]

            result = minimize(lambda x: u(x), np.zeros_like(b[0]), method='trust-constr')  # Minimizing convex functions
            max_utility_values.append(result.fun)
        return max_utility_values

    max_u = max_utility(A, b, c)

    # Objective function to maximize the weighted sum of utilities (minimization)
    def objective(x):
        u = utility(x, A, b, c)
        ratios = [u[i] / max_u[i] for i in range(n)]
        min_ratio = min(ratios)  # Minimize the minimum ratio
        return min_ratio  # No sign inversion needed

    # Initial guess for x
    x0 = current_state

    # Minimize the minimum ratio (Kalai-Smorodinsky solution)
    result = minimize(objective, x0, method='trust-constr')

    return result.x, result.fun


def egalitarian_solution(A, b, c, current_state):
    """
    Solves the Egalitarian Solution for n quadratic convex utility functions.

    Arguments:
    - A: List of n matrices A_i for each player's utility function.
    - b: List of n vectors b_i for each player's utility function.

    Returns:
    - Optimal decision vector x.
    """
    n = len(A)

    # Define the utility functions
    def utility(x, A, b, c):
        return [x.T @ A[i] @ x + b[i].T @ x + c[i] for i in range(n)]

    # Objective function to minimize the minimum utility (Egalitarian solution)
    def objective(x):
        u = utility(x, A, b, c)
        min_utility = min(u)  # Minimize the minimum utility
        return min_utility  # No sign inversion needed

    # Initial guess for x
    x0 = current_state

    # Minimize the minimum utility (Egalitarian solution)
    result = minimize(objective, x0, method='trust-constr')

    return result.x, result.fun

import numpy as np
import matplotlib.pyplot as plt

def plot_contours(A, b, c):
    plt.figure(figsize=(10, 6))
    pareto_front_x = np.linspace(-100, 100, 50)
    pareto_front_y = np.linspace(-100, 100, 50)
    X, Y = np.meshgrid(pareto_front_x, pareto_front_y)
    Z = np.zeros((X.shape[0], X.shape[1], len(A)))

    # Calculate utilities for each point on the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x1, x2 = X[i, j], Y[i, j]
            u_vals = utility(np.array([x1, x2]), A, b, c)
            Z[i, j] = u_vals

    # Plot the Pareto front using contour
    plt.contour(X, Y, Z[:, :, 0], levels=np.linspace(np.min(Z[:, :, 0]), np.max(Z[:, :, 0]), 10), cmap='viridis', alpha=0.3)
    plt.contour(X, Y, Z[:, :, 1], levels=np.linspace(np.min(Z[:, :, 1]), np.max(Z[:, :, 1]), 10), cmap='plasma', alpha=0.3)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Utiltiy Contours In 2D')
    plt.legend()
    plt.show()

def plot_solution_points(A, b, c, solutions, labels, individual_solutions):
    plt.figure(figsize=(10, 6))

    # Create the Pareto front (by generating a grid of points in the feasible region)
    pareto_font = [np.linspace(-0.25, 0.25, 50) for _ in range(len(b[0]))]

    pareto_front_x = np.linspace(-0.25, 0.25, 50)
    pareto_front_y = np.linspace(-0.25, 0.25, 50)
    X, Y = np.meshgrid(pareto_front_x, pareto_front_y)
    Z = np.zeros((X.shape[0], X.shape[1], len(A)))

    # Calculate utilities for each point on the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x1, x2 = X[i, j], Y[i, j]
            u_vals = utility(np.array([x1, x2]), A, b, c)
            Z[i, j] = u_vals

    # Add a plot for the Pareto front curve (representing trade-offs between the two players' utility)
    pareto_optimal_points = extract_pareto_front(Z[:, :, 0], Z[:, :, 1])
    plt.plot(pareto_optimal_points[:, 0], pareto_optimal_points[:, 1], 'k-', label='Pareto Front', linewidth=2)

    for i, sol in enumerate(solutions):
        u_values = utility(sol, A, b, c)
        # Increase size of KS dot and set its zorder to plot behind others
        if labels[i] == "KS":
            plt.scatter(u_values[0], u_values[1], label=f'{labels[i]} Solution', s=200, zorder=1)  # Larger size for KS
        else:
            plt.scatter(u_values[0], u_values[1], label=f'{labels[i]} Solution', zorder=2)
    plt.xlabel('Utility of Player 1')
    plt.ylabel('Utility of Player 2')
    plt.title('Solution Points and Pareto Front')
    plt.legend()
    plt.show()

def extract_pareto_front(u1_values, u2_values):
    # A simple way to extract Pareto optimal points
    pareto_front = []
    for i in range(len(u1_values)):
        for j in range(len(u2_values)):
            is_pareto = True
            for k in range(len(u1_values)):
                for l in range(len(u2_values)):
                    if u1_values[k, l] < u1_values[i, j] and u2_values[k, l] < u2_values[i, j]:
                        is_pareto = False
                        break
            if is_pareto:
                pareto_front.append([u1_values[i, j], u2_values[i, j]])

    return np.array(pareto_front)


# Example usage

np.random.seed(42)
num_agents = 2
n = 2
num_tests = 100
theta_threshold_values = [1, 0.1, 0.01, 0.001, 0.0001]
distance_dict = {}
A_set = []
b_set = []
debug = False
starting_set = []
x_min = -100
x_max = 100
for _ in range(num_tests):
    A = []
    b = []
    agent_set = []
    for _ in range(num_agents):
        A_ind, b_ind = generate_constrained_A_b(n, x_min=-100, x_max=100)  # Linear term for player 1's utility function
        A.append(A_ind)  # Example quadratic matrices
        b.append(b_ind)  # Example linear terms
    A_set.append(A)
    b_set.append(b)
    starting_state = random_starting_point(n, x_min, x_max)
    starting_set.append(starting_state)

for theta_threshold in theta_threshold_values:
    distance_list = []
    for test_num in range(num_tests):
        print(f"Theta Value {theta_threshold} Test {test_num} ")
        A = A_set[test_num]
        b = b_set[test_num]
        starting_state = starting_set[test_num].copy()
        copied_state = starting_state.copy()

        c = scale_utilities(A, b, x_min, x_max)
        individual_solutions = [find_x_i_star(A[i], b[i]) for i in range(0, len(b))]
        # print(starting_state)

        # nbs_solution = nash_bargaining_solution(A, b, c, starting_state)
        # ks_solution = kalai_smorodinsky_solution(A, b, c, starting_state)
        # egal_solution = egalitarian_solution(A, b, c, starting_state)

        # With Comparisons
        pareto_status = False
        step_size = 10
        num_iters = 1000
        comparison_based_solution = []
        state_progresssion_comparison = []
        while not pareto_status:
            comparison_based_solution, num_iters, state_progresssion_comparison_temp = fixed_point_iteration_with_comparisons(starting_state, A, b, step_size=step_size, theta_threshold = theta_threshold, num_iters=num_iters, debug=debug)
            state_progresssion_comparison.extend(state_progresssion_comparison_temp)
            pareto_status = is_pareto_optimal(comparison_based_solution, A, b)
            starting_state = comparison_based_solution
            step_size = step_size / 10
            if step_size < 1e-10:
                break

        # Without Comparisons
        pareto_status = False
        step_size = 10
        num_iters = 1000
        x_solution = []
        starting_state = copied_state.copy()
        state_progresssion_fixed = []
        # print(starting_state)
        while not pareto_status:
            x_solution, iter_counter, state_progresssion_fixed_temp = fixed_point_iteration(starting_state, A, b, step_size=step_size, num_iters=num_iters, debug=debug)
            state_progresssion_fixed.extend(state_progresssion_fixed_temp)
            pareto_status = is_pareto_optimal(x_solution, A, b)
            starting_state = x_solution
            step_size = step_size / 10
            if step_size < 1e-10:
                break

        dist = np.linalg.norm(x_solution - comparison_based_solution)
        dist_starting = np.linalg.norm(x_solution - copied_state)
        if dist > 1:
            print("Distance: ", test_num, copied_state, x_solution, comparison_based_solution, dist, dist_starting)
        distance_list.append(dist)
        if debug:
            for s in range(len(state_progresssion_comparison)):
                state_comp = state_progresssion_comparison[s]
                state_fixed = state_progresssion_fixed[s]
                print(f"State {s} {theta_threshold}: {state_comp}, {state_fixed}, {np.linalg.norm(state_comp-state_fixed)}")
                time.sleep(10)
        # print("Is Pareto?: ", pareto_status)
        # print("Nash Bargaining Solution:", nbs_solution)
        # print("Kalai-Smorodinsky Solution:", ks_solution)
        # print("Egalitarian Solution:", egal_solution)
        # print("Our Solution: ", x_solution)
        # print("Comparison Solution: ", comparison_based_solution)

        # solutions = [nbs_solution[0], ks_solution[0], egal_solution[0], x_solution, comparison_based_solution]
        # labels = ['Nash Bargaining', 'Kalai-Smorodinsky', 'Egalitarian', 'Equidistant', 'Eqidistant with Comparisons']

        # plot_solution_points(A, b, c, solutions, labels, individual_solutions)
        # plot_contours(A, b, c)
    distance_dict[theta_threshold] = distance_list

average_distances = {}
for key in distance_dict.keys():
    average_distances[key] = np.mean(distance_dict[key])

print(average_distances)

