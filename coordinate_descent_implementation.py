import numpy as np
import time


def line_search(f, x, direction, step_size=1.0, tol=1e-3):
    """Performs a line search in the given direction using function comparisons for maximization."""
    alpha, alpha_plus, alpha_minus = 0, step_size, -step_size
    num_queries = 0  # Initialize query counter

    # Check the initial queries
    num_queries += 2
    if f.query(x, direction * alpha_plus) and f.query(x, direction * alpha_minus):
        alpha_plus = 0

    num_queries += 2
    if f.query(x, direction * alpha_minus) and f.query(x, direction * alpha_plus):
        alpha_minus = 0

    # Exponentially grow alpha_plus and alpha_minus and count queries
    # Account for the inital 2 queries for the following while loops
    num_queries += 2
    while f.query(x, (direction * alpha_plus)):
        num_queries += 1
        alpha_plus = 2 * alpha_plus

    while f.query(x, (direction * alpha_minus)):
        num_queries += 1
        alpha_minus = 2 * alpha_minus

    alpha = 0.5 * (alpha_plus + alpha_minus)

    while np.abs(alpha_plus - alpha_minus) >= tol / 2:
        alpha_temp_plus = 0.5 * (alpha + alpha_plus)
        alpha_temp_minus = 0.5 * (alpha + alpha_minus)

        # Count queries for these checks
        num_queries += 1
        if f.query(x + direction * alpha, (direction * alpha_temp_plus)):
            alpha = alpha_temp_plus
            alpha_plus = alpha_plus
            alpha_minus = alpha
        else:
            num_queries += 1
            if f.query(x + direction * alpha, (direction * alpha_temp_minus)):
                alpha = alpha_temp_minus
                alpha_plus = alpha
                alpha_minus = alpha_minus
            else:
                alpha = alpha
                alpha_plus = alpha_temp_plus
                alpha_minus = alpha_temp_minus

    return alpha, num_queries  # Return alpha and the number of queries made during the line search


def comparison_based_cd_method(num_categories, agent_set, x0, num_iters=100):
    """Comparison-based coordinate descent algorithm with random directions from the unit sphere."""
    x = np.array(x0, dtype=float)
    n = num_categories
    cumulative_queries = 0
    query_list = []

    state_list = [x0.copy()]
    
    for _ in range(num_iters):
        # Sample a random direction from the unit sphere
        direction = np.random.randn(n)
        direction /= np.linalg.norm(direction)  # Normalize to get a unit vector

        steps = []

        # Perform the line search for each agent
        for agent in agent_set:
            step, num_queries = line_search(agent, x, direction)
            cumulative_queries += num_queries
            steps.append(step)  # Store the step for the current agent
        # print("agent set: ", len(agent_set), x, [agent.b for agent in agent_set], direction, steps)
        # time.sleep(10)
        if all(step > 0 for step in steps) or all(step < 0 for step in steps):
            # If all agents suggest the same direction, take the smallest step
            min_step = min(steps, key=abs)  # Find the smallest step in magnitude
            x += min_step * direction
            query_list.append(cumulative_queries)
            state_list.append(x.copy())  # Save the new state

    return x, cumulative_queries, state_list, query_list


# Example usage (without query count)
class Bargaining_Agent:
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def generate_gradient_vector(self, current_state):
        gradient = 2 * np.dot(self.A, current_state) + self.b
        return gradient / np.linalg.norm(gradient)

    def generate_directional_magnitude(self, current_state, directional_vector):
        directional_vector_unit = directional_vector / np.linalg.norm(directional_vector)
        gradient = self.generate_gradient_vector(current_state)
        return np.dot(gradient, directional_vector_unit)

    def query(self, current_state, offer):
        next_state = offer + current_state
        current_value = current_state.transpose() @ self.A @ current_state + self.b @ current_state
        next_value = next_state.transpose() @ self.A @ next_state + self.b @ next_state
        if next_value - current_value > 0:
            return True
        else:
            return False

    def utility_benefit(self, current_state, offer, flag=False):
        next_state = offer + current_state
        current_value = current_state.transpose() @ self.A @ current_state + self.b @ current_state
        next_value = next_state.transpose() @ self.A @ next_state + self.b @ next_state
        benefit = next_value - current_value

        # Debugging the actual values of next_value and current_value
        if flag:
            print(f"current_value: {current_value}, next_value: {next_value}, benefit: {benefit}")
        return benefit


# Example of testing the comparison-based method
# np.random.seed(10)
# categories_list = [3, 4, 5, 6, 7, 8, 9, 10]
# agents_list = [3, 4, 5, 6, 7, 8, 9, 10]
# progression_list = []
# offer_count = []

# for num_agents in agents_list:
#     for num_categories in categories_list:
#         agent_set = []
#         for agent_num in range(num_agents):
#             A = -1 * np.eye(num_categories)
#             b = 2 * np.random.uniform(0, 200, num_categories)
#             agent = Bargaining_Agent(A, b)
#             agent_set.append(agent)
#         x0 = np.zeros(num_categories)
#         x_opt, num_offers, state_list, offer_list = comparison_based_cd_method(num_categories, agent_set, x0=x0, num_iters=100)
#         progression_list.append(
#             {"num_agents": num_agents, "num_categories": num_categories, "x_opt": x_opt, "state_list": state_list})

        