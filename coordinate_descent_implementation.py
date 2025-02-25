import numpy as np


def line_search(f, x, direction, step_size=1.0, tol=1e-3):
    """Performs a line search in the given direction using function comparisons for maximization."""
    alpha, alpha_plus, alpha_minus = 0, step_size, -step_size

    if f.query(x, x + direction * alpha_plus) < 0 and f.query( x, x + direction * alpha_minus) > 0:
        alpha_plus = 0

    if f.query( x, x + direction * alpha_minus) < 0  and f.query( x, x + direction * alpha_plus) > 0:
        alpha_minus = 0
    # print("Checking Alpha Values, ", alpha_plus, alpha_minus)
    while f.query( x, x + (direction * alpha_plus)) > 0:
        alpha_plus = 2 * alpha_plus

    while f.query( x, x + (direction * alpha_minus)) > 0:
        alpha_minus  = 2 * alpha_minus

    # print("alphas: ", alpha_plus, alpha_minus)
    alpha = 0.5 * (alpha_plus + alpha_minus)

    while np.abs(alpha_plus - alpha_minus) >= tol / 2:
        alpha_temp_plus = 0.5 * (alpha + alpha_plus)
        alpha_temp_minus = 0.5 * (alpha + alpha_minus)
        # print("Looping: ", alpha_plus, alpha_minus)
        if f.query( x + direction * alpha, x + (direction * alpha_temp_plus)) > 0:
            alpha = alpha_temp_plus
            alpha_plus = alpha_plus
            alpha_minus = alpha
        else:
            if f.query( x + direction * alpha, x + (direction * alpha_temp_minus)) > 0:
                alpha = alpha_temp_minus
                alpha_plus = alpha
                alpha_minus = alpha_minus
            else:
                alpha = alpha
                alpha_plus = alpha_temp_plus
                alpha_minus = alpha_temp_minus
    return alpha


def comparison_based_cd(agent_set, x0, num_iters=100):
    """Comparison-based coordinate descent algorithm for maximization."""
    x = np.array(x0, dtype=float)
    n = len(x)

    state_list = []
    state_list.append(x0.copy())

    active_directions = set(range(n))  # Start with all coordinate directions
    for _ in range(num_iters):
        if not active_directions:
            break
        i = np.random.choice(list(active_directions))  # Pick a random coordinate
        direction = np.zeros(n)
        direction[i] = 1  # Move in coordinate direction
        steps = []

        for agent in agent_set:
            step = line_search(agent, x, direction)
            steps.append(step)  # Fixing the incorrect append usage

        if all(step > 0 for step in steps) or all(step < 0 for step in steps):
            min_step = min(steps, key=abs)  # Finds the element with the smallest absolute value
            x += min_step * direction
            state_list.append(x.copy())
        else:
            # Conflict detected: different objectives want opposite steps
            # Remove this direction from future consideration
            active_directions.remove(i)  # Remove this direction permanently

    return x, state_list


# Example usage
class Bargaining_Agent:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.query_count = 0

    def reset_query_count(self):
        self.query_count = 0

    def generate_gradient_vector(self, current_state):
        gradient = 2 * np.dot(self.A, current_state) + self.b
        return gradient / np.linalg.norm(gradient)

    def generate_directional_magnitude(self, current_state, directional_vector):
        directional_vector_unit = directional_vector / np.linalg.norm(directional_vector)
        gradient = self.generate_gradient_vector(current_state)
        return np.dot(gradient, directional_vector_unit)

    def query(self, current_state, next_state):
        current_value = current_state.transpose() @ self.A @ current_state + self.b @ current_state
        next_value = next_state.transpose() @ self.A @ next_state + self.b @ next_state
        if next_value - current_value > 0:
            return 1
        else:
            return -1

    def utility_benefit(self, current_state, offer, flag=False):
        next_state = offer + current_state
        current_value = current_state.transpose() @ self.A @ current_state + self.b @ current_state
        next_value = next_state.transpose() @ self.A @ next_state + self.b @ next_state
        benefit = next_value - current_value

        # Debugging the actual values of next_value and current_value
        if flag:
            print(f"current_value: {current_value}, next_value: {next_value}, benefit: {benefit}")
        return benefit

np.random.seed(10)
categories_list = [3, 4, 5, 6, 7, 8, 9, 10]
agents_list = [3, 4, 5, 6, 7, 8, 9, 10]
progression_list = []
offer_count = []

for num_agents in agents_list:
    for num_categories in categories_list:
        agent_set = []
        for agent_num in range(num_agents):
            A = -1 * np.eye(num_categories)
            b = 2 * np.random.uniform(0, 200, num_categories)
            print(b)
            agent = Bargaining_Agent(A, b)
            agent_set.append(agent)
        x0 = np.zeros(num_categories)
        x_opt, state_list = comparison_based_cd(agent_set, x0=x0, num_iters=100)
        offer_count_temp = 0
        for agent in agent_set:
            offer_count_temp += agent.query_count
        offer_count.append({"num_agents": num_agents, "num_categories": num_categories, "offer_count": offer_count_temp})

