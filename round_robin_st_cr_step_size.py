import numpy as np
import scipy
import matplotlib as plt
from scipy.optimize import minimize
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import cvxpy
from scipy.spatial import ConvexHull
import time
from mpl_toolkits.mplot3d import Axes3D
from coordinate_descent_implementation import line_search

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

class Bargaining_Agent_Linear:
    def __init__(self, c):
        self.c = c  # Linear weight vector

    def generate_gradient_vector(self, current_state):
        # For linear utility, the gradient is constant (equal to the coefficient vector c)
        gradient = self.c
        return gradient / np.linalg.norm(gradient)

    def generate_directional_magnitude(self, current_state, directional_vector):
        # For linear utility, directional magnitude is the projection of c on the direction
        directional_vector_unit = directional_vector / np.linalg.norm(directional_vector)
        gradient = self.generate_gradient_vector(current_state)
        return np.dot(gradient, directional_vector_unit)

    def query(self, current_state, offer):
        next_state = offer + current_state
        current_value = np.dot(self.c, current_state)
        next_value = np.dot(self.c, next_state)
        return next_value > current_value

    def utility_benefit(self, current_state, offer, flag=False):
        next_state = offer + current_state
        current_value = np.dot(self.c, current_state)
        next_value = np.dot(self.c, next_state)
        benefit = next_value - current_value

        if flag:
            print(f"current_value: {current_value}, next_value: {next_value}, benefit: {benefit}")
        return benefit

class Mediator:
    def __init__(self, num_agents, num_categories, cone_information):
        self.num_agents = num_agents
        self.num_categories = num_categories
        self.cone_information = cone_information

    # Offer generation for a single cone
    def generate_offers(self, agent_idx):
        center_of_cone = self.cone_information[agent_idx]["center_of_cone"]
        # Create a random orthogonal vector
        random_vector = np.random.randn(len(center_of_cone))

        # Use the Gram-Schmidt process to get an orthogonal vector
        orthogonal_vector = random_vector - (
                    np.dot(random_vector, center_of_cone) / np.dot(center_of_cone, center_of_cone)) * center_of_cone

        # Normalize the orthogonal vector
        orthogonal_vector /= np.linalg.norm(orthogonal_vector)

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
                offer_directions.append(potential_vector / np.linalg.norm(potential_vector))
        return offer_directions

    # Cone refinement for a single cone
    def refine_cone(self, agent_idx, offers, offer_responses):

        center_of_cone = self.cone_information[agent_idx]["center_of_cone"]
        theta = self.cone_information[agent_idx]["theta"]
        w_list = [center_of_cone]
        sum_value = np.zeros(self.num_categories)
        sum_value += center_of_cone / np.linalg.norm(center_of_cone)
        # Use cone update rule to determine the new cone of potential gradients
        for i in range(0, self.num_categories - 1):
            if offer_responses[i]:
                w_i = center_of_cone * np.cos(theta) + (offers[i] / np.linalg.norm(offers[i])) * np.sin(theta)
                w_list.append(np.array(w_i))
            else:
                w_i = center_of_cone * np.cos(theta) - (offers[i] / np.linalg.norm(offers[i])) * np.sin(theta)
                w_list.append(np.array(w_i))
            sum_value += (w_i / self.num_categories)
        new_center_of_cone = (sum_value / np.linalg.norm(sum_value))
        scaling_factor_theta = np.sqrt((2 * self.num_categories - 1) / (2 * self.num_categories))
        new_theta = np.arcsin(scaling_factor_theta * np.sin(theta))
        # print("Theta Update:", theta, new_theta)

        # Update cone information
        self.cone_information[agent_idx]["center_of_cone"] = new_center_of_cone
        self.cone_information[agent_idx]["theta"] = new_theta
        return new_center_of_cone, new_theta

    def find_intersection(self):
        """
        Find the intersection of dual cones by directly enforcing the constraints.

        Args:
            cones (list of dict): Each dict contains 'center' (cone center) and 'theta' (cone angle in radians).

        Returns:
            np.array or None: A vector in the intersection of the dual cones, or None if no intersection exists.
        """

        def objective(x):
            # Objective function to maximize the norm of x (or any arbitrary metric).
            # Here, we use a placeholder objective since the feasibility is handled by constraints.
            return 0

        def constraint_norm(x):
            # Enforce x to be a unit vector.
            return np.linalg.norm(x) - 1

        constraints = [{"type": "eq", "fun": constraint_norm}]

        # Add dual cone constraints: x · center >= cos(theta) for each cone.
        for cone in self.cone_information:
            center = cone["center_of_cone"]
            theta = cone["theta"]
            cos_theta = np.cos(np.pi / 2 - theta)  # Dual cone constraint
            constraints.append({"type": "ineq", "fun": lambda x, c=center, ct=cos_theta: np.dot(x, c) - ct})

        # Initial guess (random unit vector)
        n = len(self.cone_information[0]["center_of_cone"])
        x0 = np.random.randn(n)
        x0 /= np.linalg.norm(x0)

        # Optimization
        result = minimize(objective, x0, constraints=constraints, method="SLSQP")

        if result.success:
            return result.x
        else:
            return []

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

def round_robin_iteration(num_categories, agent_set, current_state, step_size=1, theta_threshold=0.001):
    cone_information = []
    thetas = []
    for agent_idx in range(len(agent_set)):
        agent = agent_set[agent_idx]
        cone_center = np.zeros(num_categories)
        for i in range(num_categories):
            initialization_offer = np.zeros(num_categories)
            initialization_offer[i] = 1
            response = agent.query(current_state, initialization_offer)
            cone_center[i] = 1 if response else -1
        cone_center /= np.linalg.norm(cone_center)
        cone_information.append({"center_of_cone": cone_center, "theta": np.arccos(1 / np.sqrt(num_categories))})
        thetas.append(np.arccos(1 / np.sqrt(num_categories)))

    mediator = Mediator(len(agent_set), num_categories, cone_information)
    trade_found = False
    num_offers = 0
    max_offers = 1000
    intersection_offer = 0
    while any(theta > theta_threshold for theta in thetas) and num_offers < max_offers:
        for agent_idx in range(len(agent_set)):

            agent = agent_set[agent_idx]
            offers = mediator.generate_offers(agent_idx)
            responses = [agent.query(current_state, offer) for offer in offers]
            responses_neg = [agent.query(current_state, -1 * offer) for offer in offers]
            num_offers += len(offers)
            prev_cone_center = mediator.cone_information[agent_idx]["center_of_cone"].copy()
            new_cone_center, new_theta = mediator.refine_cone(agent_idx, offers, responses)
            thetas[agent_idx] = new_theta
            true_gradient = agent.generate_gradient_vector(current_state)
            # if angle_between(new_cone_center, true_gradient) > new_theta:
            #     print("Offers: ", offers, responses, responses_neg)
            #     print("Failure: ", new_cone_center, new_theta, angle_between(new_cone_center, true_gradient))
            #     time.sleep(1000)

    intersection_offer = []
    step_sizes = []
    gradients = []
    for agent_idx in range(0, len(agent_set)):
        agent = agent_set[agent_idx]
        estimated_gradient = mediator.cone_information[agent_idx]["center_of_cone"].copy()
        estimated_gradient = estimated_gradient / np.linalg.norm(estimated_gradient)
        gradients.append(estimated_gradient)
        optimal_step_size, num_queries = line_search(agent, current_state, estimated_gradient)
        num_offers += num_queries
        step_sizes.append(optimal_step_size)


    step_sizes = np.array(step_sizes) / np.linalg.norm(np.array(step_sizes))
    # Convert lists to numpy arrays
    gradients = np.array(gradients)
    step_sizes = np.array(step_sizes)
    # Compute the weighted sum of gradients
    intersection_offer = np.sum(step_sizes[:, np.newaxis] * gradients, axis=0)

    return intersection_offer, num_offers, mediator

def round_robin_scenario_full(num_categories, agent_set, starting_state, step_size=10, offer_limit=10000):
    current_state = starting_state
    state_progression = []
    offer_progression = []
    state_progression.append(current_state.copy())
    total_offers = 0
    end_flag = False
    num_iterations = 10
    total_iterations = 0
    while not end_flag:
        intersection_offer, num_offers, mediator = round_robin_iteration(num_categories, agent_set, current_state,step_size=step_size)
        if len(intersection_offer) == 0 or total_iterations >= num_iterations:
            end_flag = True
        else:
            current_state += intersection_offer
            total_offers += num_offers
            total_iterations += 1
            state_progression.append(current_state.copy())
            offer_progression.append(total_offers)
        
    return current_state, total_offers, state_progression, offer_progression

def plot_cone_and_offers(previous_center, offers, responses, new_center):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(0, 0, 0, previous_center[0], previous_center[1], previous_center[2], color="blue", label="Previous Cone Center")
    for offer_idx in range(len(offers)):
        offer = offers[offer_idx]
        if responses[offer_idx]:
            ax.quiver(0, 0, 0, offer[0], offer[1], offer[2], color="green", alpha=0.7, label="Offer")
        else:
            ax.quiver(0, 0, 0, -offer[0], -offer[1], -offer[2], color="green", alpha=0.7, label="Offer")

    ax.quiver(0, 0, 0, new_center[0], new_center[1], new_center[2], color="red", label="New Cone Center")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


# Helper function to generate a gradient with a specific angle
def generate_gradient_with_angle(base_gradient, angle_deg):
    angle_rad = np.radians(angle_deg)
    dim = len(base_gradient)
    orthogonal_vector = np.random.randn(dim)  # Generate a random vector
    orthogonal_vector -= orthogonal_vector.dot(base_gradient) / np.linalg.norm(
        base_gradient) ** 2 * base_gradient  # Make orthogonal
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)  # Normalize

    new_gradient = (
            np.cos(angle_rad) * (base_gradient / np.linalg.norm(base_gradient)) +
            np.sin(angle_rad) * orthogonal_vector
    )
    return new_gradient * np.linalg.norm(base_gradient)  # Scale to match the original gradient magnitude

def generate_gradient_with_angle_linear(base_gradient, angle):
    """
    Generates a new gradient with a specified angle (in degrees) to the base gradient.
    Works correctly for any dimension.
    """
    angle_rad = np.radians(angle)
    base_gradient_unit = base_gradient / np.linalg.norm(base_gradient)

    # Create an orthogonal vector to the base_gradient
    random_vector = np.random.rand(len(base_gradient)) - 0.5
    orthogonal_vector = random_vector - np.dot(random_vector, base_gradient_unit) * base_gradient_unit
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)

    # Rotate to create the new gradient
    new_gradient = (
        np.cos(angle_rad) * base_gradient_unit + np.sin(angle_rad) * orthogonal_vector
    )
    return new_gradient * np.linalg.norm(base_gradient)  # Maintain the same magnitude as the base gradient

# # Stress testing setup
# linear_case = False
# if linear_case:
#     np.random.seed(10)
#     num_categories = 3
#     current_state = np.array([0.0, 0.0, 0.0])

#     # Base linear gradient
#     b_base = np.array([200, 200, 200])

#     angles = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]  # Angles from 0 to 180 degrees

#     for angle in angles:
#         # Generate the second gradient with the specified angle to the first gradient
#         b_2 = generate_gradient_with_angle_linear(b_base, angle)

#         # Initialize linear agents
#         agent_set = [Bargaining_Agent_Linear(b_base), Bargaining_Agent_Linear(b_2)]

#         # Run your algorithm
#         final_state, num_offers = round_robin_scenario_full(num_categories, agent_set, current_state)

#         print("Final State: ", angle, current_state, final_state, num_offers)
# else:
#     # Stress testing setup
#     np.random.seed(10)
#     num_categories = 3
#     current_state = np.array([0.0, 0.0, 0.0])
#     starting_state = current_state.copy()
#     # Base quadratic function components
#     A = -1 * np.eye(num_categories)
#     b_base = np.array([200, 200, 200])

#     angles = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]  # Angles from 0 to 180 degrees
#     # angles = [20]
#     for angle in angles:
#         current_state = np.array([0.0, 0.0, 0.0])
#         starting_state = current_state.copy()
#         # Generate the second gradient with the specified angle to the first gradient
#         b_2 = generate_gradient_with_angle(b_base, angle)
#         A_2 = -1 * np.eye(num_categories)
#         agent_set = [Bargaining_Agent(A, b_base), Bargaining_Agent(A_2, b_2)]

#         # Run Full Scenario
#         final_state, num_offers = round_robin_scenario_full(num_categories, agent_set, current_state)
#         print("Final State: ", angle, starting_state, final_state, num_offers)

