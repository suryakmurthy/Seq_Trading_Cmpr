import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm  # Progress bar library
from center_of_gravity_round_robin import round_robin_scenario_full as cog_method
from round_robin_st_cr import round_robin_scenario_full as st_cr_method
from center_of_gravity_round_robin import Bargaining_Agent
from coordinate_descent_implementation import comparison_based_cd_method

import rpy2.robjects as robjects

def calculate_utility(A, b, state, final_state):
    current_value = state.transpose() @ A @ state + b @ state
    next_value = final_state.transpose() @ A @ final_state + b @ final_state
    return next_value - current_value

def generate_averaged_data(num_offers, benefits):
    max_offers = max(max(lst) for lst in num_offers)
    all_benefits = [generate_benefit_list(offers, benefit, max_offers) for offers, benefit in zip(num_offers, benefits)]
    return np.mean(all_benefits, axis=0)

def plot_averaged_curve(data_stcr, data_cog, data_cd, categories, angle):
    plt.plot(data_stcr, linestyle='-', label='ST-CR')
    plt.plot(data_cog, linestyle='-', label='Center of Gravity')
    plt.plot(data_cd, linestyle='-', label='Comparison-Based CD')
    plt.xlabel("Number of Offers")
    plt.ylabel("Average Cumulative Benefit")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"comparison_plot_{categories}_{angle}")
    plt.clf()  # Clears the figure but keeps the window open

def calc_gradient(A, b, state):
    current_value = state.transpose() @ A @ state + b @ state
    gradient = 2 * np.dot(A, state) + b
    return gradient / np.linalg.norm(gradient)

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

def generate_benefit_list(offers, benefits, max_offers):
    benefit_list = np.zeros(max_offers + 1)
    last_value = 0
    offer_idx = 0

    for i in range(max_offers + 1):
        if offer_idx < len(offers) and i == offers[offer_idx]:
            last_value = benefits[offer_idx]
            offer_idx += 1
        benefit_list[i] = last_value  # Extend last value
    return benefit_list

# Define test parameters
# Set Python random seeds
random.seed(42)  # For Python's random module
np.random.seed(42)  # For NumPy
# Set R's random seed
robjects.r('set.seed(42)')  # Runs set.seed in R

num_categories_list = [3, 5, 7]  # Different problem sizes
angles = [40, 80, 120, 160]  # Different angles between agent utilities
num_trials = 1000  # Repeat tests for robustness

# Metrics storage
results = {"cog": [], "st_cr": [], "cd": []}

# Outer progress bar for number of categories
for num_categories in tqdm(num_categories_list, desc="Processing Category Sizes"):
    # Inner progress bar for different angles
    for angle in tqdm(angles, desc=f"Processing Angles (Categories={num_categories})", leave=False):
        cog_benefits, cog_offers = [], []
        st_cr_benefits, st_cr_offers = [], []

        soc_benefit_list_cog = []
        benefit_list_cog_1 = []
        benefit_list_cog_2 = []
        cumul_offer_list_cog = []

        soc_benefit_list_stcr = []
        benefit_list_stcr_1 = []
        benefit_list_stcr_2 = []
        cumul_offer_list_stcr = []

        soc_benefit_list_cd = []
        benefit_list_cd_1 = []
        benefit_list_cd_2 = []
        cumul_offer_list_cd = []

        # Progress bar for trials
        for _ in tqdm(range(num_trials), desc=f"Trials (Angle={angle})", leave=False):
            # Generate initial state and agents with quadratic utilities
            starting_state = np.zeros(num_categories)
            A = -1 * np.eye(num_categories)
            b_base = np.random.uniform(-200, 200, num_categories)
            b_2 = generate_gradient_with_angle(b_base, angle)
            grad_1 = calc_gradient(A, b_base, starting_state)
            grad_2 = calc_gradient(A, b_2, starting_state)


            agent_set = [Bargaining_Agent(A, b_base), Bargaining_Agent(A, b_2)]

            # Run Center of Gravity method
            final_state_cog, num_offers_cog, state_list_cog, offer_list_cog = cog_method(num_categories, agent_set, starting_state)
            benefits = [calculate_utility(A, b_base, starting_state, final_state_cog), calculate_utility(A, b_2, starting_state, final_state_cog)]
            starting_state = np.zeros(num_categories)
            benefit_list_cog_1.append([calculate_utility(A, b_base, starting_state, state) for state in state_list_cog])
            benefit_list_cog_2.append([calculate_utility(A, b_2, starting_state, state) for state in state_list_cog])
            soc_benefit_list_cog.append([calculate_utility(A, b_base, starting_state, state) + calculate_utility(A, b_2, starting_state, state) for state in state_list_cog])
            cumul_offer_list_cog.append(offer_list_cog)

            benefit_cog = sum(benefits)
            cog_benefits.append(benefit_cog)
            cog_offers.append(num_offers_cog)

            # # Run ST-CR method
            starting_state = np.zeros(num_categories)
            final_state_st_cr, num_offers_st_cr, state_list_stcr, offer_list_stcr = st_cr_method(num_categories, agent_set, starting_state)
            benefits = [calculate_utility(A, b_base, starting_state, final_state_st_cr), calculate_utility(A, b_2, starting_state, final_state_st_cr)]
            benefit_st_cr = sum(benefits)
            st_cr_benefits.append(benefit_st_cr)
            st_cr_offers.append(num_offers_st_cr)
            starting_state = np.zeros(num_categories)
            benefit_list_stcr_1.append([calculate_utility(A, b_base, starting_state, state) for state in state_list_stcr])
            benefit_list_stcr_2.append([calculate_utility(A, b_2, starting_state, state) for state in state_list_stcr])
            soc_benefit_list_stcr.append([calculate_utility(A, b_base, starting_state, state) + calculate_utility(A, b_2, starting_state, state) for state in state_list_stcr])
            # print("STCR Benefit List:", soc_benefit_list_stcr)
            cumul_offer_list_stcr.append(offer_list_stcr)

            # Run CD method
            starting_state = np.zeros(num_categories)
            final_state_cd, num_offers_cd, state_list_cd, offer_list_cd = comparison_based_cd_method(num_categories, agent_set, starting_state)
            # print("Offer_list_cd: ", offer_list_cd)
            benefits = [calculate_utility(A, b_base, starting_state, final_state_cd), calculate_utility(A, b_2, starting_state, final_state_cd)]
            benefit_cd = sum(benefits)
            st_cr_benefits.append(benefit_cd)
            st_cr_offers.append(num_offers_cd)
            starting_state = np.zeros(num_categories)
            benefit_list_cd_1.append([calculate_utility(A, b_base, starting_state, state) for state in state_list_cd])
            benefit_list_cd_2.append([calculate_utility(A, b_2, starting_state, state) for state in state_list_cd])
            soc_benefit_list_cd.append([calculate_utility(A, b_base, starting_state, state) + calculate_utility(A, b_2, starting_state, state) for state in state_list_cd])
            # print(state_list_cd)
            # print(final_state_cog, final_state_st_cr, final_state_cd)
            cumul_offer_list_cd.append(offer_list_cd)

        # Store averages
            
        # average_benefit_1_stcr = generate_averaged_data(cumul_offer_list_stcr, benefit_list_stcr_1)
        # average_benefit_2_stcr = generate_averaged_data(cumul_offer_list_stcr, benefit_list_stcr_2)

        # average_benefit_1_cog = generate_averaged_data(cumul_offer_list_cog, benefit_list_cog_1)
        # average_benefit_2_cog = generate_averaged_data(cumul_offer_list_cog, benefit_list_cog_2)

        # average_benefit_1_cd = generate_averaged_data(cumul_offer_list_cd, benefit_list_cd_1)
        # average_benefit_2_cd = generate_averaged_data(cumul_offer_list_cd, benefit_list_cd_2)
        # print(cumul_offer_list_cd)
        average_soc_benefit_stcr = generate_averaged_data(cumul_offer_list_stcr, soc_benefit_list_stcr)
        average_soc_benefit_cog = generate_averaged_data(cumul_offer_list_cog, soc_benefit_list_cog)
        average_soc_benefit_cd = generate_averaged_data(cumul_offer_list_cd, soc_benefit_list_cd)

        plot_averaged_curve(average_soc_benefit_stcr, average_soc_benefit_cog, average_soc_benefit_cd, num_categories, angle)

        results["cog"].append((num_categories, angle, np.mean(cog_benefits), np.mean(cog_offers)))
        results["st_cr"].append((num_categories, angle, np.mean(st_cr_benefits), np.mean(st_cr_offers)))
        results["cd"].append((num_categories, angle, np.mean(st_cr_benefits), np.mean(st_cr_offers)))


# Print summary results
print("Results Summary:")
for method in ["cog", "st_cr", "cd"]:
    print(f"\nMethod: {method.upper()}")
    for entry in results[method]:
        print(f"Categories: {entry[0]}, Angle: {entry[1]}, Benefit: {entry[2]:.2f}, Offers: {entry[3]}")
