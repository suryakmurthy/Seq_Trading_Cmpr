import numpy as np
from itertools import product
import itertools
import time

def generate_perturbations(num_items, perturb_range, offering_state, responding_state):
    """
    Generate all possible integer perturbations for the offering state within a specified range.
    Args:
        num_items (int): Number of item categories (dimensions of the state).
        perturb_range (tuple): A tuple specifying the range of perturbations (e.g., (-5, 5)).
        offering_state (np.array): Current state vector of the offering agent.
        responding_state (np.array): Current state vector of the responding agent.
    Returns:
        list: A list of perturbed states (current state + perturbation) that are feasible.
    """
    perturbations = product(range(perturb_range[0], perturb_range[1] + 1), repeat=num_items)
    offers_out = []
    for perturbation in perturbations:
        # Filter out infeasible offersF
        perturbation_array = np.array(perturbation)
        if np.any(offering_state + perturbation_array < 0) or np.any(offering_state + perturbation_array > 200):
            continue
        if np.any(responding_state - perturbation_array < 0) or np.any(responding_state - perturbation_array > 200) :
            continue
        offers_out.append(perturbation_array)

    return offers_out


def sample_unit_hypersphere(n, num_sampled_weights=100):
    """
    Sample points uniformly from the surface of a unit hypersphere in n dimensions.
    Args:
        n (int): Number of dimensions of the hypersphere.
        num_sampled_weights (int, optional): Number of points to sample from the hypersphere. Defaults to 100.
    Returns:
        np.array: Array of sampled points on the unit hypersphere.
    """
    # Step 1: Sample points from a normal distribution
    points = np.random.normal(0, 1, (num_sampled_weights, n))

    # Step 2: Normalize each point to lie on the unit hypersphere
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    return points

def initialize_weights_uniform(num_weights):
    """
    Initialize weights uniformly across all categories.
    Args:
        num_weights (int): Total number of weights to initialize.
    Returns:
        np.array: An array of weights, each set to 1/num_weights.
    """
    # Initialize all weights to be equal
    return np.ones(num_weights) / num_weights

def estimate_opponent_preferences(full_weights, weights_prob, current_state, rejected_offers, shrinking_factor=0.99):
    """
    Estimates the opponent's preferences given a history of rejected offers.

    Parameters:
    - rejected_offers: Matrix where each row is a rejected offer (values for each issue)
    - n_samples: Number of MCMC samples for Bayesian inference

    Returns:
    - A trace of the posterior distribution of the weights
    """
    # Assuming possible_weights and current_state are already defined
    possible_weights = full_weights.copy()

    # Identify weights to be removed
    reduction_count = 0
    for i in range(len(possible_weights)):
        red_flag = False
        for offer in rejected_offers:
            if np.dot(current_state - offer, possible_weights[i]) - np.dot(current_state, possible_weights[i]) > 0:
                weights_prob[i] = weights_prob[i] * shrinking_factor
                if not red_flag:
                    reduction_count += 1
                    red_flag = True
    weights_prob = weights_prob / np.sum(weights_prob)
    return possible_weights, weights_prob


def utility_improvement(offer, A, b, items, reduction_idx=[]):
    """
        Determine if an offer leads to a utility improvement for an agent with utility function x^TAx +x^Tb

        Args:
            offer (np.array): n-dimensional vector representing an offer from the agent's perspective.
            A (np.array): nxn matrix.
            b (np.array): n-dimensional vector.
            items (list): Current State of agent.

        Returns:
            (bool): Response value which states if the agent's utility improves with the offer
    """
    if utility_value(offer, A, b, items, reduction_idx=reduction_idx) > 0:
        return True
    else:
        return False

def utility_value(offer, A, b, items, reduction_idx=[]):
    """
        Return the value of an offer given a utility function of the form x^TAx + bx, an offer, and the current set of items.
        Args:
            offer (np.array): n-dimensional vector representing an offer from the agent's perspective.
            A (np.array): nxn matrix.
            b (np.array): n-dimensional vector.
            items (list): Current State of agent.
            reduction_idx (list, optional): set of item indices that need to be removed from consideration.

        Returns:
            (float): Utility value associated with the offer
    """
    idx_counter = 0
    true_offer = np.zeros(len(b))
    true_items = items
    for element in range(0, len(b)):
        if element not in reduction_idx:
            true_offer[element] = offer[idx_counter]
            idx_counter += 1
    next_step = true_items + true_offer
    prev_state_value = true_items.transpose() @ A @ true_items + b @ true_items
    next_state_value = next_step.transpose() @ A @ next_step + b @ next_step
    return next_state_value - prev_state_value


def greedy_concession_algorithm(estimated_weights, weights_prob, available_offers, A, b, current_state_offering,
                                current_state_receiving, reduction_idx=[]):
    """
    Implements the Greedy Concession Algorithm with the probability of opponent acceptance.
    Args:
        estimated_weights (np.array): Estimated weights for each issue reflecting the opponent's preferences.
        weights_prob (np.array): Probability distribution over the estimated weights.
        available_offers (list): Set of available offers, where each offer is a vector of issue values.
        A (np.array): nxn matrix representing the utility function of the offering agent.
        b (np.array): n-dimensional vector of constants for the offering agent's utility function.
        current_state_offering (np.array): Current state of the offering agent (vector of issue values).
        current_state_receiving (np.array): Current state of the receiving agent (vector of issue values).
        reduction_idx (list, optional): List of item indices to exclude when calculating the utility. Defaults to an empty list.
    Returns:
        list: A sequence of optimal offers, sorted in descending order based on expected utility benefit.
    """


    # Convert estimated_weights to NumPy array for vectorized operations
    estimated_weights_array = np.array(estimated_weights)

    # Precompute the dot product for current_state_receiving with all weight sets
    precomputed_dot_product = np.dot(current_state_receiving, estimated_weights_array.T)

    # Initialize list for storing optimal offers
    optimal_offers = []
    # Loop over available offers
    for offer in available_offers:
        offer_array = np.array(offer)

        # Calculate opponent's utility for all weight sets at once (vectorized)
        opponent_utility = np.dot(current_state_receiving - offer_array,
                                  estimated_weights_array.T) - precomputed_dot_product

        # Calculate acceptance probability (sum weights where opponent_utility > 0)
        acceptance_probability_offer = np.sum(weights_prob[opponent_utility > 0])

        # Calculate utility benefit for the offering agent
        utility_benefit_offering = utility_value(offer_array, A, b, current_state_offering, reduction_idx)

        # Calculate expected utility benefit, append to list if beneficial
        expected_utility_benefit_offering = utility_benefit_offering * acceptance_probability_offer
        if utility_benefit_offering > 0:
            optimal_offers.append((offer, expected_utility_benefit_offering))
    # Sort the offers by expected utility benefit in descending order
    optimal_offers.sort(key=lambda x: x[1], reverse=True)

    # Return the sorted offers, without their scores
    return [offer for offer, _ in optimal_offers]

def softmax(logits, temperature=0.02):
    """
    Compute the softmax of the input logits with a specified temperature.
    Args:
        logits (array-like): Input values for which to compute the softmax.
        temperature (float, optional): Scaling factor that influences the "sharpness" of the output distribution. Defaults to 0.02.
    Returns:
        np.array: Softmax probabilities corresponding to the input logits, normalized to sum to 1.
    """
    logits = np.array(logits)
    exp_logits = np.exp(logits / temperature)
    return exp_logits / np.sum(exp_logits)

def offer_search_GCA(num_items, current_state_offering, current_state_responding, responding_A, offering_A, responding_b, offering_b, prev_weights, prev_weights_prob, max_trade_value = 5, total_queries=1000, update_interval = 1, shrinking_factor=0.99):
    """
    Implements the Greedy Concession Algorithm (GCA) with the probability of opponent acceptance.
    Args:
        num_items (int): Total number of item categories involved in the trading scenario.
        current_state_offering (np.array): Current number of items in the offering agent's possession.
        current_state_responding (np.array): Current number of items in the responding agent's possession.
        responding_A (np.array): nxn matrix representing the utility function of the responding agent.
        offering_A (np.array): nxn matrix representing the utility function of the offering agent.
        responding_b (np.array): n-dimensional vector of constants for the responding agent's utility function.
        offering_b (np.array): n-dimensional vector of constants for the offering agent's utility function.
        prev_weights (np.array): Set of estimated weights for the responding agent's preferences.
        prev_weights_prob (np.array): Probability distribution over the estimated weights.
        max_trade_value (int, optional): Maximum number of items that can be traded from any item category. Defaults to 5.
        total_queries (int, optional): Maximum number of queries allowed during the trading process. Defaults to 1000.
        update_interval (int, optional): Frequency of updates during the offer search. Defaults to 1.
        shrinking_factor (float, optional): Factor by which weights are shrunk during updates. Defaults to 0.99.
    Returns:
        accepted (bool): Whether an acceptable offer was found.
        best_offer (list): The best offer made during the trading process.
        query_counter (int): Total number of queries made.
        weights (np.array): Updated weights representing opponent preferences.
        weights_prob (np.array): Updated probability distribution over the estimated weights.
    """

    
    weights = prev_weights.copy()
    weights_prob = prev_weights_prob.copy()
    best_offers = generate_perturbations(num_items, (-max_trade_value, max_trade_value), current_state_offering, current_state_responding)

    # Convert potential offers to a set of tuples for efficient removals
    rejected_offers = []
    accepted = False
    query_counter = 0

    best_offers = greedy_concession_algorithm(weights, weights_prob, best_offers, offering_A, offering_b,
                                             current_state_offering, current_state_responding)
    if len(best_offers) == 0:
        return False, [], query_counter, weights, weights_prob
    while not accepted and query_counter < total_queries:
        # Convert the set back to a list for the greedy algorithm
        if len(best_offers) == 0:
            return False, [], query_counter, weights, weights_prob
        
        # Obtain the best offer from the sorted list
        best_offer = best_offers.pop(0)

        # Check that the best offer benefits the offering agent
        if not utility_improvement(best_offer, offering_A, offering_b, current_state_offering):
            return False, [], query_counter, weights, weights_prob
        
        # Check if the best offer is beneficial to the responding agent
        if len(best_offer) != 0:
            query_counter += 1
            if utility_improvement(-1 * best_offer, responding_A, responding_b, current_state_responding):
                accepted = True
            else:
                rejected_offers.append(best_offer)
        else:
            return False, [], query_counter, weights, weights_prob
        
        # If the offering agent has made enough offers, update the belief probabilities and sort the remaining set of offers.
        if query_counter % update_interval == update_interval - 1:
            weights, weights_prob = estimate_opponent_preferences(weights, weights_prob, current_state_responding, rejected_offers, shrinking_factor=shrinking_factor)
            if len(weights) == 0:
                return False, [], query_counter, weights, weights_prob
            best_offers = greedy_concession_algorithm(weights, weights_prob, best_offers, offering_A,
                                                        offering_b,
                                                        current_state_offering, current_state_responding)
    
    # If query budget is exceeded, return
    if query_counter >= total_queries:
        return False, [], query_counter, weights, weights_prob

    # Update the weights before returning
    weights, weights_prob = estimate_opponent_preferences(weights, weights_prob, current_state_responding, rejected_offers, shrinking_factor=shrinking_factor)

    return True, best_offer, query_counter, weights, weights_prob


def run_trading_scenario_GCA(num_items, offering_A, offering_b, responding_A, responding_b, starting_state_offering, starting_state_responding, num_sampled_weights = 10000, max_trade_value=5, offer_budget=1000, update_interval = 1, shrinking_factor = 0.99, softmax_temp=0.02, debug=False):
    """
    Run a trading scenario using the Greedy Concession Algorithm (GCA).
    Args:
        num_items (int): Total number of item categories involved in the trading scenario.
        offering_A (np.array): nxn matrix representing the utility function of the offering agent.
        offering_b (np.array): n-dimensional vector of constants for the offering agent's utility function.
        responding_A (np.array): nxn matrix representing the utility function of the responding agent. Used for querying.
        responding_b (np.array): n-dimensional vector of constants for the responding agent's utility function. Used for querying.
        offering_items (np.array): Initial quantities of items in each category for the offering agent.
        responding_items (np.array): Initial quantities of items in each category for the responding agent.
        num_sampled_weights (int, optional): Number of sampled weights for the trade search. Defaults to 10000.
        max_trade_value (int, optional): Maximum number of items that can be traded from any item category. Defaults to 5.
        offer_budget (int, optional): Maximum number of offers allowed to the responding agent. Defaults to 1000.
        update_interval (int, optional): Frequency of updates during the trading process. Defaults to 1.
        shrinking_factor (float, optional): Factor by which weights are shrunk during updates. Defaults to 0.99.
        softmax_temp (float, optional): Temperature parameter for the softmax function. Defaults to 0.02.
        debug (bool, optional): Flag to enable debugging output. Defaults to False.
    Returns:
        log (list of dicts): Log of the trading progression including details of each trade attempt.
    """
    
    offering_items = starting_state_offering.copy()
    responding_items = starting_state_responding.copy()

    # Initialize Trading Loop
    trade_num = 0
    log = []
    cumulative_offer_count = 0
    weights = sample_unit_hypersphere(num_items, num_sampled_weights=num_sampled_weights)

    weights_prob = initialize_weights_uniform(len(weights))
    # Loop for a set number of trades or until the algorithm hits the query limit
    while trade_num <= 100 and cumulative_offer_count < offer_budget:
        trade_num += 1
        # Use the respective algorithm to find a trade
        starting_time = time.time()
        found_trade, true_out_offer, query_count, weights, weights_prob = offer_search_GCA(num_items, offering_items, responding_items, responding_A, offering_A, responding_b, offering_b, weights, weights_prob, max_trade_value = max_trade_value, total_queries = offer_budget - cumulative_offer_count, update_interval = update_interval, shrinking_factor = shrinking_factor)
        ending_time = time.time()
        weights_prob = softmax(weights_prob, temperature=softmax_temp)

        # Update the item values if a trade has been found
        information_dict = {}
        if found_trade == True:
            information_dict["found_trade"] = True
            prev_responding_items = responding_items.copy()
            prev_offering_items = offering_items.copy()
            responding_items -= true_out_offer
            offering_items += true_out_offer

            # Account for cases where items are very close to zero
            for i in range(0, len(responding_items)):
                if responding_items[i] < 0.00000001:
                    responding_items[i] = 0
                if offering_items[i] < 0.00000001:
                    offering_items[i] = 0

            # Update State Values
            prev_state_value_h = prev_responding_items.transpose() @ responding_A @ prev_responding_items + responding_b @ prev_responding_items
            prev_state_value_a = prev_offering_items.transpose() @ offering_A @ prev_offering_items + offering_b @ prev_offering_items

            next_state_value_h = responding_items.transpose() @ responding_A @ responding_items + responding_b @ responding_items
            next_state_value_a = offering_items.transpose() @ offering_A @ offering_items + offering_b @ offering_items
            if debug:
                print("GCA Result: ", found_trade, ending_time-starting_time)
                print("Accepted Offer: ", true_out_offer)
                print("New Responding Agent Item List: ", responding_items)
                print("New Offering Agent Item List: ", offering_items)
                print("Offering Agent benefit ", next_state_value_a - prev_state_value_a)
                print("Responding Agent benefit ", next_state_value_h, prev_state_value_h,
                      next_state_value_h - prev_state_value_h)
                print("Query Count: ", query_count)
                print("\n")

            # Log trade progression information
            information_dict["offer"] = true_out_offer.tolist()
            information_dict["responding_items"] = responding_items.tolist()
            information_dict["offering_items"] = offering_items.tolist()
            information_dict["responding_benefit"] = next_state_value_h - prev_state_value_h
            information_dict["offering_benefit"] = next_state_value_a - prev_state_value_a
            information_dict["query_count"] = query_count

            # Update Cumulative Query Count
            cumulative_offer_count += query_count
            prev_offer = true_out_offer.copy()
            log.append(information_dict)

            # If the trade was not beneficial to both parties, then an error has occurred.
            if next_state_value_h - prev_state_value_h < 0.000001 and next_state_value_a - prev_state_value_a < 0.000001:
                trade_num = 101
            if len(weights) == 0:
                trade_num = 101
        else:
            # If informed trading failed to find a trade, switch to pure random trading
            trade_num = 101
        if not all(i >= 0 for i in responding_items):
            break
        if not all(i >= 0 for i in offering_items):
            break
    return log


def n_dim_quad_grad(A, b, x):
    """
        Find the gradient vector of an n-dimensional quadratic function of the form x^TAx + x^Tb

        Args:
            A (np.array): nxn matrix.
            b (np.array): n-dimensional vector.
            x (list): Current State.

        Returns:
            gradient (np.array): n-dimensional vector representing the gradient of the function at state x.
    """
    gradient = 2 * np.dot(A, x) + b
    return gradient

def maximize_quadratic(A, b):
    # Solve for x using the formula x = -0.5 * A^-1 * b
    x_optimal = -0.5 * np.linalg.inv(A).dot(b)
    return x_optimal

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