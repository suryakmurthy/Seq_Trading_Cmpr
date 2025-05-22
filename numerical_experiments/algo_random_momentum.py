import numpy as np

def query(offer, A, b, responding_items):
    """
        Query an agent with utility function x^TAx + 2bx to determine if they will accept an offer.

        Args:
            offer (np.array): n-dimensional vector representing an offer from the receiving agent's perspective.
            A (np.array): nxn matrix.
            b (list): n-dimensional vector.
            responding_items (list): Current State of responding agent.

        Returns:
            flag_n_items (bool): Response value which states if the responding agent has enough items to complete the trade
            flag_utility_improvement (bool): Response value which states if the responding agent's utility improves with the offer
            min_index (int): Item index corresponding to the smallest number of items in the responding agent's possession
            min_items (int/float): Minimum number of items that the responding agent possess for any give category.
    """
    next_step = offer + responding_items
    flag_utility_improvement = utility_improvement(offer, A, b, responding_items)
    min_index = -1
    min_items = 0
    # Check if the offer is feasible for the responding agent. Threshold accounts for float errors.
    if all(i >= 0.00000001 for i in next_step):
        flag_n_items = True
    else:
        flag_n_items = False
        min_index = np.argmin(next_step)
        min_items = responding_items[min_index]
    return flag_n_items, flag_utility_improvement, min_index, min_items

def utility_improvement(offer, A, b, items):
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
    next_step = items + offer
    prev_state_value = items.transpose() @ A @ items + b @ items
    next_state_value = next_step.transpose() @ A @ next_step + b @ next_step
    # Threshold accounts for trades near optimal points.
    if next_state_value - prev_state_value > 0.000001:
        return True
    else:
        return False

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


def random_offer_search(offering_A, offering_b, responding_A, responding_b, offering_items, responding_items, max_trade_value=5, integer_constraint=True, offer_budget=1000):
    """
        Perform a random search to find an initial trade offer that both parties can accept.

        Args:
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            offer_budget (int, optional): Maximum number of offers to search for a valid offer. Defaults to 1000.

        Returns:
            tuple:
                - accepted_trade (bool): True if an acceptable trade was found, False otherwise.
                - offer (list): The trade offer that was generated.
                - offer_counter (int): Number of offers made to the receiving agent during the search.
        """
    # Initialize values
    accepted_trade = False
    offer_counter = 0
    reduction_idx = []
    # If one agent has all of an item category, remove it from consideration.
    for i in range(len(offering_items)):
        if offering_items[i] == 0 or responding_items[i] == 0:
            reduction_idx.append(i)
    while not accepted_trade and offer_counter < offer_budget:
        # Generate a random offer
        offer = generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value=max_trade_value, reduction_idx = reduction_idx, integer_constraint=integer_constraint)
        neg_offer = -1 * offer
        offer_counter += 1
        
        # Present the offer to the responding agent
        flag_n_items, flag_utility_improvement, min_index, min_items = query(neg_offer, responding_A, responding_b, responding_items)
        if flag_utility_improvement and flag_n_items:
            accepted_trade = True
        else:
            if not flag_n_items:
                max_trade_value = min_items
    return accepted_trade, offer, offer_counter


def generate_random_feasible_offer(max_trade_value, offering_items, responding_items, num_items):
    """
        Find a random vector that is feasible given the items of both parties

        Args:
            max_trade_value (int): maximum number of items that can be traded from any item category
            offering_items (list): n-dimensional vector containing the .
            responding_items (list): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            num_items (int): Number of item categories
            
        Returns:
            - scaled_vector (np.array): Vector representing a feasible trade offer
    """
    while True:
        # Generate a random unit vector
        random_vector = np.random.randn(num_items)
        random_vector /= np.linalg.norm(random_vector)

        # Scale the vector so that no entry violates the constraints
        scaling_factors = []
        for i in range(num_items):
            if random_vector[i] > 0:
                max_scale_offering = (np.inf - offering_items[i]) / random_vector[i]
                max_scale_responding = responding_items[i] / random_vector[i]
            else:
                max_scale_offering = offering_items[i] / abs(random_vector[i])
                max_scale_responding = (np.inf - responding_items[i]) / abs(random_vector[i])

            scaling_factors.append(min(max_scale_offering, max_scale_responding))

        # Find the maximum scaling factor that keeps vector within bounds
        max_scaling_factor = min(scaling_factors)

        # Ensure no entry exceeds the maximum magnitude
        scale = min(max_scaling_factor, max_trade_value / max(abs(random_vector)))

        # Scale the vector
        scaled_vector = random_vector * scale
        tolerance = 1e-10
        # Check if the resultant vector meets the requirements
        if np.all(np.isclose(scaled_vector, np.clip(scaled_vector, -max_trade_value, max_trade_value), atol=tolerance)):
            return scaled_vector

def generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value = 5, reduction_idx = [], integer_constraint=True):
    """
        Find a random offer that is feasible given the items of both parties and benefits the offering agent

        Args:
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            max_trade_value (int, optional): maximum number of items that can be traded from any item category
            reduction_idx (list, optional): set of item indices that need to be removed from consideration.
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.

        Returns:
            random_vector_full (np.array): Vector representing the full trade offer. If the random offer generator cannot find an offer that benefits the offering agent, it is near-optimal, and trading ends.
    """
    # Generate a random vector of integers of the same dimensionality as the original vector
    if integer_constraint:
        random_vector = np.random.randint(-max_trade_value, max_trade_value + 1,
                                          size=len(offering_items) - len(reduction_idx))
    else:
        mask = np.ones(len(offering_items), dtype=bool)
        mask[reduction_idx] = False
        offering_red = offering_items[mask]
        responding_red = responding_items[mask]
        random_vector = generate_random_feasible_offer(max_trade_value, offering_red, responding_red,
                                                        len(offering_items) - len(reduction_idx))

    max_iterations_a = 1000
    i_counter = 1
    random_vector_full = obtain_full_offer(random_vector, reduction_idx, len(offering_b))

    # Ensure the offer is beneficial to the offering agent
    flag_n_items, flag_utility_improvement, min_index, min_items = query(random_vector_full, offering_A, offering_b, offering_items)
    while not (flag_utility_improvement and flag_n_items) and i_counter <= max_iterations_a:

        # If the offer is not feasible for the offering agent adjust the maximum trade value
        if (not flag_n_items) and min_items < max_trade_value:
            max_trade_value = min_items
            i_counter -= 1

        # If the offer is not beneficial, generate a new one.
        if integer_constraint:
            random_vector = np.random.randint(-max_trade_value, max_trade_value+1, size=len(offering_items)-len(reduction_idx))
        else:
            mask = np.ones(len(offering_items), dtype=bool)
            mask[reduction_idx] = False
            offering_red = offering_items[mask]
            responding_red = responding_items[mask]
            random_vector = generate_random_feasible_offer(max_trade_value, offering_red, responding_red, len(offering_items)-len(reduction_idx))
        random_vector_full = obtain_full_offer(random_vector, reduction_idx, len(offering_b))
        flag_n_items, flag_utility_improvement, min_index, min_items = query(random_vector_full, offering_A, offering_b, offering_items)
        i_counter += 1

    # If no beneficial offer was found, stop trading
    if not(flag_utility_improvement and flag_n_items):
        return []
    else:
        return random_vector_full

def obtain_full_offer(offer, reduction_idx, full_size):
    """
        Given an offer that may be reduced by removing item categories with zero items, return the full offer

        Args:
            offer (np.array): Reduced offer.
            reduction_idx (list): set of item indices that need to be removed from consideration.
            full_size (int): Total number of item categories

        Returns:
            full_offer (np.array): Vector representing the full trade offer. Item categories that are not considered are filled in with 0 values.
    """
    idx_counter = 0
    full_offer = np.zeros(full_size)
    for i in range(0, full_size):
        if i not in reduction_idx:
            full_offer[i] = offer[idx_counter]
            idx_counter += 1
    return full_offer

def random_trading_momentum_search(offering_A, offering_b, responding_A, responding_b, offering_items, responding_items, max_trade_value = 5, deviation_interval=0.05, max_deviation_magnitude=5, prev_offer=[], integer_constraint=True, offer_budget=1000):
    """
        Perform a random search with momentum to find a trade offer that both parties can accept.

        Args:
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding.
            prev_offer (np.array, optional): Previously accepted trade offer, if one exits. Defaults to an empty list.
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            offer_budget (int, optional): Maximum number of iterations to search for a valid offer. Defaults to 1000.

        Returns:
            tuple:
                - accepted_trade (bool): True if an acceptable trade was found, False otherwise.
                - offer (list): The trade offer that was generated.
                - offer_counter (int): Number of offers made to the receiving agent during the search.
    """

    accepted_trade = False
    offer_counter = 0
    reduction_idx = []
    max_trade_value_original = max_trade_value
    # Remove category if any agent has all of a given item
    for i in range(len(offering_items)):
        if offering_items[i] == 0 or responding_items[i] == 0:
            reduction_idx.append(i)
    if len(reduction_idx) >= len(offering_items) - 1:
        return False, [], 0
    if len(prev_offer) == 0:
        return random_offer_search(offering_A, offering_b, responding_A, responding_b, offering_items, responding_items, max_trade_value=max_trade_value, offer_budget=offer_budget, integer_constraint=integer_constraint)

    # Initialize deviation magnitude to 0. This will offer the previously accepted trade
    deviation_magnitude = 0
    while not accepted_trade and offer_counter < offer_budget:
        offer, offering_accepted_trade = generate_beneficial_random_offer_momentum(offering_A, offering_b, offering_items, responding_items, deviation_magnitude, prev_offer, integer_constraint=integer_constraint, max_trade_value_original=max_trade_value_original, reduction_idx = reduction_idx)
        offer_counter += 1
        if not offering_accepted_trade:
            # Increase deviation magnitude if the offering agent does not accept the trade
            deviation_magnitude += deviation_interval
            if deviation_magnitude >= max_deviation_magnitude:
                deviation_magnitude = max_deviation_magnitude
            continue
        # Make the offer to the responding agent
        neg_offer = -1 * offer
        flag_n_items, flag_utility_improvement, min_index, min_items = query(neg_offer, responding_A, responding_b, responding_items)
        if flag_utility_improvement and flag_n_items and offering_accepted_trade:
            accepted_trade = True
        else:
            if not flag_n_items:
                max_trade_value = min_items

        # Increase deviation magnitude
        deviation_magnitude += deviation_interval
        if deviation_magnitude >= max_deviation_magnitude:
            deviation_magnitude = max_deviation_magnitude
    return accepted_trade, offer, offer_counter

def random_vector_with_deviation(prev_offer, deviation_magnitude, offering_items, responding_items, max_trade_value=5,
                                       integer_constraint=False):
    """
        Generate a new feasible offer given a previous offer with a random deviation of magnitude deviation_magnitude.
        Args:
            prev_offer (np.array): n-dimensional vector corresponding to a previously accepted offer.
            deviation_magnitude (int): Magnitude of the randomly sampled deviations to the previously accepted offer.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding agent.
            max_trade_value (np.array, optional): Previously accepted trade offer, if one exits. Defaults to an empty list.
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            
        Returns:
            scaled_vector (np.array): n-dimensional vector representing the previously accepted vector with a random deviation.
    """
    counter = 0
    while True:
        # Generate a random unit vector
        ball_vector = np.random.randn(len(prev_offer))
        ball_vector /= np.linalg.norm(ball_vector)

        # Limit the deviation_magnitude to a maximum value of 10
        deviation_magnitude = min(deviation_magnitude, 100)

        # Scale the vector based on the deviation_magnitude
        next_vector = (prev_offer / np.linalg.norm(prev_offer)) + (deviation_magnitude * ball_vector)
        # Normalize the next vector
        next_vector_norm = next_vector / np.linalg.norm(next_vector)

        # Calculate scaling factors to ensure constraints are met
        scaling_factors = []
        for i in range(len(prev_offer)):
            if next_vector_norm[i] > 0:
                max_scale_offering = (np.inf - offering_items[i]) / next_vector_norm[i]
                max_scale_responding = responding_items[i] / next_vector_norm[i]
            else:
                max_scale_offering = offering_items[i] / abs(next_vector_norm[i])
                max_scale_responding = (np.inf - responding_items[i]) / abs(next_vector_norm[i])

            scaling_factors.append(min(max_scale_offering, max_scale_responding))

        # Find the maximum scaling factor that keeps vector within bounds
        max_scaling_factor = min(scaling_factors)

        # Ensure no entry exceeds the maximum magnitude
        scale = min(max_scaling_factor, max_trade_value / max(abs(next_vector_norm)))

        # Scale the vector
        scaled_vector = next_vector_norm * scale
        counter += 1
        tolerance = 1e-10
        # Check if the resultant vector meets the requirements
        if np.all(np.isclose(scaled_vector, np.clip(scaled_vector, -max_trade_value, max_trade_value), atol=tolerance)):
            if integer_constraint:
                return np.round(scaled_vector)
            else:
                return scaled_vector

def generate_beneficial_random_offer_momentum(offering_A, offering_b, offering_items, responding_items, deviation_magnitude, prev_offer, max_trade_value_original = 5, integer_constraint=True, reduction_idx = []):
    """
        Generate a new offer that is beneficial to the offering agent given a previous offer and a magnitude of deviations from the previous offer.

        Args:
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding agent.
            deviation_magnitude (int): Magnitude of the randomly sampled deviations to the previously accepted offer.
            prev_offer (np.array): n-dimensional vector corresponding to a previously accepted offer.
            max_trade_value_original (int, optional): maximum number of items that can be traded from any item category.
            reduction_idx (list, optional): set of item indices that need to be removed from consideration.
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.

        Returns:
            tuple:
                - random_vector_full (np.array): Vector representing the full trade offer.
                - accepted_trade (bool): Boolean representing if the offering agent benefits from the returned offer. If the offer generator cannot find an offer that benefits the offering agent, it is near-optimal, and trading ends.
    """
    max_iterations = 1000
    i_counter = 0
    flag_n_items = False
    flag_utility_improvement = False
    while not (flag_utility_improvement and flag_n_items) and i_counter < max_iterations:
        # Generate Offer
        prev_offer_norm = prev_offer / np.linalg.norm(prev_offer)
        mask = np.ones(len(offering_items), dtype=bool)
        mask[reduction_idx] = False
        offering_red = offering_items[mask]
        responding_red = responding_items[mask]
        prev_offer_norm_red = prev_offer_norm[mask]
        random_vector = random_vector_with_deviation(prev_offer_norm_red, deviation_magnitude, offering_red, responding_red, max_trade_value=max_trade_value_original, integer_constraint=integer_constraint)
        random_vector_full = obtain_full_offer(random_vector, reduction_idx, len(offering_b))
        
        # Ensure that the offer is beneficial/feasible for the offering agent
        flag_n_items, flag_utility_improvement, min_index, min_items = query(random_vector_full, offering_A, offering_b, offering_items)
        i_counter += 1
    
    # Check if the offering agent benefits from the offer
    accepted_trade = True
    flag_n_items, flag_utility_improvement, min_index, min_items = query(random_vector_full, offering_A, offering_b, offering_items)
    
    # If a beneficial trade cannot be found, the offering agent is near an optimal point.
    if not flag_utility_improvement or not flag_n_items:
        accepted_trade = False
    return random_vector_full, accepted_trade

def run_trading_scenario_random_trading_momentum(num_items, offering_A, offering_b, responding_A, responding_b, offering_items, responding_items, offer_budget, max_trade_value=5, deviation_interval=0.05, max_deviation_magnitude=5, integer_constraint=True, debug=False):
    """
        Run a trading scenario with using the random trading with momentum baseline.

        Args:
            num_items (int): Total number of item categories in the trading scenario
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding agent.
            offer_budget (int): The total number of offers to the responding agent that the negotiating agent is allowed to make.
            max_trade_value (int, optional): maximum number of items that can be traded from any item category
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            debug (bool, optional): Boolean that controls if extra debugging information should be printed.
        Returns:
            log (list of dictionaries): List that represents trading progression for the scenario
    """
    
    trade_num = 0
    log = []
    prev_offer = []
    cumulative_offer_count = 0
    while trade_num <= 100 and cumulative_offer_count < offer_budget:
        trade_num += 1
        original_responding_gradient = list(n_dim_quad_grad(responding_A, responding_b, responding_items))
        original_offering_gradient = list(n_dim_quad_grad(offering_A, offering_b, offering_items))
        found_trade, true_out_offer, offer_count = random_trading_momentum_search(offering_A, offering_b, responding_A, responding_b, np.array(offering_items), np.array(responding_items), max_trade_value=max_trade_value, deviation_interval=deviation_interval, max_deviation_magnitude=max_deviation_magnitude, prev_offer=prev_offer, integer_constraint=integer_constraint, offer_budget=offer_budget-cumulative_offer_count)

        information_dict = {}
        if found_trade == True:
            # Update the item values if a trade has been found
            prev_responding_items = responding_items.copy()
            prev_offering_items = offering_items.copy()
            responding_items -= true_out_offer
            offering_items += true_out_offer

            # Account for values close to zero in fractional trading scenarios
            for i in range(0, len(responding_items)):
                if responding_items[i] < 0.00000001:
                    responding_items[i] = 0
                if offering_items[i] < 0.00000001:
                    offering_items[i] = 0

            # Check benefit for both parties
            neg_offer = [-1 * x for x in true_out_offer]
            prev_state_value_h = prev_responding_items.transpose() @ responding_A @ prev_responding_items + responding_b @ prev_responding_items
            prev_state_value_a = prev_offering_items.transpose() @ offering_A @ prev_offering_items + offering_b @ prev_offering_items
            next_state_value_h = responding_items.transpose() @ responding_A @ responding_items + responding_b @ responding_items
            next_state_value_a = offering_items.transpose() @ offering_A @ offering_items + offering_b @ offering_items

            # Print Trade Status to Terminal
            if debug:
                print("Random Trading with Momentum Result: ", found_trade)
                print("Accepted Offer: ", true_out_offer)
                print("Offering Agent Gradient: ", original_offering_gradient)
                print("Responding Agent Gradient: ", original_responding_gradient)
                print("New Responding Agent Item List: ", responding_items)
                print("New Offering Agent Item List: ", offering_items)
                print("Offering Agent Gradient Original: ",
                      (original_offering_gradient / np.linalg.norm(original_offering_gradient)))
                print("Offering Agent predicted benefit ",
                      np.dot(true_out_offer, (original_offering_gradient / np.linalg.norm(original_offering_gradient))))
                print("Responding Agent predicted benefit ",
                      np.dot(neg_offer, (original_responding_gradient / np.linalg.norm(original_responding_gradient))))

                print("Offering Agent benefit ", next_state_value_a - prev_state_value_a)
                print("Responding Agent benefit ", next_state_value_h, prev_state_value_h, next_state_value_h - prev_state_value_h)
                print("Offer Count: ", offer_count)
                print("\n")

            # Record Trading Progression Information
            information_dict["found_trade"] = True
            information_dict["offer"] = true_out_offer.tolist()
            information_dict["responding_items"] = responding_items.tolist()
            information_dict["offering_items"] = offering_items.tolist()
            information_dict["responding_benefit"] = next_state_value_h - prev_state_value_h
            information_dict["offering_benefit"] = next_state_value_a - prev_state_value_a
            information_dict["query_count"] = offer_count
            prev_offer = true_out_offer.copy()
            log.append(information_dict)

            # Update Cumulative Offer Budget
            cumulative_offer_count += offer_count
            # DEBUG: Check to see if the prior iteration resulted in a non-beneficial trade
            if next_state_value_h - prev_state_value_h < 0.000001 and next_state_value_a - prev_state_value_a < 0.000001:
                trade_num = 101
        else:
            trade_num = 101
        
        # DEBUG: Check to see if the prior iteration resulted in any negative item values
        if not all(i >= 0 for i in responding_items):
            break
        if not all(i >= 0 for i in offering_items):
            break
    return log