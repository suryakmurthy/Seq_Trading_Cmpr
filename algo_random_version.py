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
    flag_dot_product = utility_improvement(offer, A, b, responding_items)
    min_index = -1
    min_items = 0
    if all(i >= -0.00000001 for i in next_step):
        flag_n_items = True
    else:
        flag_n_items = False
        min_index = np.argmin(next_step)
        min_items = responding_items[min_index]
    return flag_n_items, flag_dot_product, min_index, min_items

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
    if next_state_value - prev_state_value > 0:
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

def random_offer_search(offering_A, offering_b, responding_A, responding_b, offering_items, responding_items, max_trade_value, prev_offer=[], informed = True, integer_constraint=True, offer_budget=1000):
    """
        Find a mutually beneficial trade using a random search approach

        Args:
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            prev_offer (np.array): Previously accepted offer used for heuristic trading. Default: Empty
            informed (bool, optional): Whether the algorithm should use the previously accepted trade heuristic for the first offer. Defaults to True.
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            offer_budget (int, optional): Maximum number of offers allowed to the receiving agent. Defaults to 1000.

        Returns:
            tuple:
                - accepted_trade (bool): True if an acceptable trade was found, False otherwise.
                - offer (list): The trade offer that was generated.
                - offer_counter (int): Number of offers made to the receiving agent during the search.
    """
    accepted_trade = False
    offer_counter = 0
    
    # Remove categories where one agent has all the items
    reduction_idx = []
    for i in range(len(offering_items)):
        if offering_items[i] == 0 or responding_items[i] == 0:
            reduction_idx.append(i)
    if len(reduction_idx) >= len(offering_items) - 1:
        return False, [], 0

    while not accepted_trade and offer_counter < offer_budget:
        # If the algorithm is using the previous trade heuristic, start by considering the previous trade
        if informed and offer_counter == 0:
            offer = generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value=max_trade_value, reduction_idx = reduction_idx, prev_offer = prev_offer, integer_constraint=integer_constraint)
        else:
            offer = generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value=max_trade_value, reduction_idx = reduction_idx, integer_constraint=integer_constraint)

        # If no beneficial offer was found for the offering agent, stop trading
        if len(offer) == 0:
            return False, [], offer_counter

        # Present the offer to the responding agent
        neg_offer = -1 * offer
        offer_counter += 1
        flag_n_items, flag_dot_product, min_index, min_items = query(neg_offer, responding_A, responding_b, responding_items)
        if flag_dot_product and flag_n_items:
            accepted_trade = True
        else:
            if not flag_n_items:
                offer_counter -= 1
                max_trade_value = min_items

    return accepted_trade, offer, offer_counter

def obtain_scaling(offer, offering_items, responding_items, max_trade_value):
    """
        Find a scaling factor for a potential offer results in a feasible offer that abides by the maximum trade value.

        Args:
            offer (np.array): nxn matrix used for the offering agent's utility function.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding.
            max_trade_value (int): Maximum number of items that can be traded from any item category

        Returns:
            scale (float): scaling factor for the offer that, when multiplied to a normalized offer, will result in a feasible offer for both parties.
    """
    dimensions = len(offering_items)
    scaling_factors = []
    for i in range(dimensions):
        if offer[i] > 0:
            max_scale_offering = (np.inf - offering_items[i]) / offer[i]
            max_scale_responding = responding_items[i] / offer[i]
        else:
            max_scale_offering = offering_items[i] / abs(offer[i])
            max_scale_responding = (np.inf - responding_items[i]) / abs(offer[i])

        scaling_factors.append(min(max_scale_offering, max_scale_responding))

    # Find the maximum scaling factor that keeps the offer in within the bound
    max_scaling_factor = min(scaling_factors)

    # Ensure no entry exceeds the maximum magnitude
    scale = min(max_scaling_factor, max_trade_value / max(abs(offer)))
    return scale

def generate_random_feasible_offer(max_trade_value, offering_items, responding_items, dimensions):
    """
        Find a random feasible offer for both agents

        Args:
            max_trade_value (int): Maximum number of items that can be traded from any item category.
            offer (np.array): nxn matrix used for the offering agent's utility function.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding.


        Returns:
            scaled_vector (np.array): scaled random offer that is feasible for both agents.
    """
    while True:
        # Generate a random unit vector
        random_vector = np.random.randn(dimensions)
        random_vector /= np.linalg.norm(random_vector)

        # Scale the vector so that no entry violates the constraints

        scale = obtain_scaling(random_vector,offering_items,responding_items,max_trade_value)
        # Scale the vector
        scaled_vector = random_vector * scale

        # Check if the resultant vector meets the requirements
        if all(abs(scaled_vector[i]) <= max_trade_value for i in range(dimensions)):
            return scaled_vector

def generate_random_beneficial_offer(offering_A, offering_b, offering_items, responding_items, max_trade_value = 5, reduction_idx = [], prev_offer = [], integer_constraint=True):
    """
        Find a random feasible offer that is beneficial for the offering agent

        Args:
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            offering_items (np.array): List of current items for the offering agent.
            responding_items (np.array): List of current items for the responding.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            reduction_idx (list, optional): set of item indices that need to be removed from consideration.
            prev_offer (np.array): Previously accepted offer used for heuristic trading. Default: Empty
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.


        Returns:
            random_vector_full (np.array): Scaled random offer that is beneficial for the offering agent.
    """

    # If there is no previous offer, generate a random vector of the same dimensionality as the original vector
    if len(prev_offer) == 0:
        if integer_constraint:
            random_vector = np.random.randint(-max_trade_value, max_trade_value+1, size=len(offering_items)-len(reduction_idx))
        else:
            mask = np.ones(len(offering_items), dtype=bool)
            mask[reduction_idx] = False
            offering_red = offering_items[mask]
            responding_red = responding_items[mask]
            random_vector = generate_random_feasible_offer(max_trade_value, offering_red, responding_red, len(offering_items)-len(reduction_idx))
    else:

        # If there is previous offer, ensure it is feasible.
        random_vector = prev_offer
        random_vector = random_vector/np.linalg.norm(random_vector)
        scale = obtain_scaling(random_vector, offering_items, responding_items, max_trade_value)
        random_vector *= scale
        if integer_constraint:
            random_vector = np.round(random_vector)

    # Ensure the randomly generated offer is beneficial to the offering agent. If not, generate a new offer.
    max_generations = 1000
    i_counter = 0
    random_vector_full = obtain_full_offer(random_vector, reduction_idx, len(offering_b))
    flag_n_items, flag_dot_product, min_index, min_items = query(random_vector_full, offering_A, offering_b, offering_items)
    while not (flag_dot_product and flag_n_items) and i_counter <= max_generations:
        # Ensure the offer is feasible
        if (not flag_n_items) and min_items < max_trade_value:
            max_trade_value = min_items
            i_counter -= 1

        # If the offer direction is not beneficial, generate a new offer.
        if integer_constraint:
            random_vector = np.random.randint(-max_trade_value, max_trade_value+1, size=len(offering_items)-len(reduction_idx))
        else:
            mask = np.ones(len(offering_items), dtype=bool)
            mask[reduction_idx] = False
            offering_red = offering_items[mask]
            responding_red = responding_items[mask]
            random_vector = generate_random_feasible_offer(max_trade_value, offering_red, responding_red, len(offering_items)-len(reduction_idx))
        random_vector_full = obtain_full_offer(random_vector, reduction_idx, len(offering_b))
        flag_n_items, flag_dot_product, min_index, min_items = query(random_vector_full, offering_A, offering_b, offering_items)
        i_counter += 1

    # If no beneficial offer was found, stop trading
    if not(flag_dot_product and flag_n_items):
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

def run_trading_scenario_random(num_items, offering_A, offering_b, responding_A, responding_b, offering_items, responding_items, offer_budget, max_trade_value=5, integer_constraint=True, informed = True, debug=False):
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
            informed (bool, optional): Whether the algorithm should use the previous trade heuristic. Defaults to True.
            debug (bool, optional): Boolean that controls if extra debugging information should be printed.
        Returns:
            log (list of dictionaries): List that represents trading progression for the scenario
    """

    # Initialize Trading Loop
    trade_num = 0
    log = []
    prev_offer = []
    cumulative_offer_count = 0

    # Loop for a set number of trades or until the algorithm hits the query limit
    while trade_num <= 100 and cumulative_offer_count < offer_budget:
        trade_num += 1
        original_responding_gradient = list(n_dim_quad_grad(responding_A, responding_b, responding_items))
        original_offering_gradient = list(n_dim_quad_grad(offering_A, offering_b, offering_items))

        # Use the respective algorithm to find a trade
        if informed:
            found_trade, true_out_offer, query_count = random_offer_search(offering_A, offering_b, responding_A, responding_b, np.array(offering_items), np.array(responding_items),  max_trade_value, prev_offer=prev_offer, informed=True, integer_constraint=integer_constraint, offer_budget=offer_budget-cumulative_offer_count)
        else:
            found_trade, true_out_offer, query_count = random_offer_search(offering_A, offering_b, responding_A, responding_b, np.array(offering_items), np.array(responding_items), max_trade_value, informed=False, integer_constraint=integer_constraint, offer_budget=offer_budget-cumulative_offer_count)
        
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
            neg_offer = [-1 * x for x in true_out_offer]
            prev_state_value_h = prev_responding_items.transpose() @ responding_A @ prev_responding_items + responding_b @ prev_responding_items
            prev_state_value_a = prev_offering_items.transpose() @ offering_A @ prev_offering_items + offering_b @ prev_offering_items

            next_state_value_h = responding_items.transpose() @ responding_A @ responding_items + responding_b @ responding_items
            next_state_value_a = offering_items.transpose() @ offering_A @ offering_items + offering_b @ offering_items
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
        else:
            # If informed trading failed to find a trade, switch to pure random trading
            trade_num = 101
        if not all(i >= 0 for i in responding_items):
            break
        if not all(i >= 0 for i in offering_items):
            break
    return log