import numpy as np
import math
import itertools
from scipy.optimize import minimize
import scipy
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

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
    # Generate the post-trade state
    next_step = offer + responding_items

    # Determine if the trade leads to an increase in responding agent's utility
    flag_dot_product = utility_improvement(offer, A, b, responding_items)

    # Determine if the responding agent has enough of each item to complete the trade
    min_index = -1
    min_items = 0
    if all(i >= 0 for i in next_step):
        flag_n_items = True
    else:
        flag_n_items = False
        min_index = np.argmin(next_step)
        min_items = responding_items[min_index]
    
    return flag_n_items, flag_dot_product, min_index, min_items

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


def branch_and_bound(offer, center_of_cone, offering_grad):
    """
        Given an offer, return an integer offer that is within 90 degrees of the cone's center and closest to the offering_gradient
        Args:
            offer (np.array): n-dimensional vector representing an offer from the agent's perspective.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
            offering_grad (np.array): n-dimensional vector corresponding to the offering agent's gradient.
            
        Returns:
            output_offer (np.array): Rounded Offer
    """
    
    # Generate a set of rounded offers
    rounded_list = generate_int_vectors(offer)
    theta_list = []

    # This for loop ensures that the rounded offers are aligned with the center of the cone
    for int_vector in rounded_list:
        # Represent the offer from the perspective of the responding
        int_list = list(int_vector)
        neg_list = [-1 * x for x in int_list]
        neg_list_norm = neg_list / np.linalg.norm(neg_list)
        center_of_cone_norm = center_of_cone / np.linalg.norm(center_of_cone)
        dot_product = np.dot(neg_list_norm, center_of_cone_norm)

        # If the offer and the center of the cone are not algined, then the responding agent will not accept the offer
        # In this case, we remove the offer from consideration
        if dot_product < 0:
            theta_list.append(-1 * np.inf)
        else:
            theta_list.append(np.dot(int_list/np.linalg.norm(int_list), offering_grad))
    
    # Select the rounded offer that is most closely aligned with the offering gradient direction
    output_offer = list(rounded_list[np.argmax(theta_list)])
    # If the offer is not aligned with the offering gradient, negate it's direction.
    if np.dot(output_offer, offering_grad) < 0:
        output_offer = -1 * np.array(output_offer)
    return output_offer

def generate_int_vectors(float_vector):
    """
        Given a vector of floats, return a set of vectors that represents the integer rounding of the set of floats
        Args:
            float_vector (np.array): n-dimensional float

        Returns:
            integer_combinations (list of np.array): Set of all possible roundings of the float vector
    """
    float_vector = [float(num) for num in float_vector]
    combinations = set(itertools.product(*[range(math.floor(val), math.ceil(val) + 1) for val in float_vector]))
    integer_combinations = [tuple(round(val) if not isinstance(val, int) else val for val in combo) for combo in combinations]
    icc = integer_combinations.copy()
    for combination in icc:
        if all(element == 0 for element in combination):
            integer_combinations.remove(combination)
    return integer_combinations

def find_init_offer_greedy(offering_grad, center_of_cone):
    """
        Determine an offer that is orthogonal and is in the direction of the offering agent's gradient
        Args:
            offering_grad (np.array): n-dimensional vector corresponding to the offering agent's gradient.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.

        Returns:
            offer (np.array): Projection of the offering agent's gradient on the null space of the cone center.
    """
    projection = np.dot(offering_grad, center_of_cone) / np.dot(center_of_cone, center_of_cone) * center_of_cone
    offer = offering_grad - projection
    return offer

def find_init_offer_random(offering_grad, center_of_cone):
    """
        Determine a random initial offer that is orthogonal and that is aligned with the offering agent's gradient
        Args:
            offering_grad (np.array): n-dimensional vector corresponding to the offering agent's gradient.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.

        Returns:
            offer (np.array): Random vector in the null space of the cone center
    """
    # Check if the vector is not a zero vector
    if np.linalg.norm(center_of_cone) == 0:
        raise ValueError("Input vector should not be a zero vector")

    # Create a random vector
    random_vector = np.random.randn(len(center_of_cone))

    # Use the Gram-Schmidt process to get an orthogonal vector
    orthogonal_vector = random_vector - (np.dot(random_vector, center_of_cone) / np.dot(center_of_cone, center_of_cone)) * center_of_cone

    # Normalize the orthogonal vector
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)
    if np.dot(orthogonal_vector, offering_grad) < 0:
        orthogonal_vector *= -1
    return orthogonal_vector

def find_orth_vector(vectors, offering_gradient):
    """
        Given a set of n-dimensional vectors, find a set of vectors that are orthogonal to the given set
        Args:
            vectors (list of np.array): set of vectors
            offering_gradient (np.array): n-dimensional vector corresponding to the offering agent's gradient.

        Returns:
            orthogonal_vectors (list of np.array): Set of vectors orthogonal to all the vectors in the input vector set and the center of the cone.
    """

    # Calculate the null space of the given set of vectors
    null_space = scipy.linalg.null_space(vectors)
    orthogonal_vectors = []

    # Iterate over the basis of the null space to find orthogonal vectors
    for i in range(null_space.shape[1]):
        potential_vector = null_space[:, i]
        is_orthogonal = True

        # Check that the null space vectors are orthogonal to all the vectors in the given set.
        for vector in orthogonal_vectors:
            if abs(np.dot(potential_vector, vector)) > 1e-10:  # Checking if dot product is close to zero
                is_orthogonal = False
                break
        # If the offer is orthogonal, then add it to the set of orthogonal vectors
        if is_orthogonal:
            # Ensure the direction is beneficial for the offering
            if np.dot(potential_vector, offering_gradient) < 0:
                potential_vector = -1 * potential_vector
            orthogonal_vectors.append(potential_vector)
    
    return orthogonal_vectors

def sort_orth_vectors(A, b, items, vectors, reduction_idx = []):
    """
        Sort the set of orthogonal vectors in terms of utility for a function x^TAx + b^Tx.
        Args:
            A (np.array): nxn matrix.
            b (np.array): n-dimensional vector.
            items (list): Current State of agent.
            vectors (list of np.array): vectors to be sorted
            reduction_idx (list, optional): set of item indices that need to be removed from consideration.

        Returns:
            sorted_offers (list of np.array): Set vectors sorted in decreasing order of utility value.
    """
    # This ensures that the most beneifical trades for the offering will be offered first.
    sorted_offers = sorted(vectors, key=lambda vector: utility_value(vector, A, b, items, reduction_idx=reduction_idx))
    return sorted_offers

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

def find_scaling_offering(vector, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, need_improvement = True, reduction_idx = [], int_constrained=True):
    """
        Find a scaled vector for the offering agent that is feasible

        Args:
            vector (np.array): n-dimensional vector representing the current offer.
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            offering_items (np.array): Number of items that the offering agent has in its possession from categories that are being comsidered currently.
            offering_items_original (np.array): Number of items that the offering agent has in its possession from all categories.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            need_improvement (bool, optional): Whether the offering agent needs to improve its utility with this offer Defaults to True.
            int_constrained (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            reduction_idx (list, optional): set of item indices that need to be removed from consideration.

        Returns:
            tuple:
                - scaled_vector (np.array): Scaled version of the original vector
                - scaling_factor (float): Scaling factor used to increase the magnitude of the unit vector
                - improvement (bool): Whether the offering agent is improving its utility with this offer
    """
    # Find a scaling factor that scales a given vector to a given number of items
    
    # If the offer is not aligned with the offering's gradient, reverse its direction
    offering_gradient = n_dim_quad_grad(offering_A, offering_b, offering_items_original)
    full_vector = np.zeros(len(offering_b))
    idx_counter = 0
    for element in range(0, len(offering_b)):
        if element not in reduction_idx:
            full_vector[element] = vector[idx_counter]
            idx_counter += 1
    if np.dot(full_vector, offering_gradient) < 0:
        vector = -1 * np.array(vector)
    
    # If we are looking for a trade that benefits the offering, we may need to scale down the offer to ensure we don't overshoot any optimal points.
    if need_improvement:
        improvement = False
        while not improvement:

            # Scale the offer based on the maximum amount of a given item the offering can trade (max_trade_value)
            scaled_vector = vector.copy()
            max_scaling_factor = find_scaling_factor(vector, max_trade_value, offering_items)       
            scaled_vector = [scaled_vector[i] * max_scaling_factor for i in range(len(scaled_vector))]
            max_index, max_value = max(enumerate(scaled_vector), key=lambda x: abs(x[1]))
            if int_constrained:
                scaled_vector[max_index] = round(scaled_vector[max_index])

            # Check if the offer improves the offering's utility
            improvement = utility_improvement(scaled_vector, offering_A, offering_b, offering_items_original, reduction_idx=reduction_idx)
            # If the maximum trade value is one, we cannot scale the offer down an further
            if max_trade_value == 1:
                break
            # If the offering does not benefit from the offer, but it is aligned with its gradent, then the trade mangiute is too large
            max_trade_value  = math.ceil(max_trade_value/2)
        
        return scaled_vector, max_scaling_factor, improvement
    else:
        # If improvement is not required, then scale the offer to the max_trade_value
        scaled_vector = vector.copy()
        max_scaling_factor = find_scaling_factor(vector, max_trade_value, offering_items)
        scaled_vector = [scaled_vector[i] * max_scaling_factor for i in range(len(scaled_vector))]
        improvement = utility_improvement(scaled_vector, offering_A, offering_b, offering_items_original, reduction_idx=reduction_idx)
        return scaled_vector, max_scaling_factor, improvement

def find_scaling_factor(vector, max_trade_value, offering_items):
    """
        Given a trade vector and a maximum amount of a given item that can be traded, find a scaling factor for the trade

        Args:
            vector (np.array): n-dimensional vector representing the current offer.
            offering_items (np.array): Number of items that the offering agent has in its possession from categories that are being comsidered currently.
            max_trade_value (int): Maximum number of items that can be traded from any item category.
            
        Returns:
            max_scaling_factor (float): Scaling factor for the offer to trade the maximum item amount.
    """
    abs_vector = np.abs(vector)
    # Determine the maximum scaling factor given the maximum trade value
    max_scaling_factor = max_trade_value / max(abs_vector)
    for i in range(len(vector)):
        # Account for cases that lead to negative item values
        if offering_items[i] > 0 and vector[i] != 0:
            scaling_factor = max(0, -1 * offering_items[i] / vector[i])
            if scaling_factor != 0:
                max_scaling_factor = min(max_scaling_factor, scaling_factor)
    return max_scaling_factor

def find_scaling_responding(vector, item_val, item_index):
    """
        Find a scaling factor that scales a given vector to a given number of items
        Args:
            vector (np.array): n-dimensional vector representing the current offer.
            item_val (float): Target number of items to trade
            item_index (int): Item category we want to scale up to item_val.

        Returns:
            vector_mod (np.array): Vector scaled such that it is trading item_val from item_index
    """
    vector_mod = vector.copy()
    scaling_factor = item_val / vector[item_index]
    vector_mod = [num*scaling_factor for num in vector_mod]
    return vector_mod

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

def intersection_between_hyperplanes(hyperplanes):
    """
        Given a set of hyperplanes, find the intersection point beteween the hyperplanes.
        Args:
            hyperplanes (list of tuples): Set of hyperplanes of the form ax = b

        Returns:
            intersection (np.array): point representing the intersection of the hyperplanes
    """
    num_hyperplanes = len(hyperplanes)
    dim = len(hyperplanes[0][0])

    # Initialize coefficient matrix and constant vector
    A = np.zeros((num_hyperplanes, dim))
    B = np.zeros(num_hyperplanes)

    # Populate coefficient matrix and constant vector ax = b
    for i, (normal, constant) in enumerate(hyperplanes):
        A[i] = normal
        B[i] = constant
    intersection = scipy.linalg.solve(A, B)

    return intersection

def is_in_intersection(point, halfspaces):
    """
        Check if a point is in the intersection of the given set of halfspaces
        Args:
            point (np.array): point
            halfspaces (list of tuples): Set of halfspaces of the form ax >= b

        Returns:
            (bool): Whether the point is in the intersection of the halfspaces
    """
    tolerance = 1e-10
    for a, b in halfspaces:
        if not np.dot(a, point) - b > -1 * tolerance:
            return False
    return True

def cross_prod_check(vector1, vector2):
    """
        Use the cross product to check if two vectors are parallel
        Args:
            vector1 (np.array): vector
            vector2 (np.array): vector

        Returns:
            (bool): Whether the vectors are parallel
    """

    if len(vector1) != len(vector2):
        return False 
    
    if all(x == 0 for x in vector1) or all(x == 0 for x in vector2):
        return True

    ratio = None
    for i in range(len(vector1)):
        if vector1[i] != 0:
            if ratio is None:
                ratio = vector2[i] / vector1[i]
            elif vector2[i] / vector1[i] != ratio:
                return False
        elif vector2[i] != 0:
            return False

    return True

def parallel_check(offer_set):
    """
        Check if any two vectors in the given set are parallel
        Args:
            offer_set (list of np.array): set of vectors

        Returns:
            (bool): Any two of the vectors are parallel
    """
    if not all(len(vector) == len(offer_set[0]) for vector in offer_set):
        return False
    for i, vector1 in enumerate(offer_set):
        for vector2 in offer_set[i + 1:]:
            if cross_prod_check(vector1, vector2):
                return True
    return False

def generate_corner_points(halfspaces, num_dimensions):
    """
        Given a set of halfspaces, generate corner points of the polytope defined by the halfspaces
        Args:
            halfspaces (list of tuples): set of halfspaces of the form ax >= b
            num_dimensions (int): dimensionality of the halfspaces

        Returns:
            point_set (list of np.array): Set of corner points for the polytope
    """
    # Generate Corner Points for a set of halfspaces
    point_set = []
    # Iterate over combinations of n-1 halfspaces
    for halfspace_combination in list(combinations(range(len(halfspaces)), num_dimensions)):
        halfspace_set = [halfspaces[h] for h in halfspace_combination]
        a_set = [h[0] for h in halfspace_set]
        A = np.zeros((len(halfspace_set), len(halfspace_set[0][0])))
        for i, (normal, constant) in enumerate(halfspace_set):
            A[i] = normal

        # Check if any two halfspaces in the set are parallel (In such cases, corner points do not exist)
        if not parallel_check(a_set) and not np.isclose(np.linalg.det(A), 0):

            # Calculate the intersection point
            intersection_point = intersection_between_hyperplanes(halfspace_set)

            # Check that the point is within the given set of halfspaces
            if is_in_intersection(intersection_point, halfspaces):
                intersection_tuple = tuple(intersection_point)
                if intersection_tuple not in map(tuple, point_set):
                    point_set.append(intersection_point)
    return point_set

def generate_hypercube(radius_val, basis_set):
    """
        Given a radius of a circle centered at the origin and a basis set, create a hypercube that encloses the circle
        Args:
            radius_val (float): radius of the circle
            basis_set (list of np.array): Set of basis vectors for the space
        Returns:
            hypercube_halfspace_set (list of tuples): Set of halfspaces of the form ax >= b that define the hypercube
    """
    hypercube_halfspace_set = []
    len_val = len(basis_set)

    # Iterate over dimensions
    for i in range(0, len_val):
        output_vals = np.zeros(len_val)
        output_vals[i] = 1

        # Format of halfspace contrants is ax >= b
        for sign in [-1, 1]:
            a = sign * output_vals
            b = -1 * radius_val
            hypercube_halfspace_set.append((a, b))
    return hypercube_halfspace_set

def qr_decomposition(normal_vector):
    """
        Use QR decomposition to obtain a basis set of the orthogonal space of a given vector
        Args:
            normal_vector (np.array): normal vector to the basis set.
        Returns:
            basis_vectors (np.array): Set of basis vectors that map from n-dimensional space to the n-1 dimensional orthogonal space of the normal_vector.
    """
    Q, _ = np.linalg.qr(np.column_stack([normal_vector] + [np.random.randn(len(normal_vector)) for _ in range(len(normal_vector) - 1)]))
    basis_vectors = np.array(Q[:, 1:])

    return basis_vectors.T

def vector_projection(v, basis_vectors):
    """
        Project a vector into the space defined by a set of basis vectors
        Args:
            v (np.array): n-dimensional vector to be projected onto the n-1 dimensional space
            basis_vectors (np.array): Set of basis vectors that map from n-dimensional space to the n-1 dimensional orthogonal space of the normal_vector.
        Returns:
            projection (np.array): n-1 dimensional projection of v onto the space defined by the basis vectors.
    """
    projection = []
    for basis in basis_vectors:
        norm_b_squared = np.linalg.norm(basis) ** 2
        proj_component = np.dot(v, basis) / norm_b_squared
        projection.append(proj_component)
    return projection

def generate_halfspaces(offer_set, basis, center_of_cone):
    """
        Generate halfspaces given a set of offers
        Args:
            offer_set (list of np.array): set of n-dimensional offers
            basis (np.array): Set of basis vectors that map from n-dimensional space to the n-1 dimensional orthogonal space of the normal_vector.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
        Returns:
            halfspace_set (list of tuples): Set of halfspaces corresponding to the set of offers.
    """
    halfspace_set = []
    for offer in offer_set:
        halfspace_set.append(calc_projected_halfspace(offer, basis, center_of_cone))        
    return halfspace_set

def calc_projected_halfspace(offer, basis, center_of_cone):
    """
        Given an n-dimensional offer, generate a corresponding halfspace constraint in the n-1 dimensional null space of the cone center defined by a set of basis vectors.
        Args:
            offer (np.array): n-dimensional offers
            basis (np.array): Set of basis vectors that map from n-dimensional space to the n-1 dimensional orthogonal space of the normal_vector.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
        Returns:
            (projected_vector, b_val) (list of tuples): Halfspace corresponding to the offer in the form projected_vector x >= b_val.
    """
    # Transform an offer into a halfspace contraint. ax >= b

    # The offer's direction is the normal vector the halfspace
    offer_norm = offer / np.linalg.norm(offer)
    projected_vector = np.array(vector_projection(offer_norm, basis))

    # Halfspace offset is determined by the angle offset of the offer and the orthogonal plane
    angle_offset = angle_between(offer_norm, center_of_cone)
    angle_minus_90 = angle_offset - (np.pi / 2)
    b_val = np.tan(angle_minus_90) * np.linalg.norm(projected_vector)

    return (projected_vector, b_val)

def rotate_vector(r_vector, d_vector, theta):
    """
        Rotate a vector in a given direction by an angle theta
        Args:
            r_vector (np.array): Vector to be rotated.
            d_vector (np.array): Rotation direction.
            theta (float): Rotation angle in radians.
        Returns:
            (np.array): r_vector rotated in the direction of d_vector by an angle theta
    """
    # Gram-Schmidt orthogonalization
    n1 = r_vector / np.linalg.norm(r_vector)
    v2 = d_vector - np.dot(n1,d_vector) * n1
    n2 = v2 / np.linalg.norm(v2)
        
    # rotation by pi/2
    a = theta
        
    I = np.identity(len(n2))
        
    R = I + (np.outer(n2,n1) - np.outer(n1,n2)) * np.sin(a) + ( np.outer(n1,n1) + np.outer(n2,n2)) * (np.cos(a)-1)

    # check result
    return np.matmul(R,n1)

def is_separating(hyperplane, points):
    """
        Determine if a hyperplane is separating two points
        Args:
            hyperplane (tuple): Hyperplane of the form ax = b
            points (np.array, np.array): The two points to be separated.
        Returns:
            (bool): Whether the hyperplane is separating the two points
    """
    tolerance = 1e-10

    # Check if the two points are on different sides of the hyperplane
    val_1 = np.dot(points[0], hyperplane[0]) - hyperplane[1]
    val_2 = np.dot(points[1], hyperplane[0]) - hyperplane[1]
    if np.abs(val_1) <= tolerance or np.abs(val_2) <= tolerance:
        return False
    if np.sign(val_1) != np.sign(val_2):
        return True
    else:
        return False


def farthest_points(points):
    """
        Given a set of points, return the two points that are farthest apart
        Args:
            points (list of np.array): Set of two points
        Returns:
            tuple:
                - farthest_pair (tuple): The two farthest points
                - max_distance (float): Distance between the two points
    """
    # Given a set of points, return the two points that are farthest apart
    max_distance = 0
    farthest_pair = []

    # Iterate through all pairs of points
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i != j:  # Exclude comparing the same point
                distance = np.linalg.norm(point1 - point2)
                if distance > max_distance:
                    max_distance = distance
                    farthest_pair = (point1, point2)

    return farthest_pair, max_distance

def calculate_new_cone_integer_contstrained(offer_set, hypercube, basis_set, center_of_cone):
    """
        Given a set of integer offers and the hypercube enclosing the current cone, determine a new cone of potential gradients.
        Args:
            offer_set (list of np.array): Set of past integer offers
            hypercube (list of tuples): Halfspace constraints corresponding to the hypercube that encloses the current cone
            basis_set (np.array): Set of basis vectors that map from n-dimensional space to the n-1 dimensional orthogonal space of the normal_vector.
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
        Returns:
            tuple:
                - center_of_circle (np.array): n-1 dimensional center of the hypersphere that encloses the space of potential gradients
                - potential_new_theta (float): Angle of the cone that encloses the hypersphere
                - corner_points (list of np.array): Farthest corner points of the polytope of potential gradients
                - (bool): Boolean that signals if the polytope of potential gradients is empty.
    """
    num_dim = len(basis_set)

    # Generate Corner Points of the Polyhedron
    halfspace = generate_halfspaces(offer_set, basis_set, center_of_cone)
    full_halfspace_set = halfspace + hypercube
    point_set = generate_corner_points(full_halfspace_set, num_dim)
    corner_points, point_dist = farthest_points(point_set)

    # If the halfspace contraints do not allow for corner points, return an error case
    if len(corner_points) == 0:
        return None, None, None, True
    
    # Calculate new circle parameters
    center_of_circle = np.mean(corner_points, axis=0)
    radius_of_circle = (point_dist)/2 
    radius_of_circle = np.sqrt(3) * radius_of_circle
    center_norm = center_of_circle / np.linalg.norm(center_of_circle)

    # Calculate new angle of opening (Theta)
    point_a = center_of_circle - (radius_of_circle * center_norm)
    point_b = center_of_circle + (radius_of_circle * center_norm)
    d_a = np.sqrt(1 + (np.linalg.norm(point_a)**2))
    d_b = np.sqrt(1 + (np.linalg.norm(point_b)**2))
    diameter = point_dist
    ratio = ((diameter**2) - (d_a**2) - (d_b**2))/(-2*d_a*d_b)
    potential_new_theta = np.arccos(ratio) / 2

    return center_of_circle, potential_new_theta, corner_points, False


def calculate_new_cone(offer_set, theta, center_of_cone, num_items):
    """
       Given a set of fractional offers and the current cone, determine a new cone of potential gradients.
       Args:
           offer_set (list of np.array): Set of past fractional offers
           theta (float): semi-vertical angle of the current cone
           center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
           num_items (int): Total number of item categories
        Returns:
           tuple:
               - new_center_of_cone (np.array): Direction of the new cone of potential gradients
               - potential_new_theta (float): Angle of the new cone of potential gradients
   """
    # Return New Cone Given the current cuts (vector set)
    w_list = [center_of_cone]
    sum_value = np.zeros(num_items)

    # Use cone update rule to determine the new cone of potential gradients
    for i in range(0, num_items-1):
        w_i = center_of_cone * np.cos(theta) + (offer_set[i]/np.linalg.norm(offer_set[i])) * np.sin(theta)
        w_list.append(np.array(w_i))
        sum_value += (w_i / num_items)
    new_center_of_cone = (sum_value / np.linalg.norm(sum_value))
    scaling_factor_theta = np.sqrt((2 * num_items - 1) / (2 * num_items))
    potential_new_theta = np.arcsin(scaling_factor_theta * np.sin(theta))
    return new_center_of_cone, potential_new_theta

def make_heuristic_offer(heuristic_offer, center_of_cone, offering_gradient, responding_items_original, offering_A, offering_b, responding_A, responding_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value):
    """
       Make a heuristic offer to the responding agent
       Args:
            heuristic_offer (np.array): Heuristic offer
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
            offering_gradient (np.array): n-dimensional vector corresponding to the offering agent's gradient.
            responding_items_original (np.array): List of current items from all categories for the responding agent
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items (np.array):List of current items with excluded categories for the offering agent
            offering_items_original (np.array):List of current items from all categories for the offering agent
            reduction_idx (np.array): List of item categories to be excluded from trading
            int_constrained (bool): Whether the trade should be restricted to integer values. Defaults to True.
            max_trade_value (int): Maximum number of items that can be traded from any item category

        Returns:
           tuple:
               - bool: Whether the offer was accepted
               - offer (np.array): The accepted offer
               - num_queries (int): Total number of offers made to the receiving agent at this stage.
   """
    num_queries = 0
    # Scale and round the offer
    offer, scaling_factor, improvement = find_scaling_offering(heuristic_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
    if int_constrained:
        offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
    
    # If the iterate until responding agent rejects the offer
    rejected_dot_product = False
    while not rejected_dot_product and improvement == True:

        # Make the offer to the responding agent
        neg_offer = -1 * np.array(offer)
        num_queries += 1
        neg_offer_q = np.zeros(len(responding_items_original))
        neg_offer_c = 0
        for index_val in range(0, len(neg_offer_q)):
            if index_val not in reduction_idx:
                neg_offer_q[index_val] = neg_offer[neg_offer_c]
                neg_offer_c += 1
        response_n, response_product, min_index, min_items = query(neg_offer_q, responding_A, responding_b, responding_items_original)

        # If the responding agent accepts the offer, return success
        if response_product:
            if response_n:
                return True, offer, num_queries
            else:
                # If the responding agent does not have enough items to complete the trade, scale to meet responding agent's items
                # We note that, since the receiving agent has access to the responding agent's state, this should not count toward the total offers
                num_queries -= 1
                for idx in range(0, min_index):
                    if idx in reduction_idx:
                        min_index -= 1
                offer = find_scaling_responding(offer, min_items, min_index)
                if int_constrained:
                    offer = branch_and_bound(offer, center_of_cone, offering_gradient)
                    improvement = utility_improvement(offer, offering_A, offering_b, offering_items_original, reduction_idx=reduction_idx)
                continue
        else:
            # If the responding agent rejects the offer, return failure
            rejected_dot_product = True
    return False, offer, num_queries

def obain_full_offer(offer, reduction_idx, full_size):
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

def round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value):
    """
        Given a fractional offer, return an integer offer that benefits the offering agent

        Args:
            offer (np.array): Fractional offer
            center_of_cone (np.array): n-dimensional vector corresponding to the center of the gradient cone.
            offering_gradient (np.array): n-dimensional vector corresponding to the offering agent's gradient.
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            offering_items (np.array):List of current items with excluded categories for the offering agent
            offering_items_original (np.array):List of current items from all categories for the offering agent
            reduction_idx (np.array): List of item categories to be excluded from trading
            int_constrained (bool): Whether the trade should be restricted to integer values. Defaults to True.
            max_trade_value (int): Maximum number of items that can be traded from any item category

        Returns:
            tuple:
                - offer (np.array): Rounded Offer
                - improvement (boolean): Whether the offering agent benefits fromt he offer
    """
    # Round the offer to contain only integer values
    offer = branch_and_bound(offer, center_of_cone, offering_gradient)
    full_offer = obain_full_offer(offer, reduction_idx, len(offering_items_original))

    # Scale the offer in accordance with the offering's items
    while any(element < 0 for element in offering_items_original + full_offer):
        offer, scaling_factor, improvement = find_scaling_offering(offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
        offer = branch_and_bound(offer, center_of_cone, offering_gradient)
        full_offer = obain_full_offer(offer, reduction_idx, len(offering_items_original))
    improvement = utility_improvement(offer, offering_A, offering_b, offering_items_original, reduction_idx=reduction_idx)

    return offer, improvement


def obtain_responding_offer(offer, responding_items_original, reduction_idx):
    """
        Given an offer that may not be for the responding agent, return a scaled down offer that is feasible

        Args:
            offer (np.array): Offer from the
            responding_items_original (np.array):List of current items from all categories for the responding agent
            reduction_idx (np.array): List of item categories to be excluded from trading
        Returns:
            tuple:
                - neg_offer_q (np.array): Scaled down offer for the responding agent
    """
    neg_offer = [-1 * x for x in offer]
    neg_offer_q = np.zeros(len(responding_items_original))
    neg_offer_c = 0
    for index_val in range(0, len(neg_offer_q)):
        if index_val not in reduction_idx:
            neg_offer_q[index_val] = neg_offer[neg_offer_c]
            neg_offer_c += 1
    next_step = neg_offer_q + responding_items_original

    # Scale down the offer to match the responding agent's items
    if not all(i >= 0 for i in next_step):
        flag_n_items = False
        min_index = np.argmin(next_step)
        min_items = responding_items_original[min_index]
        neg_offer_q[min_index] = -1 * min_items
    return neg_offer_q

def offer_search(offering_A, offering_b, responding_A, responding_b, offering_items_original, responding_items_original, num_items, center_of_cone, theta, max_trade_value, theta_closeness, int_constrained = True, prev_offer = [], prev_offer_flag=False, center_of_cone_flag=False, average_flag=False, offering_grad_flag = False, offer_budget = 1000):
    """
        Use ST-CR to find a mutually beneficial offer
        Args:
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items_original (np.array): List of current items for the offering agent across all item categories.
            responding_items_original (np.array): List of current items for the responding agent across all item categories.
            center_of_cone (np.array): n-dimensional vector corresponding to the current center of the cone of potential gradients.
            theta (float): Angle of the cone of potential gradients in radians.
            num_items (int): Total number of item categories.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            theta_closeness (float): Hyperparamter used by ST-CR to stop trading.
            prev_offer (np.array): Previously accepted offer used for heuristic trading. Default: Empty
            prev_offer_flag (bool, optional): Whether ST-CR will use the previously accepted trade heuristic
            center_of_cone_flag (bool, optional): Whether ST-CR will use the center of the cone as heuristic offer
            average_flag (bool, optional): Whether ST-CR will use the average between the offering agent's gradient and the center of the cone as a heuristic offer
            offering_grad_flag (bool, optional): Whether ST-CR will use the offering agent's gradient as a heuristic offer.
            int_constrained (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            offer_budget (int, optional): Maximum number of offers allowed to the receiving agent. Defaults to 1000.

        Returns:
            tuple:
                - found_trade (bool): Whether a mutually beneficial offer was found
                - offer (np.array): The mutually beneficial offer (if found)
                - offer_count (int): Number of offers made to the responding agent,
                - iterations (int): Number of cone refinements
                - center_of_cone (np.array): n-dimensional vector corresponding to the center of the cone of potential gradients after searching for a mutually beneficial offer.
                - theta (float): Angle (in radians) of the cone of potential gradients after trading.
                - edge_case_break (bool): Whether ST-CR stopped trading due to an edge case.
    """
    # Initialize Gradient Values (The responding agent's gradient is only used with respect to the query function)
    original_responding_gradient = list(n_dim_quad_grad(responding_A, responding_b, responding_items_original))
    original_offering_gradient = list(n_dim_quad_grad(offering_A, offering_b, offering_items_original))

    # Account for cases where the number of items for a given category are zero
    new_grad_h = list(original_responding_gradient.copy())
    new_grad_a = list(original_offering_gradient.copy())
    responding_items_mod = list(responding_items_original)
    offering_items_mod = list(offering_items_original)
    reduction_idx = []
    center_list = list(center_of_cone)


    # Remove Item Categories that have zero items
    for i in range(len(offering_items_original)):
        if offering_items_original[i] == 0 or responding_items_original[i] == 0:
            reduction_idx.append(i)
    reduction_num = len(reduction_idx)
    prev_offer = list(prev_offer)
    for i in sorted(reduction_idx, reverse=True):
        a = new_grad_a.pop(i)
        h = new_grad_h.pop(i)
        prev_trade = prev_offer.pop(i)
        item_a = offering_items_mod.pop(i)
        item_h = responding_items_mod.pop(i)
        c = center_list.pop(i)
    center_of_cone = np.array(center_list)
    offering_items = np.array(offering_items_mod)
    offering_gradient = new_grad_a / np.linalg.norm(new_grad_a)

    # Initialize iteration variables
    offer_count = 0
    iterations = 0
    edge_case_break = False
    # If there is only one item left to trade, end the cone refinement.
    if reduction_num >= num_items-1:
        return False, [], offer_count, iterations, center_of_cone, theta, edge_case_break
    # If the offering's gradient is zero, end the cone refinement
    if all(grad_entry == 0 for grad_entry in new_grad_a):
        return False, [], offer_count, iterations, center_of_cone, theta, edge_case_break
    remaining_items = num_items - reduction_num

    # Make Heuristic Offers Based on Previous Information
    heuristic_offers = []

    # Offer the Previously Accepted Offer
    if prev_offer_flag:
        if len(prev_offer) != 0:
            if not all(item == 0 for item in prev_offer):
                prev_info_offer = prev_offer / np.linalg.norm(prev_offer)
                response, offer, num_queries = make_heuristic_offer(prev_info_offer, center_of_cone, offering_gradient, responding_items_original, offering_A, offering_b, responding_A, responding_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                offer_count += num_queries
                heuristic_offers.append(offer)
                if response:
                    return True, offer, offer_count, iterations, center_of_cone, theta, edge_case_break
    
    # Offer the Center of the Cone
    if center_of_cone_flag and theta < np.pi:
        center_of_cone_norm = -1 * (center_of_cone / np.linalg.norm(center_of_cone))
        prev_info_offer = center_of_cone_norm
        if not all(item == 0 for item in prev_info_offer):
            if np.dot(-1 * center_of_cone_norm, offering_gradient) >= 0:
                response, offer, num_queries = make_heuristic_offer(prev_info_offer, center_of_cone, offering_gradient, responding_items_original, offering_A, offering_b, responding_A, responding_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                offer_count += num_queries
                heuristic_offers.append(offer)
                if response:
                    return True, offer, offer_count, iterations, center_of_cone, theta, edge_case_break

    # Offer the Average of the Cone Center and the offering Gradient
    if average_flag and theta < np.pi:
        center_of_cone_norm = -1 * (center_of_cone / np.linalg.norm(center_of_cone))
        prev_info_offer = (center_of_cone_norm + offering_gradient) / 2
        if not all(item == 0 for item in prev_info_offer):
            response, offer, num_queries = make_heuristic_offer(prev_info_offer, center_of_cone, offering_gradient, responding_items_original, offering_A, offering_b, responding_A, responding_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
            offer_count += num_queries
            heuristic_offers.append(offer)
            if response:
                return True, offer, offer_count, iterations, center_of_cone, theta, edge_case_break

    # Offer the offering Gradient Heuristic
    if offering_grad_flag:
        prev_info_offer = offering_gradient
        if not all(item == 0 for item in prev_info_offer):
            response, offer, num_queries = make_heuristic_offer(prev_info_offer, center_of_cone, offering_gradient, responding_items_original, offering_A, offering_b, responding_A, responding_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
            offer_count += num_queries
            heuristic_offers.append(offer)
            if response:
                return True, offer, offer_count, iterations, center_of_cone, theta, edge_case_break



    # Determine Initial Quadrant of the responding agent's gradient
    if theta >= np.pi:
        quadrant = np.zeros(num_items - len(reduction_idx))
        for i in range(0, num_items - len(reduction_idx)):
            og_offer = np.zeros(num_items - len(reduction_idx))
            og_offer[i] = 1
            offer, scaling_factor, improvement = find_scaling_offering(og_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
            # Round the offer to integer values
            if int_constrained:
                offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)

            # Make the offer to the responding
            neg_offer_q = obtain_responding_offer(offer, responding_items_original, reduction_idx)
            offer_count += 1
            response_n, response_product, min_index, min_items = query(neg_offer_q, responding_A, responding_b, responding_items_original) # query_gpt(neg_offer, item_list)
            
            # Return the accepted offer
            if response_product:
                offer = -1 * neg_offer_q
                offer = list(offer)
                for i in sorted(reduction_idx, reverse=True):
                    offer_out = offer.pop(i)
                return True, np.array(offer), offer_count, 0, center_of_cone, theta, False 
            
            # Otherwise, use the rejected offer to inform the quadrant
            else:
                if offer[i] < 0:
                    quadrant[i] = -1
                else:
                    quadrant[i] = 1

        # If all offers were rejected, initialize the center of cone and angle
        center_of_cone = (1/np.sqrt(num_items - len(reduction_idx))) * quadrant
        theta =  np.arccos(1/np.sqrt(num_items - len(reduction_idx)))

    # Generate basis set and hypercube for Calculating Next Cone Update
    basis_set = qr_decomposition(center_of_cone)
    hypercube = generate_hypercube(np.abs(np.tan(theta)) , basis_set)
    found_trade = False

    # Begin Cone Refinement Loop
    while found_trade == False and edge_case_break == False and theta >= theta_closeness and offer_count < offer_budget:
        # Generate an initial offer
        offering_gradient_norm = offering_gradient/np.linalg.norm(offering_gradient)
        center_of_cone_norm = center_of_cone/np.linalg.norm(center_of_cone)

        # If the offering gradient is exactly aligned with the center of the cone, then we cannot generate beneficial offers for the offering
        # Return as an edge case
        diff_vector = offering_gradient_norm - center_of_cone_norm
        if all(entry == 0 for entry in diff_vector):
            edge_case_break = True
            refined_cone = True
            break

        # Find an initial offer by determining the orthogonal offer (to the cone center) that is closest to the offering's gradient
        original_offer = find_init_offer_random(offering_gradient, center_of_cone)

        # Determine the other orthogonal vectors that will be used for cone refinement
        orthogonal_vectors = find_orth_vector(np.array([center_of_cone, original_offer]), offering_gradient)
        orthogonal_vectors = sort_orth_vectors(offering_A, offering_b, offering_items_original, orthogonal_vectors, reduction_idx=reduction_idx)

        # Scale up the offer while accounting for the items the offering possesses
        offer, scaling_factor, improvement = find_scaling_offering(original_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
        
        # Round the offer to integer values
        if int_constrained:
            offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)

        # Initialize set of queries used to determine the updated center of the cone
        i = 0
        offer_set = []
        offer_set_og = []
        offer_set_og.append(center_of_cone)
        offer_set.append(center_of_cone)

        # Refine the cone using n-1 requests
        refined_cone = False
        while not refined_cone:
            # Present the trade offer to the responding
            neg_offer = [-1 * x for x in offer]
            offer_count += 1
            neg_offer_q = np.zeros(len(responding_items_original))
            neg_offer_c = 0
            for index_val in range(0, len(neg_offer_q)):
                if index_val not in reduction_idx:
                    neg_offer_q[index_val] = neg_offer[neg_offer_c]
                    neg_offer_c += 1
            response_n, response_product, min_index, min_items = query(neg_offer_q, responding_A, responding_b, responding_items_original)
            
            # Check if the responding agent likes the direction of the trade (Fix This after remaining comments/changes have been implemented)
            if response_n == False:
                # Check if the responding agent has enough of each item to complete the trade
                # If not use information from the responding agent to scale the offer again and resubmit the offer
                # In our testing, we assume that the offering agent has access to the responding agent's state, so this should not contribute to total offers
                offer_count -= 1
                for idx in range(0, min_index):
                    if idx in reduction_idx:
                        min_index -= 1
                offer = find_scaling_responding(offer, min_items, min_index)
                if int_constrained:
                    offer = branch_and_bound(offer, center_of_cone, offering_gradient)
                    improvement = utility_improvement(offer, offering_A, offering_b, offering_items_original, reduction_idx=reduction_idx)
                continue

            else:
                # If the responding agent accepted the offer, negate the offer to ensure that the appropriate direction of refinement is maintained.
                if response_product == True:
                    found_trade = True
                    refined_cone = True
                    out_offer = offer.copy()
                    offer = -1 * np.array(offer)
                
                # Check if any offers have been repeated. Since all offers are orthogonal, this should only occur if the previous offer failed to refine the search space.
                # In such cases, we break on an edge case
                if any(np.array_equal(offer, v) for v in offer_set):
                    edge_case_break = True
                    refined_cone = True
                    break
                # Add the offer to the set of previous offers used to refine the cone
                else:
                    offer_set.append(offer)
                    offer_set_og.append(original_offer)
                potential_new_theta = np.inf
                i += 1
                
                # If we have made n-1 offer, we can begin refining the cone
                if i >= remaining_items - 1:
                    
                    # Calculate the new cone using the prior offers
                    if int_constrained:
                        center_of_circle, potential_new_theta, corner_points, error_case = calculate_new_cone_integer_contstrained(offer_set[1:], hypercube, basis_set, center_of_cone)
                        # If there are no corner points enclosed by the cone, we have made an incorrect cut in a prior trade and we must increase the angle to account for the lost trade directions.
                        while error_case == True:
                            theta += 0.01
                            hypercube = generate_hypercube(np.abs(np.tan(theta)), basis_set)
                            center_of_circle, potential_new_theta, corner_points, error_case = calculate_new_cone_integer_contstrained(
                                offer_set[1:], hypercube, basis_set, center_of_cone)
                    else:
                        # Without integer constraints
                        potential_center_of_cone, potential_new_theta = calculate_new_cone(offer_set[1:], theta, center_of_cone, num_items - len(reduction_idx))
                    # If the set of contraints does not contain any corner points, it is likely due to a sign error in a prior refinement.
                    # In such cases, we increase the cone's angle of opening and perform the calculation again.

                    # If the new cone is smaller than the current cone, we can make a cone refinement
                    if potential_new_theta < theta:
                        if int_constrained:
                            theta = potential_new_theta

                            # Obtain a new center of the cone by rotating the old cone center in the direction of the circle center
                            rotation_direction_n = center_of_circle @ basis_set
                            new_center_of_cone = rotate_vector(center_of_cone, rotation_direction_n, np.arctan(np.linalg.norm(center_of_circle)))
                            center_of_cone = new_center_of_cone

                            # Obtain the basis and hypercube for the new cone
                            basis_set = qr_decomposition(center_of_cone)
                            hypercube = generate_hypercube(np.abs(np.tan(theta)), basis_set)
                            refined_cone = True
                        else:
                            # Make an unconstrained cone refinement
                            theta = potential_new_theta
                            center_of_cone = potential_center_of_cone
                            basis_set = qr_decomposition(center_of_cone)
                            hypercube = generate_hypercube(np.abs(np.tan(theta)), basis_set)
                            refined_cone = True

                    
                    # If the new cone is not smaller than the prior cone, we must find a new offer to refine the cone.
                    else:
                        # The new offer is obtained by finding the hyperplane that separates the two corner points.
                        # Equations found here: https://math.stackexchange.com/questions/933266/compute-the-bisecting-normal-hyperplane-between-two-n-dimensional-points
                        
                        bisecting_hyperplane_a = (corner_points[0] - corner_points[1])/np.linalg.norm(corner_points[0] - corner_points[1]) 
                        bisecting_hyperplane_b = ((np.linalg.norm(corner_points[0])**2) - (np.linalg.norm(corner_points[1])**2))/(2 * (np.linalg.norm(corner_points[0] - corner_points[1])))
                        dist_from_origin = np.abs(bisecting_hyperplane_b) / np.linalg.norm(bisecting_hyperplane_a)
                        theta_offset = np.arctan(dist_from_origin)
                        if bisecting_hyperplane_b >= 0:
                            theta_offset = -1 * theta_offset

                        # Once the hyperplane that bisects the two farthest corner points is obtained, we turn it into an n-dimesional offer
                        offer_in_n_dimensions = bisecting_hyperplane_a @ basis_set
                        # Rotate the offer to account for offset from origin in n-1 dimensions
                        original_offer = rotate_vector(offer_in_n_dimensions, center_of_cone, theta_offset)

                        # Scale and round the offer
                        scaled_offer, scaling_factor, improvement = find_scaling_offering(original_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
                        if int_constrained:
                            offer, improvement = round_offer(scaled_offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                        
                        # Due to rounding, the bisecting offer may no longer separate the corner points.
                        # In this case, we must scale up the offer to reduce the impact of roudning on the trade direction.
                        mtv = 5
                        last_offer = offer.copy()
                        while not is_separating(calc_projected_halfspace(offer, basis_set, center_of_cone), corner_points):
                            
                            # Scale up and round the offer
                            mtv *= 2
                            offer, scaling_factor, improvement = find_scaling_offering(original_offer, offering_A, offering_b, offering_items, offering_items_original, mtv, reduction_idx=reduction_idx)
                            if int_constrained:
                                offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                            
                            # If the offer doesn't change after being scaled up, it means that there is insufficiant quantities of each item to be scaled furthur.
                            # In such scenarios, we can no longer refine the cone, and we exit the algorithm
                            if np.array_equal(offer, last_offer) or mtv > 200:
                                edge_case_break = True
                                refined_cone = True
                                break
                            else:
                                last_offer = offer.copy()
                            
                # If we haven't made n-1 offers yet, we use the orthogonal offers to inform the next trade
                else:
                    original_offer = orthogonal_vectors.pop()
                    offer, scaling_factor, improvement = find_scaling_offering(original_offer, offering_A, offering_b, offering_items, offering_items_original, max_trade_value, reduction_idx=reduction_idx, int_constrained=int_constrained)
                    if int_constrained:
                        offer, improvement = round_offer(offer, center_of_cone, offering_gradient, offering_A, offering_b, offering_items, offering_items_original, reduction_idx, int_constrained, max_trade_value)
                        
                        # Ensure that the orthogonal offers are properly scaled after rounding
                        unscaled_offer = offer.copy()
                        for mtv in range(2, 6):
                            max_scaling_factor = find_scaling_factor(unscaled_offer, mtv, offering_items)
                            temp_offer = list(max_scaling_factor * np.array(unscaled_offer))
                            temp_offer_rounded = branch_and_bound(temp_offer, center_of_cone, offering_gradient)
                            after_comp_state = np.array(offering_items) + np.array(temp_offer_rounded)
                            improvement = utility_improvement(temp_offer_rounded, offering_A, offering_b, offering_items_original, reduction_idx=reduction_idx)
                            if improvement and not any(entry < 0 for entry in after_comp_state):
                                offer = temp_offer_rounded.copy()
                            else:
                                break

                # Increment the number cone refinement iterations
                iterations += 1
        # If a mutually beneficial trade has been found, return the trade
        if found_trade == True:
            return True, out_offer, offer_count, iterations, center_of_cone, theta, edge_case_break
    # If the algorithm failed to find a trade, return false
    return False, [], offer_count, iterations, center_of_cone, theta, edge_case_break

def run_trading_scenario_stcr(num_items, offering_A, offering_b, responding_A, responding_b, offering_items, responding_items, offer_budget, max_trade_value, theta_closeness, integer_constraint=True, average_flag=False, center_of_cone_flag=False, prev_offer_flag=False, offering_grad_flag=False, debug=False):
    """
        Run a trading scenario using ST-CR
        Args:
            num_items (int): Total number of item categories
            offering_A (np.array): nxn matrix used for the offering agent's utility function.
            offering_b (np.array): n-dimensional vector constants for the offering agent's utility function.
            responding_A (np.array): nxn matrix used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            responding_b (np.array): n-dimensional vector used for the responding agent's utility function. This is only used for querying the simulated responding agent.
            offering_items (np.array): Initial number of items in each category for the offering agent.
            responding_items (np.array): Initial number of items in each category for the responding agent.
            offer_budget (int): Maximum number of offers allowed to the receiving agent. Defaults to 1000.
            max_trade_value (int): Maximum number of items that can be traded from any item category
            theta_closeness (float): Hyperparamter used by ST-CR to stop trading.
            integer_constraint (bool, optional): Whether the trade should be restricted to integer values. Defaults to True.
            prev_offer_flag (bool, optional): Whether ST-CR will use the previously accepted trade heuristic. Defaults to False.
            center_of_cone_flag (bool, optional): Whether ST-CR will use the center of the cone as heuristic offer. Defaults to False.
            average_flag (bool, optional): Whether ST-CR will use the average between the offering agent's gradient and the center of the cone as a heuristic offer. Defaults to False.
            offering_grad_flag (bool, optional): Whether ST-CR will use the offering agent's gradient as a heuristic offer. Defaults to False.
        Returns:
            log (list of dicts): Log of trading progression
        """

    # Initialize all algorithm objects
    center_of_cone = np.zeros(num_items)
    center_of_cone[0] = 1
    theta = np.pi
    trade_num = 0
    beta = 0.01
    prev_offer = []
    error_flag = False
    cumulative_offer_count = 0
    random_flag = False

    # Initialize the trading log
    log = []
    
    # Iterate for a set number of trades.
    # If the algorithm makes more than 1000 queries to the responding, end the trading scenario.
    while trade_num <= 100 and cumulative_offer_count < offer_budget:

        # Initialize the query count for the trading iteration
        offer_count = 0
        trade_num += 1
        original_responding_gradient = list(n_dim_quad_grad(responding_A, responding_b, responding_items))
        original_offering_gradient = list(n_dim_quad_grad(offering_A, offering_b, offering_items))

        # Account for cases where one party has all of a given item.
        # In such cases, remove the item categories from consideration when trading
        reduction_idx = []
        for i in range(len(original_offering_gradient)):
            if offering_items[i] == 0 or responding_items[i] == 0:
                reduction_idx.append(i)
        
        ## Search for a mutually beneficial trade offer
        found_trade, out_offer, offer_count, iterations, center_of_cone, theta, edge_case_break = offer_search(offering_A, offering_b, responding_A, responding_b, np.array(offering_items), np.array(responding_items), num_items, center_of_cone, theta, max_trade_value, theta_closeness, int_constrained=integer_constraint, prev_offer = prev_offer, average_flag=average_flag, center_of_cone_flag=center_of_cone_flag, prev_offer_flag=prev_offer_flag, offering_grad_flag=offering_grad_flag, offer_budget = offer_budget - cumulative_offer_count)
        
        # Obtain the full offer for the state transition
        true_out_offer = []
        if len(out_offer) != 0:
            true_out_offer = obain_full_offer(out_offer, reduction_idx, num_items)
        true_center_of_cone = obain_full_offer(center_of_cone, reduction_idx, num_items)
        center_of_cone = true_center_of_cone
        # Update the previously accepted offer
        if found_trade:
            prev_offer = true_out_offer

        # If a successful trade occured, update the state information
        information_dict = {}
        if found_trade == True:
            information_dict["found_trade"] = True

            # Increase the cone's angle of opening to account for changes in the responding agent's gradient
            theta += (beta * np.linalg.norm(true_out_offer))
            if theta > np.pi:
                theta = np.pi
            else:
                if theta > np.arccos(1/np.sqrt(num_items)):
                    theta = np.arccos(1/np.sqrt(num_items))
            
            # Use the accepted offer to transition to the next state
            prev_responding_items = responding_items.copy()
            prev_offering_items = offering_items.copy()
            responding_items = responding_items - true_out_offer
            offering_items += true_out_offer

            for i in range(0, len(responding_items)):
                if responding_items[i] < 0.00000001:
                    responding_items[i] = 0
                if offering_items[i] < 0.00000001:
                    offering_items[i] = 0

            cumulative_offer_count += offer_count

            # Record utility benefits
            neg_offer = [-1 * x for x in true_out_offer]
            prev_state_value_h = prev_responding_items.transpose() @ responding_A @ prev_responding_items + responding_b @ prev_responding_items
            prev_state_value_a = prev_offering_items.transpose() @ offering_A @ prev_offering_items + offering_b @ prev_offering_items

            next_state_value_h = responding_items.transpose() @ responding_A @ responding_items + responding_b @ responding_items
            next_state_value_a = offering_items.transpose() @ offering_A @ offering_items + offering_b @ offering_items
            
            # Print Trade Status to Terminal
            if debug:
                print("ST-CR Result: ", found_trade)
                if prev_offer_flag:
                    print("Previous Trade Heuristic Enabled")
                print("Offer: ", true_out_offer)
                print("Offering Agent Gradient: ", original_offering_gradient)
                print("Responding Agent Gradient: ", original_responding_gradient)
                print("New Responding Agent Item List: ", responding_items)
                print("New Offering Agent Item List: ", offering_items)
                print("Center of Cone: ", center_of_cone)
                print("Offering Agent Gradient Original: ", (original_offering_gradient / np.linalg.norm(original_offering_gradient)))
                print("Offering Agent predicted benefit ", np.dot(true_out_offer, (original_offering_gradient / np.linalg.norm(original_offering_gradient))))
                print("Responding Agent predicted benefit ", np.dot(neg_offer, (original_responding_gradient / np.linalg.norm(original_responding_gradient))))
                print("Theta: ", theta)
                print("Offering Agent benefit ", next_state_value_a - prev_state_value_a)
                print("Responding Agent benefit ", next_state_value_h, prev_state_value_h, next_state_value_h - prev_state_value_h)
                print("Offer Count: ", offer_count)
                print("\n")

            # Log information regarding the current trade
            information_dict["offer"] = true_out_offer.tolist()
            information_dict["responding_items"] = responding_items.tolist()
            information_dict["offering_items"] = offering_items.tolist()
            information_dict["responding_benefit"] = next_state_value_h - prev_state_value_h
            information_dict["offering_benefit"] = next_state_value_a - prev_state_value_a
            information_dict["edge_case"] = edge_case_break
            information_dict["query_count"] = offer_count
            information_dict["random_flag"] = random_flag
            log.append(information_dict)
            if next_state_value_h - prev_state_value_h < 0.000001 and next_state_value_a - prev_state_value_a < 0.000001:
                trade_num = 101
        else:
            trade_num = 101

        # DEBUG: Check to see if the prior iteration resulted in any negative item values
        if not all(i >= 0 for i in responding_items):
            break
        if not all(i >= 0 for i in offering_items):
            break
    return log, error_flag