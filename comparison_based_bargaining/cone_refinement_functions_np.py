import numpy as np
import scipy
import time
import torch

def softmax(x):
    e = torch.exp(x - torch.max(x))  # Subtract max for numerical stability
    return e / torch.sum(e)

def new_softmax_transform(x_space_vector):
    x_n = -torch.sum(x_space_vector)
    x_full = torch.cat([x_space_vector, x_n.unsqueeze(0)])
    return softmax(x_full)

def new_softmax_inverse_transform(w_space_vector, epsilon=1e-12):
    w_clipped = torch.clamp(w_space_vector, min=epsilon, max=1.0)
    n = w_clipped.shape[0]
    log_w = torch.log(w_clipped)
    sum_value = torch.mean(log_w)
    output = log_w - sum_value
    return output[:n-1]

def query(Sigma, lambda_mu, current_state, offer):
    # Move in the unconstrained space, then map to simplex
    new_state = current_state + offer

    w_current = new_softmax_transform(current_state)
    w_next = new_softmax_transform(new_state)
    # print("Query: ",offer, w_current - w_next)
    # time.sleep(10)
    current_value = w_current.T @ Sigma @ w_current - lambda_mu @ w_current
    next_value = w_next.T @ Sigma @ w_next - lambda_mu @ w_next

    return next_value > current_value  # Lower is better in this formulation

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

def true_markowitz_gradient(x, Sigma, lambda_mu):
    """
    Compute ∇_x f(softmax(x)) using chain rule:
    ∇_x f = J_softmax(x)^T @ ∇_w f(w)
    """
    w = new_softmax_transform(x)
    quad_term = torch.dot(w, Sigma @ w)
    linear_term = torch.dot(lambda_mu, w)
    expression = quad_term - linear_term

    # Backward to get gradient
    expression.backward()
    gradient = x.grad
    return gradient

def true_markowitz_gradient_w(x, Sigma, lambda_mu):
    """
    Compute ∇_x f(softmax(x)) using chain rule:
    ∇_x f = J_softmax(x)^T @ ∇_w f(w)
    """
    w = new_softmax_transform(x)
    grad_w = 2 * Sigma @ w - lambda_mu  # gradient in simplex space
    return grad_w

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

def estimate_gradient_f_i_comparisons(x, Sigma, lambda_mu, theta_threshold = 0.001):
    """Compute the gradient of f_i(x) = x^T Sigma x + lambda_mu^T x."""
    ## Initalize Cone:
    # print(f"Theta Threshold Values: {theta_threshold}")
    num_dim = len(x)
    cone_center = np.zeros(num_dim)
    for index in range(num_dim):
        initialization_offer = np.zeros(num_dim)
        initialization_offer[index] = 0.00000001
        response = query(Sigma, lambda_mu, x, initialization_offer)
        cone_center[index] = 1 if response else -1

    cone_center = np.array(cone_center) / np.linalg.norm(cone_center)
    theta = np.arccos(1 / np.sqrt(num_dim))

    while theta > theta_threshold:
        offers = generate_offers(cone_center)
        responses = []
        neg_responses = []
        for offer in offers:
            response = query(Sigma, lambda_mu, x, offer)
            neg_response = query(Sigma, lambda_mu, x, -1 * offer)
            scale_down = 0.000000001

            while response == neg_response and scale_down > 1e-10:
                scaled_offer = scale_down * offer

                response = query(Sigma, lambda_mu, x, scaled_offer)
                neg_response = query(Sigma, lambda_mu, x, -1 * scaled_offer)
                scale_down *= 0.1
                # print(offer, scaled_offer, scale_down, response, neg_response)
            responses.append(response)
            neg_responses.append(neg_response)


        new_cone_center, new_theta = refine_cone(cone_center, theta, offers, responses)

        cone_center = new_cone_center
        theta = new_theta

    return cone_center

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