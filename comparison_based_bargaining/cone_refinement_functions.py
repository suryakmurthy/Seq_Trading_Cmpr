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
    current_value = torch.dot(w_current, Sigma @ w_current) - torch.dot(lambda_mu, w_current)
    next_value = torch.dot(w_next, Sigma @ w_next) - torch.dot(lambda_mu, w_next)

    return next_value > current_value  # Lower is better in this formulation


def angle_between(v1, v2):
    """
    Return the angle between two vectors in radians.

    Args:
        v1 (torch.Tensor): n-dimensional vector.
        v2 (torch.Tensor): n-dimensional vector.

    Returns:
        angle (torch.Tensor): Angle in radians.
    """
    dot_product = torch.dot(v1, v2)
    m1 = torch.norm(v1)
    m2 = torch.norm(v2)
    cos_theta = dot_product / (m1 * m2)

    # Clamp to ensure numerical stability (e.g., for rounding errors)
    cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
    angle = torch.acos(cos_theta_clamped)
    return angle

def true_markowitz_gradient(x, Sigma, lambda_mu):
    """
    Compute ∇_x f(softmax(x)) using chain rule:
    ∇_x f = J_softmax(x)^T @ ∇_w f(w)
    """
    if not x.is_leaf:
        x.retain_grad()
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
    # Create a random vector and orthogonalize it
    random_vector = torch.randn_like(center_of_cone)
    projection = torch.dot(random_vector, center_of_cone) / torch.dot(center_of_cone, center_of_cone)
    orthogonal_vector = random_vector - projection * center_of_cone
    orthogonal_vector = orthogonal_vector / torch.norm(orthogonal_vector) * step_size_orth

    vectors = torch.stack([center_of_cone, orthogonal_vector])
    offer_directions = [orthogonal_vector]

    # Find a basis for the null space using SVD
    u, s, vh = torch.linalg.svd(vectors, full_matrices=True)
    null_space_basis = vh[s.size(0):]

    for i in range(null_space_basis.size(0)):
        potential_vector = null_space_basis[i]
        is_orthogonal = True
        for vector in offer_directions:
            if torch.abs(torch.dot(potential_vector, vector)) > 1e-10:
                is_orthogonal = False
                break
        if is_orthogonal:
            normed = step_size_orth * potential_vector / torch.norm(potential_vector)
            offer_directions.append(normed)

    return offer_directions


def estimate_gradient_f_i_comparisons(x, Sigma, lambda_mu, theta_threshold=0.001):
    num_dim = x.shape[0]
    cone_center = torch.zeros(num_dim)
    # true_gradient = true_markowitz_gradient(x, Sigma, lambda_mu)
    # print("Checking True Gradient: ", true_gradient)
    # time.sleep(100)
    step_size_init = 0.001
    for index in range(num_dim):
        init_offer = torch.zeros(num_dim)
        init_offer[index] = step_size_init

        response = query(Sigma, lambda_mu, x, init_offer)
        neg_response = query(Sigma, lambda_mu, x, -1 * init_offer)
        while response == neg_response and step_size_init > 1e-20:
            step_size_init *= 0.1
            init_offer = torch.zeros(num_dim)
            init_offer[index] = step_size_init
            response = query(Sigma, lambda_mu, x, init_offer)
            neg_response = query(Sigma, lambda_mu, x, -1 * init_offer)
        # print("Response: ", response)
        cone_center[index] = 1.0 if response else -1.0

    cone_center = cone_center / torch.norm(cone_center)
    theta = torch.acos(torch.tensor(1.0) / torch.sqrt(torch.tensor(float(num_dim))))
    # if angle_between(cone_center, true_gradient) - theta > -1e-6:
        # print("Failure Initial: ", angle_between(cone_center, true_gradient), cone_center, true_gradient, theta)
        # time.sleep(100)
    while theta > theta_threshold:
        offers = generate_offers(cone_center)

        responses = []
        neg_responses = []
        scale_values = []
        for offer in offers:
            response = query(Sigma, lambda_mu, x, offer)
            neg_response = query(Sigma, lambda_mu, x, -1 * offer)
            scale_down = 0.001

            while response == neg_response and scale_down > 1e-20:
                scaled_offer = scale_down * offer
                response = query(Sigma, lambda_mu, x, scaled_offer)
                neg_response = query(Sigma, lambda_mu, x, -1 * scaled_offer)
                scale_down *= 0.1
            scale_values.append(scale_down)
            responses.append(response)
            neg_responses.append(neg_response)

        cone_center, theta = refine_cone(cone_center, theta, offers, responses)
        # if angle_between(cone_center, true_gradient) - theta > -1e-6:
        #     print("Failure: ", scale_values, angle_between(cone_center, true_gradient) - theta, responses, neg_responses)
            # time.sleep(100)

    return cone_center

def refine_cone(center_of_cone, theta, offers, offer_responses):
    w_list = [center_of_cone]
    sum_value = center_of_cone / torch.norm(center_of_cone)

    for i in range(len(offer_responses)):
        direction = offers[i] / torch.norm(offers[i])
        if offer_responses[i]:
            w_i = center_of_cone * torch.cos(theta) + direction * torch.sin(theta)
        else:
            w_i = center_of_cone * torch.cos(theta) - direction * torch.sin(theta)
        w_list.append(w_i)
        sum_value += w_i / len(center_of_cone)

    new_center = sum_value / torch.norm(sum_value)
    scaling_factor = torch.sqrt(
        torch.tensor((2 * len(center_of_cone) - 1) / (2 * len(center_of_cone)), dtype=center_of_cone.dtype))
    new_theta = torch.arcsin(scaling_factor * torch.sin(theta))

    return new_center, new_theta

