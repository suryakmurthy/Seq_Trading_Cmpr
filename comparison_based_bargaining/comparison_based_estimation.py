import torch
from helper_functions import from_subspace_to_simplex, angle_between, compute_subspace_gradient

def query(Sigma, lambda_mu, current_state, offer, barrier_coeff=1e-6):
    """
    Comparison query between current state and offered perturbation, using barrier penalties.
    """
    new_state = current_state + offer
    w_current = from_subspace_to_simplex(current_state)
    w_next = from_subspace_to_simplex(new_state)

    def barrier(w, v):
        if torch.any(w <= 0.0) or torch.sum(v) >= 1.0:
            return torch.tensor(1e6, dtype=w.dtype, device=w.device)
        eps = 1e-6
        return -torch.sum(torch.log(w + eps)) - torch.log(1.0 - torch.sum(v) + eps)

    current_val = torch.dot(w_current, Sigma @ w_current) - torch.dot(lambda_mu, w_current)
    next_val = torch.dot(w_next, Sigma @ w_next) - torch.dot(lambda_mu, w_next)

    current_total = current_val + barrier_coeff * barrier(w_current, current_state)
    next_total = next_val + barrier_coeff * barrier(w_next, new_state)
    return next_total > current_total

def generate_offers(center_of_cone: torch.Tensor, step_size_orth=0.001):
    """
    Generate orthogonal perturbations around the cone center.
    """
    d = center_of_cone.shape[0]
    center = center_of_cone / torch.norm(center_of_cone)
    Q = torch.eye(d)
    Q[:, 0] = center
    Q, _ = torch.linalg.qr(Q)
    offers = Q[:, 1:].T * step_size_orth
    return [v for v in offers]

def refine_cone(center_of_cone, theta, offers, offer_responses):
    """
    Refine cone center using polar updates depending on offer outcomes.
    """
    sum_value = center_of_cone / torch.norm(center_of_cone)
    for i in range(len(offer_responses)):
        direction = offers[i] / torch.norm(offers[i])
        sign = 1.0 if offer_responses[i] else -1.0
        w_i = center_of_cone * torch.cos(theta) + sign * direction * torch.sin(theta)
        sum_value += w_i

    new_center = sum_value / torch.norm(sum_value)
    scaling_factor = torch.sqrt(torch.tensor((2 * len(center_of_cone) - 1) / (2 * len(center_of_cone))))
    new_theta = torch.arcsin(scaling_factor * torch.sin(theta))
    return new_center, new_theta

def estimate_gradient_f_i_comparisons(x, Sigma, lambda_mu, theta_threshold=0.001):
    """
    Estimate gradient via cone refinement using binary comparison oracle.
    """
    num_dim = x.shape[0]
    cone_center = torch.zeros(num_dim)
    true_gradient = compute_subspace_gradient(x, Sigma, lambda_mu)
    query_count = 0
    step_size_init = 0.001

    for index in range(num_dim):
        init_offer = torch.zeros(num_dim)
        init_offer[index] = step_size_init
        response = query(Sigma, lambda_mu, x, init_offer)
        neg_response = query(Sigma, lambda_mu, x, -1 * init_offer)
        while response == neg_response and step_size_init > 1e-20:
            step_size_init *= 0.1
            init_offer[index] = step_size_init
            response = query(Sigma, lambda_mu, x, init_offer)
            neg_response = query(Sigma, lambda_mu, x, -1 * init_offer)
            query_count += 2
        cone_center[index] = 1.0 if response else -1.0

    cone_center = cone_center / torch.norm(cone_center)
    theta = torch.acos(torch.tensor(1.0) / torch.sqrt(torch.tensor(float(num_dim))))

    while theta > theta_threshold:
        offers = generate_offers(cone_center)
        responses = []
        for offer in offers:
            response = query(Sigma, lambda_mu, x, offer)
            neg_response = query(Sigma, lambda_mu, x, -1 * offer)
            query_count += 2
            scale_down = 1e-1
            while response == neg_response and scale_down > 1e-5:
                scaled_offer = scale_down * offer
                response = query(Sigma, lambda_mu, x, scaled_offer)
                neg_response = query(Sigma, lambda_mu, x, -1 * scaled_offer)
                query_count += 2
                scale_down *= 0.1
            responses.append(response)
        cone_center, theta = refine_cone(cone_center, theta, offers, responses)

    return cone_center, query_count