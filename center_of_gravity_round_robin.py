import numpy as np
import scipy
import matplotlib as plt
from scipy.optimize import minimize
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import time
from mpl_toolkits.mplot3d import Axes3D
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from scipy.spatial.distance import pdist, squareform
import random
import itertools
import cvxpy as cp

numpy2ri.activate()  # Ensure automatic NumPy to R conversion

volesti = importr("volesti")  # Import the volesti package
numpy2ri.activate()

# Import the R package and functions
ro.r('''
library(volesti)
set.seed(42)

estimate_centroid_volesti <- function(halfspaces, num_samples = 10000) {
  # Convert halfspaces into A_matrix (coefficients of inequalities) and b_vector (right-hand side)
  A_matrix <- matrix(0, nrow = length(halfspaces), ncol = length(halfspaces[[1]][[1]]))
  b_vector <- numeric(length(halfspaces))
  
  for (i in 1:length(halfspaces)) {
    A_matrix[i, ] <- halfspaces[[i]]$coefficients
    b_vector[i] <- halfspaces[[i]]$rhs
  }
  
  # Create the H-polytope using volesti
  H_polytope <- Hpolytope(A = A_matrix, b = b_vector)
  
  # Sample points uniformly inside the polytope
  samples <- sample_points(H_polytope, num_samples)
  
  # Compute the centroid as the mean of the sampled points
  centroid <- rowMeans(samples)
  return(centroid)
}

# Function to estimate the volume of a polytope using volesti
estimate_volume_volesti <- function(halfspaces) {
  # Convert halfspaces into A_matrix (coefficients of inequalities) and b_vector (right-hand side)
  A_matrix <- matrix(0, nrow = length(halfspaces), ncol = length(halfspaces[[1]][[1]]))
  b_vector <- numeric(length(halfspaces))
  
  for (i in 1:length(halfspaces)) {
    A_matrix[i, ] <- halfspaces[[i]]$coefficients
    b_vector[i] <- halfspaces[[i]]$rhs
  }
  
  # Create the H-polytope using volesti
  H_polytope <- Hpolytope(A = A_matrix, b = b_vector)
  
  # Compute the volume of the polytope
  vol <- volume(H_polytope)
  return(vol)
}
''')

#### HELPER FUNCTIONS

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

def is_in_cone(a_vectors, v, atol = 1e-6):
    """
    Check if the vector v lies within the cone defined by the set of halfspaces
    represented by the a-vectors.
    
    Parameters:
    - a_vectors: List of numpy arrays, each representing an a-vector for a halfspace.
    - v: numpy array representing the vector to check.
    
    Returns:
    - True if v is within the cone, False otherwise.
    """
    for a in a_vectors:
        # print(a, v)
        if np.dot(a, v) < 0 + atol:  # If a^T v < 0, v is outside the cone defined by this halfspace
            return False
    return True

def plot_halfspaces_and_centroid(existing_halfspaces, centroid, new_halfspace, x_range=None):
    """
    Plots existing halfspaces, the centroid, and a new halfspace in 2D.
    
    Args:
        existing_halfspaces (list of tuples): Each tuple contains a normal vector and a bias value for the halfspaces.
                                              Example: [((1, 1), 5), ((-1, 2), 3)]
        centroid (array-like): The (x, y) coordinates of the centroid to plot.
        new_halfspace (tuple): The normal vector and bias value for the new halfspace to add.
                               Example: ((2, -1), 2)
        x_range (array-like, optional): The range of x values to plot. Default is None, which sets it to np.linspace(-5, 5, 500).
    """
    if x_range is None:
        x_range = np.linspace(-5, 5, 500)
    
    # Function to plot a halfspace given its normal vector and b value
    def plot_halfspace(ax, normal, b_val, x_range, type=0):
        a, b = normal
        c = b_val
        y_vals = (-a * x_range + c) / b
        if type == 0:
            ax.plot(x_range, y_vals, label="Halfspace", color="blue")
            ax.fill_between(x_range, y_vals, np.max(y_vals), color='blue', alpha=0.1)
        else:
            ax.plot(x_range, y_vals, label="New Halfspace", color="green", linestyle="--")
            ax.fill_between(x_range, y_vals, np.max(y_vals), color='green', alpha=0.1)

    # Function to plot the centroid
    def plot_centroid(ax, centroid):
        ax.scatter(centroid[0], centroid[1], color='red', zorder=5, label="Centroid")

    # Function to plot the new halfspace
    def plot_new_halfspace(ax, normal, b_val, x_range):
        plot_halfspace(ax, normal, b_val, x_range, type=1)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the existing halfspaces
    for normal, b_val in existing_halfspaces:
        plot_halfspace(ax, normal, b_val, x_range)

    # Plot the centroid
    plot_centroid(ax, centroid)

    # Plot the new halfspace
    plot_new_halfspace(ax, new_halfspace[0], new_halfspace[1], x_range)

    # Set plot limits and labels
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_aspect('equal', 'box')
    ax.legend()

    # Show the plot
    plt.show()

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
        self.polytope_information = []
        self.initalize_polytopes()

    def initalize_polytopes(self):
        for i in range(self.num_agents):
            center_of_cone = self.cone_information[i]["center_of_cone"]
            basis_set = qr_decomposition(center_of_cone)
            halfspaces = []

            for offer in self.cone_information[i]["offers"]:
                offer_norm = offer / np.linalg.norm(offer)
                a_vec = np.array(vector_projection(offer_norm, basis_set))

                angle_offset = angle_between(offer_norm, center_of_cone)
                angle_minus_90 = angle_offset - (np.pi / 2)
                b_val = np.tan(angle_minus_90) * np.linalg.norm(a_vec)

                halfspaces.append((-a_vec / np.linalg.norm(a_vec), -b_val))  # Convert to Ax <= b
            self.polytope_information.append({"basis_set": basis_set, "halfspaces": halfspaces})

    def estimate_volume(self, agent_idx, num_samples = 10000):
        halfspaces = self.polytope_information[agent_idx]["halfspaces"]
        # Convert halfspaces into R matrix form
        # Convert halfspaces from Python format to R format
        halfspaces_r = ro.ListVector({
            str(i): ro.ListVector({
                'coefficients': ro.FloatVector(h[0]),
                'rhs': ro.FloatVector([h[1]])
            }) for i, h in enumerate(halfspaces)
        })
        
        # Call the R function and get the volume
        volume_r = ro.r['estimate_volume_volesti'](halfspaces_r)
        
        # Convert the volume from R to a Python float and return
        return float(volume_r[0])
    
    def determine_vertices(self, agent_idx, atol=1e-6):
        # Extract coefficients and RHS
        halfspaces = self.polytope_information[agent_idx]["halfspaces"]
        A = np.array([hs[0] for hs in halfspaces])  # Coefficients of inequalities
        b = np.array([hs[1] for hs in halfspaces])  # Right-hand side values
        
        # Number of dimensions (based on the number of columns in A)
        dim = A.shape[1]
        
        # List to store the vertices
        vertices = []
        
        # Solve for vertices by iterating through combinations of constraints
        for indices in itertools.combinations(range(len(halfspaces)), dim):
            # Sub-matrix A for the chosen constraints
            A_sub = A[list(indices), :]
            b_sub = b[list(indices)]
            
            # Solve the linear system A_sub * x = b_sub
            try:
                x = np.linalg.solve(A_sub, b_sub)  # Solve for the intersection point
                # print("Potential Vertex: ", x, np.dot(A, x), b)
                
                # Check if the solution satisfies all the constraints with a tolerance
                if np.all(np.dot(A, x) <= b + atol):  # Account for floating-point error
                    # Check if this vertex is a unique solution
                    is_unique = True
                    for v in vertices:
                        if np.allclose(v, x, atol=atol):  # Use atol for numerical tolerance
                            is_unique = False
                            break
                    if is_unique:
                        vertices.append(x)
            except np.linalg.LinAlgError:
                # If the system is singular (no unique solution), skip this combination
                continue
        
        return np.array(vertices)

    def estimate_centroid(self, agent_idx, num_samples=10000):
        halfspaces = self.polytope_information[agent_idx]["halfspaces"]

        # Convert halfspaces from Python format to R format
        halfspaces_r = ro.ListVector({
            str(i): ro.ListVector({
                'coefficients': ro.FloatVector(h[0]),
                'rhs': ro.FloatVector([h[1]])
            }) for i, h in enumerate(halfspaces)
        })
        
        # Call the R function and get the centroid
        centroid_r = ro.r['estimate_centroid_volesti'](halfspaces_r, num_samples)
        
        # Convert the centroid from R to a NumPy array and return
        return list(centroid_r)

    def generate_offer(self, agent_idx):
        center_of_cone = self.cone_information[agent_idx]["center_of_cone"]
        basis_set = self.polytope_information[agent_idx]["basis_set"]
        halfspaces  = self.polytope_information[agent_idx]["halfspaces"]

        centroid = self.estimate_centroid(agent_idx)
        vertices = self.determine_vertices(agent_idx)
        if len(vertices) <= 2:
            return []
        # print("Checking: ", centroid, vertices)
        # Compute pairwise distances between vertices
        # print("Error Check", vertices)
        distance_matrix = squareform(pdist(vertices))
        
        # Find the indices of the two farthest vertices
        farthest_pair = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        
        # Extract the farthest vertices
        v1, v2 = vertices[farthest_pair[0]], vertices[farthest_pair[1]]

        # Compute the direction vector (connecting the two farthest vertices)
        direction_vector = v2 - v1

        # print("Checking Farthest Vectors: ", vertices, v2, v1, direction_vector)

        # Normalize the direction vector to get the normal to the plane
        a_vec = direction_vector / np.linalg.norm(direction_vector)

        # Compute b for the half-space equation a^T x = b
        b_value = np.dot(a_vec, centroid)

        dist_from_origin = np.abs(b_value) / np.linalg.norm(a_vec)
        theta_offset = np.arctan(dist_from_origin)
        if b_value >= 0:
            theta_offset = -1 * theta_offset

        # Once the hyperplane that bisects the two farthest corner points is obtained, we turn it into an n-dimesional offer
        offer_in_n_dimensions = a_vec @ basis_set
        # Rotate the offer to account for offset from origin in n-1 dimensions
        original_offer = rotate_vector(offer_in_n_dimensions, center_of_cone, theta_offset)

        # Return the offer corresponding to the bisecting halfspace
        return original_offer

    # Cone refinement for a single cone
    def refine_cone(self, agent_idx, offer, offer_response):

        # Inialize relevant variables
        basis_set = self.polytope_information[agent_idx]["basis_set"]
        center_of_cone = self.cone_information[agent_idx]["center_of_cone"]

        prev_volume = self.estimate_volume(agent_idx)

        # Refine cone
        if not offer_response:
            offer = -1 * offer
        self.cone_information[agent_idx]["offers"].append(offer)

        offer_norm = offer / np.linalg.norm(offer)
        a_vec = np.array(vector_projection(offer_norm, basis_set))

        angle_offset = angle_between(offer_norm, center_of_cone)
        angle_minus_90 = angle_offset - (np.pi / 2)
        b_val = np.tan(angle_minus_90) * np.linalg.norm(a_vec)
        self.polytope_information[agent_idx]["halfspaces"].append((-a_vec / np.linalg.norm(a_vec), -b_val)) 

        new_volume = self.estimate_volume(agent_idx)
        # print("Checking for refinement: ", prev_volume, new_volume)

        return new_volume

    def find_intersection(self):
        """
        Find the intersection of dual cones using scipy.optimize.

        Returns:
            np.array or None: A normalized vector in the intersection of the dual cones, or None if no intersection exists.
        """
        centroids = []
        for agent_idx in range(self.num_agents):
            centroid = np.array(self.estimate_centroid(agent_idx))
            center_of_cone = self.cone_information[agent_idx]["center_of_cone"]
            basis_set = self.polytope_information[agent_idx]["basis_set"]
            rotation_direction_n = centroid @ basis_set
            centroid_vector = rotate_vector(center_of_cone, rotation_direction_n, np.arctan(np.linalg.norm(centroid)))
            centroids.append(centroid_vector)
        centroids = np.array(centroids)
        n = centroids.shape[1]  # Dimension of space

        # Objective function: We just need a feasibility problem, so we use a dummy function
        def objective(x):
            return 0  # Since we only care about constraints

        # Constraints: x^T c_i >= epsilon for all centroids c_i
        epsilon = 1e-3
        constraints = [{'type': 'ineq', 'fun': lambda x, c=centroids[i]: np.dot(x, c) - epsilon} for i in range(len(centroids))]

        # Enforce unit norm constraint
        constraints.append({'type': 'eq', 'fun': lambda x: np.linalg.norm(x) - 1})

        # Initial guess (random unit vector)
        x0 = np.random.randn(n)
        x0 /= np.linalg.norm(x0)

        # Solve optimization
        result = minimize(objective, x0, method='SLSQP', constraints=constraints, options={'disp': False})

        if result.success:
            return result.x
        else:
            return []


    def find_intersection_prev(self):
        """
        Find the intersection of dual cones by directly enforcing the constraints.

        Args:
            cones (list of dict): Each dict contains 'center' (cone center) and 'theta' (cone angle in radians).

        Returns:
            np.array or None: A vector in the intersection of the dual cones, or None if no intersection exists.
        """
        centroids = []
        for agent_idx in range(0, self.num_agents):
            centroid = np.array(self.estimate_centroid(agent_idx))
            center_of_cone = self.cone_information[agent_idx]["center_of_cone"]
            basis_set = self.polytope_information[agent_idx]["basis_set"]
            # print(centroid, basis_set)
            rotation_direction_n = centroid @ basis_set
            centroid_vector = rotate_vector(center_of_cone, rotation_direction_n, np.arctan(np.linalg.norm(centroid)))
            centroids.append(centroid_vector)
        centroids = np.array(centroids)
        # print("centroids: ", centroids)
        n = centroids.shape[1]  # Dimension of space

        # Define the optimization variable
        x = cp.Variable(n)

        # Define a small positive margin for robustness
        epsilon = 1e-3

        # Constraints: x^T c_i >= epsilon for all centroids c_i
        constraints = [x @ centroids[i] >= epsilon for i in range(len(centroids))]
        
        # Enforce unit norm constraint
        constraints.append(cp.norm(x, 2) <= 1)

        # Define and solve the feasibility problem
        prob = cp.Problem(cp.Maximize(0), constraints)
        result = prob.solve(solver=cp.GUROBI)

        if prob.status == cp.OPTIMAL:
            return x.value
        else:
            return []


def round_robin_iteration(num_categories, agent_set, current_state, step_size=1):
    volume_reduction = []
    uncertainty_reduction = []
    cone_information = []
    offer_list = []
    for agent_idx in range(len(agent_set)):
        volume_reduction.append([])
        uncertainty_reduction.append([])
        agent = agent_set[agent_idx]
        cone_center = np.zeros(num_categories)
        offers = []
        for i in range(num_categories):
            initialization_offer = np.zeros(num_categories)
            initialization_offer[i] = 1
            response = agent.query(current_state, initialization_offer)
            cone_center[i] = 1 if response else -1
            offers.append(initialization_offer) if response else offers.append(-1*initialization_offer)
        offer_list.append(offers.copy())
        cone_center /= np.linalg.norm(cone_center)
        cone_information.append({"center_of_cone": cone_center, "offers": offers})

    mediator = Mediator(len(agent_set), num_categories, cone_information)
    for agent_idx in range(len(agent_set)):
        volume_reduction[agent_idx].append(mediator.estimate_volume(agent_idx))

        # Find Uncertainty Reduction
        vertices = mediator.determine_vertices(agent_idx)
        distance_matrix = squareform(pdist(vertices))
        farthest_pair = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        v1, v2 = vertices[farthest_pair[0]], vertices[farthest_pair[1]]
        uncertainty_reduction[agent_idx].append(np.linalg.norm(v1 - v2))

    trade_found = False
    num_offers = 0
    max_offers = 40
    intersection_offer = 0
    while not trade_found and num_offers < max_offers:
        # print("Iteration: ", num_offers, max_offers)
        for agent_idx in range(len(agent_set)):
            agent = agent_set[agent_idx]
            offer = mediator.generate_offer(agent_idx)
            if len(offer) == 0:
                continue
            response = agent.query(current_state, offer)
            if response:
                offer_list[agent_idx].append(offer)
            else:
                offer_list[agent_idx].append(-1 * offer)

            num_offers += 1
            prev_cone_center = mediator.cone_information[agent_idx]["center_of_cone"].copy()
            # print("Offer: ", offer, len(offer_list[agent_idx]))
            new_volume = mediator.refine_cone(agent_idx, offer, response)
            
            volume_reduction[agent_idx].append(new_volume)
            
            vertices = mediator.determine_vertices(agent_idx)

            if len(vertices) >= 2:
                distance_matrix = squareform(pdist(vertices))
                farthest_pair = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
                v1, v2 = vertices[farthest_pair[0]], vertices[farthest_pair[1]]
                uncertainty_reduction[agent_idx].append(np.linalg.norm(v1 - v2))
            else:
                uncertainty_reduction[agent_idx].append(0)
            true_gradient = agent.generate_gradient_vector(current_state)
            # print("Checking if gradient is contained in cone: ", true_gradient, len(offer_list[agent_idx]), is_in_cone(offer_list[agent_idx], true_gradient))
            # time.sleep(3)
            # if angle_between(new_cone_center, true_gradient) > new_theta:
            #     print("Offers: ", offers, responses, responses_neg, agent.generate_directional_magnitude(current_state, offers[0]), agent.generate_directional_magnitude(current_state, offers[1]))
            #     print("Failure: ", new_cone_center, new_theta, angle_between(new_cone_center, true_gradient))
            #     time.sleep(1000)

        intersection_offer = []
        # TODO Add Nash Update here instead of feasibility
        intersection_offer = mediator.find_intersection()
        if len(intersection_offer) != 0:
            intersection_offer = np.array(intersection_offer) / np.linalg.norm(intersection_offer)
            scaling_value = False
            while not scaling_value:
                scaled_intersection_offer = step_size * intersection_offer
                responses = [agent.query(current_state, scaled_intersection_offer) for agent in agent_set]
                num_offers += 1
                if all(responses) == True:
                    trade_found = True
                    intersection_offer = scaled_intersection_offer
                    scaling_value = True
                else:
                    if step_size > 0.0001:
                        step_size = 1/2 * step_size
                    else:
                        intersection_offer = []
                        scaling_value = True

    return intersection_offer, num_offers, mediator, volume_reduction, uncertainty_reduction

def round_robin_scenario_full(num_categories, agent_set, starting_state, step_size=10, offer_limit=1000):
    current_state = starting_state
    state_progression = []
    offer_progression = []
    state_progression.append(current_state.copy())

    total_offers = 0
    end_flag = False
    while not end_flag:
        intersection_offer, num_offers, mediator, volume_reduction, uncertainty_reduction = round_robin_iteration(num_categories, agent_set, current_state,step_size=step_size)
        # print("Offer: ", intersection_offer, current_state)
        if len(intersection_offer) == 0 or total_offers >= offer_limit:
            end_flag = True
        else:
            current_state += intersection_offer
            total_offers += num_offers
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
# np.random.seed(10)
# linear_case = False
# if linear_case:
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
#     num_categories_list = [3, 5, 7, 9]
#     for num_categories in num_categories_list:
#         current_state = np.array([0.0 for i in range(num_categories)])
#         starting_state = current_state.copy()
#         # Base quadratic function components
#         A = -1 * np.eye(num_categories)
#         b_base = np.array([random.randint(-200, 200) for i in range(num_categories)])

#         angles = [20]  # Angles from 0 to 180 degrees
#         # angles = [20]
#         for angle in angles:
#             current_state = np.array([0.0 for i in range(num_categories)])
#             starting_state = current_state.copy()
#             # Generate the second gradient with the specified angle to the first gradient
#             b_2 = generate_gradient_with_angle(b_base, angle)
#             A_2 = -1 * np.eye(num_categories)
#             agent_set = [Bargaining_Agent(A, b_base), Bargaining_Agent(A_2, b_2)]

#             # Run Full Scenario
#             final_state, num_offers = round_robin_scenario_full(num_categories, agent_set, current_state)
#             print("Final State: ", angle, starting_state, final_state, num_offers)
#             # x_values = range(len(volume_reduction[0]))
#             # plt.figure(figsize=(8,5))
#             # plt.plot(x_values, volume_reduction[0], label = "Agent 1", marker="o")
#             # plt.plot(x_values, volume_reduction[1], label = "Agent 2", marker="s")

#             # plt.xlabel("Iteration")
#             # plt.ylabel("Polytope Volume")
#             # plt.yscale("log")
#             # plt.title(f"Volume Reduction Curve ({num_categories} Categories)")
#             # plt.legend()
#             # plt.grid(True)
#             # plt.savefig(f"Volume Reduction {num_categories}")
#             # plt.show()

#             # plt.figure(figsize=(8,5))
#             # plt.plot(x_values, uncertainty_reduction[0], label = "Agent 1", marker="o")
#             # plt.plot(x_values, uncertainty_reduction[1], label = "Agent 2", marker="s")

#             # plt.xlabel("Iteration")
#             # plt.ylabel("Distance Between Farthest Vertices")
#             # plt.yscale("log")
#             # plt.title(f"Uncertainty Reduction Curve ({num_categories} Categories)")
#             # plt.legend()
#             # plt.grid(True)
#             # plt.savefig(f"Uncertainty Reduction{num_categories}")
#             # plt.show()
    
        

