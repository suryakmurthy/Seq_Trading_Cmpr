import math
import random
import numpy as np
from scipy.optimize import minimize
import scipy
from pyoptsparse import SLSQP, Optimization
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_halfspaces(halfspace_set, hypercube, corner_points=None, center_of_circle=None, x_range=(-2.5, 2.5), y_range=(-2.5, 2.5)):
    """
    Plots a set of halfspaces and hypercube constraints, with options to add corner points and a center of circle.

    Args:
        halfspace_set (list of tuples): Non-hypercube halfspace constraints of the form (A, b).
        hypercube (list of tuples): Hypercube halfspace constraints of the form (A, b).
        corner_points (list of np.array, optional): Points marking the farthest corners of the polytope. Defaults to None.
        center_of_circle (np.array, optional): Center of the circle to be plotted. Defaults to None.
        x_range (tuple, optional): Range of x-axis for plotting. Defaults to (-10, 10).
        y_range (tuple, optional): Range of y-axis for plotting. Defaults to (-10, 10).
    """
    x = np.linspace(x_range[0], x_range[1], 500)  # Generate x values
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot hypercube halfspaces without shading
    for A, b in hypercube:
        A = np.array(A).flatten()
        assert A.size == 2, "A must be a 2D vector for 2D plotting"

        a1, a2 = A
        if a2 != 0:
            y = (b - a1 * x) / a2
            ax.plot(x, y, 'k--', linewidth=1, label=f"Hypercube {A} >= {b}")
        elif a1 != 0:
            x_val = b / a1
            ax.axvline(x_val, color='k', linestyle='--', linewidth=1, label=f"Hypercube {A} >= {b}")

    # Plot other halfspaces with shading
    for A, b in halfspace_set:
        A = np.array(A).flatten()
        assert A.size == 2, "A must be a 2D vector for 2D plotting"

        a1, a2 = A
        if a2 != 0:
            y = (b - a1 * x) / a2
            if a2 > 0:
                ax.fill_between(x, y, y_range[0], alpha=0.3, label=f"Halfspace {A} >= {b}")
            else:
                ax.fill_between(x, y, y_range[1], alpha=0.3, label=f"Halfspace {A} >= {b}")
        elif a1 != 0:
            x_val = b / a1
            if a1 > 0:
                ax.fill_betweenx(np.linspace(y_range[0], y_range[1], 500), x_val, x_range[1], alpha=0.3, label=f"Halfspace {A} >= {b}")
            else:
                ax.fill_betweenx(np.linspace(y_range[0], y_range[1], 500), x_range[0], x_val, alpha=0.3, label=f"Halfspace {A} >= {b}")

    # Plot the center of the circle
    if center_of_circle is not None:
        ax.plot(center_of_circle[0], center_of_circle[1], 'ro', label='Center of Circle')

    # Plot the corner points
    if corner_points is not None:
        corner_points = np.array(corner_points)
        ax.scatter(corner_points[:, 0], corner_points[:, 1], color='blue', label='Corner Points')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True)
    # plt.legend(loc='upper left')
    plt.title("Halfspaces Plot")
    plt.show()

#make class named Bargainingame, which stores dimensions of the problem and number of agents
class BargainingGame:
    def __init__(self, n_agents, n_dims):
        self.n_agents = n_agents
        self.n_dims = n_dims
        
#make subclass named OfferGenerator, which takes cone-center (vector) estimates of players and generates offers
class OfferGenerator(BargainingGame):
    def __init__(self, n_agents, n_dims, cone_centers):
        super().__init__(n_agents, n_dims)
        self.cone_centers = cone_centers
        self.offers = []  #stores previous offers made during a cone refinement procedure
                
    def generate_initial_offer_obj(self, xdict):
        x = xdict["vars"]
        assert len(x) == self.n_dims
        funcs = {}
        denom = np.linalg.norm(x)
        funcs["obj"] = sum([abs(sum([x_i*cone_center_i for x_i, cone_center_i in zip(x, cone_center)])) for cone_center in self.cone_centers])/denom
        # return sum([sum([x_i*cone_center_i for x_i, cone_center_i in zip(x, cone_center)]) for cone_center in self.cone_centers])/denom
        conval = [0]
        conval[0] = np.linalg.norm(x)
        funcs["con"] = conval
        fail = False
        return funcs, fail
        
    def generate_subsequent_offer_obj(self, xdict):
        n_offers = len(self.offers)
        assert n_offers > 0
        x = xdict["vars"]
        assert len(x) == self.n_dims
        funcs = {}
        denom = np.linalg.norm(x)
        funcs["obj"] = sum([abs(sum([x_i*cone_center_i for x_i, cone_center_i in zip(x, cone_center)])) for cone_center in self.cone_centers])/denom
        conval = [0] * (n_offers + 1)
        conval[0] = np.linalg.norm(x)
        for i in range(n_offers):
            conval[i+1] = np.dot(x, self.offers[i])
        funcs["con"] = conval
        fail = False
        return funcs, fail

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
        # print(basis_vectors, v)
        proj_component = np.dot(v, basis) / norm_b_squared
        projection.append(proj_component)
    return projection

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
    # print("Checking Vector Projection: ", offer, offer_norm)
    projected_vector = np.array(vector_projection(offer_norm, basis))

    # Halfspace offset is determined by the angle offset of the offer and the orthogonal plane
    angle_offset = angle_between(offer_norm, center_of_cone)
    angle_minus_90 = angle_offset - (np.pi / 2)
    b_val = np.tan(angle_minus_90) * np.linalg.norm(projected_vector)

    return (projected_vector, b_val)

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
    # print("Offer Set: ", offer_set)
    halfspace = generate_halfspaces(offer_set, basis_set, center_of_cone)
    full_halfspace_set = halfspace + hypercube
    point_set = generate_corner_points(full_halfspace_set, num_dim)
    corner_points, point_dist = farthest_points(point_set)

    # If the halfspace contraints do not allow for corner points, return an error case
    if len(corner_points) == 0:
        return None, None, None, True
    # print("Halfspaces and Corner Points: ", full_halfspace_set, corner_points)
    # Calculate new circle parameters
    center_of_circle = np.mean(corner_points, axis=0)
    radius_of_circle = (point_dist)/2
    radius_of_circle = np.sqrt(3) * radius_of_circle
    # print("Corner Points: ", full_halfspace_set, corner_points)
    if all(c == 0 for c in center_of_circle):
        center_norm = np.zeros(num_dim)
        center_norm[0] = 1
    else:
        center_norm = center_of_circle / np.linalg.norm(center_of_circle)
    # plot_halfspaces(full_halfspace_set, hypercube, center_of_circle=center_of_circle, corner_points=point_set)
    # Calculate new angle of opening (Theta)
    point_a = center_of_circle - (radius_of_circle * center_norm)
    point_b = center_of_circle + (radius_of_circle * center_norm)

    # Points to vectors
    direction_a = point_a @ basis_set
    direction_b = point_b @ basis_set

    # Is this right?

    full_direction_a = center_of_cone + direction_a
    full_direction_b = center_of_cone + direction_b
    full_direction_a = full_direction_a / np.linalg.norm(full_direction_a)
    full_direction_b = full_direction_b / np.linalg.norm(full_direction_b)

    new_cone_center = (full_direction_a + full_direction_b) / 2
    new_cone_center = new_cone_center / np.linalg.norm(new_cone_center)
    new_theta = angle_between(new_cone_center, full_direction_a)


    # print(point_a, point_b, direction_a / np.linalg.norm(direction_a), direction_b / np.linalg.norm(direction_b))
    # d_a = np.sqrt(1 + (np.linalg.norm(point_a)**2))
    # d_b = np.sqrt(1 + (np.linalg.norm(point_b)**2))
    # diameter = point_dist
    #
    # ratio = ((diameter**2) - (d_a**2) - (d_b**2))/(-2*d_a*d_b)
    # # print("Point Distance Calculation: ", point_a, point_b, d_a, d_b, diameter)
    # print(ratio, d_a, d_b)
    # potential_new_theta = np.arccos(ratio) / 2
    # print("Potential New Theta: ", center_of_circle, potential_new_theta)
    return new_cone_center, new_theta, corner_points, False

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

random.seed(10)
np.random.seed(10)
game = BargainingGame(2, 3)
offer_in_game = OfferGenerator(2, 3, [np.random.rand(3), np.random.rand(3)])
print("Number of agents: ", game.n_agents)
print("Number of dimensions: ", game.n_dims)
print("Cone center for agent 1: ", offer_in_game.cone_centers[0])
print("Cone center for agent 2: ", offer_in_game.cone_centers[1])

print("Generating Initial Offer")

opt_prob = Optimization("Bargaining Game", offer_in_game.generate_initial_offer_obj)
opt_prob.addVarGroup("vars", 3, type="c", value=1.0)
opt_prob.addConGroup("con", 1, lower=1.0, upper=1.0)
opt_prob.addObj("obj")
# print(opt_prob)
optOptions = {"IPRINT": -1}
opt = SLSQP(options=optOptions)
sol = opt(opt_prob, sens="FD")
# print(sol)

vectors = offer_in_game.cone_centers
offer_vec = sol.getDVs()["vars"]

offer_in_game.offers.append(offer_vec)

opt_prob = Optimization("Bargaining Game", offer_in_game.generate_subsequent_offer_obj)
opt_prob.addVarGroup("vars", 3, type="c", value=1.0)
low_arr = [0.0]*(len(offer_in_game.offers) + 1)
low_arr[0] = 1.0
opt_prob.addConGroup("con", len(offer_in_game.offers) + 1, lower=low_arr, upper=low_arr)
opt_prob.addObj("obj")
# print(opt_prob)
optOptions = {"IPRINT": -1}
opt = SLSQP(options=optOptions)
sol = opt(opt_prob, sens="FD")
new_offer_vec = sol.getDVs()["vars"]
offer_in_game.offers.append(new_offer_vec)


#Potting cone centers and offer
print("Offer: ", offer_vec, math.degrees(angle_between(offer_in_game.cone_centers[0], offer_vec)), math.degrees(angle_between(offer_in_game.cone_centers[1], offer_vec)))
print("New Offer: ", new_offer_vec, math.degrees(angle_between(offer_in_game.cone_centers[0], new_offer_vec)), math.degrees(angle_between(offer_in_game.cone_centers[1], new_offer_vec)))
print("Angle: ", math.degrees(angle_between(offer_vec, new_offer_vec)))

# Insert cone refinement here
test_flag_1 = True
if test_flag_1:
    offer_set = [offer_vec, new_offer_vec]
    theta = np.arccos(1 / np.sqrt(3))

    basis_agent_1 = qr_decomposition(offer_in_game.cone_centers[0])
    basis_agent_2 = qr_decomposition(offer_in_game.cone_centers[1])

    hypercube_agent_1 = generate_hypercube(np.abs(np.tan(theta)), basis_agent_1)
    hypercube_agent_2 = generate_hypercube(np.abs(np.tan(theta)), basis_agent_2)

    center_of_cone_1, potential_new_theta, corner_points, error_case = calculate_new_cone_integer_contstrained(offer_set, hypercube_agent_1, basis_agent_1, offer_in_game.cone_centers[0])

    print("Checking if New Cone is Possible: ", center_of_cone_1, theta, potential_new_theta, corner_points, error_case)

    center_of_cone_2, potential_new_theta, corner_points, error_case = calculate_new_cone_integer_contstrained(offer_set, hypercube_agent_2, basis_agent_2, offer_in_game.cone_centers[1])

    print("Checking if New Cone is Possible: ", center_of_cone_2, theta, potential_new_theta, corner_points, error_case)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
count_for_legend = 0

for vec in vectors:
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], length=1, normalize=True, label="Cone Center for Agent " +str(count_for_legend+1))
    count_for_legend = count_for_legend + 1
    print("Check value " + str(count_for_legend) + ": ", np.dot(vec, offer_vec))
if test_flag_1:
    ax.quiver(0, 0, 0, center_of_cone_1[0], center_of_cone_1[1], center_of_cone_1[2], length=1, normalize=True, color='r', label="New Cone Center for Agent 1")
    ax.quiver(0, 0, 0, center_of_cone_2[0], center_of_cone_2[1], center_of_cone_2[2], length=1, normalize=True, color='r', label="New Cone Center for Agent 1")

ax.quiver(0, 0, 0, offer_vec[0], offer_vec[1], offer_vec[2], length=1, normalize=True, color='g', label="Initial Offer")
ax.quiver(0, 0, 0, new_offer_vec[0], new_offer_vec[1], new_offer_vec[2], length=1, normalize=True, color='g', label="Subsequent Offer")
print("Check value for subsequent offer", np.dot(new_offer_vec, offer_vec))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1, 1, 1])  # Aspect ratio for the axes
plt.legend()
plt.show()
