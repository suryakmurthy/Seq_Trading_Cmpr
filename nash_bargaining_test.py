import numpy as np
from scipy.optimize import minimize

def find_intersection_direct(cones):
    """
    Find the intersection of dual cones by directly enforcing the constraints.

    Args:
        cones (list of dict): Each dict contains 'center' (cone center) and 'theta' (cone angle in radians).

    Returns:
        np.array or None: A vector in the intersection of the dual cones, or None if no intersection exists.
    """
    def objective(x):
        # Objective function to maximize the norm of x (or any arbitrary metric).
        # Here, we use a placeholder objective since the feasibility is handled by constraints.
        return 0

    def constraint_norm(x):
        # Enforce x to be a unit vector.
        return np.linalg.norm(x) - 1

    constraints = [{"type": "eq", "fun": constraint_norm}]

    # Add dual cone constraints: x · center >= cos(theta) for each cone.
    for cone in cones:
        center = cone["center"]
        theta = cone["theta"]
        cos_theta = np.cos(np.pi / 2 - theta)  # Dual cone constraint
        constraints.append({"type": "ineq", "fun": lambda x, c=center, ct=cos_theta: np.dot(x, c) - ct})

    # Initial guess (random unit vector)
    n = len(cones[0]["center"])
    x0 = np.random.randn(n)
    x0 /= np.linalg.norm(x0)

    # Optimization
    result = minimize(objective, x0, constraints=constraints, method="SLSQP")

    if result.success:
        return result.x
    else:
        return None

# Input parameters
center1 = np.array([0.57901105, 0.57800146, 0.57503089])
theta1 = 0.009376056278886992

center2 = np.array([-0.12327593, -0.29596421, -0.94721077])
theta2 = 0.009376056278886992

cones = [
    {"center": center1 / np.linalg.norm(center1), "theta": theta1},
    {"center": center2 / np.linalg.norm(center2), "theta": theta2},
]

# Solve for intersection
intersection = find_intersection_direct(cones)

# Output results
if intersection is not None:
    print("Intersection found at:", intersection)
    print("Norm of intersection vector:", np.linalg.norm(intersection))
    print("Dot products with centers:")
    for cone in cones:
        print(f"Dot with center: {np.dot(intersection, cone['center'])}")
else:
    print("No intersection found.")
