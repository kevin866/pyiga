import numpy as np
import torch
import matplotlib.pyplot as plt

# Custom NURBS evaluation function with index safety checks
def basis_function(i, k, u, knot_vector):
    if k == 0:
        if i >= len(knot_vector) - 1:  # Avoid out-of-range access
            return 0.0
        return 1.0 if knot_vector[i] <= u < knot_vector[i + 1] else 0.0
    else:
        denom1 = knot_vector[i + k] - knot_vector[i] if i + k < len(knot_vector) else 0
        denom2 = knot_vector[i + k + 1] - knot_vector[i + 1] if i + k + 1 < len(knot_vector) else 0
        
        term1 = ((u - knot_vector[i]) / denom1) * basis_function(i, k - 1, u, knot_vector) if denom1 != 0 else 0
        term2 = ((knot_vector[i + k + 1] - u) / denom2) * basis_function(i + 1, k - 1, u, knot_vector) if denom2 != 0 else 0
        return term1 + term2


# Function to calculate NURBS points
def calculate_nurbs_points(ctrlpts, weights, knotvector, degree, num_points=100, epsilon=1e-6):
    u_values = np.linspace(knotvector[degree], knotvector[-degree-1], num_points)
    nurbs_points = []
    for u in u_values:
        numerator = torch.zeros(2, dtype=ctrlpts.dtype, device=ctrlpts.device)
        denominator = torch.tensor(0.0, dtype=ctrlpts.dtype, device=ctrlpts.device)
        for i in range(len(ctrlpts)):
            b = basis_function(i, degree, u, knotvector)
            numerator += b * weights[i] * ctrlpts[i]
            denominator += b * weights[i]
        nurbs_points.append(numerator / (denominator + epsilon))
    return torch.stack(nurbs_points)

# Control points and weights
P = np.array([[5, 2, 2, 6, 10, 10, 7], 
              [1, 3, 8, 10, 8, 3, 1]])
weights = [1, 1, 1, 1, 1, 1, 1]
knotvector = [0, 0, 0, 0.1, 0.2, 0.5, 0.7, 1, 1, 1]
degree = 3  # Degree of the curve

# Convert control points and weights to tensors
ctrlpts = torch.tensor(P.T, dtype=torch.float32)
weights = torch.tensor(weights, dtype=torch.float32)

# Calculate NURBS points
nurbs_points = calculate_nurbs_points(ctrlpts, weights, knotvector, degree)

# Convert NURBS points to numpy for visualization
nurbs_points_np = nurbs_points.cpu().numpy()

# Plotting the NURBS curve
plt.figure(figsize=(8, 6))
plt.plot(ctrlpts[:, 0].cpu().numpy(), ctrlpts[:, 1].cpu().numpy(), 'ro-', label='Control Points')
plt.plot(nurbs_points_np[:, 0], nurbs_points_np[:, 1], 'b-', label='NURBS Curve')
plt.title("NURBS Curve")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
