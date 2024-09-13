import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_structure import NURBSGenerator
from pyiga import approx, bspline
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
superformula_points = np.load('nurbs_models\data\superformula_points.npy')
superformula_params = np.load('nurbs_models\data\superformula_params.npy')
print(superformula_points.shape)
print(superformula_params.shape)

# Custom NURBS evaluation function
def basis_function(i, k, u, knot_vector):
    if k == 0:
        return 1.0 if knot_vector[i] <= u < knot_vector[i + 1] else 0.0
    else:
        denom1 = knot_vector[i + k] - knot_vector[i]
        denom2 = knot_vector[i + k + 1] - knot_vector[i + 1]
        term1 = ((u - knot_vector[i]) / denom1) * basis_function(i, k - 1, u, knot_vector) if denom1 != 0 else 0
        term2 = ((knot_vector[i + k + 1] - u) / denom2) * basis_function(i + 1, k - 1, u, knot_vector) if denom2 != 0 else 0
        return term1 + term2

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

# Initialize NURBS parameters
degree = 3
n_kv = 50
knotvector = bspline.make_knots(degree, 0.0, 1.0, n_kv)
num_ctrlpts = knotvector.numdofs
knotvector = knotvector.kv
input_dim = 4  # Superformula parameters
output_dim = num_ctrlpts * 2 + num_ctrlpts  # Control points and weights
model = NURBSGenerator(input_dim, output_dim, degree, num_ctrlpts)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Initialize model and move to GPU
model = NURBSGenerator(input_dim, output_dim, degree, num_ctrlpts).to(device)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    tot_loss = 0
    for params, superformula_pts in zip(superformula_params, superformula_points):
        # Move tensors to GPU
        params = torch.tensor(params, dtype=torch.float32).to(device)
        superformula_pts = torch.tensor(superformula_pts, dtype=torch.float32).to(device)
        
        # Forward pass
        output = model(params)
        ctrlpts, weights = output.split([num_ctrlpts*2, num_ctrlpts])
        ctrlpts = ctrlpts.view(num_ctrlpts, 2)
        weights = weights.view(num_ctrlpts)
        
        # Ensure requires_grad is True
        ctrlpts = ctrlpts.requires_grad_()
        weights = weights.requires_grad_()
        
        # Calculate NURBS points using custom function
        nurbs_points = calculate_nurbs_points(ctrlpts, weights, knotvector, degree)
        
        # Calculate loss
        loss = criterion(nurbs_points, superformula_pts[:nurbs_points.shape[0]])
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tot_loss += loss.item()
        
    avg_loss = tot_loss / len(superformula_params)
    print(f'Epoch [{epoch}/{num_epochs}], Avg Loss: {avg_loss:.4f}')
