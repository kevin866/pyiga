import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from geomdl import BSpline
from geomdl import utilities
import matplotlib.pyplot as plt

# Superformula generation function
def superformula(m, n1, n2, n3, a=1, b=1, num_points=100):
    phi = np.linspace(0, 2 * np.pi, num_points)
    r = (np.abs(np.cos(m * phi / 4) / a)**n2 + np.abs(np.sin(m * phi / 4) / b)**n3)**(-1 / n1)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.vstack((x, y)).T

# Generate a dataset of superformula points
superformula_params = [(m, n1, n2, n3) for m in range(1, 10) for n1 in range(1, 5) for n2 in range(1, 5) for n3 in range(1, 5)]
superformula_points = [superformula(*params) for params in superformula_params]

# Initialize NURBS parameters
degree = 3
num_ctrlpts = 10
knotvector = utilities.generate_knot_vector(degree, num_ctrlpts)

# Function to calculate distance
def calculate_distance(nurbs_points, superformula_points):
    if isinstance(nurbs_points, np.ndarray):
        nurbs_points = torch.tensor(nurbs_points, dtype=torch.float32, requires_grad=True)
    if isinstance(superformula_points, np.ndarray):
        superformula_points = torch.tensor(superformula_points, dtype=torch.float32)
    
    distance = torch.norm(nurbs_points - superformula_points, dim=1).sum()
    return distance

# Placeholder NURBS points calculation (implement proper NURBS evaluation)
def calculate_nurbs_points(ctrlpts, weights, knotvector, num_points=1000):
    curve = BSpline.Curve()
    curve.degree = degree
    curve.ctrlpts = ctrlpts.tolist()
    curve.knotvector = knotvector
    curve.weights = weights.tolist()
    curve_points = np.array(curve.evalpts)
    return curve_points

# Custom NURBS Generator with valid knot vector layer
class NURBSGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, degree, num_ctrlpts):
        super(NURBSGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim - (degree + num_ctrlpts + 1))
        )

    def forward(self, x):
        params = self.fc(x)
        return params

# Initialize model, loss function, and optimizer
input_dim = 4  # Superformula parameters
output_dim = num_ctrlpts * 2 + num_ctrlpts + degree + num_ctrlpts + 1  # Control points, weights, and knot vector
model = NURBSGenerator(input_dim, output_dim, degree, num_ctrlpts)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to hook gradients
def print_grad(grad):
    print('Gradient:', grad)

# # Attach hooks to model parameters
# for param in model.parameters():
#     param.register_hook(print_grad)

# Simple training loop to check gradients
params = torch.tensor([1, 1, 1, 1], dtype=torch.float32)  # Simple test case
superformula_pts = torch.tensor(superformula(1, 1, 1, 1), dtype=torch.float32)

for epoch in range(10):
    output = model(params)
    ctrlpts, weights = output.split([num_ctrlpts*2, num_ctrlpts])
    ctrlpts = ctrlpts.view(num_ctrlpts, 2)
    weights = weights.view(num_ctrlpts)
    # print(type(ctrlpts))
    nurbs_points = calculate_nurbs_points(ctrlpts.detach().numpy(), weights.detach().numpy(), knotvector)
    nurbs_points = torch.from_numpy(nurbs_points)
    print(ctrlpts)
    break
    # Placeholder loss function to test gradient flow
    loss = (ctrlpts - superformula_pts[:num_ctrlpts]).pow(2).sum()
    
    optimizer.zero_grad()
    loss.backward()
    
    # for name, param in model.named_parameters():
        # if param.grad is not None:
        #     # print(f'Gradients for {name} at Epoch {epoch}: {param.grad}')
        # else:
        #     # print(f'No gradients for {name} at Epoch {epoch}')
    
    optimizer.step()
    
    # print(f'Epoch [{epoch}], Loss: {loss.item():.4f}')
