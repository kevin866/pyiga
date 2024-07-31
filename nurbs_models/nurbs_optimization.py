import numpy as np

# Superformula generation function
def superformula(m, n1, n2, n3, a=1, b=1, num_points=1000):
    phi = np.linspace(0, 2 * np.pi, num_points)
    r = (np.abs(np.cos(m * phi / 4) / a)**n2 + np.abs(np.sin(m * phi / 4) / b)**n3)**(-1 / n1)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.vstack((x, y)).T

# Generate a dataset of superformula points
superformula_params = [(m, n1, n2, n3) for m in range(1, 10) for n1 in range(1, 5) for n2 in range(1, 5) for n3 in range(1, 5)]
superformula_points = [superformula(*params) for params in superformula_params]

from geomdl import BSpline
from geomdl import utilities

# Initialize NURBS parameters
degree = 3
num_ctrlpts = 10
ctrlpts = np.random.rand(num_ctrlpts, 2)
weights = np.ones(num_ctrlpts)
knotvector = utilities.generate_knot_vector(degree, num_ctrlpts)

# Function to calculate distance
def calculate_distance(nurbs_points, superformula_points):
    return np.linalg.norm(nurbs_points - superformula_points, axis=1).sum()

# Placeholder NURBS points calculation (implement proper NURBS evaluation)
def calculate_nurbs_points(ctrlpts, weights, knotvector, num_points=1000):
    # Create a B-Spline curve
    curve = BSpline.Curve()
    curve.degree = degree
    curve.ctrlpts = ctrlpts.tolist()
    print(knotvector)
    curve.knotvector = knotvector
    curve.weights = weights.tolist()
    # Evaluate curve
    curve_points = np.array(curve.evalpts)
    return curve_points

import torch
import torch.nn as nn
import torch.optim as optim

class ValidKnotVectorLayer(nn.Module):
    def __init__(self, degree, num_ctrlpts):
        super(ValidKnotVectorLayer, self).__init__()
        self.degree = degree
        self.num_ctrlpts = num_ctrlpts
        self.num_knots = self.degree + self.num_ctrlpts + 1

    def forward(self, x):
        # Ensure non-decreasing order
        sorted_x, _ = torch.sort(x, dim=-1)
        # Create valid knot vector with required multiplicities at the ends
        knot_vector = torch.cat([
            torch.zeros(self.degree + 1),
            sorted_x[self.degree + 1:self.num_knots - (self.degree + 1)],
            torch.ones(self.degree + 1) * sorted_x[-1]
        ], dim=-1)
        return knot_vector

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
        self.knot_vector_layer = ValidKnotVectorLayer(degree, num_ctrlpts)

    def forward(self, x):
        params = self.fc(x)
        knot_vector = self.knot_vector_layer(params[:, -self.knot_vector_layer.num_knots:])
        return torch.cat((params[:, :-self.knot_vector_layer.num_knots], knot_vector), dim=-1)

# Initialize model, loss function, and optimizer
input_dim = 4  # Superformula parameters
output_dim = num_ctrlpts * 2 + num_ctrlpts + degree + num_ctrlpts + 1  # Control points, weights, and knot vector
model = NURBSGenerator(input_dim, output_dim, degree, num_ctrlpts)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    for params, superformula_pts in zip(superformula_params, superformula_points):
        params = torch.tensor(params, dtype=torch.float32)
        superformula_pts = torch.tensor(superformula_pts, dtype=torch.float32)
        
        # Forward pass
        output = model(params)
        ctrlpts, weights, knotvector = output.split([num_ctrlpts*2, num_ctrlpts, len(knotvector)])
        ctrlpts = ctrlpts.view(num_ctrlpts, 2)
        
        # Calculate NURBS points
        nurbs_points = calculate_nurbs_points(ctrlpts.detach().numpy(), weights.detach().numpy(), knotvector.detach().numpy())
        
        # Calculate loss
        loss = calculate_distance(nurbs_points, superformula_pts.detach().numpy())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
