import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_structure import NURBSGenerator
from pyiga import approx, bspline

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

def cal_c(r, a, L0):
    return np.sqrt(2)*np.sqrt(np.pi*(3+3*r**2+2*r)*a*L0)/(np.pi*(3+3*r**2+2*r))
def superformula(r, L0, n, a=0.5, d=1, num_res=100):
    theta = np.linspace(0, 2 * np.pi, num_res)
    c = cal_c(r,a,L0)
    result = c*((1+r)-d*(-1)**((n+2)/2)*(r-1)*np.cos(n*theta))
    x = result * np.cos(theta)
    y = result * np.sin(theta)
    return np.vstack((x, y)).T

# Generate a dataset of superformula points
superformula_params = [(round(r, 2), L0, n, (round(a, 2))) for r in np.arange(0.2, 0.9, 0.1).tolist() for L0 in np.arange(15, 35, 5).tolist()
                       for n in np.arange(2, 12, 2).tolist() for a in np.arange(0.2, 0.7, 0.1).tolist()]
superformula_points = [superformula(*params) for params in superformula_params]

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
# print(dir(knotvector))
# print(knotvector.kv)
# print(knotvector.p)
# # Function to hook gradients
# def print_grad(grad):
#     print('Gradient:', grad)

# # Attach hooks to model parameters
# for param in model.parameters():
#     param.register_hook(print_grad)


# Training loop
num_epochs = 300
for epoch in range(num_epochs):
    tot_loss = 0
    for params, superformula_pts in zip(superformula_params, superformula_points):
        params = torch.tensor(params, dtype=torch.float32)
        superformula_pts = torch.tensor(superformula_pts, dtype=torch.float32)
        
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
        
        # # Print gradients
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f'Gradients for {name} at Epoch {epoch}: {param.grad}')
        #     else:
        #         print(f'No gradients for {name} at Epoch {epoch}')
        
        optimizer.step()
        
        tot_loss += loss.item()
        
    avg_loss = tot_loss / len(superformula_params)
    print(f'Epoch [{epoch}/{num_epochs}], Avg Loss: {avg_loss:.4f}')
    
    # if epoch % 10 == 0:
    #     print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
    #     # plot_control_points(ctrlpts, epoch)
# Save the model
torch.save(model.state_dict(), 'nurbs_generator_50.pth')
print("Model saved successfully.")


