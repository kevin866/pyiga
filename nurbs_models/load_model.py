from model_structure import NURBSGenerator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pyiga.geometry import *
from pyiga import approx, bspline
degree = 3
n_kv = 9
# knotvector = np.concatenate(([0] * (degree + 1), np.linspace(0, 1, num_ctrlpts - degree), [1] * (degree + 1)))
knotvector = bspline.make_knots(1, 0.0, 1.0, n_kv)
num_ctrlpts = knotvector.numdofs
input_dim = 4  # Superformula parameters
output_dim = num_ctrlpts * 2 + num_ctrlpts  # Control points and weights

# Initialize the model structure
model = NURBSGenerator(input_dim, output_dim, degree, num_ctrlpts)

# Load the saved model weights
model.load_state_dict(torch.load('nurbs_generator.pth'))
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully.")

# Use the model to generate NURBS control points and weights from new superformula parameters
new_params = torch.tensor([0.5, 20, 4, 0.3], dtype=torch.float32)
output = model(new_params)
ctrlpts, weights = output.split([num_ctrlpts*2, num_ctrlpts])
ctrlpts = ctrlpts.view(num_ctrlpts, 2)
weights = weights.view(num_ctrlpts)



# Calculate NURBS points using custom function
# nurbs_points = calculate_nurbs_points(ctrlpts, weights, knotvector, degree)
from pyiga.geometry import *
from pyiga import approx, bspline
from pyiga import bspline, assemble, vform, geometry, vis, solvers
ctrlpts_np = ctrlpts.detach().numpy()
weights_np = weights.detach().numpy()

nurbs = NurbsFunc((knotvector,), ctrlpts_np.copy(), weights=weights_np)
vis.plot_geo(nurbs,res=500, linewidth=None, color='black')
# plt.axis('equal');
