import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Custom NURBS Generator with valid knot vector layer
    
class NURBSGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, degree, num_ctrlpts):
        super(NURBSGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),  # New layer
            nn.ReLU(),
            nn.Linear(512, 256),  # New layer
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        params = self.fc(x)
        return params

