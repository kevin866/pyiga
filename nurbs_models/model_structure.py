import torch
import torch.nn as nn

class NURBSGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, degree, num_ctrlpts):
        super(NURBSGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.InstanceNorm1d(256),  # Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.3),      # Dropout for regularization
            
            nn.Linear(256, 512),
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1024),  # Increased layer size
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Reshape 1D input to 2D by adding batch dimension
        params = self.fc(x)
        return params
