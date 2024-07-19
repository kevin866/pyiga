import numpy
import torch
import torch.nn as nn
import torch.optim as optim

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiffusionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Define the forward diffusion process (adding noise)
def forward_diffusion(x, noise_level):
    noise = torch.randn_like(x) * noise_level
    return x + noise

# Training loop
def train_diffusion_model(model, data, epochs, noise_levels, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for x in data:
            for noise_level in noise_levels:
                noisy_x = forward_diffusion(x, noise_level)
                optimizer.zero_grad()
                denoised_x = model(noisy_x)
                loss = criterion(denoised_x, x)
                loss.backward()
                optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Example usage
input_dim = 10  # Example dimension
hidden_dim = 128
model = DiffusionModel(input_dim, hidden_dim)
data = [torch.randn(input_dim) for _ in range(1000)]  # Example data
noise_levels = [0.1, 0.2, 0.5]  # Example noise levels
train_diffusion_model(model, data, epochs=100, noise_levels=noise_levels)
