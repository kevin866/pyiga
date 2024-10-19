import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_model_structure import SuperformulaDiffusionModel
def diffusion_loss(pred_control_points, pred_weights, true_control_points, true_weights):
    # MSE loss for control points and weights
    loss_control_points = F.mse_loss(pred_control_points, true_control_points)
    loss_weights = F.mse_loss(pred_weights, true_weights)
    return loss_control_points + loss_weights


def train_diffusion_model(model, data_loader, optimizer, epochs=10, timesteps=100):
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (cartesian_points, true_control_points, true_weights) in enumerate(data_loader):
            optimizer.zero_grad()
            
            # Random timestep t for each batch
            t = torch.randint(0, timesteps, (1,), device=cartesian_points.device).item()
            
            # Forward diffusion: add noise to true control points and weights
            noisy_control_points, noisy_weights, _ = model(cartesian_points, t)
            
            # Reverse diffusion: predict the denoised control points and weights
            pred_control_points, pred_weights = model.reverse_process(noisy_control_points, noisy_weights, t)
            
            # Compute loss
            loss = diffusion_loss(pred_control_points, pred_weights, true_control_points, true_weights)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        
        
from torch.utils.data import DataLoader, Dataset

class SuperformulaDataset(Dataset):
    def __init__(self, cartesian_points, control_points, weights):
        self.cartesian_points = cartesian_points
        self.control_points = control_points
        self.weights = weights

    def __len__(self):
        return len(self.cartesian_points)

    def __getitem__(self, idx):
        return self.cartesian_points[idx], self.control_points[idx], self.weights[idx]

# Assuming you have loaded your dataset
cartesian_points = torch.randn(12, 200)  # 12 samples, 100 points * 2 coordinates
control_points = torch.randn(12, 10, 2)  # Corresponding control points
weights = torch.randn(12, 10)  # Corresponding weights

dataset = SuperformulaDataset(cartesian_points, control_points, weights)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training
model = SuperformulaDiffusionModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_diffusion_model(model, data_loader, optimizer)
