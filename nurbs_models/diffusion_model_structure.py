import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperformulaToNURBS(nn.Module):
    def __init__(self, input_dim=200, latent_dim=32, num_control_points=10):
        super(SuperformulaToNURBS, self).__init__()

        # Encoder: MLP to compress 100x2 Cartesian points to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder: MLP to expand latent space to NURBS control points, weights, and knot vector
        self.decoder_control_points = nn.Linear(latent_dim, num_control_points * 2)  # Control points (x, y)
        self.decoder_weights = nn.Linear(latent_dim, num_control_points)  # Weights
        self.decoder_knots = nn.Linear(latent_dim, num_control_points + 1)  # Knot vector (for a uniform B-spline)

    def forward(self, x):
        # Encode input to latent space
        latent = self.encoder(x)
        
        # Decode latent space into control points, weights, and knot vectors
        control_points = self.decoder_control_points(latent).view(-1, 10, 2)  # 10 control points with 2D coords
        weights = self.decoder_weights(latent)  # 10 weights
        knots = self.decoder_knots(latent)  # Knot vector of size 11 (num_control_points + 1)

        return control_points, weights, knots

# Example input: 100x2 Cartesian coordinates
example_input = torch.randn(1, 200)  # (Batch size 1, 100 points * 2 coordinates)
model = SuperformulaToNURBS()
control_points, weights, knots = model(example_input)

print("Control Points:", control_points)
print("Weights:", weights)
print("Knot Vector:", knots)




import numpy as np

class DiffusionScheduler:
    def __init__(self, timesteps=100):
        self.timesteps = timesteps
        self.betas = self.linear_beta_schedule(timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        
    def linear_beta_schedule(self, timesteps):
        return np.linspace(0.0001, 0.02, timesteps)
    
    def get_alpha_cumprod(self, t):
        return self.alpha_cumprod[t]

# Example usage
scheduler = DiffusionScheduler(timesteps=100)



import torch
import torch.nn.functional as F

class SuperformulaDiffusionModel(nn.Module):
    def __init__(self, timesteps=100, input_dim=200, latent_dim=32, num_control_points=10):
        super(SuperformulaDiffusionModel, self).__init__()
        
        # Encoder-decoder as before
        self.model = SuperformulaToNURBS(input_dim, latent_dim, num_control_points)
        self.timesteps = timesteps  # Number of diffusion steps
        self.scheduler = DiffusionScheduler(timesteps)

    def forward(self, x, t):
        # Generate initial NURBS parameters
        control_points, weights, knots = self.model(x)
        
        # Get the cumulative product of alpha for current timestep
        alpha_cumprod_t = torch.tensor(self.scheduler.get_alpha_cumprod(t), device=x.device)
        
        # Add noise to the control points and weights
        noise_control_points = torch.randn_like(control_points) * torch.sqrt(1 - alpha_cumprod_t)
        noisy_control_points = control_points * torch.sqrt(alpha_cumprod_t) + noise_control_points
        
        noise_weights = torch.randn_like(weights) * torch.sqrt(1 - alpha_cumprod_t)
        noisy_weights = weights * torch.sqrt(alpha_cumprod_t) + noise_weights
        
        return noisy_control_points, noisy_weights, knots

    def reverse_process(self, noisy_control_points, noisy_weights, t):
        # Predict the original control points and weights by removing noise
        alpha_cumprod_t = torch.tensor(self.scheduler.get_alpha_cumprod(t), device=noisy_control_points.device)
        
        # Apply the reverse of the noise step
        denoised_control_points = noisy_control_points / torch.sqrt(alpha_cumprod_t)
        denoised_weights = noisy_weights / torch.sqrt(alpha_cumprod_t)
        
        return denoised_control_points, denoised_weights
