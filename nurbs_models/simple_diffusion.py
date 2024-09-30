import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the model (a simple UNet-like architecture for diffusion)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.out_conv = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = self.out_conv(x)
        return x

# Schedule for diffusion (linear schedule)
def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)


# Add noise to the images
def add_noise(img, t, noise):
    # Reshape t to be broadcastable to the image shape
    t = t[:, None, None, None]
    return torch.sqrt(1 - t) * img + torch.sqrt(t) * noise

# Training loop for the diffusion model
def train_diffusion_model(model, dataloader, epochs=10, timesteps=1000, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    betas = linear_beta_schedule(timesteps).to(device)
    
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            t = torch.randint(0, timesteps, (data.size(0),), device=device).float()
            noise = torch.randn_like(data).to(device)
            noisy_data = add_noise(data, betas[t.long()], noise)

            optimizer.zero_grad()
            output = model(noisy_data, t)
            loss = F.mse_loss(output, noise)  # Loss between predicted and actual noise
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Set up data (e.g., MNIST) and transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize and train the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleUNet().to(device)

# Train the diffusion model
train_diffusion_model(model, trainloader, epochs=5)
