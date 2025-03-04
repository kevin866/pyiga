import numpy as np
import torch
import math
from torch.utils.data import DataLoader, Dataset
# from .interpolation import interpolate
from .interpolation import interpolate
class UIUCAirfoilDataset(Dataset):
    r"""UIUC Airfoil Dataset. 

    Args:
        np_data: Dataset of numpy array format.
        N: Number of data points.
        k: Degree of spline.
        D: Shifting constant. The higher the more uniform the data points are.
    Shape:
        Output: `(N, D, DP)` where D is the dimension of each point and DP is the number of data points.
    """

    def __init__(self, np_data, N=192, k=3, D=20, device='cpu'):
        super().__init__()
        self.device = device
        self.airfoils = torch.tensor(
            np_data.transpose((0, 2, 1)), 
            device=device, dtype=torch.float
        )
        if (N, k, D) == (192, 3, 20):
            self.N = N; self.k = k; self.D = D
        else:
            self.refresh(N, k, D)

    def refresh(self, N, k, D):
        self.N = N; self.k = k; self.D = D
        self.airfoils = torch.tensor(
            [interpolate(airfoil, N, k, D) for airfoil in self.airfoils],
            device=device, dtype=torch.float
        )
    
    def __getitem__(self, index):
        return self.airfoils[index]
    
    def __len__(self):
        return len(self.airfoils)
    
    def __str__(self):
        return '<UIUC Airfoil Dataset (size={}, resolution={}, spline degree={}, uniformity={})>'.format(
            self.__len__(), self.N, self.k, self.D
        )

class NoiseGenerator:
    def __init__(self, batch: int, sizes: list=[4, 10], noise_type: list=['u', 'n'], output_prob: bool=False, device='cpu'):
        super().__init__()
        self.batch = batch
        self.sizes = sizes
        self.noise_type = noise_type
        self.output_prob = output_prob
        self.device = device
        
    def __call__(self):
        noises = []
        for size, n_type in zip(self.sizes, self.noise_type):
            if n_type == 'u':
                noises.append(torch.rand(self.batch, size))
            elif n_type == 'n':
                noises.append(torch.randn(self.batch, size))
        if self.output_prob:
            return torch.cat(noises, dim=1).to(self.device), self._cal_prob(noises).to(self.device)
        else:
            return torch.cat(noises, dim=1).to(self.device)
    
    def _cal_prob(self, noises):
        n_noise = torch.cat([noises[i] for i, n_t in enumerate(self.noise_type) if n_t == 'n'], dim=1)
        d = n_noise.shape[1]
        return (2 * math.pi) ** (-d / 2) * torch.exp(-torch.norm(n_noise, dim=1, keepdim=True) / 2)