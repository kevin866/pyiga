import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
_eps = 1e-7

import torch
import torch.nn as nn

class NURBSLayer(nn.Module):
    def __init__(self, in_features: int, n_control_points: int, n_data_points: int) -> None:
        super(NURBSLayer, self).__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.degree = 3
        self.EPSILON = 1e-7
        zeros_part = torch.zeros(self.degree + 1)
        linspace_part = torch.linspace(0, 1, n_control_points - self.degree)
        ones_part = torch.ones(self.degree + 1)
        self.knots = torch.cat((zeros_part, linspace_part, ones_part))

        ub = np.linspace(0.0, 1.0, self.n_data_points)

        ub = torch.tensor(ub, dtype=torch.float32).to('cuda:0')

        xi = ub
        p = len(self.knots) - self.n_control_points - 1
        js = torch.zeros_like(xi).to('cuda:0')
        js = js.to(torch.int)
        for i in range(self.n_data_points):
            x = xi[i]  # Get current xi
            j = self.find_span(self.n_control_points, p, x, self.knots)  # Find knot span
            js[i] = j
        self.ub = ub
        self.js = js
        self.N = self.cal_N(xi, js)

    def find_span(self, n, p, u, U):
        """
        Find the knot span index for one variable u, NURBS-Book (algorithm A2.1).

        Parameters:
            n (int): Number of basis functions - 1
            p (int): Degree of the basis functions
            u (float): Evaluation point
            U (list): Knot vector (1D list)

        Returns:
            int: Index of knot span
        """
        if u == U[n ]:
            return n - 1

        low = p
        high = n + 1
        mid = (low + high) // 2

        while u < U[mid] or u >= U[mid + 1]:
            if u < U[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2

        return mid


    def basis_fun(self, i, u, p, U):
        """
        Evaluate nonzero basis functions, NURBS-Book (algorithm A2.2).

        Parameters:
            i (int): Current knot span
            u (float): Evaluation point
            p (int): Degree of the basis functions
            U (list): Knot vector (1D list)

        Returns:
            np.ndarray: Row vector (dimension p + 1) with values of the basis
                        functions N_(i-p) ... N_(i) at the evaluation point u
        """
        N = torch.zeros(p + 1)
        N[0] = 1.0
        left = torch.zeros(p + 1)
        right = torch.zeros(p + 1)

        for j in range(1, p + 1):

            left[j] = u - U[i + 1 - j]
            right[j] = U[i + j] - u
            saved = 0.0

            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r])
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp

            N[j] = saved

        return N

    import numpy as np


    def cal_N(self, ub, js):
        """
        Compute a NURBS curve based on control points and weights.

        Parameters:
            cp (np.ndarray): Control points (Nx2 array).
            w (np.ndarray): Weights (Nx1 array).

        Returns:
            np.ndarray: Computed NURBS curve points.
        """

        # Number of control points
        n = self.n_control_points

        # Generate the knot vector
        Xi = self.knots.to('cuda:0')
        # Number of knots
        k = len(Xi)

        # Order of basis
        p = k - n - 1

        Xi_store = ub
        N = torch.zeros((n, len(Xi_store)))  # Basis functions evaluated at each xi in Xi_store
        for i in range(len(Xi_store)):
            xi = Xi_store[i]  # Get current xi
            j = js[i]
            N_loc = self.basis_fun(j, xi, p, Xi)  # Get non-zero basis functions
            N[j - p:j + 1, i] = N_loc  # Store the non-zero basis functions
        N = N.to('cuda:0')
        return N

    def forward(self, input: torch.Tensor, control_points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

        w = weights.squeeze(1)
        W = torch.mm(w.to('cuda:0'), self.N)
        res = (w.unsqueeze(2) / W.unsqueeze(1))

        R = self.N * res


        dp = torch.bmm(control_points,R)

        return dp


class BSplineLayer(nn.Module):
    def __init__(self, in_features: int, n_control_points: int, n_data_points: int) -> None:
        super(BSplineLayer, self).__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.degree = 5
        self.EPSILON = 1e-7
        self.knots = self._compute_open_knot_vector()
        # print(self.degree)31

        # Generate intervals similar to BezierLayer
        self.generate_intervals = nn.Sequential(
            nn.Linear(in_features, n_data_points - 1),
            nn.Softmax(dim=1),
            nn.ConstantPad1d([1, 0], 0)
        )

    def _compute_open_knot_vector(self):
        knots = [0.0] * (self.degree + 1) + \
                [i / (self.n_control_points - self.degree) for i in range(1, self.n_control_points - self.degree)] + \
                [1.0] * (self.degree + 1)
        return torch.tensor(knots, dtype=torch.float32)

    def basis_function(self, t, i, p, cache=None):
        if cache is None:
            cache = {}

        t_key = '_'.join(map(str, t.detach().cpu().numpy().flatten()))


        key = (t_key, i, p)
        if key in cache:
            return cache[key]


        if p == 0:
            result = ((self.knots[i] <= t) & (t < self.knots[i + 1])).float()
        else:
            A = ((t - self.knots[i]) / (self.knots[i + p] - self.knots[i] + self.EPSILON)) * \
                self.basis_function(t, i, p - 1, cache)
            B = ((self.knots[i + p + 1] - t) / (self.knots[i + p + 1] - self.knots[i + 1] + self.EPSILON)) * \
                self.basis_function(t, i + 1, p - 1, cache)
            result = A + B

        cache[key] = result
        return result

    def forward(self, input: torch.Tensor, control_points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # print(input.shape,control_points.shape,weights.shape)torch.Size([512, 256]) torch.Size([512, 2, 32]) torch.Size([512, 1, 32])
        intvls = self.generate_intervals(input)
        ub = torch.cumsum(intvls, -1).clamp(0, 1).unsqueeze(1)

        N = []
        cache = {}
        for j in range(self.n_control_points):
            N_j = self.basis_function(ub, j, self.degree, cache)
            N.append(N_j)

        N = torch.stack(N, dim=-2)  # Shape: [16, 1, 32, 192]

        # Adjust the shapes of cp_w and then compute dp and N_w

        cp_w = control_points * weights
        cp_w_expanded = cp_w.unsqueeze(-1)  # Shape: [16, 2, 32, 1]
        # print(N.shape,cp_w_expanded.shape)torch.Size([16, 1, 32, 192]) torch.Size([16, 2, 32, 1])
        # print(cp_w_expanded.squeeze(-1).shape)torch.Size([16, 2, 32])
        dp = torch.sum(N * cp_w_expanded, dim=2)  # Shape: [16, 32, 192]
        N_w = (N * weights.unsqueeze(-1)).sum(dim=2)  # Shape: [16, 32, 192]
        # print(dp)
        dp = dp / (N_w + self.EPSILON)


        return dp, ub, intvls

        # torch.Size([16, 32, 192])


class BezierLayer(nn.Module):
    r"""Produces the data points on the Bezier curve, together with coefficients
        for regularization purposes.

    Args:
        in_features: size of each input sample.
        n_control_points: number of control points.
        n_data_points: number of data points to be sampled from the Bezier curve.

    Shape:
        - Input:
            - Input Features: `(N, H)` where H = in_features.
            - Control Points: `(N, D, CP)` where D stands for the dimension of Euclidean space,
            and CP is the number of control points. For 2D applications, D = 2.
            - Weights: `(N, 1, CP)` where CP is the number of control points.
        - Output:
            - Data Points: `(N, D, DP)` where D is the dimension and DP is the number of data points.
            - Parameter Variables: `(N, 1, DP)` where DP is the number of data points.
            - Intervals: `(N, DP)` where DP is the number of data points.
    """

    def __init__(self, in_features: int, n_control_points: int, n_data_points: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.generate_intervals = nn.Sequential(
            nn.Linear(in_features, n_data_points-1),
            nn.Softmax(dim=1),
            nn.ConstantPad1d([1,0], 0)
        )

    def forward(self, input: Tensor, control_points: Tensor, weights: Tensor) -> Tensor:
        cp, w = self._check_consistency(control_points, weights) # [N, d, n_cp], [N, 1, n_cp]
        bs, pv, intvls = self.generate_bernstein_polynomial(input) # [N, n_cp, n_dp]
        dp = (cp * w) @ bs / (w @ bs) # [N, d, n_dp]
        return dp, pv, intvls

    def _check_consistency(self, control_points: Tensor, weights: Tensor) -> Tensor:
        assert control_points.shape[-1] == self.n_control_points, 'The number of control points is not consistent.'
        assert weights.shape[-1] == self.n_control_points, 'The number of weights is not consistent.'
        assert weights.shape[1] == 1, 'There should be only one weight corresponding to each control point.'
        return control_points, weights

    def generate_bernstein_polynomial(self, input: Tensor) -> Tensor:
        intvls = self.generate_intervals(input) # [N, n_dp]
        pv = torch.cumsum(intvls, -1).clamp(0, 1).unsqueeze(1) # [N, 1, n_dp]
        pw1 = torch.arange(0., self.n_control_points, device=input.device).view(1, -1, 1) # [1, n_cp, 1]
        pw2 = torch.flip(pw1, (1,)) # [1, n_cp, 1]
        lbs = pw1 * torch.log(pv+_eps) + pw2 * torch.log(1-pv+_eps) \
            + torch.lgamma(torch.tensor(self.n_control_points, device=input.device)+_eps).view(1, -1, 1) \
            - torch.lgamma(pw1+1+_eps) - torch.lgamma(pw2+1+_eps) # [N, n_cp, n_dp]
        bs = torch.exp(lbs) # [N, n_cp, n_dp]
        # print(bs.shape)torch.Size([16, 32, 192])
        return bs, pv, intvls

    def extra_repr(self) -> str:
        return 'in_features={}, n_control_points={}, n_data_points={}'.format(
            self.in_features, self.n_control_points, self.n_data_points
        )

class _Combo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def forward(self, input):
        return self.model(input)

class LinearCombo(_Combo):
    r"""Regular fully connected layer combo.
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(alpha)
        )

class Deconv1DCombo(_Combo):
    r"""Regular deconvolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2
        ):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(alpha)
        )

class Deconv2DCombo(_Combo):
    r"""Regular deconvolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2
        ):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha)
        )

class Conv1DCombo(_Combo):
    r"""Regular convolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2, dropout=0.4
        ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout)
        )