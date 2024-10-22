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
        self.knots = self._compute_open_knot_vector()

        # Generate intervals similar to BezierLayer
        self.generate_intervals = nn.Sequential(
            nn.Linear(in_features, n_data_points - 1),
            nn.Softmax(dim=1),
            nn.Linear(n_data_points - 1, n_data_points - 1),
            nn.Softmax(dim=1),
            nn.Linear(n_data_points - 1, n_data_points - 1),
            nn.Softmax(dim=1),
            nn.ConstantPad1d([1, 0], 0)
        )

    def _compute_open_knot_vector(self):
        # Beginning knots, remain the same to keep the curve open.
        knots = [0.0] * (self.degree + 1)

        total_internal_knots = self.n_control_points - self.degree - 1
        for i in range(1, total_internal_knots):
            # Example of a non-linear (quadratic) distribution, modify this formula as needed.
            knot_value = (i / total_internal_knots) ** 3
            knots.append(knot_value)

        # Ending knots, remain the same to keep the curve open.
        knots += [1.0] * (self.degree + 1)

        return torch.tensor(knots, dtype=torch.float32)

    def basis_function(self, t, i, p):
        if p == 0:
            return ((self.knots[i] <= t) & (t < self.knots[min(i + 1, len(self.knots) - 1)])).float()
        else:
            A_denom = self.knots[min(i + p, len(self.knots) - 1)] - self.knots[i] + self.EPSILON
            B_denom = self.knots[min(i + p + 1, len(self.knots) - 1)] - self.knots[
                min(i + 1, len(self.knots) - 1)] + self.EPSILON
            A = ((t - self.knots[i]) / A_denom) * self.basis_function(t, i, p - 1)
            B = ((self.knots[min(i + p + 1, len(self.knots) - 1)] - t) / B_denom) * self.basis_function(t, min(i + 1,
                                                                                                               len(self.knots) - 1),
                                                                                                        p - 1)
            return A + B

    def forward(self, input: torch.Tensor, control_points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # self.knots = self._compute_open_knot_vector(control_points)
        # print('input, cp, w',input.shape,control_points.shape,weights.shape)
        intvls = self.generate_intervals(input)
        # print('intvl',intvls.shape)
        ub = torch.cumsum(intvls, -1).clamp(0, 1).unsqueeze(1)
        # print('ub',ub.shape)


        N = [self.basis_function(ub, j, self.degree) for j in range(self.n_control_points)]
        N = torch.stack(N, dim=-2)
        # print('N',N.shape)

        cp_w = control_points * weights
        # print(weights.shape)
        # print('cpw',cp_w.shape)
        cp_w_expanded = cp_w.unsqueeze(-1)
        # print('cp_w',cp_w_expanded.shape)
        dp = torch.sum(N * cp_w_expanded, dim=2)
        # print('DP',dp.shape)
        N_w = (N * weights.unsqueeze(-1)).sum(dim=2)
        # print('N_w', N_w.shape)

        dp = dp / (N_w + self.EPSILON)

        return dp, ub, intvls

# This is just the implementation of the efficient layer, and does not include a complete model or training routine.
# However, this should be more efficient than the original implementation.

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
        # generate fixed intervals
        ub = np.linspace(0.0, 1.0, input)
        intvls = np.cumsum(ub[::-1])[::-1] 
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