import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from . import SCI2layers as layers
from .utils import first_element
from thop import profile

import torch.nn as nn
import os
import torch
import torch.nn as nn
import json

import torch.nn as nn

class AdvancedMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, layer_width: list, activation=nn.ReLU, dropout_rate=0,
                 use_residual=False, n_heads=1):
        super(AdvancedMLP, self).__init__()
        self.use_residual = use_residual
        layers = []
        current_in_features = in_features

        for width in layer_width:
            layers.append(nn.Linear(current_in_features, width))
            layers.append(nn.LayerNorm(width))  # Replacing BatchNorm with LayerNorm
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
            current_in_features = width

        # MultiheadAttention layer
        self.self_attention = nn.MultiheadAttention(current_in_features, n_heads)

        # Final linear layer
        layers.append(nn.Linear(current_in_features, out_features))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        for layer in self.model:
            if isinstance(layer, nn.Linear) and self.use_residual:
                x = layer(x) + identity
            else:
                x = layer(x)

            if isinstance(layer, nn.Dropout):
                x, _ = self.self_attention(x, x, x)

                if self.use_residual:
                    x = x + identity

            identity = x

        return x


class TransformerLikeMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, layer_width: list, activation=nn.LeakyReLU, dropout_rate=0,
                 n_heads=1):
        super(TransformerLikeMLP, self).__init__()
        # print(layer_width)
        self.self_attention = nn.MultiheadAttention(in_features, n_heads)
        self.norm1 = nn.LayerNorm(in_features)
        self.dropout1 = nn.Dropout(dropout_rate)

        layers = []
        current_in_features = in_features
        for width in layer_width:
            layers.append(nn.Linear(current_in_features, width))
            layers.append(activation())
            layers.append(
                nn.Linear(width, in_features))  # Ensuring the output matches the input size for residual connections
            current_in_features = in_features  # Reset to match input size for residual connections

        self.mlp = nn.Sequential(*layers)
        self.norm2 = nn.LayerNorm(in_features)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.final_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Attention sub-layer
        x1 = self.norm1(x)
        x2, _ = self.self_attention(x1, x1, x1)
        x = x + self.dropout1(x2)

        # MLP sub-layer
        x1 = self.norm2(x)
        x2 = self.mlp(x1)
        x = x + self.dropout2(x2)

        # Final linear layer
        x = self.final_linear(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, layer_width: list, activation=nn.ReLU, norm_layer=None):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        current_in_features = in_features
        for width in layer_width:
            layers.append(nn.Linear(current_in_features, width))
            if norm_layer is not None:
                layers.append(norm_layer(width))
            layers.append(activation())
            current_in_features = width

        # Output layer
        layers.append(nn.Linear(current_in_features, out_features))

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Conv1DNetwork(nn.Module):
    """The 1D convolutional front end.

    Args:
        in_channels: The number of channels of each input feature.
        in_features: The number of input features.
        conv_channels: The number of channels of each conv1d layer.

    Shape:
        - Input: `(N, C, H_in)` where C = in_channel and H_in = in_features.
        - Output: `(N, H_out)` where H_out is calculated based on in_features.
    """
    def __init__(
        self, in_channels: int, in_features: int, conv_channels: list, 
        combo = layers.Conv1DCombo
        ):
        super().__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.m_features = self._calculate_m_features(conv_channels)
        self.conv = self._build_conv(conv_channels, combo)

    def forward(self, input):
        return self.conv(input)
    
    def _calculate_m_features(self, channels):
        n_l = len(channels)
        m_features = self.in_features // (2 ** n_l) * channels[-1]
        return m_features

    def _build_conv(self, channels, combo):
        conv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(zip(
            [self.in_channels] + channels[:-1], channels
            )):
            conv.add_module(
                str(idx), combo(
                    in_chnl, out_chnl, 
                    kernel_size=4, stride=2, padding=1
                    )
                )
            conv.add_module(str(idx+1), nn.Flatten())
        return conv

class CPWGenerator(nn.Module):
    """Generate given number of control points and weights for Bezier Layer.

    Args:
        in_features: The number of input features.
        n_control_points: The number of control point and weights to be output.
            Should be even.
        dense_layers: The widths of the hidden layers of the MLP connecting
            input features and deconvolutional layers.
        deconv_channels: The number of channels deconvolutional layers have.

    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output:
            - Control Points: `(N, 2, H_out)` where H_out = n_control_points.
            - Weights: `(N, 1, H_out)` where H_out = n_control_points.
    """
    def __init__(
        self, in_features: int, n_control_points: int,
        dense_layers: list = [1024,],
        deconv_channels: list = [ 96],
        ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points

        self.in_chnl, self.in_width = self._calculate_parameters(n_control_points, deconv_channels)

        print(in_features, self.in_chnl,self.in_width, dense_layers)

        self.dense = TransformerLikeMLP(in_features, self.in_chnl*self.in_width, dense_layers)
        # self.deconv = self._build_deconv(deconv_channels)
        self.cp_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 2, 1), nn.Tanh())
        self.w_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 1, 1), nn.Sigmoid())
        self.view_channels = deconv_channels[-1]
        self.view_control_points = n_control_points // (2 ** (len(deconv_channels) - 1))

    def forward(self, input):

        # print('input',input.shape)
        x = self.dense(input)
        # print('x1',x.shape)
        x = x.view(-1, self.view_channels, self.view_control_points)
        # print('x2',x.shape)
        # x = self.deconv(x)
        # print('x3',x.shape)
        cp = self.cp_gen(x)
        w = self.w_gen(x)
        return cp, w

    def _calculate_parameters(self, n_control_points, channels):
        print('n_control_points,channels',n_control_points,channels)
        n_l = len(channels) - 1
        in_chnl = channels[0]
        in_width = n_control_points // (2 ** n_l)
        print('in_chnl, in_width',in_chnl, in_width)
        assert in_width >= 4, 'Too many deconvolutional layers ({}) for the {} control points.'\
            .format(n_l, self.n_control_points)
        return in_chnl, in_width

    # def _build_deconv(self, channels):
    #     deconv = nn.Sequential()
    #     for idx, (in_chnl, out_chnl) in enumerate(zip(channels[:-1], channels[1:])):
    #         deconv.add_module(
    #             str(idx), layers.Deconv1DCombo(
    #                 in_chnl, out_chnl,
    #                 kernel_size=4, stride=2, padding=1
    #                 )
    #             )
    #     return deconv
import pickle
class BezierGenerator(nn.Module):

    def __init__(self, in_features: int, n_control_points: int, n_data_points: int,
                 m_features: int = 256, feature_gen_layers: list = [1024, ],
                 dense_layers: list = [1024, ], deconv_channels: list = [768, 384, 192, 96]):
        super().__init__()
        # ... [rest of your initialization code] ...

        # Print parameters in self.feature_generator


        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

        self.feature_generator = TransformerLikeMLP(in_features, m_features, feature_gen_layers)
        self.cpw_generator = CPWGenerator(in_features, n_control_points, dense_layers, deconv_channels)
        print('M',m_features,n_control_points,n_data_points)
        self.B = layers.BSLayer(m_features, n_control_points, n_data_points)
        self.save_count = 0
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Print number of trainable parameters in self.feature_generator
        print("Trainable parameters in self.feature_generator:", count_parameters(self.feature_generator))

        # Print number of trainable parameters in self.cpw_generator
        print("Trainable parameters in self.cpw_generator:", count_parameters(self.cpw_generator))

        # Print number of trainable parameters in self.B
        print("Trainable parameters in self.B:", count_parameters(self.B))


    def forward(self, input):

        features = self.feature_generator(input)
        cp, w = self.cpw_generator(input)
        # print(cp.shape,w.shape)torch.Size([500, 2, 32]) torch.Size([500, 1, 32])

        # self.save_cp_w(cp, w)
        # print(cp,w)

        dp, ub, intvls = self.B(features, cp)
        # flops, params = profile(self.B, inputs=(features, cp, w))
        # print(f"FLOPs: {flops}")
        # print(f"Parameters: {params}")
        # FLOPs: 25327616.0
        # Parameters: 49087.0
        return dp, cp,  ub, intvls
    
    def extra_repr(self) -> str:
        return 'in_features={}, n_control_points={}, n_data_points={}'.format(
            self.in_features, self.n_control_points, self.n_data_points
        )

    def save_cp_w(self, cp, w, separator='|'):
        """Saves control points and weights to a readable file, separated by a symbol.

        Args:
            cp: Control points tensor.
            w: Weights tensor.
            separator: Symbol used to separate values.
        """
        filename = 'cpw.json'
        # print("Attempting to save to:", filename)

        # Convert tensors to lists and then to strings separated by the specified symbol
        cp_str = separator.join(map(str, cp.detach().cpu().tolist()))
        w_str = separator.join(map(str, w.detach().cpu().tolist()))

        # Increment save count and use it as a tag
        self.save_count += 1
        save_tag = f"save_{self.save_count}"

        # Prepare data to save
        data_to_save = {save_tag: {'control_points': cp_str, 'weights': w_str}}

        # Check if file exists to append or create new
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
            existing_data.update(data_to_save)
        else:
            existing_data = data_to_save

        # Save or update the JSON file
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)

        # print(f"Data saved under tag '{save_tag}' in '{filename}'.")



    # def save_cp_w(self, cp, w, filename):
    #     """Saves control points and weights to a file.
    #
    #     Args:
    #         cp: Control points tensor.
    #         w: Weights tensor.
    #         filename: The filename to save to.
    #     """
    #     with open(filename, 'wb') as f:
    #         pickle.dump({'control_points': cp.cpu().numpy(), 'weights': w.cpu().numpy()}, f)


class Critics1D(Conv1DNetwork):
    """Regular discriminator for GANs.

    Args: 
        in_channels: The number of channels of each input feature.
        in_features: The number of input features.
        n_critics: The number of critics.
        conv_channels: The number of channels of each conv1d layer.
        crt_layers: The widths of fully connected hidden layers of critics.

    Shape:
        - Input: `(N, C, H)` where C = in_channel and H = in_features.
        - Output: `(N, NC, 2)` where NC is the number of critics.
    """
    def __init__(
        self, in_channels: int, in_features: int, n_critics: int, 
        conv_channels: list, crt_layers: list
        ):
        super().__init__(in_channels, in_features, conv_channels=conv_channels)
        self.n_critics = n_critics
        self.critics = nn.Sequential(
            MLP(self.m_features, crt_layers[-1], crt_layers[:-1]),
            nn.Linear(crt_layers[-1], n_critics)
        )

    def forward(self, input):
        x = super().forward(input)
        critics = self.critics(x)
        return critics

class InfoDiscriminator1D(Critics1D):
    """Discriminator for GANs equiped with mutual information maximization.

    Args: 
        in_channels: The number of channels of each input feature.
        in_features: The number of input features.
        n_critics: The number of critics.
        latent_dim: The number of latent variables
        conv_channels: The number of channels of each conv1d layer.
        crt_layers: The widths of fully connected hidden layers of critics.
        pred_layers: The widths of fully connected hidden layers of latent code predictor.

    Shape:
        - Input: `(N, C, H)` where C = in_channel and H = in_features.
        - Output: 
            - Critics: `(N, NC)` where NC = n_critics.
            - Latent Code: `(N, NL, 2)` where NL = latent_dim.
    """
    def __init__(
        self, in_channels: int, in_features: int, n_critics: int, latent_dim: int,
        conv_channels: list = [64, 64*2, 64*4, 64*8, 64*16, 64*32],
        crt_layers: list = [1024,],
        pred_layers: list = [512,]
        ):
        super().__init__(in_channels, in_features, n_critics, conv_channels, crt_layers)
        self.latent_dim = latent_dim
        self.latent_predictor = nn.Sequential(
            MLP(self.m_features, pred_layers[-1], pred_layers[:-1]),
            nn.Linear(pred_layers[-1], latent_dim * 2)
        )
    
    # def forward(self, input):
    #     x = self.conv(input)
    #     critics = self.critics(x)
    #     latent_code = self.latent_predictor(x).reshape([-1, self.latent_dim, 2])
    #     # print(latent_code.shape)
    #     return critics, latent_code


class AdaptiveCost(nn.Module):
    def __init__(self, feature_gen, p=2):
        super().__init__()
        self.feature_gen = feature_gen
        self.p = p
    
    def forward(self, x, y):
        ft_x = first_element(self.feature_gen(x))
        ft_y = first_element(self.feature_gen(y))
        return torch.cdist(ft_x, ft_y, p=self.p)