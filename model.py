import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# class OccupancyMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, pos_encoding):
#         super(OccupancyMLP, self).__init__()
#         self.pos_encoding = pos_encoding
#         encoded_dim = self.pos_encoding.get_output_dim(input_dim)
#         layers = [nn.Linear(encoded_dim, hidden_dim), nn.ReLU()]
#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#         layers.append(nn.Linear(hidden_dim, output_dim))
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.pos_encoding.encode(x)
#         return self.model(x)
    


class OccupancyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(OccupancyMLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def siren_init(layer, w0=30):
    with torch.no_grad():
        if isinstance(layer, nn.Linear):
            num_input = layer.weight.size(-1)
            # First layer
            if num_input == 3:
                layer.weight.uniform_(-1 / num_input, 1 / num_input)
            else:
                layer.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)

    
class OccupancySIREN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(OccupancySIREN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        layers = [nn.Linear(input_dim, hidden_dim), Sine()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Sine())
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.model.apply(siren_init)

    def forward(self, x):
        return self.model(x)
    
# class PositionalEncoding:
#     def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True):
#         self.num_encoding_functions = num_encoding_functions
#         self.include_input = include_input
#         self.log_sampling = log_sampling

#     def encode(self, inputs):
#         encoding = [inputs] if self.include_input else []
#         for i in range(self.num_encoding_functions):
#             for fn in [torch.sin, torch.cos]:
#                 if self.log_sampling:
#                     encoding.append(fn(2.**i * inputs))
#                 else:
#                     encoding.append(fn(2.**i * inputs))
#         return torch.cat(encoding, dim=-1)

#     def get_output_dim(self, input_dim):
#         multiplier = 1 + 2 * self.num_encoding_functions if self.include_input else 2 * self.num_encoding_functions
#         return input_dim * multiplier

def siren_init(layer, w0=30):
    with torch.no_grad():
        if isinstance(layer, nn.Linear):
            num_input = layer.weight.size(-1)
            if num_input == 3:
                layer.weight.uniform_(-1 / num_input, 1 / num_input)
            else:
                layer.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)
            if layer.bias is not None:
                layer.bias.zero_()

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class OccupancySIREN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, pos_encoding):
        super(OccupancySIREN, self).__init__()
        self.pos_encoding = pos_encoding
        encoded_dim = self.pos_encoding.get_output_dim(input_dim)
        layers = [nn.Linear(encoded_dim, hidden_dim), Sine()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Sine())
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.model.apply(siren_init)

    def forward(self, x):
        x = self.pos_encoding.encode(x)
        return self.model(x)

