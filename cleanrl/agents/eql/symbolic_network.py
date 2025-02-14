"""Contains the symbolic regression neural network architecture."""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import activation
from . import functions
from .functions import separate_single_double_funcs
import sympy as sp
from .pretty_print import filter_expr2, round_expr


class SymbolicLayer(nn.Module):
    """Neural network layer for symbolic regression where activation functions correspond to primitive functions.
    Can take multi-input activation functions (like multiplication)"""
    def __init__(self, funcs, in_dim):
        """
        funcs: List of activation functions, using utils.functions
        initial_weight: (Optional) Initial value for weight matrix
        variable: Boolean of whether initial_weight is a variable or not
        init_stddev: (Optional) if initial_weight isn't passed in, this is standard deviation of initial weight
        """
        super().__init__()

        if funcs is None:
            funcs = functions.default_func
        self.n_funcs = len(funcs)                       # Number of activation functions (and number of layer outputs)
        self.funcs = [func.torch for func in funcs]     # Convert functions to list of PyTorch functions
        self.n_double = functions.count_double(funcs)   # Number of activation functions that take 2 inputs
        self.n_single = self.n_funcs - self.n_double    # Number of activation functions that take 1 input
        self.func_in_dim = self.n_funcs + self.n_double
        self.transform = nn.Linear(in_dim, self.func_in_dim)

    def forward(self, x):  # used to be __call__
        """Multiply by weight matrix and apply activation units"""

        g = self.transform(x)
        self.output = []

        in_i = 0    # input index
        out_i = 0   # output index
        # Apply functions with only a single input
        while out_i < self.n_single:
            self.output.append(self.funcs[out_i](g[:, in_i]))
            in_i += 1
            out_i += 1
        # Apply functions that take 2 inputs and produce 1 output
        while out_i < self.n_funcs:
            self.output.append(self.funcs[out_i](g[:, in_i], g[:, in_i+1]))
            in_i += 2
            out_i += 1

        self.output = torch.stack(self.output, dim=1)

        return self.output

    def get_weights(self):
        """Return list of weight matrices"""
        # First part is iterating over hidden weights. Then append the output weight.
        return [i.cpu().detach().numpy() for i in self.get_weights_tensor()]

    def get_weights_tensor(self):
        """Return list of weight matrices as tensors"""
        return [i for i in self.parameters()]


class SymbolicNet(nn.Module):
    """Symbolic regression network with multiple layers. Produces one output."""
    def __init__(self, symbolic_depth, funcs, in_dim, out_dim):
        super(SymbolicNet, self).__init__()

        self.depth = symbolic_depth     # Number of hidden layers
        self.funcs = funcs

        layers = []
        for i in range(self.depth):
            layers.append(SymbolicLayer(funcs=self.funcs, in_dim=in_dim))
            in_dim = layers[-1].n_funcs
        layers.append(nn.Linear(in_dim, out_dim))
            # Initialize weights for last layer (without activation functions)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

    def get_weights(self):
        """Return list of weight matrices"""
        # First part is iterating over hidden weights. Then append the output weight.
        return [i.cpu().detach().numpy() for i in self.get_weights_tensor()]

    def get_weights_tensor(self):
        """Return list of weight matrices as tensors"""
        return [i for i in self.parameters()]

class SymbolicNetSimplified(nn.Module):
    """Simplified SymbolicNet with one layer"""
    def __init__(self, funcs, in_dim, out_dim, use_bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.single_funcs, self.double_funcs = separate_single_double_funcs(funcs)
        self.single_funcs_sympy, self.double_funcs_sympy = separate_single_double_funcs(funcs, return_func_type="sp")

        self.use_bias = use_bias
        
        # calculate input dims to linear layer
        linear_in_dim = 0
        if use_bias:
            linear_in_dim += 1
        # add single
        linear_in_dim += len(self.single_funcs) * in_dim
        # add double as cross-prod
        linear_in_dim += len(self.double_funcs) * in_dim ** 2
        
        self.linear = nn.Linear(linear_in_dim, out_dim)


    def forward(self, x):
        # Input tensor shape: (batch_size, in_dim)
        batch_size = x.size(0)

        # Start building the input to the linear layer
        linear_input = []

        # Step 1: Add bias (if enabled)
        if self.use_bias:
            # Add a column of ones for the bias
            linear_input.append(torch.ones((batch_size, 1), device=x.device))

        # Step 2: Apply single-input activation functions
        for func in self.single_funcs:
            linear_input.append(func(x))  # Apply func element-wise to x

        # Step 3: Apply double-input activation functions
        if len(self.double_funcs) > 0:
            # Compute pairwise combinations for double-input functions
            x_i = x.unsqueeze(2)  # Shape: (batch_size, in_dim, 1)
            x_j = x.unsqueeze(1)  # Shape: (batch_size, 1, in_dim)

            for func in self.double_funcs:
                pairwise_results = func(x_i, x_j)
                linear_input.append(pairwise_results.view(batch_size, -1))

        # Step 4: Concatenate all parts of the input
        linear_input = torch.cat(linear_input, dim=1)  # Final shape: (batch_size, linear_in_dim)

        # Step 5: Pass through the linear layer
        output = self.linear(linear_input)  # Shape: (batch_size, out_dim)

        return output

    def get_weights(self):
        """Return list of weight matrices"""
        # First part is iterating over hidden weights. Then append the output weight.
        return [i.cpu().detach().numpy() for i in self.get_weights_tensor()]

    def get_weights_tensor(self):
        """Return list of weight matrices as tensors"""
        return [i for i in self.parameters()]

    def pretty_print(self, variable_names, output_names, threshold=0.01, accuracy=0.01):
        """Print the symbolic computation as equations using SymPy."""
        
        assert len(variable_names) == self.in_dim, "Variable names must match input dimension"
        
        symbols = sp.symbols(variable_names)
        expr_list = []

        # Collect terms for the linear input
        linear_input = []
        
        if self.use_bias:
            linear_input.append(1)  # Bias term
        
        # Single functions
        for func in self.single_funcs_sympy:
            linear_input.extend([func(var) for var in symbols])
        
        # Double functions
        for func in self.double_funcs_sympy:
            for i in range(self.in_dim):
                for j in range(self.in_dim):
                    linear_input.append(func(symbols[i], symbols[j]))
        
        # Convert to SymPy expressions
        weights = self.get_weights_tensor()[0].detach().cpu().numpy()
        for out_idx in range(self.out_dim):
            equation = sum(w * term for w, term in zip(weights[out_idx], linear_input))
            expr = sp.simplify(equation)
            expr = filter_expr2(expr, threshold)
            expr = round_expr(expr, accuracy)
            # Incorporate the output name into the expression string
            final_expr = f"logits_{output_names[out_idx]} = {str(expr)}"
            expr_list.append(final_expr)
            # Print the equation
            print(f"Output {out_idx + 1}: {final_expr}")
        return expr_list
        

class SymbolicLayerL0(SymbolicLayer):
    def __init__(self, in_dim=None, funcs=None, initial_weight=None, init_stddev=0.1,
                 bias=False, droprate_init=0.5, lamba=1.,
                 beta=2 / 3, gamma=-0.1, zeta=1.1, epsilon=1e-6):
        super().__init__(in_dim=in_dim, funcs=funcs, initial_weight=initial_weight, init_stddev=init_stddev)

        self.droprate_init = droprate_init if droprate_init != 0 else 0.5
        self.use_bias = bias
        self.lamba = lamba
        self.bias = None
        self.in_dim = in_dim
        self.eps = None

        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.epsilon = epsilon

        if self.use_bias:
            self.bias = nn.Parameter(0.1 * torch.ones((1, self.out_dim)))
        self.qz_log_alpha = nn.Parameter(torch.normal(mean=np.log(1 - self.droprate_init) - np.log(self.droprate_init),
                                                      std=1e-2, size=(in_dim, self.out_dim)))

    def quantile_concrete(self, u):
        """Quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + self.qz_log_alpha) / self.beta)
        return y * (self.zeta - self.gamma) + self.gamma

    def sample_u(self, shape, reuse_u=False):
        """Uniform random numbers for concrete distribution"""
        if self.eps is None or not reuse_u:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.eps = torch.rand(size=shape).to(device) * (1 - 2 * self.epsilon) + self.epsilon
        return self.eps

    def sample_z(self, batch_size, sample=True):
        """Use the hard concrete distribution as described in https://arxiv.org/abs/1712.01312"""
        if sample:
            eps = self.sample_u((batch_size, self.in_dim, self.out_dim))
            z = self.quantile_concrete(eps)
            return torch.clamp(z, min=0, max=1)
        else:  # Mean of the hard concrete distribution
            pi = torch.sigmoid(self.qz_log_alpha)
            return torch.clamp(pi * (self.zeta - self.gamma) + self.gamma, min=0.0, max=1.0)

    def get_z_mean(self):
        """Mean of the hard concrete distribution"""
        pi = torch.sigmoid(self.qz_log_alpha)
        return torch.clamp(pi * (self.zeta - self.gamma) + self.gamma, min=0.0, max=1.0)

    def sample_weights(self, reuse_u=False):
        z = self.quantile_concrete(self.sample_u((self.in_dim, self.out_dim), reuse_u=reuse_u))
        mask = torch.clamp(z, min=0.0, max=1.0)
        return mask * self.W

    def get_weight(self):
        """Deterministic value of weight based on mean of z"""
        return self.W * self.get_z_mean()

    def loss(self):
        """Regularization loss term"""
        return torch.sum(torch.sigmoid(self.qz_log_alpha - self.beta * np.log(-self.gamma / self.zeta)))

    def forward(self, x, sample=True, reuse_u=False):
        """Multiply by weight matrix and apply activation units"""
        if sample:
            h = torch.matmul(x, self.sample_weights(reuse_u=reuse_u))
        else:
            w = self.get_weight()
            h = torch.matmul(x, w)

        if self.use_bias:
            h = h + self.bias

        # shape of h = (?, self.n_funcs)

        output = []
        # apply a different activation unit to each column of h
        in_i = 0  # input index
        out_i = 0  # output index
        # Apply functions with only a single input
        while out_i < self.n_single:
            output.append(self.funcs[out_i](h[:, in_i]))
            in_i += 1
            out_i += 1
        # Apply functions that take 2 inputs and produce 1 output
        while out_i < self.n_funcs:
            output.append(self.funcs[out_i](h[:, in_i], h[:, in_i + 1]))
            in_i += 2
            out_i += 1
        output = torch.stack(output, dim=1)
        return output


class SymbolicNetL0(nn.Module):
    """Symbolic regression network with multiple layers. Produces one output."""

    def __init__(self, symbolic_depth, in_dim=1, funcs=None, initial_weights=None, init_stddev=0.1):
        super(SymbolicNetL0, self).__init__()
        self.depth = symbolic_depth  # Number of hidden layers
        self.funcs = funcs

        layer_in_dim = [in_dim] + self.depth * [len(funcs)]
        if initial_weights is not None:
            layers = [SymbolicLayerL0(funcs=funcs, initial_weight=initial_weights[i],
                                      in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())
        else:
            # Each layer initializes its own weights
            if not isinstance(init_stddev, list):
                init_stddev = [init_stddev] * self.depth
            layers = [SymbolicLayerL0(funcs=funcs, init_stddev=init_stddev[i], in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            # Initialize weights for last layer (without activation functions)
            self.output_weight = nn.Parameter(torch.rand(size=(self.hidden_layers[-1].n_funcs, 1)) * 2)
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input, sample=True, reuse_u=False):
        # connect output from previous layer to input of next layer
        h = input
        for i in range(self.depth):
            h = self.hidden_layers[i](h, sample=sample, reuse_u=reuse_u)

        h = torch.matmul(h, self.output_weight)     # Final output (no activation units) of network
        return h

    def get_loss(self):
        return torch.sum(torch.stack([self.hidden_layers[i].loss() for i in range(self.depth)]))

    def get_weights(self):
        """Return list of weight matrices"""
        # First part is iterating over hidden weights. Then append the output weight.
        return [self.hidden_layers[i].get_weight().cpu().detach().numpy() for i in range(self.depth)] + \
               [self.output_weight.cpu().detach().numpy()]


