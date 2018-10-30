import torch
import torch.nn as nn

class LinearMLP(nn.Module):
    """Create a Multilayer Perceptron (MLP) with an arbitrary number of linear
    functions for the hidden layers.
    
    Using `torch.nn.Module`, this creates a fully connected, feedforward linear
    neural network for classifying data. This implementation only uses `torch.nn.
    Linear` for the hidden layers. It can be *shallow* (only a single hidden layer)
    or *deep* (a stricly greater than one number of hidden layers).

    This aims at providing a visual interpretation of why linear models cannot
    classify data that follow a non-linear distribution and why stacking more
    linear functions on top of other linear functions does not help the classification.
    """
    
    def __init__(self, classes, hidden_layers=1, hidden_units=128):
        super(LinearMLP, self).__init__()

        self.hidden_units = hidden_units
        linear_layers = [nn.Linear(2, hidden_units)]
        linear_layers.extend([nn.Linear(self._hidden_units(i), self._hidden_units(i + 1))
                         for i in range(hidden_layers - 1)])

        self.hidden_layers = nn.Sequential(*linear_layers)
        self.output_layer = nn.Linear(int(hidden_units / (2 ** hidden_layers - 1)), len(classes))

    def _hidden_units(self, i):
        return int(self.hidden_units / (2 ** i))

    def forward(self, x):
        out = self.hidden_layers(x)
        out = self.output_layer(out)

        return out


class NonLinearMLP(nn.Module):
    """Create a Multilayer Perceptron (MLP) with an arbitrary number of nonlinear
    functions for the hidden layers.
    
    Using `torch.nn.Module`, this creates a fully connected, feedforward nonlinear
    neural network for classifying data. This implementation uses `torch.nn.Linear`
    for the hidden layers as well as incorporating a nonlinearality through the use
    of `torch.nn.ReLU`. It can be *shallow* (only a single hidden layer) or *deep*
    (a stricly greater than one number of hidden layers).
    """
    
    def __init__(self, classes, hidden_layers=1, hidden_units=128):
        super(NonLinearMLP, self).__init__()

        self.hidden_units = hidden_units
        nonlinear_layers = [nn.Linear(2, hidden_units), nn.ReLU()]
        nonlinear_layers.extend([nn.Sequential(nn.Linear(self._hidden_units(i),
                                                         self._hidden_units(i + 1)), nn.ReLU)
                                for i in range(hidden_layers - 1)])

        self.hidden_layers = nn.Sequential(*nonlinear_layers)
        self.output_layer = nn.Linear(self._hidden_units(hidden_layers - 1), len(classes))

    def _hidden_units(self, i):
        return int(self.hidden_units / (2 ** i))

    def forward(self, x):
        out = self.hidden_layers(x)
        out = self.output_layer(out)

        return out
