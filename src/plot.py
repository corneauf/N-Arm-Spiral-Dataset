#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import NArmSpiral
from models.mlp import LinearMLP, NonLinearMLP
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Plot the n-arm spiral dataset')
    parser.add_argument('filename', type=str, help='Name of the file containing the dataset')
    parser.add_argument('actions', nargs='+', help='Plot actions to execute', default=['dataset'])
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--linear-file', type=str, default='linear_model.pt')
    parser.add_argument('--nonlinear-file', type=str, default='nonlinear_model.pt')
    parser.add_argument('--hidden-units', type=int, default=128, help='Number of hidden units for \
    the model')
    parser.add_argument('--hidden-layers', type=int, default=1, help='Number of hidden layers')
    
    args = parser.parse_args()

    n_arm_spiral = NArmSpiral(args.filename)

    model = load_model(n_arm_spiral.classes, args)

    for action in args.actions:
        globals()[action](n_arm_spiral, args, model)
        plt.clf()

def load_model(classes, args):
    """Load the model from a file

    Create the model from LinearMLP or NonLinearMLP depending on the `--linear` flag then
    load the weights from the file (also determined through the `--linear` flag).

    Parameters
    ----------
    classes: list
        List of all the classes in the dataset.
    args:
        Object returned by `argparse.ArgumentParser.parse_args().

    Returns
    -------
    LinearMLP or NonLinearMLP
        Instance of the corresponding class from the `--linear` flag.

    Notes
    -----
    --linear is False by default.
    """
    if args.linear:
        model = LinearMLP(classes)
        weights = torch.load(args.linear_file)
    else:
        model = NonLinearMLP(classes)
        weights = torch.load(args.nonlinear_file)

    model.load_state_dict(weights)
    print(model.eval())
    return model

def view_dataset(dataset, args, *unused):
    """Plot the points of a dataset

    Create a visual representation of the NArmSpiral dataset by coloring each class (arm)
    using a different color.

    Parameters
    ----------
    dataset: NArmSpiral
        The dataset containing the points to plot.
    args:
        Object returned by `argparse.ArgumentParser.parse_args().
    unused: list
        Unused objects.
    """
    # Split the dataset into n subarray where n is the number of classes
    arms = np.split(dataset.data, len(dataset.classes))

    for arm in arms:
        # Since the values of arm are all the same class, just take the class of the first one
        plt.plot(arm[:, 0], arm[:, 1], '.', str(arm[0, 2]))

    plt.show()

def view_test(dataset, args, model):
    """Plot the test dataset as classified by the model

    Classify the test dataset then, plot each point on the graph and color them according to the
    class the model has predicted.

    Parameters
    ----------
    dataset: NArmSpiral
        The dataset containing the points to plot.
    args:
        Object returned by `argparse.ArgumentParser.parse_args().
    model: LinearMLP or NonLinearMLP
        The model use to classify the test dataset.
    """
    classified = classify_test_data(args, model)
    
    for _class, points in classified.items():
        numpy_points = np.array(points).transpose()
        plt.plot(*numpy_points, '.', str(_class))
    
    plt.show()

def classify_test_data(args, model):
    """Classify the test data.

    Iterate through the test dataset and pass each point to the model to classify it.

    Parameters
    ----------
    args:
        Object returned by `argparse.ArgumentParser.parse_args().
    model: LinearMLP or NonLinearMLP
        The model use to classify the test dataset.

    Returns
    -------
    dict:
        Dictionary of classified points where the keys are the classes and the value are a
        list of points belonging to that class.
    """
    test_dataset = NArmSpiral(args.filename, train=False)
    loader = DataLoader(test_dataset)

    classified = {_class: [] for _class in test_dataset.classes}

    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            point = inputs.data[0].numpy()
            _class = predicted.item()
            classified[_class].append(point)

    return classified

if __name__ == '__main__':
    main()
