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
    
    args = parser.parse_args()

    n_arm_spiral = NArmSpiral(args.filename)

    model = load_model(n_arm_spiral, args)

    for action in args.actions:
        globals()[action](n_arm_spiral, args, model)
        plt.clf()

def load_model(n_arm_spiral, args):
    if args.linear:
        model = LinearMLP(n_arm_spiral.classes)
        weights = torch.load(args.linear_file)
    else:
        model = NonLinearMLP(n_arm_spiral.classes)
        weights = torch.load(args.nonlinear_file)

    model.load_state_dict(weights)
    print(model.eval())
    return model

def view_dataset(dataset, args, *unused):
    arms = np.split(dataset.data, len(dataset.classes))

    for arm in arms:
        plt.plot(arm[:, 0], arm[:, 1], '.', str(arm[0, 2]))

    plt.show()

def view_test(dataset, args, model):
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
    
    for _class, points in classified.items():
        numpy_points = np.array(points).transpose()
        plt.plot(*numpy_points, '.', str(_class))
    
    plt.show()

if __name__ == '__main__':
    main()
