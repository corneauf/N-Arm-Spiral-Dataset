#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import NArmSpiral
from models.mlp import LinearMLP, NonLinearMLP
from torch.utils.data import DataLoader

def train_model(model, train_dataset, device, args):
    model.to(device)
    loader = DataLoader(train_dataset, shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        running_loss = 0.0

        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 500 == 499:
                print('[{}, {}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

def test_model(model, test_dataset):
    correct = 0
    total = 0

    loader = DataLoader(test_dataset, num_workers=2)
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test points: {:.2f}%'.format(100 * correct / total))

def main():
    parser = argparse.ArgumentParser(description='Run the different models')
    parser.add_argument('--epochs', type=int, help='Number of epochs to run',
                        default=1000)
    parser.add_argument('--lr', type=float, help='SGD learning rate',
                        default=0.001)
    parser.add_argument('--linear-file', type=str, help='Name of the file storing the linear \
    model', default='linear_model.pt')
    parser.add_argument('--nonlinear-file', type=str, help='Name of the file storing the nonlinear \
    model', default='nonlinear_model.pt')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Using GPU")

    train_dataset = NArmSpiral('n_arm_spiral.csv')
    test_dataset = NArmSpiral('n_arm_spiral.csv', train=False)

    print('#### Training shallow linear model')
    model = LinearMLP(train_dataset.classes)
    print('#### Linear shallow MLP:')
    print(model)

    train_model(model, train_dataset, device, args)
    test_model(model, test_dataset)

    print('#### Training shallow nonlinear model')
    model = NonLinearMLP(train_dataset.classes)
    print('#### Non linear shallow MLP:')
    print(model)

    train_model(model, train_dataset, device, args)
    test_model(model, test_dataset)

if __name__ == '__main__':
    main()
