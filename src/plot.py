#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import numpy as np

from dataset import NArmSpiral

def main():
    parser = argparse.ArgumentParser(description='Plot the n-arm spiral dataset')
    parser.add_argument('filename', type=str, help='Name of the file containing the dataset')
    
    args = parser.parse_args()

    n_arm_spiral = NArmSpiral(args.filename)
    arms = np.split(n_arm_spiral.data, len(n_arm_spiral.classes))

    for arm in arms:
        plt.plot(arm[:, 0], arm[:, 1], '.', str(arm[0,2]))

    plt.show()


if __name__ == '__main__':
    main()
