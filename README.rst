********************
N-Arm Spiral Dataset
********************

This project aims at providing a way to generate a toy dataset from scratch
that cannot be classified with a linear classifier. I decided to generate
a spiral inspired from the 2-Arm Spiral dataset but extended to support more
than two arms.

As a way to showcase the difference between linear classifier and non-linear
ones, toy examples are also provided. These examples will try to classify
the dataset with a simple shallow Multilayer Perceptron (MLP). The first
classifier will only use a linear function as part of its only hidden layer.
On the other hand, the second classifier will add a non-linear activation
function to the hidden layer (a Rectified Linear Unit, or ReLU for short).


Installation
============

To build the dataset you only need numpy and scipy which you can install by
running :
``pip install numpy scipy``

To run the plotting script, you will need matplotlib:
``pip install matplotlib``

And finally, to run the classifier you will need PyTorch. You can download
it by looking at the documentation here: `https://github.com/pytorch/pytorch`

License
=======

The project is licensed under the BSD license.
