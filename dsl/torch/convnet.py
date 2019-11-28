""" This is getting started with PyTorch (3/5)

It is a simple feed-forward network. It takes the input, feeds it through
several layers one after the other, and then finally gives the output.

A typical training procedure for a neural network is as follows:

    * Define the neural network that has some learnable parameters (or weights)
    * Iterate over a dataset of inputs
    * Process input through the network
    * Compute the loss (how far is the output from being correct)
    * Propagate gradients back into the network’s parameters
    * Update the weights of the network, typically using a simple update rule:
        weight = weight - learning_rate * gradient

    DEF:
    ---

    * torch.Tensor - A multi-dimensional array with support for autograd
      operations like backward(). Also holds the gradient w.r.t. the tensor.

    * nn.Module - Neural network module. Convenient way of encapsulating
      parameters, with helpers for moving them to GPU, exporting, loading, etc.

    * nn.Parameter - A kind of Tensor, that is automatically registered as a
      parameter when assigned as an attribute to a Module.

    * autograd.Function - Implements forward and backward definitions of an
      autograd operation. Every Tensor operation creates at least a single
      Function node that connects to functions that created a Tensor and 
      encodes its history.


    REF
    ---
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        #  1 input image channel, 6 output channels,
        #  3x3 square convolution
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        #  an affine transformation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6 * 6 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #   Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #  If the size is a square we can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

    def num_flat_features(self, x):
        size = x.size()[1:]  #  all dimensions except a batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = ConvNet()
print(net)
if __name__ == '__main__':

    print("""The learnable parameters of a model are returned by 
    net.parameters()""")
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    print("""Let try a random 32x32 input.
Note: expected input size of this net (LeNet) is 32x32.
To use this net on MNIST dataset, please resize the images
from the dataset to 32x32.\n""")
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
    print("""This example is not working, why the net class
has the callable property?""")
