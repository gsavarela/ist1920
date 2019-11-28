""" This is getting started with PyTorch

    REF
    ---
    https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
"""
from __future__ import print_function
import torch


print("x----Construct an empty matrix----x")
x = torch.empty(5, 3)
print(x)


print("x----Construct a zero matrix----x")
x = torch.zeros(5, 3, dtype=torch.long)
print(x)


print("x----Construct a tensor from data----x")
x = torch.tensor([5.5, 3])
print(x)


print("x----Create a tensor from a tensor----x")
x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float) # override dtype!
print(x)  # result has the same size

print("x----Get size----x")
print(x.size())


