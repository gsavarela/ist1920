""" This is getting started with PyTorch (1/5)

    REF
    ---
    https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
"""
from __future__ import print_function
import torch


print("x----There are multiple syntax for operation----x")
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print("x----Syntax 1: overloading----x")
print(x + y)

print("x----Syntax 2: torch module----x")
print(torch.add(x, y))

print("x----Syntax 3: torch module & provide output argument----x")
z = torch.empty(5, 3)
torch.add(x, y, out=z)
print(z)


print("x----Syntax 4: inplace----x")
# Any operation that mutates a tensor in-place is post-fixed with an _.
# For example: x.copy_(y), x.t_(), will change x.
print(y.add_(x))


print("x----NumPy-like:Slicing----x")
print(x[:, 1])

print("x----NumPy-like:Resizing----x")
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) # the -1 size is inferred from the other dimension
print(x.size(), y.size(), z.size())


print("x---NumPy Bridge:Get a NumPy number----x")
x = torch.randn(1)
print(x)
print(x.item())

print("x----NumPy Bridge: tensors -> ndarrays----x")
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

print("x----NumPy Bridge: ndarrays -> tensors----x")
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
print('''NOTE: All the Tensors on the CPU except a CharTensor
      support converting to NumPy and back.''')

