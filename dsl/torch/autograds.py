""" This is getting started with PyTorch (2/5)

    REF
    ---
    https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
"""
from __future__ import print_function
import torch


print("""Create a tensor and set requires_grad=True 
      to track computation with it.\n""")
x = torch.ones(2, 2, requires_grad=True)
print(x)

print("""y was created as a result of an operation,
      so it has a grad_fn.\n""")
y = x + 2
print(y)
print(y.grad_fn)

print("""Do more operations on y\n""")
z = y * y * 3
out = z.mean()

print(z, out)

print(""".requires_grad_( ... ) changes an existing Tensor’s
    requires_grad flag in-place. The input flag defaults to
    False if not given.\n""")

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

print("Gradients\n\n")

print("""Let’s backprop now.
Because out contains a single scalar, out.backward() is equivalent to
out.backward(torch.tensor(1.)).\n""")

out.backward()
print(x.grad)


print("""Backprop is a tool for computing the Jacobian product
on the form J * v = x -- such form has benefits (???)\n""")
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

print("""Now in this case y is no longer a scalar.
torch.autograd could not compute the full Jacobian directly,
but if we just want the vector-Jacobian product,
simply pass the vector to backward as argument:\n""")

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.double)
y.backward(v)
print(x.grad)
