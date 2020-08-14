---
title: Automatic Differentiation

---

# Automatic Differentiation in EagerPy

EagerPy uses a functional approach to automatic differentiation. You first define a function that will then be differentiated with respect to its inputs. This function is then passed to `ep.value_and_grad` to evaluate both the function and its gradient. More generally, you can also use `ep.value_aux_and_grad` if your function has additional auxiliary outputs and `ep.value_and_grad_fn` if you want the gradient function without immediately evaluating it at some point `x`.

Using `ep.value_and_grad` for automatic differentiation in EagerPy:

```python
import torch
x = torch.tensor([1., 2., 3.])

# The following code works for any framework, not just Pytorch!

import eagerpy as ep
x = ep.astensor(x)

def loss_fn(x):
    # this function takes and returns an EagerPy tensor
    return x.square().sum()

print(loss_fn(x))
# PyTorchTensor(tensor(14.))

print(ep.value_and_grad(loss_fn, x))
# (PyTorchTensor(tensor(14.)), PyTorchTensor(tensor([2., 4., 6.])))
```
