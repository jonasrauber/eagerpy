---
title: Generic Functions

---

# Implementing Generic Framework-Agnostic Functions

Using the conversion functions shown in [Converting](./converting.md), we can already define a simple framework-agnostic function.

```python
import eagerpy as ep

def norm(x):
    x = ep.astensor(x)
    result = x.square().sum().sqrt()
    return result.raw
```

This function can be called with a native tensor from any framework and it will return the norm of that tensor, again as a native tensor from that framework.

Calling the `norm` function using a PyTorch tensor:
```python
import torch
norm(torch.tensor([1., 2., 3.]))
# tensor(3.7417)
```

Calling the `norm` function using a TensorFlow tensor:
```python
import tensorflow as tf
norm(tf.constant([1., 2., 3.]))
# <tf.Tensor: shape=(), dtype=float32, numpy=3.7416575>
```

If we would call the above `norm` function with an EagerPy tensor, the `ep.astensor` call would simply return its input. The `result.raw` call in the last line would however still extract the underlying native tensor. Often it is preferably to implement a generic function that not only transparently handles any native tensor but also EagerPy tensors, that is the return type should always match the input type. This is particularly useful in libraries like Foolbox that allow users to work with EagerPy and native tensors. To achieve that, EagerPy comes with two derivatives of the above conversion functions: `ep.astensor_` and `ep.astensors_`. Unlike their counterparts without an underscore, they return an additional inversion function that restores the input type. If the input to `astensor_` is a native tensor, `restore_type` will be identical to `.raw`, but if the original input was an EagerPy tensor, `restore_type` will not call `.raw`. With that, we can write generic framework-agnostic functions that work transparently for any input.


An improved framework-agnostic `norm` function:
```python
import eagerpy as ep

def norm(x):
    x, restore_type = ep.astensor_(x)
    result = x.square().sum().sqrt()
    return restore_type(result)
```

Converting and restoring multiple inputs using `ep.astensors_`:
```python
import eagerpy as ep

def example(x, y, z):
    (x, y, z), restore_type = ep.astensors_(x, y, z)
    result = (x + y) * z
    return restore_type(result)
```
