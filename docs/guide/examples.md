---
title: Examples

---

# Examples :tada:

## A framework-agnostic `norm` function

Write your function using EagerPy:

```python
import eagerpy as ep

def norm(x):
    x = ep.astensor(x)
    result = x.square().sum().sqrt()
    return result.raw
```

You can now **use** the `norm` function **with native tensors** and arrays from PyTorch, TensorFlow, JAX and NumPy with **virtually no overhead compared to native code**. Of course, it also works with **GPU tensors**.

```python
import torch
norm(torch.tensor([1., 2., 3.]))
# tensor(3.7417)
```

```python
import tensorflow as tf
norm(tf.constant([1., 2., 3.]))
# <tf.Tensor: shape=(), dtype=float32, numpy=3.7416575>
```

```python
import jax.numpy as np
norm(np.array([1., 2., 3.]))
# DeviceArray(3.7416575, dtype=float32)
```

```python
import numpy as np
norm(np.array([1., 2., 3.]))
# 3.7416573867739413
```

::: tip NOTE
EagerPy already comes with a [builtin implementation of `norm`](/api/norms.md#l2).
:::
