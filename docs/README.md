---
home: true
heroImage: /logo.png
heroText: EagerPy
tagline: Writing Code That Works Natively with<br />PyTorch, TensorFlow, JAX, and NumPy
actionText: Get Started →
actionLink: /guide/
features:
- title: Native Performance
  details: EagerPy operations get directly translated into the corresponding native operations.
- title: Fully Chainable
  details: All functionality is available as methods on the tensor objects and as EagerPy functions.
- title: Type Checking
  details: Catch bugs before running your code thanks to EagerPy's extensive type annotations.
footer: Copyright © 2020 Jonas Rauber

---

### What is EagerPy?

**EagerPy** is a **Python framework** that lets you write code that automatically works natively with [**PyTorch**](https://pytorch.org), [**TensorFlow**](https://www.tensorflow.org), [**JAX**](https://github.com/google/jax), and [**NumPy**](https://numpy.org).

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

### Getting Started

You can install the latest release from [PyPI](https://pypi.org/project/eagerpy/) using `pip`:

```bash
python3 -m pip install eagerpy
```

::: warning NOTE
EagerPy requires Python 3.6 or newer.
:::
