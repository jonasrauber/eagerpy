---
title: Examples

---

# Examples :tada:

Write your function using EagerPy:

```python
import eagerpy as ep

def norm(x, axis=None):
    x = ep.astensor(x)
    result = x.square().sum(axis=axis).sqrt()
    return result.raw
```

And now you can use it with PyTorch, TensorFlow, JAX and NumPy:

```python
import torch

norm(torch.tensor([1., 2., 3.]))
```

```
import tensorflow as tf

norm(tf.constant([1., 2., 3.]))
```

```
import jax.numpy as np

norm(np.array([1., 2., 3.]))
```

```
import numpy as np

norm(np.array([1., 2., 3.]))
```
