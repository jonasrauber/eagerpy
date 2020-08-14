---
title: Converting

---

# Converting Between EagerPy and Native Tensors

A native tensor could be a PyTorch GPU or CPU tensor, a TensorFlow tensor, a JAX array, or a NumPy array.

**A native PyTorch tensor:**
```python
import torch
x = torch.tensor([1., 2., 3., 4., 5., 6.])
```

**A native TensorFlow tensor:**
```python
import tensorflow as tf
x = tf.constant([1., 2., 3., 4., 5., 6.])
```

**A native JAX array:**
```python
import jax.numpy as np
x = np.array([1., 2., 3., 4., 5., 6.])
```

**A native NumPy array:**
```python
import numpy as np
x = np.array([1., 2., 3., 4., 5., 6.])
```

No matter which native tensor you have, it can always be turned into the appropriate EagerPy tensor using `ep.astensor`. This will automatically wrap the native tensor with the correct EagerPy tensor class. The original native tensor can always be accessed using the `.raw` attribute.

```python
# x should be a native tensor (see above)
# for example:
import torch
x = torch.tensor([1., 2., 3., 4., 5., 6.])

# Any native tensor can easily be turned into an EagerPy tensor
import eagerpy as ep
x = ep.astensor(x)

# Now we can perform any EagerPy operation
x = x.square()

# And convert the EagerPy tensor back into a native tensor
x = x.raw
# x will now again be a native tensor (e.g. a PyTorch tensor)
```

Especially in functions, it is common to convert all inputs to EagerPy tensors. This could be done using individual calls to `ep.astensor`, but using `ep.astensors` this can be written even more compactly.

```python
# x, y should be a native tensors (see above)
# for example:
import torch
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])

import eagerpy as ep
x, y = ep.astensors(x, y)  # works for any number of inputs
```
