=======
EagerPy
=======

EagerPy is a thin wrapper around **PyTorch**, **TensorFlow Eager** and
**NumPy** that unifies their interface and thus allows writing code that
works natively across all of them.

Warning: this is work in progress; the tests should run through just fine,
but lot's of features are still missing. Let me know if this project is useful
to you and which features are needed.

Installation
------------

.. code-block:: bash

   pip install eagerpy


Example
-------

.. code-block:: python

   import eagerpy as ep

   import torch
   x = torch.tensor([1., 2., 3.])
   x = ep.PyTorchTensor(x)

   import tensorflow as tf
   x = tf.constant([1., 2., 3.])
   x = ep.TensorFlowTensor(x)

   import numpy as np
   x = np.array([1., 2., 3.])
   x = ep.NumPyTensor(x)

   # In all cases, the resulting EagerPy tensor provides the same
   # interface. This makes it possible to write code that works natively
   # independent of the underlying framework.

   # EagerPy tensors provide a lot of functionality through methods, e.g.
   x.sum()
   x.sqrt()
   x.clip(0, 1)

   # but EagerPy also provides them as functions, e.g.
   ep.sum(x)
   ep.sqrt(x)
   ep.clip(x, 0, 1)
   ep.uniform(x, (3, 3), low=-1., high=1.)  # x is needed to infer the framework
