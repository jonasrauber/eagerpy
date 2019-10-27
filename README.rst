=======
EagerPy
=======

EagerPy is a thin wrapper around **PyTorch** and **TensorFlow Eager** that unifies their interface and thus allows writing code that works with both.

Warning: this is work in progress; the tests should run through just fine, but lot's of features are still missing. Let me know if this project is useful to you and which features are needed.

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

   # In both cases, the resulting EagerPy tensor provides the same
   # interface and a library build on top of the interface provided
   # by EagerPy will work with both PyTorch and TensorFlow tensors.

   # EagerPy tensors provide a lot of functionality through methods, e.g.
   x.sum()
   x.sqrt()
   x.clip(0, 1)

   # but EagerPy also provides them as functions, e.g.
   ep.sum(x)
   ep.sqrt(x)
   ep.clip(x, 0, 1)
   ep.uniform((3, 3), low=-1., high=1.)
