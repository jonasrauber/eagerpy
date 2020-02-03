.. image:: https://badge.fury.io/py/eagerpy.svg
    :target: https://badge.fury.io/py/eagerpy

.. image:: https://codecov.io/gh/jonasrauber/eagerpy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/jonasrauber/eagerpy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black


=======
EagerPy
=======

EagerPy is a thin wrapper around **PyTorch**, **TensorFlow Eager**, **JAX** and
**NumPy** that unifies their interface and thus allows writing code that
works natively across all of them.

Learn more about in the `documentation <https://jonasrauber.github.io/eagerpy/>`_.

**EagerPy is now in active use** to develop `Foolbox Native <https://github.com/jonasrauber/foolbox-native>`_.

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

   import jax.numpy as np
   x = np.array([1., 2., 3.])
   x = ep.JAXTensor(x)

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


Compatibility
-------------

We currently test with the following versions:

* PyTorch 1.3.1
* TensorFlow 2.0.0
* JAX 0.1.57
* NumPy 1.18.1
