.. raw:: html

   <a href="https://eagerpy.jonasrauber.de"><img src="https://raw.githubusercontent.com/jonasrauber/eagerpy/master/docs/.vuepress/public/logo_small.png" align="right" /></a>

.. image:: https://badge.fury.io/py/eagerpy.svg
   :target: https://badge.fury.io/py/eagerpy

.. image:: https://codecov.io/gh/jonasrauber/eagerpy/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/jonasrauber/eagerpy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black

======================================================================================
EagerPy: PyTorch, TensorFlow, JAX and NumPy — all of them natively using the same code
======================================================================================

`EagerPy <https://eagerpy.jonasrauber.de>`_ is a **Python framework** that let's you write code that automatically works natively with `PyTorch <https://pytorch.org>`_, `TensorFlow <https://www.tensorflow.org>`_, `JAX <https://github.com/google/jax>`_, and `NumPy <https://numpy.org>`_. EagerPy is **also great when you work with just one framework** but prefer a clean and consistent API that is fully chainable, provides extensive type annotions and let's you write beautiful code.


🔥 Design goals
----------------

- **Native Performance**: EagerPy operations get directly translated into the corresponding native operations.
- **Fully Chainable**: All functionality is available as methods on the tensor objects and as EagerPy functions.
- **Type Checking**: Catch bugs before running your code thanks to EagerPy's extensive type annotations.


📖 Documentation
-----------------

Learn more about in the `documentation <https://eagerpy.jonasrauber.de>`_.


🚀 Quickstart
--------------

.. code-block:: bash

   pip install eagerpy


🎉 Example
-----------

.. code-block:: python

   import torch
   x = torch.tensor([1., 2., 3., 4., 5., 6.])

   import tensorflow as tf
   x = tf.constant([1., 2., 3., 4., 5., 6.])

   import jax.numpy as np
   x = np.array([1., 2., 3., 4., 5., 6.])

   import numpy as np
   x = np.array([1., 2., 3., 4., 5., 6.])

   # No matter which framwork you use, you can use the same code
   import eagerpy as ep

   # Just wrap a native tensor using EagerPy
   x = ep.astensor(x)

   # All of EagerPy's functionality is available as methods
   x = x.reshape((2, 3))
   x.flatten(start=1).square().sum(axis=-1).sqrt()
   # or just: x.flatten(1).norms.l2()

   # and as functions (yes, we gradients are also supported!)
   loss, grad = ep.value_and_grad(loss_fn, x)
   ep.clip(x + eps * grad, 0, 1)

   # You can even write functions that work transparently with
   # Pytorch tensors, TensorFlow tensors, JAX arrays, NumPy arrays

   def my_universal_function(a, b, c):
       # Convert all inputs to EagerPy tensors
       a, b, c = ep.astensors(a, b, c)

       # performs some computations
       result = (a + b * c).square()

       # and return a native tensor
       return result.raw


🗺 Use cases
------------

`Foolbox Native <https://github.com/bethgelab/foolbox>`_, the latest version of
Foolbox, a popular adversarial attacks library, has been rewritten from scratch
using EagerPy instead of NumPy to achieve native performance on models
developed in PyTorch, TensorFlow and JAX, all with one code base.


🐍 Compatibility
-----------------

We currently test with the following versions:

* PyTorch 1.4.0
* TensorFlow 2.1.0
* JAX 0.1.57
* NumPy 1.18.1
