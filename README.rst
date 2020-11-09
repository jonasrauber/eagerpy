.. raw:: html

   <a href="https://eagerpy.jonasrauber.de"><img src="https://raw.githubusercontent.com/jonasrauber/eagerpy/master/docs/.vuepress/public/logo_small.png" align="right" /></a>

.. image:: https://badge.fury.io/py/eagerpy.svg
   :target: https://badge.fury.io/py/eagerpy

.. image:: https://codecov.io/gh/jonasrauber/eagerpy/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/jonasrauber/eagerpy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black

==================================================================================
EagerPy: Writing Code That Works Natively with PyTorch, TensorFlow, JAX, and NumPy
==================================================================================

`EagerPy <https://eagerpy.jonasrauber.de>`_ is a **Python framework** that lets you write code that automatically works natively with `PyTorch <https://pytorch.org>`_, `TensorFlow <https://www.tensorflow.org>`_, `JAX <https://github.com/google/jax>`_, and `NumPy <https://numpy.org>`_. EagerPy is **also great when you work with just one framework** but prefer a clean and consistent API that is fully chainable, provides extensive type annotions and lets you write beautiful code.


🔥 Design goals
----------------

- **Native Performance**: EagerPy operations get directly translated into the corresponding native operations.
- **Fully Chainable**: All functionality is available as methods on the tensor objects and as EagerPy functions.
- **Type Checking**: Catch bugs before running your code thanks to EagerPy's extensive type annotations.


📖 Documentation
-----------------

Learn more about EagerPy in the `documentation <https://eagerpy.jonasrauber.de>`_.


🚀 Quickstart
--------------

.. code-block:: bash

   pip install eagerpy

EagerPy requires Python 3.6 or newer. Besides that, all essential dependencies are automatically installed. To use it with PyTorch, TensorFlow, JAX, or NumPy, the respective framework needs to be installed separately. These frameworks are not declared as dependencies because not everyone wants to use and thus install all of them and because some of these packages have different builds for different architectures and `CUDA <https://developer.nvidia.com/cuda-zone>`_ versions.

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

   # and as functions (yes, gradients are also supported!)
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

EagerPy is also used by other frameworks to reduce code duplication (e.g. `GUDHI <https://github.com/GUDHI/gudhi-devel>`_) or to `compare the performance of different frameworks <https://github.com/jonasrauber/uniformly-sampling-nd-ball>`_.

📄 Citation
------------

If you use EagerPy, please cite our `paper <https://arxiv.org/abs/2008.04175>`_ using the this BibTex entry:

.. code-block::

   @article{rauber2020eagerpy,
     title={{EagerPy}: Writing Code That Works Natively with {PyTorch}, {TensorFlow}, {JAX}, and {NumPy}},
     author={Rauber, Jonas and Bethge, Matthias and Brendel, Wieland},
     journal={arXiv preprint arXiv:2008.04175},
     year={2020},
     url={https://eagerpy.jonasrauber.de},
   }


🐍 Compatibility
-----------------

We currently test with the following versions:

* PyTorch 1.4.0
* TensorFlow 2.1.0
* JAX 0.1.57
* NumPy 1.18.1
