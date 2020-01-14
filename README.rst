.. image:: https://badge.fury.io/py/eagerpy.svg
    :target: https://badge.fury.io/py/eagerpy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black


=======
EagerPy
=======

EagerPy is a thin wrapper around **PyTorch**, **TensorFlow Eager**, **JAX** and
**NumPy** that unifies their interface and thus allows writing code that
works natively across all of them.

Warning: this is work in progress; the tests should run through just fine,
but lot's of features are still missing. Let me know if this project is useful
to you and which features are needed.

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


Development
-----------

For development, it is recommended to do a an editable installation of Foolbox
and Foolbox native using :code:`pip install -e .` in the corresponding folders (after
cloning the two repositories). Unfortunately, `pip` has a
`bug <https://github.com/pypa/pip/issues/7265>`_ with editable installs and
namespace packages like Foolbox Native. A simple workaround is to add a symlink
to the :code:`foolbox/ext/native` folder of Foolbox Native in the :code:`foolbox/ext/` folder
of Foolbox itself.
