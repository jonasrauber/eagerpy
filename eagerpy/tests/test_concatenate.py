import numpy as np
import torch
import tensorflow as tf
import jax.numpy as jnp

import eagerpy as ep


def test_concatenate_tensorflow():
    x = np.arange(12).astype(np.float32).reshape(4, 3)
    x = tf.constant(x)
    x = ep.astensor(x)
    assert ep.concatenate((x, x[:2])).shape == (6, 3)
    assert ep.concatenate((x, x[:, :2]), axis=1).shape == (4, 5)


def test_concatenate_pytorch():
    x = np.arange(12).astype(np.float32).reshape(4, 3)
    x = torch.from_numpy(x)
    x = ep.astensor(x)
    assert ep.concatenate((x, x[:2])).shape == (6, 3)
    assert ep.concatenate((x, x[:, :2]), axis=1).shape == (4, 5)


def test_concatenate_pytorch_cuda():
    x = np.arange(12).astype(np.float32).reshape(4, 3)
    x = torch.from_numpy(x).cuda()
    x = ep.astensor(x)
    assert ep.concatenate((x, x[:2])).shape == (6, 3)
    assert ep.concatenate((x, x[:, :2]), axis=1).shape == (4, 5)


def test_concatenate_numpy():
    x = np.arange(12).astype(np.float32).reshape(4, 3)
    x = ep.astensor(x)
    assert ep.concatenate((x, x[:2])).shape == (6, 3)
    assert ep.concatenate((x, x[:, :2]), axis=1).shape == (4, 5)


def test_concatenate_jax():
    x = jnp.arange(12).astype(np.float32).reshape(4, 3)
    x = ep.astensor(x)
    assert ep.concatenate((x, x[:2])).shape == (6, 3)
    assert ep.concatenate((x, x[:, :2]), axis=1).shape == (4, 5)
