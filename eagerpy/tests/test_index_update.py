import numpy as np
import torch
import tensorflow as tf
import jax.numpy as jnp

import eagerpy as ep


def test_index_update_tensorflow():
    x = np.ones((3, 4))
    ind = np.array([0, 1, 2, 1])
    x = tf.constant(x)
    ind = tf.constant(ind)
    x = ep.astensor(x)
    ind = ep.astensor(ind)

    a = np.ones((3, 4))
    a[1] = 66.0
    assert (ep.index_update(x, ep.index[1], ep.ones(x, 4) * 66.0).numpy() == a).all()

    a = np.ones((3, 4))
    a[:, 1] = 55.0
    assert (ep.index_update(x, ep.index[:, 1], ep.ones(x, 3) * 55.0).numpy() == a).all()

    a = np.ones((3, 4))
    a[ind.numpy(), np.arange(4)] = 33.0
    assert (
        ep.index_update(x, ep.index[ind, ep.arange(x, 4)], ep.ones(x, 4) * 33.0).numpy()
        == a
    ).all()


def test_index_update_pytorch():
    x = np.ones((3, 4))
    ind = np.array([0, 1, 2, 1])
    x = torch.from_numpy(x)
    ind = torch.from_numpy(ind)
    x = ep.astensor(x)
    ind = ep.astensor(ind)

    a = np.ones((3, 4))
    a[1] = 66.0
    assert (ep.index_update(x, ep.index[1], ep.ones(x, 4) * 66.0).numpy() == a).all()

    a = np.ones((3, 4))
    a[:, 1] = 55.0
    assert (ep.index_update(x, ep.index[:, 1], ep.ones(x, 3) * 55.0).numpy() == a).all()

    a = np.ones((3, 4))
    a[ind.numpy(), np.arange(4)] = 33.0
    assert (
        ep.index_update(x, ep.index[ind, ep.arange(x, 4)], ep.ones(x, 4) * 33.0).numpy()
        == a
    ).all()


def test_index_update_pytorch_cuda():
    x = np.ones((3, 4))
    ind = np.array([0, 1, 2, 1])
    x = torch.from_numpy(x).cuda()
    ind = torch.from_numpy(ind).cuda()
    x = ep.astensor(x)
    ind = ep.astensor(ind)

    a = np.ones((3, 4))
    a[1] = 66.0
    assert (ep.index_update(x, ep.index[1], ep.ones(x, 4) * 66.0).numpy() == a).all()

    a = np.ones((3, 4))
    a[:, 1] = 55.0
    assert (ep.index_update(x, ep.index[:, 1], ep.ones(x, 3) * 55.0).numpy() == a).all()

    a = np.ones((3, 4))
    a[ind.numpy(), np.arange(4)] = 33.0
    assert (
        ep.index_update(x, ep.index[ind, ep.arange(x, 4)], ep.ones(x, 4) * 33.0).numpy()
        == a
    ).all()


def test_index_update_numpy():
    x = np.ones((3, 4))
    ind = np.array([0, 1, 2, 1])
    x = ep.astensor(x)
    ind = ep.astensor(ind)

    a = np.ones((3, 4))
    a[1] = 66.0
    assert (ep.index_update(x, ep.index[1], ep.ones(x, 4) * 66.0).numpy() == a).all()

    a = np.ones((3, 4))
    a[:, 1] = 55.0
    assert (ep.index_update(x, ep.index[:, 1], ep.ones(x, 3) * 55.0).numpy() == a).all()

    a = np.ones((3, 4))
    a[ind.numpy(), np.arange(4)] = 33.0
    assert (
        ep.index_update(x, ep.index[ind, ep.arange(x, 4)], ep.ones(x, 4) * 33.0).numpy()
        == a
    ).all()


def test_index_update_jax():
    x = np.ones((3, 4))
    ind = np.array([0, 1, 2, 1])
    x = jnp.asarray(x)
    ind = jnp.asarray(ind)
    x = ep.astensor(x)
    ind = ep.astensor(ind)

    a = np.ones((3, 4))
    a[1] = 66.0
    assert (ep.index_update(x, ep.index[1], ep.ones(x, 4) * 66.0).numpy() == a).all()

    a = np.ones((3, 4))
    a[:, 1] = 55.0
    assert (ep.index_update(x, ep.index[:, 1], ep.ones(x, 3) * 55.0).numpy() == a).all()

    a = np.ones((3, 4))
    a[ind.numpy(), np.arange(4)] = 33.0
    assert (
        ep.index_update(x, ep.index[ind, ep.arange(x, 4)], ep.ones(x, 4) * 33.0).numpy()
        == a
    ).all()
