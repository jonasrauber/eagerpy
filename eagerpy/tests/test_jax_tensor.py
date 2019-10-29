import numpy as np
import pytest

import torch
import tensorflow as tf
import jax

from eagerpy import TensorFlowTensor
from eagerpy import PyTorchTensor
from eagerpy import NumPyTensor
from eagerpy import JAXTensor
import eagerpy as ep


@pytest.fixture
def a1():
    x = np.arange(5).astype(np.float32)
    return x


@pytest.fixture
def tf1():
    x = np.arange(5).astype(np.float32)
    x = tf.constant(x)
    x = TensorFlowTensor(x)
    return x


@pytest.fixture
def th1():
    x = np.arange(5).astype(np.float32)
    x = torch.from_numpy(x)
    x = PyTorchTensor(x)
    return x


@pytest.fixture
def th1g():
    x = np.arange(5).astype(np.float32)
    x = torch.from_numpy(x).cuda()
    x = PyTorchTensor(x)
    return x


@pytest.fixture
def np1():
    x = np.arange(5).astype(np.float32)
    x = NumPyTensor(x)
    return x


@pytest.fixture
def jax1():
    x = np.arange(5).astype(np.float32)
    x = jax.numpy.asarray(x)
    x = JAXTensor(x)
    return x


@pytest.fixture
def a2():
    x = np.arange(7, 17, 2).astype(np.float32)
    return x


@pytest.fixture
def tf2():
    x = np.arange(7, 17, 2).astype(np.float32)
    x = tf.constant(x)
    x = TensorFlowTensor(x)
    return x


@pytest.fixture
def th2():
    x = np.arange(7, 17, 2).astype(np.float32)
    x = torch.from_numpy(x)
    x = PyTorchTensor(x)
    return x


@pytest.fixture
def th2g():
    x = np.arange(7, 17, 2).astype(np.float32)
    x = torch.from_numpy(x).cuda()
    x = PyTorchTensor(x)
    return x


@pytest.fixture
def np2():
    x = np.arange(7, 17, 2).astype(np.float32)
    x = NumPyTensor(x)
    return x


@pytest.fixture
def jax2():
    x = np.arange(7, 17, 2).astype(np.float32)
    x = jax.numpy.asarray(x)
    x = JAXTensor(x)
    return x


@pytest.fixture(params=["jax1", "jax2"])
def ta(request, tf1, th1, th1g, np1, jax1, a1, tf2, th2, th2g, np2, jax2, a2):
    return {
        "tf1": (tf1, a1),
        "th1": (th1, a1),
        "th1g": (th1g, a1),
        "np1": (np1, a1),
        "jax1": (jax1, a1),
        "tf2": (tf2, a2),
        "th2": (th2, a2),
        "th2g": (th2g, a2),
        "np2": (np2, a2),
        "jax2": (jax2, a2),
    }[request.param]


@pytest.fixture(params=["jax"])
def ttaa(request, tf1, th1, th1g, np1, jax1, a1, tf2, th2, th2g, np2, jax2, a2):
    return {
        "tf": (tf1, tf2, a1, a2),
        "th": (th1, th2, a1, a2),
        "thg": (th1g, th2g, a1, a2),
        "np": (np1, np2, a1, a2),
        "jax": (jax1, jax2, a1, a2),
    }[request.param]


def test_len(ta):
    t, a = ta
    assert len(t) == len(a)


def test_bool(ta):
    t, a = ta
    assert bool(t.sum() == 0) == bool(a.sum() == 0)


def test_neg(ta):
    t, a = ta
    assert ((-t).numpy() == -a).all()


def test_add(ttaa):
    t1, t2, a1, a2 = ttaa
    assert ((t1 + t2).numpy() == (a1 + a2)).all()


def test_sub(ttaa):
    t1, t2, a1, a2 = ttaa
    assert ((t1 - t2).numpy() == (a1 - a2)).all()


def test_mul(ttaa):
    t1, t2, a1, a2 = ttaa
    assert ((t1 * t2).numpy() == (a1 * a2)).all()


def test_div(ttaa):
    t1, t2, a1, a2 = ttaa
    np.testing.assert_allclose((t1 / t2).numpy(), (a1 / a2))


def test_getitem(ta):
    t, a = ta
    for i in range(len(t)):
        assert ((t[i]).numpy() == a[i]).all()
    assert ((t[1:3]).numpy() == a[1:3]).all()


def test_pow(ta):
    t, a = ta
    np.testing.assert_allclose((t ** 3).numpy(), (a ** 3))


def test_square(ta):
    t, a = ta
    assert (t.square().numpy() == np.square(a)).all()


def test_sqrt(ta):
    t, a = ta
    assert (t.sqrt().numpy() == np.sqrt(a)).all()


def test_shape(ta):
    t, a = ta
    assert t.shape == a.shape


def test_reshape(ta):
    t, a = ta
    t_shape = (1,) + t.shape + (1,)
    a_shape = (1,) + a.shape + (1,)
    assert (t.reshape(t_shape).numpy() == a.reshape(a_shape)).all()


def test_clip(ta):
    t, a = ta
    assert (t.clip(2, 3.5).numpy() == a.clip(2, 3.5)).all()


def test_sign(ta):
    t, a = ta
    assert (t.sign().numpy() == np.sign(a)).all()


def test_sum(ta):
    t, a = ta
    assert t.sum().numpy() == a.sum()
    assert t.sum(axis=0).numpy() == a.sum(axis=0)
    assert (t.sum(axis=0, keepdims=True).numpy() == a.sum(axis=0, keepdims=True)).all()


def test_mean(ta):
    t, a = ta
    assert t.mean().numpy() == a.mean()
    assert t.mean(axis=0).numpy() == a.mean(axis=0)
    assert (
        t.mean(axis=0, keepdims=True).numpy() == a.mean(axis=0, keepdims=True)
    ).all()


def test_tanh(ta):
    t, a = ta
    np.testing.assert_allclose(t.tanh().numpy(), np.tanh(a), rtol=1e-06)


def test_abs(ta):
    t, a = ta
    assert (abs(t).numpy() == abs(a)).all()
    assert (t.abs().numpy() == abs(a)).all()


def test_min(ta):
    t, a = ta
    assert t.min().numpy() == a.min()
    assert t.min(axis=0).numpy() == a.min(axis=0)
    assert (t.min(axis=0, keepdims=True).numpy() == a.min(axis=0, keepdims=True)).all()


def test_max(ta):
    t, a = ta
    assert t.max().numpy() == a.max()
    assert t.max(axis=0).numpy() == a.max(axis=0)
    assert (t.max(axis=0, keepdims=True).numpy() == a.max(axis=0, keepdims=True)).all()


def test_minimum(ttaa):
    t1, t2, a1, a2 = ttaa
    assert (ep.minimum(t1, t2).numpy() == np.minimum(a1, a2)).all()


def test_maximum(ttaa):
    t1, t2, a1, a2 = ttaa
    assert (ep.maximum(t1, t2).numpy() == np.maximum(a1, a2)).all()


def test_argmin(ta):
    t, a = ta
    assert t.argmin().numpy() == a.argmin()
    assert t.argmin(axis=0).numpy() == a.argmin(axis=0)


def test_argmax(ta):
    t, a = ta
    assert t.argmax().numpy() == a.argmax()
    assert t.argmax(axis=0).numpy() == a.argmax(axis=0)


def test_argsort(ta):
    t, a = ta
    assert (t.argsort().numpy() == a.argsort()).all()
