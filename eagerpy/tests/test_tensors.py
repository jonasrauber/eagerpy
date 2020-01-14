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


@pytest.fixture(params=["tf1", "th1", "th1g", "np1", "tf2", "th2", "th2g", "np2"])
def ta(request, tf1, th1, th1g, np1, a1, tf2, th2, th2g, np2, a2):
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


@pytest.fixture(params=["tf", "th", "thg", "np"])
def ttaa(request, tf1, th1, th1g, np1, a1, tf2, th2, th2g, np2, a2):
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


def test_transpose(ta):
    t, a = ta
    assert (t.transpose().numpy() == np.transpose(a)).all()
    if a.ndim == 3:
        axes = (1, 2, 0)
        assert (t.transpose(axes=axes).numpy() == np.transpose(a, axes=axes)).all()


def test_all(ta):
    t, a = ta
    t = t < 10
    a = a < 10
    assert t.all().numpy() == a.all()
    assert t.all(axis=0).numpy() == a.all(axis=0)
    assert (t.all(axis=0, keepdims=True).numpy() == a.all(axis=0, keepdims=True)).all()


def test_any(ta):
    t, a = ta
    t = t < 0
    a = a < 0
    assert t.any().numpy() == a.any()
    assert t.any(axis=0).numpy() == a.any(axis=0)
    assert (t.any(axis=0, keepdims=True).numpy() == a.any(axis=0, keepdims=True)).any()


def test_logical_and(ta):
    t, a = ta
    t1 = t < 3
    t2 = t >= 3
    a1 = a < 3
    a2 = a >= 3
    assert (t1.logical_and(t2).numpy() == np.logical_and(a1, a2)).all()


def test_logical_or(ta):
    t, a = ta
    t1 = t < 3
    t2 = t >= 3
    a1 = a < 3
    a2 = a >= 3
    assert (t1.logical_or(t2).numpy() == np.logical_or(a1, a2)).all()


def test_logical_not(ta):
    t, a = ta
    t1 = t < 3
    t2 = t >= 3
    a1 = a < 3
    a2 = a >= 3
    assert (t1.logical_not() == t2).all()
    assert (t2.logical_not() == t1).all()
    assert (t1.logical_not().numpy() == np.logical_not(a1)).all()
    assert (t2.logical_not().numpy() == np.logical_not(a2)).all()


def test_exp(ta):
    t, a = ta
    np.testing.assert_allclose(np.exp(a), t.exp().numpy())


def test_log(ta):
    t, a = ta
    np.testing.assert_allclose(np.log(a), t.log().numpy())


def test_log2(ta):
    t, a = ta
    np.testing.assert_allclose(np.log2(a), t.log2().numpy())


def test_log10(ta):
    t, a = ta
    np.testing.assert_allclose(np.log10(a), t.log10().numpy())


def test_log1p(ta):
    t, a = ta
    np.testing.assert_allclose(np.log1p(a), t.log1p().numpy())


def test_where(ta):
    t, a = ta
    assert (ep.where(t >= 3, t, -t).numpy() == np.where(a >= 3, a, -a)).all()


def test_tile(ta):
    t, a = ta
    m = (3,) * a.ndim
    assert (ep.tile(t, m).numpy() == np.tile(a, m)).all()


def test_matmul(ta):
    t, a = ta
    if t.ndim == 1:
        t = t[None]
        a = a[None]
        assert (ep.matmul(t, t.T).numpy() == np.matmul(a, a.T)).all()
        t = t[0, :, None]
        a = a[0, :, None]
        assert (ep.matmul(t, t.T).numpy() == np.matmul(a, a.T)).all()
    elif t.ndim == 2:
        assert (ep.matmul(t, t.T).numpy() == np.matmul(a, a.T)).all()
    elif t.ndim > 2:
        axes = list(range(t.ndim - 2))
        t = t.sum(axes)
        a = a.sum(axes)
        assert (ep.matmul(t, t.T).numpy() == np.matmul(a, a.T)).all()


def test_softmax(ta):
    t, a = ta
    s = t.softmax(axis=-1)
    np.testing.assert_allclose(s.sum(axis=-1).numpy(), 1.0, rtol=1e-6)
    assert (s.numpy() >= 0).all()
    assert (s.numpy() <= 1).all()


def test_stack(ttaa):
    t1, t2, a1, a2 = ttaa
    assert (ep.stack([t1, t2]).numpy() == np.stack([a1, a2])).all()
    assert (ep.stack([t1, t2], axis=1).numpy() == np.stack([a1, a2], axis=1)).all()


def test_expand_dims(ta):
    t, a = ta
    assert t.expand_dims(axis=0).shape == np.expand_dims(a, axis=0).shape
    assert t.expand_dims(axis=1).shape == np.expand_dims(a, axis=1).shape


def test_squeeze(ta):
    t, a = ta
    assert (
        t.expand_dims(axis=0).squeeze(axis=0).shape
        == np.expand_dims(a, axis=0).squeeze(axis=0).shape
    )
    assert (
        t.expand_dims(axis=0).squeeze().shape
        == np.expand_dims(a, axis=0).squeeze().shape
    )
    assert (
        t.expand_dims(axis=0).expand_dims(axis=1).squeeze(axis=0).shape
        == np.expand_dims(np.expand_dims(a, axis=0), axis=1).squeeze(axis=0).shape
    )
    assert (
        t.expand_dims(axis=0).expand_dims(axis=1).squeeze(axis=1).shape
        == np.expand_dims(np.expand_dims(a, axis=0), axis=1).squeeze(axis=1).shape
    )
    assert (
        t.expand_dims(axis=0).expand_dims(axis=1).squeeze(axis=(0, 1)).shape
        == np.expand_dims(np.expand_dims(a, axis=0), axis=1).squeeze(axis=(0, 1)).shape
    )
    assert (
        t.expand_dims(axis=0).expand_dims(axis=1).squeeze().shape
        == np.expand_dims(np.expand_dims(a, axis=0), axis=1).squeeze().shape
    )


def test_ones_like(ta):
    t, a = ta
    assert ep.ones_like(t).numpy().shape == np.ones_like(a).shape
    assert (ep.ones_like(t).numpy() == np.ones_like(a)).all()


def test_zeros_like(ta):
    t, a = ta
    assert ep.zeros_like(t).numpy().shape == np.zeros_like(a).shape
    assert (ep.zeros_like(t).numpy() == np.zeros_like(a)).all()


def test_full_like(ta):
    t, a = ta
    assert ep.full_like(t, 5).numpy().shape == np.full_like(a, 5).shape
    assert (ep.full_like(t, 5).numpy() == np.full_like(a, 5)).all()


def test_ones(ta):
    t, a = ta
    assert ep.ones(t, (2, 3)).numpy().shape == np.ones((2, 3)).shape
    assert (ep.ones(t, (2, 3)).numpy() == np.ones((2, 3))).all()
    assert ep.ones(t, 5).numpy().shape == np.ones(5).shape


def test_zeros(ta):
    t, a = ta
    assert ep.zeros(t, (2, 3)).numpy().shape == np.zeros((2, 3)).shape
    assert (ep.zeros(t, (2, 3)).numpy() == np.zeros((2, 3))).all()
    assert ep.zeros(t, 5).numpy().shape == np.zeros(5).shape


def test_uniform(ta):
    t, a = ta
    assert ep.uniform(t, (2, 3)).numpy().shape == np.random.uniform(size=(2, 3)).shape
    assert ep.uniform(t, 5).numpy().shape == np.random.uniform(size=5).shape


def test_normal(ta):
    t, a = ta
    assert ep.normal(t, (2, 3)).numpy().shape == np.random.normal(size=(2, 3)).shape
    assert ep.normal(t, 5).numpy().shape == np.random.normal(size=5).shape


def test_arctanh(ta):
    t, a = ta
    np.testing.assert_allclose(t.arctanh().numpy(), np.arctanh(a), rtol=1e-06)


def test_full(ta):
    t, a = ta
    assert ep.full(t, (2, 3), 4.0).numpy().shape == np.full((2, 3), 4.0).shape
    assert (ep.full(t, (2, 3), 4.0).numpy() == np.full((2, 3), 4.0)).all()
    assert ep.full(t, 5, 4.0).numpy().shape == np.full(5, 4.0).shape


def test_arange(ta):
    t, a = ta
    assert ep.arange(t, 6).numpy().shape == np.arange(6).shape
    assert (ep.arange(t, 6).numpy() == np.arange(6)).all()


def test_cumsum(ta):
    t, a = ta
    assert (ep.cumsum(t, axis=0).numpy() == np.cumsum(a, axis=0)).all()
    assert (ep.cumsum(t, axis=-1).numpy() == np.cumsum(a, axis=-1)).all()
    assert (ep.cumsum(t, axis=None).numpy() == np.cumsum(a, axis=None)).all()
    assert (ep.cumsum(t).numpy() == np.cumsum(a, axis=None)).all()


def test_flip(ta):
    t, a = ta
    assert (ep.flip(t, axis=0).numpy() == np.flip(a, axis=0)).all()
    assert (ep.flip(t, axis=-1).numpy() == np.flip(a, axis=-1)).all()
    assert (ep.flip(t, axis=None).numpy() == np.flip(a, axis=None)).all()
    assert (ep.flip(t).numpy() == np.flip(a, axis=None)).all()


def test_meshgrid(ta):
    t, a = ta
    t = ep.arange(t, 5)
    a = np.arange(5)
    t2 = ep.arange(t, 3)
    a2 = np.arange(3)
    assert len(ep.meshgrid(t, t2)) == len(np.meshgrid(a, a2)) == 2
    assert (ep.meshgrid(t, t2)[0].numpy() == np.meshgrid(a, a2)[0]).all()
    assert (ep.meshgrid(t, t2)[1].numpy() == np.meshgrid(a, a2)[1]).all()
    assert (
        ep.meshgrid(t, t2, indexing="ij")[0].numpy()
        == np.meshgrid(a, a2, indexing="ij")[0]
    ).all()
    assert (
        ep.meshgrid(t, t2, indexing="ij")[1].numpy()
        == np.meshgrid(a, a2, indexing="ij")[1]
    ).all()


def test_pad(ta):
    t, a = ta
    a = np.arange(120).reshape((2, 3, 4, 5)).astype(np.float32)
    t = ep.from_numpy(t, a)
    assert (
        ep.pad(t, ((0, 0), (0, 0), (2, 3), (1, 2))).numpy()
        == np.pad(a, ((0, 0), (0, 0), (2, 3), (1, 2)))
    ).all()
    assert (
        ep.pad(t, ((0, 0), (0, 0), (2, 3), (1, 2)), value=-2).numpy()
        == np.pad(a, ((0, 0), (0, 0), (2, 3), (1, 2)), constant_values=-2)
    ).all()
    assert (
        ep.pad(t, ((0, 0), (0, 0), (2, 3), (1, 2)), mode="reflect").numpy()
        == np.pad(a, ((0, 0), (0, 0), (2, 3), (1, 2)), mode="reflect")
    ).all()


def test_isnan(ta):
    t, a = ta
    assert (t.isnan().numpy() == np.isnan(a)).all()
    t = t + np.nan
    a = a + np.nan
    assert (t.isnan().numpy() == np.isnan(a)).all()


def test_isinf(ta):
    t, a = ta
    assert (t.isinf().numpy() == np.isinf(a)).all()
    t = t + np.inf
    a = a + np.inf
    assert (t.isinf().numpy() == np.isinf(a)).all()
    t, a = ta
    t = t - np.inf
    a = a - np.inf
    assert (t.isinf().numpy() == np.isinf(a)).all()


def test_crossentropy(ta):
    t, a = ta
    np.random.seed(22)
    a = np.random.uniform(size=(10, 5)).astype(np.float32)
    tx = ep.from_numpy(t, a)  # EagerPy t-like tensor
    tn = ep.astensor(a)  # EagerPy NumPy tensor
    txl = tx.argmax(axis=-1)
    tnl = tn.argmax(axis=-1)
    np.testing.assert_allclose(
        tx.crossentropy(txl).numpy(), tn.crossentropy(tnl).numpy(), rtol=1e-6
    )
