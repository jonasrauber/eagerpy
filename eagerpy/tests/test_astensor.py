import numpy as np
import torch
import tensorflow as tf

import eagerpy as ep


def test_from_tensorflow():
    x = np.arange(5).astype(np.float32)
    x = tf.constant(x)
    x = ep.astensor(x)
    assert isinstance(x, ep.TensorFlowTensor)


def test_from_pytorch():
    x = np.arange(5).astype(np.float32)
    x = torch.from_numpy(x)
    x = ep.astensor(x)
    assert isinstance(x, ep.PyTorchTensor)


def test_from_pytorch_cuda():
    x = np.arange(5).astype(np.float32)
    x = torch.from_numpy(x).cuda()
    x = ep.astensor(x)
    assert isinstance(x, ep.PyTorchTensor)


def test_from_numpy():
    x = np.arange(5).astype(np.float32)
    x = ep.astensor(x)
    assert isinstance(x, ep.NumPyTensor)
