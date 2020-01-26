import eagerpy as ep


class _Indexable:
    __slots__ = ()

    def __getitem__(self, index):
        return index


index = _Indexable()


def get_dummy(framework):
    if framework == "pytorch":
        x = ep.torch.zeros(0)
        assert isinstance(x, ep.PyTorchTensor)
    elif framework == "pytorch-gpu":
        x = ep.torch.zeros(0, device="cuda:0")
        assert isinstance(x, ep.PyTorchTensor)
    elif framework == "tensorflow":
        x = ep.tensorflow.zeros(0)
        assert isinstance(x, ep.TensorFlowTensor)
    elif framework == "jax":
        x = ep.jax.numpy.zeros(0)
        assert isinstance(x, ep.JAXTensor)
    elif framework == "numpy":
        x = ep.numpy.zeros(0)
        assert isinstance(x, ep.NumPyTensor)
    else:
        raise ValueError(f"unknown framework: {framework}")  # pragma: no cover
    return x.float32()
