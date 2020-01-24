import pytest
import eagerpy as ep


def pytest_addoption(parser):
    parser.addoption("--backend")


@pytest.fixture(scope="session")
def dummy(request):
    backend = request.config.option.backend
    if backend is None:
        pytest.skip()
    if backend == "pytorch":
        x = ep.torch.zeros(0)
        assert isinstance(x, ep.PyTorchTensor)
    elif backend == "pytorch-gpu":
        x = ep.torch.zeros(0, device="cuda:0")
        assert isinstance(x, ep.PyTorchTensor)
    elif backend == "tensorflow":
        x = ep.tensorflow.zeros(0)
        assert isinstance(x, ep.TensorFlowTensor)
    elif backend == "jax":
        x = ep.jax.numpy.zeros(0)
        assert isinstance(x, ep.JAXTensor)
    elif backend == "numpy":
        x = ep.numpy.zeros(0)
        assert isinstance(x, ep.NumPyTensor)
    else:
        raise ValueError(f"unknown backend: {backend}")
    return x


@pytest.fixture(scope="session")
def t1(dummy):
    return ep.arange(dummy, 5).float32()


@pytest.fixture(scope="session")
def t2(dummy):
    return ep.arange(dummy, 7, 17, 2).float32()


@pytest.fixture(scope="session", params=["t1", "t2"])
def t(request, t1, t2):
    return {"t1": t1, "t2": t2}[request.param]
