import pytest
import eagerpy as ep


def pytest_addoption(parser):
    parser.addoption("--backend")


@pytest.fixture(scope="session")
def dummy(request):
    backend = request.config.option.backend
    if backend is None:
        pytest.skip()
    return ep.utils.get_dummy(backend)


@pytest.fixture(scope="session")
def t1(dummy):
    return ep.arange(dummy, 5).float32()


@pytest.fixture(scope="session")
def t2(dummy):
    return ep.arange(dummy, 7, 17, 2).float32()


@pytest.fixture(scope="session", params=["t1", "t2"])
def t(request, t1, t2):
    return {"t1": t1, "t2": t2}[request.param]
