from importlib import import_module
import inspect
from types import ModuleType
import functools
from . import astensor


def wrap(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        try:
            result = astensor(result)
        except ValueError:
            pass
        return result

    return wrapper


class ModuleWrapper(ModuleType):
    """A wrapper for modules that delays the import until it is needed
    and wraps the output of functions as EagerPy tensors"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.__doc__ is None:
            self.__doc__ = f"EagerPy wrapper of the '{self.__name__}' module"

    def __dir__(self):
        # makes sure tab completion works
        return import_module(self.__name__).__dir__()

    def __getattr__(self, name):
        attr = getattr(import_module(self.__name__), name)
        if callable(attr):
            attr = wrap(attr)
        elif inspect.ismodule(attr):
            attr = ModuleWrapper(attr.__name__)
        return attr


torch = ModuleWrapper("torch")
tensorflow = ModuleWrapper("tensorflow")
jax = ModuleWrapper("jax")
numpy = ModuleWrapper("numpy")
