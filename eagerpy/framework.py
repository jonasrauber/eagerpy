from . import istensor

newaxis = None
inf = float("inf")
nan = float("nan")


def clip(t, *args, **kwargs):
    return t.clip(*args, **kwargs)


def abs(t):
    return t.abs()


def sign(t):
    return t.sign()


def sqrt(t):
    return t.sqrt()


def square(t):
    return t.square()


def pow(t, exponent):
    return t.pow(exponent)


def tanh(t):
    return t.tanh()


def arctanh(t):
    return t.arctanh()


def sum(t, *args, **kwargs):
    return t.sum(*args, **kwargs)


def mean(t, *args, **kwargs):
    return t.sum(*args, **kwargs)


def min(t, *args, **kwargs):
    return t.min(*args, **kwargs)


def max(t, *args, **kwargs):
    return t.max(*args, **kwargs)


def minimum(x, y):
    if not istensor(x):
        return y.minimum(x)
    return x.minimum(y)


def maximum(x, y):
    if not istensor(x):
        return y.maximum(x)
    return x.maximum(y)


def argmin(t, *args, **kwargs):
    return t.argmin(*args, **kwargs)


def argmax(t, *args, **kwargs):
    return t.argmax(*args, **kwargs)


def argsort(t, *args, **kwargs):
    return t.argsort(*args, **kwargs)


def uniform(t, *args, **kwargs):
    return t.uniform(*args, **kwargs)


def normal(t, *args, **kwargs):
    return t.normal(*args, **kwargs)


def ones(t, *args, **kwargs):
    return t.ones(*args, **kwargs)


def zeros(t, *args, **kwargs):
    return t.zeros(*args, **kwargs)


def ones_like(t, *args, **kwargs):
    return t.ones_like(*args, **kwargs)


def zeros_like(t, *args, **kwargs):
    return t.zeros_like(*args, **kwargs)


def full_like(t, *args, **kwargs):
    return t.full_like(*args, **kwargs)


def onehot_like(t, *args, **kwargs):
    return t.onehot_like(*args, **kwargs)


def from_numpy(t, *args, **kwargs):
    return t.from_numpy(*args, **kwargs)


def concatenate(tensors, axis=0):
    t = tensors[0]
    return t._concatenate(tensors, axis=axis)


def transpose(t, axes=None):
    return t.transpose(axes=axes)


def logical_and(x, y):
    if not istensor(x):
        return y.logical_and(x)
    return x.logical_and(y)


def logical_or(x, y):
    if not istensor(x):
        return y.logical_or(x)
    return x.logical_or(y)


def logical_not(t):
    return t.logical_not()


def exp(t):
    return t.exp()


def log(t):
    return t.log()


def log2(t):
    return t.log2()


def log10(t):
    return t.log10()


def log1p(t):
    return t.log1p()


def where(condition, x, y):
    return condition.where(x, y)


def tile(t, multiples):
    return t.tile(multiples)


def matmul(x, y):
    return x.matmul(y)


def softmax(t, axis=-1):
    return t.softmax(axis=axis)


def log_softmax(t, axis=-1):
    return t.log_softmax(axis=axis)


def stack(tensors, axis=0):
    t = tensors[0]
    return t._stack(tensors, axis=axis)


def squeeze(t, *args, **kwargs):
    return t.squeeze(*args, **kwargs)


def expand_dims(t, *args, **kwargs):
    return t.expand_dims(*args, **kwargs)


def full(t, *args, **kwargs):
    return t.full(*args, **kwargs)


def index_update(t, *args, **kwargs):
    return t.index_update(*args, **kwargs)


def arange(t, *args, **kwargs):
    return t.arange(*args, **kwargs)


def cumsum(t, *args, **kwargs):
    return t.cumsum(*args, **kwargs)


def flip(t, *args, **kwargs):
    return t.flip(*args, **kwargs)


def meshgrid(t, *args, **kwargs):
    return t.meshgrid(*args, **kwargs)


def pad(t, *args, **kwargs):
    return t.pad(*args, **kwargs)


def isnan(t):
    return t.isnan()


def isinf(t):
    return t.isinf()


def all(t, *args, **kwargs):
    return t.all(*args, **kwargs)


def any(t, *args, **kwargs):
    return t.any(*args, **kwargs)


def crossentropy(logits, labels):
    return logits.crossentropy(labels)


def value_and_grad_fn(t, f, has_aux=False):
    return t._value_and_grad_fn(f, has_aux=has_aux)


def value_and_grad(f, t, *args, **kwargs):
    return t.value_and_grad(f, *args, **kwargs)


def value_aux_and_grad(f, t, *args, **kwargs):
    return t.value_aux_and_grad(f, *args, **kwargs)


def reshape(t, shape):
    return t.reshape(shape)
