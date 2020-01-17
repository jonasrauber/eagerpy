from . import inf


def l0(x, axis=None, keepdims=False):
    return (x != 0).sum(axis=axis, keepdims=keepdims)


def l1(x, axis=None, keepdims=False):
    return x.abs().sum(axis=axis, keepdims=keepdims)


def l2(x, axis=None, keepdims=False):
    return x.square().sum(axis=axis, keepdims=keepdims).sqrt()


def linf(x, axis=None, keepdims=False):
    return x.abs().max(axis=axis, keepdims=keepdims)


def lp(x, p, axis=None, keepdims=False):
    if p == 0:
        return l0(x, axis=axis, keepdims=keepdims)
    if p == 1:
        return l1(x, axis=axis, keepdims=keepdims)
    if p == 2:
        return l2(x, axis=axis, keepdims=keepdims)
    if p == inf:
        return linf(x, axis=axis, keepdims=keepdims)
    return x.abs().pow(p).sum(axis=axis, keepdims=keepdims).pow(1.0 / p)
