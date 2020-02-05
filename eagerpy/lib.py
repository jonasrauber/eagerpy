from .tensor import TensorType


def kl_div_with_logits(
    logits_p: TensorType, logits_q: TensorType, axis: int = -1, keepdims: bool = False
) -> TensorType:
    log_p = logits_p.log_softmax(axis=axis)
    log_q = logits_q.log_softmax(axis=axis)
    p = logits_p.softmax(axis=-1)
    return (p * (log_p - log_q)).sum(axis=axis, keepdims=keepdims)
