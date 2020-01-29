from . import Tensor


def kl_div_with_logits(
    logits_p: Tensor, logits_q: Tensor, axis=-1, keepdims=False
) -> Tensor:
    log_p = logits_p.log_softmax(axis=axis)
    log_q = logits_q.log_softmax(axis=axis)
    p = logits_p.softmax(axis=-1)
    return (p * (log_p - log_q)).sum(axis=axis, keepdims=keepdims)
