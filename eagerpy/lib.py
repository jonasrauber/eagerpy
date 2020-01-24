def kl_div_with_logits(logits_p, logits_q, axis=-1, keepdims=False):
    log_p = logits_p.log_softmax(axis=axis)
    log_q = logits_q.log_softmax(axis=axis)
    p = logits_p.softmax(axis=-1)
    return (p * (log_p - log_q)).sum(axis=axis, keepdims=keepdims)
