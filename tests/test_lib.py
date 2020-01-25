import eagerpy as ep


def test_kl_div_with_logits():
    logits_p = logits_q = ep.numpy.arange(12).float32().reshape((3, 4))
    assert (ep.kl_div_with_logits(logits_p, logits_q) == 0).all()
