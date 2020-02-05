import pytest
import eagerpy as ep


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_kl_div_with_logits(dummy: ep.Tensor, axis: int) -> None:
    logits_p = logits_q = ep.arange(dummy, 12).float32().reshape((3, 4))
    assert (ep.kl_div_with_logits(logits_p, logits_q, axis=axis) == 0).all()
