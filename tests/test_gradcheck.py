# test_finite_diff.py
import pytest
import torch
import torch.nn as nn

from bergson import FiniteDiff


def make_model():
    lin1 = nn.Linear(5, 3, bias=False)
    lin2 = nn.Linear(3, 4, bias=False)
    return nn.Sequential(lin1, lin2)


def clone_params(model):
    "Return a dict of *detached* clones of all parameters."
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def forward_loss(model, x):
    """Dummy forward that attaches a .loss attribute for FiniteDiff."""
    y = model(x)
    return (y**2).mean()


@pytest.mark.parametrize("step_size", [0.05, -0.05])  # check both signs
def test_reversibility(step_size):
    torch.manual_seed(0)

    model = make_model()
    fd = FiniteDiff(model)

    x = torch.randn(7, 5)
    loss = forward_loss(model, x)

    # 1. backward & remember grads/params
    model.zero_grad(set_to_none=True)
    loss.backward()

    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    orig = clone_params(model)

    # 2. compute finite differences
    fd.store(step_size)

    # after compute: parameters should be unchanged, grads gone
    for n, p in model.named_parameters():
        assert torch.allclose(p, orig[n]), f"{n} changed during compute"
        assert p.grad is None, f"{n}.grad not cleared"

    # 3. byte-level swap in context manager
    with fd.apply():
        for n, p in model.named_parameters():
            if n in fd.params:
                expected = orig[n] + step_size * grads[n]
                assert torch.allclose(p, expected), f"{n} not updated inside ctx"

    # 4. on exit, parameters must be *exactly* as before (bit-wise)
    for n, p in model.named_parameters():
        assert torch.equal(p, orig[n]), f"{n} not restored exactly"

    # 5. cleanup code-path
    fd.clear()
    assert fd.params == {}, "params dict not cleared"


def test_multiple_swaps():
    """Calling swap() twice manually should be a no-op overall."""
    model = make_model()
    fd = FiniteDiff(model)

    # Create grads
    x = torch.randn(3, 5)
    out = forward_loss(model, x)
    model.zero_grad()
    out.backward()

    fd.store(0.1)
    before = clone_params(model)

    fd.swap()  # first swap --> weights changed
    fd.swap()  # second swap --> back to original

    after = clone_params(model)
    for n in before:
        assert torch.equal(before[n], after[n]), f"{n} not identical after 2 swaps"
