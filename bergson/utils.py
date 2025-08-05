from typing import Any, Type, TypeVar, cast

import torch
from torch import nn, Tensor
from transformers import PreTrainedModel

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)  # type: ignore[return-value]


def get_layer_list(model: PreTrainedModel) -> nn.ModuleList:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        mod
        for mod in model.base_model.modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]

@torch.compile
def unpack_bits(bits: Tensor, dtype: torch.dtype) -> Tensor:
    """Unpack a flat bit-packed tensor of dtype `torch.uint8` into a tensor of the given dtype."""
    assert bits.dtype == torch.uint8
    assert bits.ndim == 1
    result = torch.empty(bits.shape[0] * 8, dtype=dtype, device=bits.device)
    for i in range(8):
        result[i::8] = bits & (1 << i)
    return result
