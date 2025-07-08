from typing import Any, Callable, Dict

import torch
from torch import Tensor


class TensorDict:
    """Wrapper for a dictionary of tensors to allow for easy tensor operations"""

    def __init__(self, tensors: Dict[str, Tensor]):
        self.tensors = tensors

    def _check_keys_match(self, other: "TensorDict"):
        if set(self.tensors.keys()) != set(other.tensors.keys()):
            raise ValueError(
                f"Keys don't match: {set(self.tensors.keys())} vs {set(other.tensors.keys())}"
            )

    def _apply_unary(self, op: Callable[[Tensor], Tensor]) -> "TensorDict":
        """Apply unary operation to all tensors"""
        return TensorDict({k: op(v) for k, v in self.tensors.items()})

    def _apply_binary(
        self, other: "TensorDict", op: Callable[[Tensor, Tensor], Tensor]
    ) -> "TensorDict":
        """Apply binary operation between corresponding tensors"""
        self._check_keys_match(other)
        return TensorDict(
            {k: op(self.tensors[k], other.tensors[k]) for k in self.tensors}
        )

    # Arithmetic dunder methods
    def __add__(self, other):
        if isinstance(other, TensorDict):
            return self._apply_binary(other, torch.add)
        # Handle scalar addition
        return self._apply_unary(lambda t: t + other)

    def __sub__(self, other):
        if isinstance(other, TensorDict):
            return self._apply_binary(other, torch.sub)
        return self._apply_unary(lambda t: t - other)

    def __mul__(self, other):
        if isinstance(other, TensorDict):
            return self._apply_binary(other, torch.mul)
        return self._apply_unary(lambda t: t * other)

    def __truediv__(self, other):
        if isinstance(other, TensorDict):
            return self._apply_binary(other, torch.div)
        return self._apply_unary(lambda t: t / other)

    def __radd__(self, other: Any) -> "TensorDict":
        """Handle right-side addition, e.g., for sum([td1, td2])."""
        if isinstance(other, (int, float)) or (
            torch.is_tensor(other) and other.numel() == 1
        ):
            return self._apply_unary(lambda t: other + t)  # Fixed: was applyunary
        return NotImplemented

    def __getattr__(self, name):
        """Automatically forward torch operations and tensor methods"""
        # First check if it's a torch function
        if hasattr(torch, name):
            torch_func = getattr(torch, name)

            def wrapper(*args, **kwargs):
                # If first arg is a TensorDict, it's likely a binary operation
                if args and isinstance(args[0], TensorDict):
                    return self._apply_binary(
                        args[0], lambda x, y: torch_func(x, y, *args[1:], **kwargs)
                    )
                # Otherwise, it's a unary operation
                return self._apply_unary(lambda t: torch_func(t, *args, **kwargs))

            return wrapper

        # Check if all tensors have this method/attribute
        if all(hasattr(tensor, name) for tensor in self.tensors.values()):

            def wrapper(*args, **kwargs):
                return self._apply_unary(lambda t: getattr(t, name)(*args, **kwargs))

            return wrapper

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # Dict-like access
    def __getitem__(self, key):
        return self.tensors[key]

    def __setitem__(self, key, value):
        self.tensors[key] = value

    def keys(self):
        return self.tensors.keys()

    def values(self):
        return self.tensors.values()

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __repr__(self):
        return f"TensorDict({dict(self.tensors)})"

    def to_dict(self) -> Dict[str, Tensor]:
        """Convert back to regular Dict[str, Tensor]"""
        return self.tensors


# Eigenvectors for the activation covariance matrix.
ACTIVATION_EIGENVECTORS_NAME = "activation_eigenvectors"
# Eigenvalues for the activation covariance matrix.
ACTIVATION_EIGENVALUES_NAME = "activation_eigenvalues"
# Eigenvectors for the pseudo-gradient covariance matrix.
GRADIENT_EIGENVECTORS_NAME = "gradient_eigenvectors"
# Eigenvalues for the pseudo-gradient covariance matrix.
GRADIENT_EIGENVALUES_NAME = "gradient_eigenvalues"

# A list of factors to keep track of when performing Eigendecomposition.
EIGENDECOMPOSITION_FACTOR_NAMES = [
    ACTIVATION_EIGENVECTORS_NAME,
    ACTIVATION_EIGENVALUES_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    GRADIENT_EIGENVALUES_NAME,
]
