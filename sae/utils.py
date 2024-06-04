from typing import Any, Type, TypeVar, cast

from torch import Tensor
import torch.distributed as dist

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def maybe_all_reduce(x: Tensor, op: str = "mean") -> Tensor:
    """Reduce a tensor across all processes."""
    if not dist.is_initialized():
        return x

    if op == "sum":
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    elif op == "mean":
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    else:
        raise ValueError(f"Unknown reduction op '{op}'")

    return x