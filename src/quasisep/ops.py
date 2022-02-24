# -*- coding: utf-8 -*-

__all__ = ["elementwise_add", "elementwise_mul"]

from typing import Optional, Tuple, TypeVar, Union

from quasisep.quasisep import (
    QSM,
    DiagQSM,
    LowerTriQSM,
    SquareQSM,
    StrictLowerTriQSM,
    StrictUpperTriQSM,
    SymmQSM,
    UpperTriQSM,
)

AnyQSM = Union[
    QSM,
    SymmQSM,
    SquareQSM,
    LowerTriQSM,
    UpperTriQSM,
    StrictLowerTriQSM,
    StrictUpperTriQSM,
    DiagQSM,
]
F = TypeVar("F", DiagQSM, StrictLowerTriQSM, StrictUpperTriQSM)


def elementwise_add(a: AnyQSM, b: AnyQSM) -> Optional[AnyQSM]:
    diag_a, lower_a, upper_a = deconstruct(a)
    diag_b, lower_b, upper_b = deconstruct(b)

    diag = add_two(diag_a, diag_b)
    lower = add_two(lower_a, lower_b)
    upper = add_two(upper_a, upper_b)

    is_symm_a = isinstance(a, SymmQSM) or isinstance(a, DiagQSM)
    is_symm_b = isinstance(b, SymmQSM) or isinstance(b, DiagQSM)
    return construct(diag, lower, upper, is_symm_a and is_symm_b)


def elementwise_mul(a: AnyQSM, b: AnyQSM) -> Optional[AnyQSM]:
    diag_a, lower_a, upper_a = deconstruct(a)
    diag_b, lower_b, upper_b = deconstruct(b)

    diag = mul_two(diag_a, diag_b)
    lower = mul_two(lower_a, lower_b)
    upper = mul_two(upper_a, upper_b)

    is_symm_a = isinstance(a, SymmQSM) or isinstance(a, DiagQSM)
    is_symm_b = isinstance(b, SymmQSM) or isinstance(b, DiagQSM)
    return construct(diag, lower, upper, is_symm_a and is_symm_b)


def deconstruct(
    a: AnyQSM,
) -> Tuple[
    Optional[DiagQSM], Optional[StrictLowerTriQSM], Optional[StrictUpperTriQSM]
]:
    diag = a if isinstance(a, DiagQSM) else getattr(a, "diag", None)
    lower = (
        a if isinstance(a, StrictLowerTriQSM) else getattr(a, "lower", None)
    )
    upper = None
    if isinstance(a, StrictUpperTriQSM):
        upper = a
    elif isinstance(a, SymmQSM):
        upper = a.lower.transpose()
    elif hasattr(a, "upper"):
        upper = getattr(a, "upper")
    return diag, lower, upper


def construct(
    diag: Optional[DiagQSM],
    lower: Optional[StrictLowerTriQSM],
    upper: Optional[StrictUpperTriQSM],
    symm: bool,
) -> Optional[AnyQSM]:
    if lower is None and upper is None:
        return diag

    if symm:
        assert diag is not None
        assert lower is not None
        return SymmQSM(diag=diag, lower=lower)

    if lower is None and upper is None:
        return diag

    if lower is None:
        if diag is None:
            return upper
        else:
            assert upper is not None
            return UpperTriQSM(diag=diag, upper=upper)

    elif upper is None:
        if diag is None:
            return lower
        else:
            assert lower is not None
            return LowerTriQSM(diag=diag, lower=lower)

    elif diag is None:
        # We would hit here if we add a StrictLower to a StrictUpper; is this an
        # ok way to handle that?
        return None

    return SquareQSM(diag=diag, lower=lower, upper=upper)


def add_two(a: Optional[F], b: Optional[F]) -> Optional[F]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a.self_add(b)


def mul_two(a: Optional[F], b: Optional[F]) -> Optional[F]:
    if a is None or b is None:
        return None
    return a.self_mul(b)
