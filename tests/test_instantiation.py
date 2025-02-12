# -*- coding: utf-8 -*-

import re

import pytest

from future_number import FutureNumber
from future_number.models.base import (
    FutureComparisonResult,
    FutureScalarNumber,
    FutureVectorNumber,
)

str_reprs = {
    "empty": {
        "instance": FutureNumber(),
        "cls": FutureScalarNumber,
        "repr": "FutureScalarNumber()",
        "str": re.compile(r"^Unknown<[0-9a-f]{10,}>$"),
    },
    "empty_with_placeholder": {
        "instance": FutureNumber("?"),
        "cls": FutureScalarNumber,
        "repr": "FutureScalarNumber('?')",
        "str": "?",
    },
    "with_value": {
        "instance": FutureNumber(inner=42),
        "cls": FutureScalarNumber,
        "repr": "FutureScalarNumber(inner=42)",
        "str": "42",
    },
    "with_value_and_placeholder": {
        "instance": FutureNumber("?", inner=42),
        "cls": FutureScalarNumber,
        "repr": "FutureScalarNumber('?', inner=42)",
        "str": "42",
    },
    "vector": {
        "instance": FutureNumber("X") + FutureNumber("Y"),
        "cls": FutureVectorNumber,
        "repr": "FutureVectorNumber(FutureScalarNumber('X'), FutureScalarNumber('Y'), operator=FutureArithmaticOperator('+'))",
        "str": "X + Y",
    },
    "vector_with_values": {
        "instance": FutureNumber("X", inner=42) + FutureNumber("Y", inner=24),
        "cls": FutureVectorNumber,
        "repr": "FutureVectorNumber(FutureScalarNumber('X', inner=42), FutureScalarNumber('Y', inner=24), operator=FutureArithmaticOperator('+'))",
        "str": "42 + 24",
    },
    "vector_with_values_and_scalar": {
        "instance": FutureNumber("X", inner=42) + 24,
        "cls": FutureVectorNumber,
        "repr": "FutureVectorNumber(FutureScalarNumber('X', inner=42), 24, operator=FutureArithmaticOperator('+'))",
        "str": "42 + 24",
    },
    "comparison": {
        "instance": FutureNumber("X") < FutureNumber("Y"),
        "cls": FutureComparisonResult,
        "repr": "FutureComparisonResult(FutureScalarNumber('X'), FutureScalarNumber('Y'), operator=FutureCompareOperator('<'))",
        "str": "X < Y",
    },
    "comparison_with_values": {
        "instance": FutureNumber("X", inner=42) < FutureNumber("Y", inner=24),
        "cls": FutureComparisonResult,
        "repr": "FutureComparisonResult(FutureScalarNumber('X', inner=42), FutureScalarNumber('Y', inner=24), operator=FutureCompareOperator('<'))",
        "str": "42 < 24",
    },
    "comparison_with_values_and_scalar": {
        "instance": FutureNumber("X", inner=42) < 24,
        "cls": FutureComparisonResult,
        "repr": "FutureComparisonResult(FutureScalarNumber('X', inner=42), 24, operator=FutureCompareOperator('<'))",
        "str": "42 < 24",
    },
}


@pytest.mark.parametrize(
    ("instance", "cls", "expected"),
    [
        pytest.param(
            test["instance"],
            test["cls"],
            test["repr"],
            id=name,
        )
        for name, test in str_reprs.items()
    ],
)
def test_repr(instance: FutureNumber, cls: type[FutureNumber], expected: str):
    assert type(instance) is cls, (
        f"The instance should be of type {cls}, found {type(instance).__name__} instead."
    )
    assert repr(instance) == expected, (
        f"The repr of {cls} should be {expected}, found {repr(instance)}"
    )


@pytest.mark.parametrize(
    ("instance", "cls", "expected"),
    [
        pytest.param(
            test["instance"],
            test["cls"],
            test["str"],
            id=name,
        )
        for name, test in str_reprs.items()
    ],
)
def test_str(
    instance: FutureNumber,
    cls: type[FutureNumber],
    expected: str | re.Pattern,
):
    assert type(instance) is cls, (
        f"The instance should be of type {cls}, found {type(instance).__name__} instead."
    )

    if isinstance(expected, re.Pattern):
        assert expected.match(str(instance)), (
            f"The str of {instance!r} should match {expected}, found {str(instance)}"
        )
    else:
        assert str(instance) == expected, (
            f"The str of {instance!r} should be {expected}, found {str(instance)}"
        )
