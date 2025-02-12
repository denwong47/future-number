# -*- coding: utf-8 -*-
"""
A enum class to represent the different types of operators that can be used on
the :class:`FutureNumber` instances.
"""

from enum import Enum
from typing import Generic, TypeVar

from . import protocols

TA = TypeVar("TA", bound=protocols.SupportsArithmeticOperators)
TC = TypeVar("TC", bound=protocols.SupportsRichComparison)


class FutureOperator(str, Enum):
    """
    An abstract class representing a future operator that can be applied to a
    :class:`FutureNumber`, and will be evaluated when the final value is needed.
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"{type(self).__name__}({self.value!r})"


class FutureArithmaticOperator(Generic[TA], FutureOperator):
    """
    A class representing a future arithmatic operator that can be applied to a
    :class:`FutureNumber`, and will be evaluated when the final value is needed.
    """

    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    FLOOR_DIVIDE = "//"
    MODULO = "%"
    POWER = "**"

    def operate(self, lhs: TA, rhs: TA) -> TA:
        """
        Apply the operator to the two values and return the result.
        """
        match self:
            case FutureArithmaticOperator.ADD:
                return lhs + rhs
            case FutureArithmaticOperator.SUBTRACT:
                return lhs - rhs
            case FutureArithmaticOperator.MULTIPLY:
                return lhs * rhs
            case FutureArithmaticOperator.DIVIDE:
                return lhs / rhs
            case FutureArithmaticOperator.FLOOR_DIVIDE:
                return lhs // rhs
            case FutureArithmaticOperator.MODULO:
                return lhs % rhs
            case FutureArithmaticOperator.POWER:
                return lhs**rhs
            case _:
                raise ValueError(f"Unsupported operator: {self}")


class FutureCompareOperator(Generic[TC], FutureOperator):
    """
    A class representing a future arithmatic operator that can be applied to a
    :class:`FutureNumber`, and will be evaluated when the final value is needed.
    """

    LESS_THAN = "<"
    LESS_THAN_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_THAN_EQUAL = ">="

    def operate(self, lhs: TC, rhs: TC) -> bool:
        """
        Apply the operator to the two values and return the result.
        """
        match self:
            case FutureCompareOperator.LESS_THAN:
                return lhs < rhs
            case FutureCompareOperator.LESS_THAN_EQUAL:
                return lhs <= rhs
            case FutureCompareOperator.EQUAL:
                return lhs == rhs
            case FutureCompareOperator.NOT_EQUAL:
                return lhs != rhs
            case FutureCompareOperator.GREATER_THAN:
                return lhs > rhs
            case FutureCompareOperator.GREATER_THAN_EQUAL:
                return lhs >= rhs
            case _:
                raise ValueError(f"Unsupported operator: {self}")
