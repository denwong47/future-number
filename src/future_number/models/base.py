# -*- coding: utf-8 -*-
"""
The main model class for export.
"""

import numbers
import threading
from typing import Any, Callable, Generic, ParamSpec, TypeVar, cast

from . import protocols
from .operators import FutureArithmaticOperator, FutureCompareOperator, FutureOperator

T = TypeVar("T")
TA = TypeVar("TA", bound=protocols.SupportsArithmeticOperators)
TC = TypeVar("TC", bound=protocols.SupportsRichComparison)

R = TypeVar("R")


def _arithmatic_wrapper(
    operator: FutureArithmaticOperator,
    *,
    invert: bool = False,
) -> Callable[["FutureNumber[TA]", "FutureNumber[TA]" | TA], "FutureVectorNumber[TA]"]:
    """
    Create a wrapper function for an arithmatic operator.

    This function will create a new :class:`FutureVectorNumber` instance
    with the current number as the left hand side, and the other number as
    the right hand side.

    Parameters
    ----------
    operator : FutureArithmaticOperator
        The operator to apply to the two numbers.

    Returns
    -------
    Callable[[Self, Self], FutureVectorNumber[T]]
        A wrapper function that will create a new :class:`FutureVectorNumber`
        instance.
    """

    def wrapper(
        self: "FutureNumber[TA]", other: "FutureNumber[TA]" | TA
    ) -> "FutureVectorNumber[TA]":
        if invert:
            return FutureVectorNumber(other, self, operator)
        return FutureVectorNumber(self, other, operator)

    return wrapper


def _comparison_wrapper(
    operator: FutureCompareOperator,
) -> Callable[["FutureNumber[TC]", object], "FutureComparisonResult[TC]"]:
    """
    Create a wrapper function for a comparison operator.

    This function will create a new :class:`FutureComparisonResult` instance
    with the current number as the left hand side, and the other number as
    the right hand side.

    Parameters
    ----------
    operator : FutureCompareOperator
        The operator to apply to the two numbers.

    Returns
    -------
    Callable[[Self, Self], FutureComparisonResult[T]]
        A wrapper function that will create a new :class:`FutureComparisonResult`
        instance.
    """

    def wrapper(
        self: "FutureNumber[TC]", other: object
    ) -> "FutureComparisonResult[TC]":
        return FutureComparisonResult(self, cast("FutureNumber[TC]", other), operator)

    return wrapper


class FutureNumber(Generic[T]):
    """
    A class representing a number that is lazy evaluated.

    This allows further expressions to be recorded on the number, and only be
    evaluated when the final value is needed.

    The most basic operation is to create a number with a placeholder, and then
    set the value later::

        >>> from future_number import FutureNumber
        >>> x = FutureNumber("X")
        >>> y = x + 42
        >>> print(y)
        X + 42

    At this point the value of ``x`` has not been set, so the value of ``y`` is
    still a placeholder. You can set the value of ``x`` later::

        >>> x.set(24)
        >>> print(y)
        24 + 42
        >>> print(y.evaluate())
        66

    This is typically used with :class:`Generator` instances. For example, if
    you have a generator that return numbers, and you need to create a normalised
    sequence of those numbers based on some aggregation that could not be
    determined before the whole sequence is exhausted, you can::

        >>> from future_number import FutureNumber
        >>> from collections import defaultdict
        >>> sequence = iter([("red", 1), ("blue", 2), ("red", 2), ("green", 3), ("blue", 2), ("red", 2)])
        >>> categorised_totals = defaultdict(lambda: FutureNumber(inner=0))
        >>> normalised_within_category = defaultdict(list)
        >>> for category, value in sequence:
        ...     normalised_within_category[category].append(
        ...         # accumulate the total of each category inside a `FutureNumber`
        ...         value / categorised_totals[category].mut_add(value)
        ...     )
        >>> normalised_evaluated = {
        ...     k: [v.evaluate() for v in l]
        ...     for k, l in normalised_within_category.items()
        ... }
        >>> normalised_evaluated
        {'red': [0.2, 0.4, 0.4], 'blue': [0.5, 0.5], 'green': [1.0]}

    Without the :class:`FutureNumber` class, you would have to accumulate the
    total of each category in a separate dictionary, and then re-iterate over the
    sequence to normalise the values, which is not memory efficient.

    .. note::
        You still need to consider any other lazy evaluation rules, and whether the
        number at the point of evaluation is set to the correct value.

        For example, if you create another :class:`Generator` that returns the
        normalised values, you need to ensure the original sequence is exhausted
        before you can evaluate the normalised values::

            >>> from future_number import FutureNumber
            >>> total = FutureNumber(inner=0)
            >>> sequence = iter([1, 3, 4])
            >>> normalised = (value / total.mut_add(value) for value in sequence)
            >>> [value.evaluate() for value in normalised]
            [1.0, 0.75, 0.5]

        In the above example, the total value has not finished accumulating at
        the point of evaluation, so the normalised values are incorrect::

            [1 / 1, 3 / 4, 4 / 8] = [1.0, 0.75, 0.5]

        However if you use a :class:`list` comprehension, the total value will
        be set before the normalised values are evaluated::

            >>> from future_number import FutureNumber
            >>> total = FutureNumber(inner=0)
            >>> sequence = iter([1, 3, 4])
            >>> normalised = [value / total.mut_add(value) for value in sequence]
            >>> [value.evaluate() for value in normalised]
            [0.125, 0.375, 0.5]

        Which yields the correct normalised values::

            [1 / 8, 3 / 8, 4 / 8] = [0.125, 0.375, 0.5]

    This class will work on any type that supports the operators that are used
    on the numbers, and will raise a :class:`TypeError` if the operator is not
    supported.

    Specifically, a :class:`FutureNumber` can be a :class:`numpy.ndarray` and
    arithmetic operators are supported::

        >>> import numpy as np
        >>> from future_number import FutureNumber
        >>> arr = FutureNumber("A")
        >>> new_arr = arr * 3 + np.ones((5,), dtype=int)
        >>> arr.set(np.arange(5, dtype=int))
        >>> new_arr.evaluate()
        array([ 1,  4,  7, 10, 13])
    """

    def __new__(cls, *args, **kwargs) -> "FutureNumber":
        """
        Create a new :class:`FutureNumber` instance.

        The returned instance will be a :class:`FutureScalarNumber` instance,
        unless the class is being called from a subclass, in which case the
        subclass will be returned.

        .. warning:;
            Subclasses need to reimplement this method to return the correct
            subclass instance.
        """
        return object.__new__(FutureScalarNumber)

    def __bool__(self) -> bool:
        """
        The truthiness of a :class:`FutureNumber` is the truthiness of the
        inner value if its set, otherwise it raises a :class:`ValueError`.
        """
        if self.is_set():
            return bool(self.evaluate())

        raise ValueError(f"The value of `{self}` has not been set.")

    def evaluate(self) -> Any:
        """
        Evaluate the number and return the result.
        """
        raise NotImplementedError

    def is_set(self) -> bool:
        """
        Check if the number has been set.
        """
        raise NotImplementedError

    @staticmethod
    def _resolve(value: Any):
        """
        Resolve a value to a number.
        """
        if isinstance(value, FutureNumber):
            return value.evaluate()

        return value

    @staticmethod
    def _is_set(value: Any):
        """
        Check if a value is set.
        """
        if isinstance(value, FutureNumber):
            return value.is_set()

        return True

    __add__ = _arithmatic_wrapper(FutureArithmaticOperator.ADD)
    __sub__ = _arithmatic_wrapper(FutureArithmaticOperator.SUBTRACT)
    __mul__ = _arithmatic_wrapper(FutureArithmaticOperator.MULTIPLY)
    __truediv__ = _arithmatic_wrapper(FutureArithmaticOperator.DIVIDE)
    __floordiv__ = _arithmatic_wrapper(FutureArithmaticOperator.FLOOR_DIVIDE)
    __mod__ = _arithmatic_wrapper(FutureArithmaticOperator.MODULO)
    __pow__ = _arithmatic_wrapper(FutureArithmaticOperator.POWER)

    __radd__ = _arithmatic_wrapper(FutureArithmaticOperator.ADD, invert=True)
    __rsub__ = _arithmatic_wrapper(FutureArithmaticOperator.SUBTRACT, invert=True)
    __rmul__ = _arithmatic_wrapper(FutureArithmaticOperator.MULTIPLY, invert=True)
    __rtruediv__ = _arithmatic_wrapper(FutureArithmaticOperator.DIVIDE, invert=True)
    __rfloordiv__ = _arithmatic_wrapper(
        FutureArithmaticOperator.FLOOR_DIVIDE, invert=True
    )
    __rmod__ = _arithmatic_wrapper(FutureArithmaticOperator.MODULO, invert=True)
    __rpow__ = _arithmatic_wrapper(FutureArithmaticOperator.POWER, invert=True)

    __lt__ = _comparison_wrapper(FutureCompareOperator.LESS_THAN)
    __le__ = _comparison_wrapper(FutureCompareOperator.LESS_THAN_EQUAL)
    # Type checking mandates these to be booleans, so we need to ignore the
    # type check here.
    __eq__ = _comparison_wrapper(FutureCompareOperator.EQUAL)  # type: ignore
    __ne__ = _comparison_wrapper(FutureCompareOperator.NOT_EQUAL)  # type: ignore
    __gt__ = _comparison_wrapper(FutureCompareOperator.GREATER_THAN)
    __ge__ = _comparison_wrapper(FutureCompareOperator.GREATER_THAN_EQUAL)


def _mutable_arithmatic_wrapper(
    func: Callable[[T, T], T],
) -> Callable[["FutureScalarNumber[T]", T], "FutureScalarNumber[T]"]:
    """
    Create a wrapper function for a internal mutating arithmatic operator.
    """

    def wrapper(self: "FutureScalarNumber[T]", other: T) -> "FutureScalarNumber[T]":
        if self.inner is None:
            raise ValueError(f"The value of `{self}` has not been set.")

        with self._lock:
            self.inner = func(self.inner, other)
        return self

    return wrapper


class FutureScalarNumber(FutureNumber[T], Generic[T]):
    """
    A class representing a static number that at the time of instantiation,
    we do not know the value of.

    This allows further expressions to be recorded on the number, and only be
    evaluated when the final value is needed.

    .. note::
        You typically do not need to instantiate this class directly, but
        instead use the :class:`FutureNumber` class::

            >>> from future_number import FutureNumber
            >>> x = FutureNumber("X")
            >>> y = x + 42
            >>> print(y)
            X + 42
            >>> x.set(24)
            >>> print(y)
            24 + 42
            >>> print(y.evaluate())
            66
    """

    placeholder: str | None
    inner: T | None

    _lock: threading.Lock

    def __init__(self, name: str | None = None, *, inner: T | None = None):
        """
        Create a new :class:`FutureScalarNumber` instance.

        Typically this will be called with no arguments, and the value will be
        set later.

        Parameters
        ----------
        name : str, optional
            The placeholder to show when the value has not been set, by default "?".

            Use this to represent your variable in the string representation of the
            number.

        inner : T, optional
            The inner value of the number, by default None

        """
        # DO NOT CALL `super().__init__`: FutureNumber.__init__ calls this method
        # instead!
        self.inner = inner

        if isinstance(name, numbers.Number):
            raise TypeError(
                "The name parameter cannot be a number; "
                f"did you mean to set the inner value like `{type(self).__name__}(inner={name!r})`?"
            )
        self.placeholder = name

        self._lock = threading.Lock()

    @property
    def _placeholder(self) -> str:
        """
        The placeholder to show when the value has not been set.
        """
        return self.placeholder or f"Unknown<{id(self):010x}>"

    def __repr__(self) -> str:
        attrs = []
        if self.placeholder is not None:
            attrs.append(repr(self._placeholder))

        if self.inner is not None:
            attrs.append(f"inner={self.inner!r}")

        return f"{type(self).__name__}({', '.join(attrs)})"

    def __str__(self) -> str:
        if self.inner is None:
            return self._placeholder

        return str(self.inner)

    def set(self, value: T) -> None:
        """
        Set the value of the number.
        """
        self.inner = value

    def unset(self) -> None:
        """
        Unset the value of the number.
        """
        self.inner = None

    def evaluate(self) -> T:
        """
        Evaluate the number and return the result.
        """
        if self.inner is None:
            raise ValueError(f"The value of `{self}` has not been set.")

        # If the inner value is a future number, evaluate it.
        # This does NOT support `FutureComparisonResult` instances.
        if isinstance(self.inner, (FutureScalarNumber, FutureVectorNumber)):
            return self.inner.evaluate()

        return self.inner

    def is_set(self) -> bool:
        """
        Check if the number has been set.
        """
        return self.inner is not None

    def transform(
        self,
        func: Callable[[T], R],
        *,
        name: str | None = None,
    ) -> "FutureScalarNumber[R]":
        """
        Apply a function to the inner value of the number, and return a new
        :class:`FutureScalarNumber` instance with the transformed value.

        Contrary to :meth:`mutate`, this method will:
        - not change the inner value of the current instance, and will return a new
          instance with the transformed value;
        - not lock the value in a Mutex, but other observers will not be able to
          reference the new value; and
        - not require the given function to return the same type as the inner value.

        Parameters
        ----------
        func : Callable[[T], R]
            The function to apply to the inner value.

        name : str, optional
            The placeholder to show when the value has not been set, by default it
            will use the same placeholder as the current instance.

            Use this to represent your variable in the string representation of the
            number.

        Returns
        -------
        FutureScalarNumber[R]
            A new instance of :class:`FutureScalarNumber` with the transformed value.

        Raises
        ------
        ValueError
            If the inner value has not been set.
        """
        if self.inner is None:
            raise ValueError(f"The value of `{self}` has not been set.")

        return cast(
            # For some bizarre reason, pyright is not able to infer the type of the
            # returned value, so we need to cast it.
            FutureScalarNumber[R],
            FutureScalarNumber(
                self.placeholder if name is None else name, inner=func(self.inner)
            ),
        )

    def mutate(
        self,
        func: Callable[[T], T],
    ) -> "FutureScalarNumber[T]":
        """
        Apply a function to the inner value of the number if it has been set.

        The value will be locked in a Mutex from the point of retrieving the
        inner value to setting the new value, ensuring that the value is not
        changed by another thread in the meantime, at the cost of some
        performance.

        The same instance will be returned, allowing for chaining of methods.
        This is different from :meth:`transform`, which will return a new
        instance with the transformed value.

        Parameters
        ----------
        func : Callable[[T], T]
            The function to apply to the inner value.

        Returns
        -------
        Self
            The current instance.

        Raises
        ------
        ValueError
            If the inner value has not been set.
        """
        if self.inner is None:
            raise ValueError(f"The value of `{self}` has not been set.")

        with self._lock:
            self.inner = func(self.inner)
        return self

    mut_add = _mutable_arithmatic_wrapper(lambda a, b: a + b)
    """
    Add a value to the inner value of the number if it has been set.

    For this method to work, the inner value type must support ``+``
    operator (i.e. :meth:`__add__`).

    Parameters
    ----------
    value : T
        The value to add to the inner value.

    Returns
    -------
    Self
        The current instance.

    Raises
    ------
    ValueError
        If the inner value has not been set.
    """

    mut_sub = _mutable_arithmatic_wrapper(lambda a, b: a - b)
    """
    Subtract a value from the inner value of the number if it has been set.

    For this method to work, the inner value type must support ``-``
    operator (i.e. :meth:`__sub__`).

    Parameters
    ----------
    value : T
        The value to subtract from the inner value.

    Returns
    -------
    Self
        The current instance.

    Raises
    ------
    ValueError
        If the inner value has not been set.
    """

    mut_mul = _mutable_arithmatic_wrapper(lambda a, b: a * b)
    """
    Multiply the inner value of the number by a value if it has been set.

    For this method to work, the inner value type must support ``*``
    operator (i.e. :meth:`__mul__`).

    Parameters
    ----------
    value : T
        The value to multiply the inner value by.

    Returns
    -------
    Self
        The current instance.

    Raises
    ------
    ValueError
        If the inner value has not been set.
    """

    mut_truediv = _mutable_arithmatic_wrapper(lambda a, b: a / b)
    """
    Divide the inner value of the number by a value if it has been set.

    For this method to work, the inner value type must support ``/``
    operator (i.e. :meth:`__truediv__`).

    Parameters
    ----------
    value : T
        The value to divide the inner value by.

    Returns
    -------
    Self
        The current instance.

    Raises
    ------
    ValueError
        If the inner value has not been set.
    """

    mut_floordiv = _mutable_arithmatic_wrapper(lambda a, b: a // b)
    """
    Divide the inner value of the number by a value if it has been set.

    For this method to work, the inner value type must support ``//``
    operator (i.e. :meth:`__floordiv__`).

    Parameters
    ----------
    value : T
        The value to divide the inner value by.

    Returns
    -------
    Self
        The current instance.

    Raises
    ------
    ValueError
        If the inner value has not been set.
    """

    mut_mod = _mutable_arithmatic_wrapper(lambda a, b: a % b)
    """
    Get the modulus of the inner value of the number by a value if it has been set.

    For this method to work, the inner value type must support ``%``
    operator (i.e. :meth:`__mod__`).

    Parameters
    ----------
    value : T
        The value to get the modulus of the inner value by.

    Returns
    -------
    Self
        The current instance.

    Raises
    ------
    ValueError
        If the inner value has not been set.
    """

    mut_max = _mutable_arithmatic_wrapper(max)
    """
    Set the inner value to the maximum between the given number and the existing
    inner value if it has been set.

    For this method to work, the inner value type must support ``max``
    function, which is typically available for numbers.

    Parameters
    ----------
    value : T
        The value to compare with the inner value.

    Returns
    -------
    Self
        The current instance.

    Raises
    ------
    ValueError
        If the inner value has not been set.
    """

    mut_min = _mutable_arithmatic_wrapper(min)
    """
    Set the inner value to the minimum between the given number and the existing
    inner value if it has been set.

    For this method to work, the inner value type must support ``min``
    function, which is typically available for numbers.

    Parameters
    ----------
    value : T
        The value to compare with the inner value.

    Returns
    -------
    Self
        The current instance.

    Raises
    ------
    ValueError
        If the inner value has not been set.
    """


Op = TypeVar("Op", bound=FutureOperator)


class FutureOperatedNumber(FutureNumber[T], Generic[Op, T]):
    """
    An intemediary class defining the shared attributes between the two
    classes that operate on numbers.
    """

    left: FutureNumber[T] | T
    right: FutureNumber[T] | T
    operator: Op

    def __new__(cls, *args, **kwargs):
        # Forces the creation of a new instance of this class, not using the
        # superclass.
        return object.__new__(cls)

    def __str__(self):
        return f"{self.left} {self.operator} {self.right}"

    def __repr__(self):
        return f"{type(self).__name__}({self.left!r}, {self.right!r}, operator={self.operator!r})"

    def is_set(self):
        """
        Check if the number has been set.
        """
        return self._is_set(self.left) and self._is_set(self.right)


class FutureVectorNumber(
    FutureOperatedNumber[FutureArithmaticOperator, TA],
    Generic[TA],
):
    """
    A class representing a vector of numbers that are lazy evaluated.
    """

    left: FutureNumber[TA] | TA
    right: FutureNumber[TA] | TA
    operator: FutureArithmaticOperator

    def __init__(
        self,
        left: FutureNumber[TA] | TA,
        right: FutureNumber[TA] | TA,
        operator: FutureArithmaticOperator,
    ):
        """
        Create a new :class:`FutureVectorNumber` instance.

        Parameters
        ----------
        left : FutureNumber[TA]
            The left hand side of the vector.

        right : FutureNumber[TA]
            The right hand side of the vector.

        operator : FutureOperator
            The operator to apply to the two numbers.
        """
        self.left = left
        self.right = right
        self.operator = operator

    def evaluate(self) -> TA:
        """
        Evaluate the number and return the result.
        """
        lhs = FutureNumber._resolve(self.left)
        rhs = FutureNumber._resolve(self.right)

        return self.operator.operate(lhs, rhs)


class FutureComparisonResult(
    FutureOperatedNumber[FutureCompareOperator, TC],
    Generic[TC],
):
    """
    A class representing the result of a comparison between two numbers.
    """

    left: FutureNumber[TC] | TC
    right: FutureNumber[TC] | TC
    operator: FutureCompareOperator

    def __init__(
        self,
        left: FutureNumber[TC] | TC,
        right: FutureNumber[TC] | TC,
        operator: FutureCompareOperator,
    ):
        """
        Create a new :class:`FutureComparisonResult` instance.

        Parameters
        ----------
        left : FutureNumber[TC]
            The left hand side of the comparison.

        right : FutureNumber[TC]
            The right hand side of the comparison.

        operator : FutureCompareOperator
            The operator to apply to the two numbers.
        """
        self.left = left
        self.right = right
        self.operator = operator

    def evaluate(self) -> bool:
        """
        Evaluate the number and return the result.
        """
        lhs = FutureNumber._resolve(self.left)
        rhs = FutureNumber._resolve(self.right)

        return self.operator.operate(lhs, rhs)


P = ParamSpec("P")


class FutureAbitraryMethodResult(
    FutureNumber[T],
    Generic[P, T, R],
):
    """
    A class representing the result of a comparison between two numbers.

    This class is used to represent the result of an arbitrary method call on
    a number, and is used to store the parameters of the method call.

    .. warning::
        This class is not currently used in the library, and is here for
        future expansion.

        In order for this to work, we need a separate class of
        ``FutureArbitraryMethodExecutor`` that will be returned upon
        ``__getattr__`` of :class:`FutureNumber` instances, but that comes
        with its own set of problems, such as how to handle attributes,
        properties, and methods that are not callable.
    """

    inner: FutureNumber[T]
    method_getter: Callable[[T], Callable[P, R]]
    args: tuple
    kwargs: dict[str, Any]

    def __new__(cls, *args, **kwargs):
        # Forces the creation of a new instance of this class, not using the
        # superclass.
        return object.__new__(cls)

    def __init__(
        self,
        inner: FutureNumber[T],
        method_getter: Callable[[T], Callable[P, R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        """
        Create a new :class:`FutureAbitraryMethodResult` instance.

        Parameters
        ----------
        inner : FutureNumber[T]
            The inner number to apply the method to.

        method_getter : Callable[[T], Callable[P, R]]
            The method to apply to the inner number.

        args : tuple
            The arguments to pass to the method.

        kwargs : dict
            The keyword arguments to pass to the method.
        """
        self.inner = inner
        self.method_getter = method_getter
        self.args = args
        self.kwargs = kwargs

    def evaluate(self) -> R:
        """
        Evaluate the number and return the result.
        """
        inner = FutureNumber._resolve(self.inner)
        method = self.method_getter(inner)

        return method(*self.args, **self.kwargs)
