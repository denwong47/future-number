# Future Number
A class representing a number that is lazy evaluated.

![CI](https://github.com/denwong47/future-number/actions/workflows/ci-lint.yml/badge.svg?branch=main)

This allows further expressions to be recorded on the number, and only be
evaluated when the final value is needed.

------

### Usage


The most basic operation is to create a number with a placeholder, and then
set the value later::

```python
    >>> from future_number import FutureNumber
    >>> x = FutureNumber("X")
    >>> y = x + 42
    >>> print(y)
    X + 42
```

At this point the value of ``x`` has not been set, so the value of ``y`` is
still a placeholder. You can set the value of ``x`` later::

```python
    >>> x.set(24)
    >>> print(y)
    24 + 42
    >>> print(y.evaluate())
    66
```

This is typically used with `Generator` instances. For example, if
you have a generator that return numbers, and you need to create a normalised
sequence of those numbers based on some aggregation that could not be
determined before the whole sequence is exhausted, you can::

```python
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
```

Without the `FutureNumber` class, you would have to accumulate the
total of each category in a separate dictionary, and then re-iterate over the
sequence to normalise the values, which is not memory efficient.

> [!TIP]
>
> You still need to consider any other lazy evaluation rules, and whether the
> number at the point of evaluation is set to the correct value.
>
> For example, if you create another `Generator` that returns the
> normalised values, you need to ensure the original sequence is exhausted
> before you can evaluate the normalised values::
>
> ```python
>    >>> from future_number import FutureNumber
>    >>> total = FutureNumber(inner=0)
>    >>> sequence = iter([1, 3, 4])
>    >>> normalised = (value / total.mut_add(value) for value in sequence)
>    >>> [value.evaluate() for value in normalised]
>    [1.0, 0.75, 0.5]
> ```
>
> In the above example, the total value has not finished accumulating at
> the point of evaluation, so the normalised values are incorrect::
>
> ```python
>    [1 / 1, 3 / 4, 4 / 8] = [1.0, 0.75, 0.5]
> ```
>
> However if you use a `list` comprehension, the total value will
> be set before the normalised values are evaluated::
>
> ```python
>    >>> from future_number import FutureNumber
>    >>> total = FutureNumber(inner=0)
>    >>> sequence = iter([1, 3, 4])
>    >>> normalised = [value / total.mut_add(value) for value in sequence]
>    >>> [value.evaluate() for value in normalised]
>    [0.125, 0.375, 0.5]
> ```
>
> Which yields the correct normalised values::
>
> ```python
>    [1 / 8, 3 / 8, 4 / 8] = [0.125, 0.375, 0.5]
> ```

This class will work on any type that supports the operators that are used
on the numbers, and will raise a `TypeError` if the operator is not
supported.

Specifically, a `FutureNumber` can be a `numpy.ndarray` and
arithmetic operators are supported::

```python
    >>> import numpy as np
    >>> from future_number import FutureNumber
    >>> arr = FutureNumber("A")
    >>> new_arr = arr * 3 + np.ones((5,), dtype=int)
    >>> arr.set(np.arange(5, dtype=int))
    >>> new_arr.evaluate()
    array([ 1,  4,  7, 10, 13])
```
