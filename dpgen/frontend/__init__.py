import enum
from typing import Union, Callable

import numpy as np

Numeric = Union[int, float, np.number]
ConstraintFunction = Callable[[...], bool]
Bound = tuple[Numeric, Numeric]
BoundFunction = Callable[[...], Bound]


class ListBound(enum.Enum):
    ALL_DIFFER = 1
    ONE_DIFFER = 2
