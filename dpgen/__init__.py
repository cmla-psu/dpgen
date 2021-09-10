import ast
import logging
from typing import Callable, Union

import coloredlogs

from dpgen.frontend import ListBound, Bound, BoundFunction, ConstraintFunction
from dpgen.frontend.preprocess import preprocess
from dpgen.frontend.sketch import generate_sketch
from dpgen.frontend.transform import transform

coloredlogs.install('DEBUG', fmt='%(asctime)s [0x%(process)x] %(levelname)s %(message)s')

logger = logging.getLogger(__name__)


def privatize(
        f: Callable,
        fx: Callable,
        privates: set[str],
        bound: str = 'epsilon',
        constraint: ConstraintFunction = None,
        original_bounds: dict[str, Union[Bound, BoundFunction]] = None,
        related_bounds: dict[str, Union[Bound, ListBound]] = None
) -> ast.FunctionDef:
    function_ast = preprocess(f, privates, bound, constraint, original_bounds, related_bounds)
    logger.debug(f'original program: \n{ast.unparse(function_ast)}')
    sketch = generate_sketch(function_ast, privates, bound)
    logger.debug(f'sketch program: \n{ast.unparse(sketch)}')
    fx()
    print(f'sketch program: \n{ast.unparse(sketch)}')
    # transformed = transform(sketch, privates)
    # logger.debug(f'transformed program: \n{ast.unparse(transformed)}')
    return sketch
