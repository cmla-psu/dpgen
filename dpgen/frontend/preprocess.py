import ast
import inspect
from typing import Callable, Union

from dpgen.frontend import Bound, BoundFunction, ListBound, ConstraintFunction
from dpgen.frontend.symbols import PREFIX, OUTPUT


def preprocess(
        f: Callable,
        private: set[str],
        bound: str = 'epsilon',
        constraint: ConstraintFunction = None,
        original_bounds: dict[str, Union[Bound, BoundFunction]] = None,
        related_bounds: dict[str, Union[Bound, ListBound]] = None
) -> ast.FunctionDef:
    # parse the function
    function_source = inspect.getsource(f)
    function_ast = ast.parse(function_source).body[0]

    if not isinstance(function_ast, ast.FunctionDef):
        raise ValueError(f'parsing error: expected ast.FunctionDef, got {type(function_ast)}')

    # first do sanity checks on the parameters
    arg_names = {arg.arg for arg in function_ast.args.args}

    # bound variable must be a fresh variable
    if bound in arg_names:
        raise ValueError('bound variable is already defined in the arguments')

    # constraint must have same number of arguments with the function
    if constraint and (count := len(inspect.signature(constraint).parameters)) != len(arg_names):
        raise ValueError(
            f'constraint function has incompatible number of arguments: expected {len(arg_names)}, got {count}')

    # private must be a subset of function arguments
    if not private.issubset(arg_names):
        raise ValueError(f'private argument set contains undefined arguments: {private - arg_names}')

    # original_bounds must be a subset of function arguments
    if not arg_names.issuperset(original_bounds.keys()):
        raise ValueError(f'original_bounds set contains undefined arguments: {set(original_bounds.keys()) - arg_names}')

    # same for related_bounds
    if not arg_names.issuperset(related_bounds.keys()):
        raise ValueError(f'related_bounds set contains undefined arguments: {set(related_bounds.keys()) - arg_names}')

    # checks for argument type annotations:
    # (1) All arguments must have a type annotation;
    # (2) All argument annotations can only contain "list", "float" or "int";
    # (3) No nested type annotation is allowed (i.e., "list[list[int]]").
    for arg in function_ast.args.args:
        if not arg.annotation:
            raise ValueError(f"function arguments must have a type annotation, missing '{arg.arg}'")
        if isinstance(arg.annotation, ast.Subscript) and not isinstance(arg.annotation.slice, ast.Name):
            raise ValueError(f'function argument type annotations cannot be nested: {ast.unparse(arg)}')
        for name in filter(lambda node: isinstance(node, ast.Name), ast.walk(arg)):
            if name.id not in {'list', 'int', 'float'}:
                raise ValueError(f"argument type annotation can only use '('list', 'int', 'float')', got {name}")

    # Here we walk through the entire AST and check for unsupported language features or variable name collisions.
    for n in ast.walk(function_ast):
        # check for variable name collisions
        if isinstance(n, ast.Name) and (n.id.startswith(PREFIX) or n.id == bound):
            raise ValueError(f"Line {n.lineno}: use of reserved variable name: {n.id}")

        # check for unsupported complex expression of subscript (e.g., "(a + b)[0]")
        elif isinstance(n, ast.Subscript) and not isinstance(n.value, ast.Name):
            raise ValueError(f"Line {n.lineno}: unsupported complex expression in subscript: {ast.unparse(n.value)}")

        # check for unsupported function calls
        elif isinstance(n, ast.Call):
            # Currently, only output function call is supported to represent outputting a value.
            if not isinstance(n.func, ast.Name):
                raise ValueError(
                    f"Line {n.lineno}: unsupported complex expression in function call: {ast.unparse(n.func)}"
                )

            if n.func.id != OUTPUT:
                raise ValueError(
                    f"Line {n.lineno}: unsupported function call: {ast.unparse(n.func)}, currently only the provided"
                    f"{OUTPUT} function call is supported."
                )

        # check for complex expressions involving function calls, i.e., "output(1) + 1" is not allowed
        elif isinstance(n, ast.Expr):
            # Here, any complex standalone expressions should not contain function calls
            # Simple function calls are allowed (e.g., "output(1)"), the validity of the function name will be checked
            # by the branch above.
            if isinstance(n.value, ast.Call):
                continue

            # Otherwise, there should not be any function calls
            if any(isinstance(node, ast.Call) for node in ast.walk(n.value)):
                raise ValueError(
                    f"Line {n.lineno}: function calls can not be used in complex expressions: {ast.unparse(n.value)}"
                )

        # check for imbalanced assignments (e.g., "a, b = c") since currently we do not support tuple unpacking
        elif isinstance(n, ast.Assign):
            # Python supports multiple assignments (e.g., "a = b = 1"), so here n.targets could contain multiple nodes.
            # The unpacked variables are represented by a ast.Tuple node consisting of multiple ast.Name nodes.
            # See https://docs.python.org/3/library/ast.html#ast.Assign for detailed explanation.

            # find the length of value
            value_len = len(n.value.elts) if isinstance(n.value, ast.Tuple) else 1

            for target in n.targets:
                # find the length of _each_ target and compare it with the length of value
                # By default, the target is a single element, which means the target length is 1
                target_len = 1
                if isinstance(target, ast.Tuple):
                    # Currently, we do not support nested tuple, so we run another check for that.
                    for element in target.elts:
                        for child in ast.walk(element):
                            if isinstance(child, ast.Tuple):
                                raise NotImplementedError(
                                    f'Line {child.lineno}: \"{ast.get_source_segment(function_source, target)}\" nested tuple '
                                    f'is current not supported'
                                )
                    # reset the target length
                    target_len = len(target.elts)

                # The length of value and the length of target _must_ be the same.
                if value_len != target_len:
                    raise ValueError(
                        f"Line {n.lineno}: imbalanced assignment: \"{ast.unparse(n)}\", we do not currently support "
                        f"tuple unpacking")

    return function_ast
