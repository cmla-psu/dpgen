import ast
from copy import deepcopy
from typing import Callable, Union

import numba
from sympy import simplify

import dpgen.frontend.symbols as symbols
from dpgen.frontend.typesystem import TypeSystem


def try_simplify(expr):
    try:
        expr = str(simplify(expr))
    finally:
        return expr


class DistanceGenerator(ast.NodeTransformer):
    def __init__(self, types):
        self._types = types

    def generic_visit(self, node):
        # TODO: should handle cases like -(-(-(100)))
        raise NotImplementedError

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.operand, ast.Constant):
            return '0', '0'
        else:
            raise NotImplementedError

    def visit_Constant(self, n):
        return '0', '0'

    def visit_Name(self, node: ast.Name):
        align, shadow, *_ = self._types.get_types(node.id)
        align = f'({symbols.ALIGNED_DISTANCE}_{node.id})' if align == '*' else align
        shadow = f'({symbols.SHADOW_DISTANCE}_{node.id})' if shadow == '*' else shadow
        return align, shadow

    def visit_Subscript(self, node: ast.Subscript):
        assert isinstance(node.value, ast.Name)
        var_name, subscript = node.value.id, ast.unparse(node.slice)
        align, shadow, *_ = self._types.get_types(var_name)
        align = f'({symbols.ALIGNED_DISTANCE}_{var_name}[{subscript}])' if align == '*' else align
        shadow = f'({symbols.SHADOW_DISTANCE}_{var_name}[{subscript}])' if shadow == '*' else shadow
        return align, shadow

    def visit_BinOp(self, node: ast.BinOp):
        return tuple(
            try_simplify(f'{left} {node.op} {right}')
            for left, right in zip(self.visit(node.left), self.visit(node.right))
        )


class ExpressionReplacer(ast.NodeTransformer):
    def __init__(self, type_system: TypeSystem, is_aligned: bool):
        self._type_system = type_system
        self._is_aligned = is_aligned

    def _replace(self, node: Union[ast.Name, ast.Subscript]):
        assert isinstance(node, ast.Name) or (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name))

        # find the variable name and get its distance from the type system
        name = node.id if isinstance(node, ast.Name) else node.value.id
        aligned, shadow, *_ = self._type_system.get_types(name)
        distance = aligned if self._is_aligned else shadow

        # Zero distance variables should remain the same.
        if distance == '0':
            return node

        # Otherwise, we replace the variable with x^aligned or x^shadow.

        # construct the distance variable
        distance_var_name = f'{symbols.ALIGNED_DISTANCE if self._is_aligned else symbols.SHADOW_DISTANCE}_{name}'
        if isinstance(node, ast.Name):
            right = ast.Name(id=distance_var_name)
        else:
            right = ast.Subscript(value=ast.Name(id=distance_var_name), slice=node.slice)

        # form "original variable + variable distance"
        return ast.BinOp(op='+', left=node, right=right)

    def visit_Name(self, node: ast.Name):
        return self._replace(node)

    def visit_Subscript(self, node: ast.Subscript):
        return self._replace(node)


def is_divergent(type_system: TypeSystem, condition: ast.expr) -> list[bool]:
    # if the condition contains star variable it means the aligned/shadow branch will diverge
    results = []
    for type_index in range(2):
        for node in ast.walk(condition):
            if isinstance(node, ast.Name) and type_system.get_types(node.id)[type_index] == '*':
                results.append(True)
                continue
        results.append(False)
    return results


def get_variable_name(node: ast.expr):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Subscript):
        # We disallow using complex expressions in .value (e.g., "(a + b)[0]"), this will be filtered in the
        # preprocessor, therefore here we simply assert that it is an ast.Name node.
        assert isinstance(node.value, ast.Name)
        return node.value.id
    else:
        # not possible, just to be safe
        raise ValueError(f'unexpected node type: {type(node)}')


def is_ast_equal(node1: ast.AST, node2: ast.AST) -> bool:
    if type(node1) is not type(node2):
        return False

    if isinstance(node1, ast.AST):
        for k, v in vars(node1).items():
            if k in ("lineno", "col_offset", "ctx", "end_lineno", "end_col_offset"):
                continue
            if not is_ast_equal(v, getattr(node2, k)):
                return False
        return True

    return node1 == node2


def extract_nodes(node: ast.AST, func: Callable[[ast.AST], bool]) -> set[ast.AST]:
    class Visitor(ast.NodeVisitor):
        def __init__(self, f: Callable[[ast.AST], bool]):
            self._f = f
            self.found = set()

        def generic_visit(self, n: ast.AST):
            if self._f(n):
                self.found.add(n)
                return

            # Here we only go deeper if the node has not been found
            super(Visitor, self).generic_visit(n)

    visitor = Visitor(func)
    visitor.visit(node)
    return visitor.found


def has_node(node: ast.AST, func: Callable[[ast.AST], bool]) -> bool:
    class Visitor(ast.NodeVisitor):
        def __init__(self, f: Callable[[ast.AST], bool]):
            self._f = f
            self.has_found = False

        def generic_visit(self, n: ast.AST):
            # skip the check if the node is already found
            if self.has_found:
                return

            if self._f(n):
                self.has_found = True
                return

            # We only go deeper if the node has not been found yet
            super(Visitor, self).generic_visit(n)

    visitor = Visitor(func)
    visitor.visit(node)
    return visitor.has_found


@numba.njit
def dpgen_assert(cond):
    return 0 if cond else 1


def add_numba_njit(tree: ast.AST) -> ast.AST:
    cloned = deepcopy(tree)
    numba_import = ast.Import(names=[ast.alias(name='numba')])
    decorator = ast.Attribute(value=ast.Name(id='numba', ctx=ast.Load()), attr='njit', ctx=ast.Load())
    cloned.body.insert(0, numba_import)
    cloned.body[1].decorator_list.append(decorator)
    ast.fix_missing_locations(cloned)
    return cloned
