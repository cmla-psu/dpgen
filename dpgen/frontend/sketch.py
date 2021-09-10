import ast
from copy import deepcopy
from typing import Optional, Union

import dpgen.frontend.symbols as symbols
from dpgen.frontend.utils import extract_nodes, get_variable_name


def _variable_type_error(node: ast.AST):
    return f'expected ast.Name or ast.Subscript, got {type(node)}'


class OffendingVariableFinder(ast.NodeVisitor):
    def __init__(self, private: set[str]):
        self._private = private
        self._tainted = set(private)
        self._offending_variables = set()

    def _store_offending_variables(self, expr: ast.expr):
        is_tainted = False

        # Here we first extract the components in the expressions.
        components = extract_nodes(expr, lambda n: isinstance(n, ast.Name) or isinstance(n, ast.Subscript))

        # check if any component contains tainted variables
        for component in components:
            # standalone variable (e.g., "a")
            if isinstance(component, ast.Name):
                is_tainted = component.id in self._tainted
            # subscript component (e.g., "q[i]")
            elif isinstance(component, ast.Subscript):
                names = extract_nodes(component, lambda n: isinstance(n, ast.Name))
                is_tainted = any(name.id in self._tainted for name in names)
            else:
                # not possible, but just to be safe
                raise TypeError(_variable_type_error(component))

            # break once tainted variables are found
            if is_tainted:
                break

        # store the offending variables to the set
        if is_tainted:
            self._offending_variables.update(components)

    def offending_variables(self) -> set[ast.AST]:
        return self._offending_variables

    def visit_Assign(self, node: ast.Assign):
        is_tainted = self._store_offending_variables(node.value)

        # mark the targets as tainted if the value contains tainted variables
        if not is_tainted:
            return

        for target in node.targets:
            for name in filter(lambda n: isinstance(n, ast.Name), ast.walk(target)):
                self._tainted.add(name.id)

    def visit_If(self, node: ast.If):
        self._store_offending_variables(node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self._store_offending_variables(node.test)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        # We only care about function calls here
        if not isinstance(node.value, ast.Call):
            return

        # This is already checked by the preprocessor.
        assert isinstance(node.value.func, ast.Name) and node.value.func.id == symbols.OUTPUT

        for arg in node.value.args:
            self._store_offending_variables(arg)


class SketchGenerator(ast.NodeTransformer):
    def __init__(self, offending_variables: set[ast.AST], privates: set[str], bound: str):
        # This dictionary keeps track of whether an offending variable has been replaced or not in the value. This is
        # useful in determining the assignment of replaced variables:
        # (1) replaced_a = "a" + eta_1 for the first time adding a random variable, or
        # (2) replaced_a = "replaced_a" + eta_2 if replaced_a has already been declared.
        self._offending_variables: dict[ast.AST, bool] = {variable: False for variable in offending_variables}
        self._privates = privates
        self._bound = bound

        # We need to keep track of the live variables as we traverse to determine whether a noise is needed for a
        # particular offending variable at a particular location
        self._live_variables: set[str] = set()

        # TODO: consider subscript replacement, e.g., adding random noise to "q[0]"
        # TODO: we need to keep track of different offending variables with subscripts, i.e., "q[i]" vs "q[0]" should be
        # TODO: different offending variables, but right now they are treated as the same (both replaced with ("replaced_q")).

        # store the public parameters to create templates for scales
        self._template_elements: list[str] = []

        # keep track of the indices of generated templates (the # of random variables and # of holes in the templates)
        self._random_index = 1
        self._lambda_index = 0

    def _create_random_variable(self, node: ast.AST) -> Optional[list[ast.AST]]:
        for offending_variable, has_declared in self._offending_variables.items():
            should_create = any((
                (
                        isinstance(node, ast.arg) and
                        isinstance(offending_variable, ast.Name) and
                        node.arg == offending_variable.id
                ),
                (
                        isinstance(node, ast.Name) and
                        isinstance(offending_variable, ast.Name) and
                        node.id == offending_variable.id
                ),
                (
                        isinstance(node, ast.Subscript) and
                        isinstance(offending_variable, ast.Subscript) and
                        node == offending_variable
                )
            ))

            if not should_create:
                continue

            variable_name = get_variable_name(offending_variable)

            if variable_name not in self._live_variables:
                continue

            template = ' + '.join(
                f'{symbols.LAMBDA_HOLE}[{index + self._lambda_index}] * {element}'
                for index, element in enumerate(self._template_elements)
            )
            inserted = [
                ast.parse(
                    f"{symbols.RANDOM_VARIABLE}_{self._random_index} = {symbols.LAPLACE}(({template})/{self._bound})"),

            ]

            # Here we properly update the replaced variable, but first we need to check if the replaced variable has
            # been declared or not.
            original_variable = f'{symbols.REPLACED_VARIABLE}_{variable_name}' if has_declared else \
                ast.unparse(offending_variable)
            inserted.append(
                ast.parse(
                    f'{symbols.REPLACED_VARIABLE}_{variable_name} = '
                    f'{original_variable} + {symbols.RANDOM_VARIABLE}_{self._random_index}'
                )
            )

            # mark the replaced variable as declared
            self._offending_variables[offending_variable] = True

            # update the current index for random variable and current index for lambda (the holes in the scales of
            # random distribution) for next time
            self._random_index += 1
            self._lambda_index += len(self._template_elements)

            return inserted

        # If the given node does not match any offending variable, we insert no statements
        return []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # first add the parameters to the live variable set and store the public parameters

        # add a constant element 1 in the template set to simplify the logic later
        self._template_elements.append('1')

        for name in node.args.args:
            self._live_variables.add(name.arg)

            # create the template for random variable scales
            if name.arg not in self._privates:
                self._template_elements.append(name.arg)

        # Now that the live variable set is properly set up, we insert random variables at the beginning of the function
        # body based on offending variable set.
        inserted = []
        for name in node.args.args:
            inserted.extend(self._create_random_variable(name))

        # visit children first to avoid visiting inserted statements
        node = self.generic_visit(node)

        assert isinstance(node, ast.FunctionDef)
        node.body = inserted + node.body

        # add lambda holes to the function arguments
        node.args.args.append(ast.arg(
            arg=symbols.LAMBDA_HOLE,
            annotation=ast.Subscript(
                value=ast.Name(id='list'),
                slice=ast.Name(id='int')
            )
        ))

        return node

    def visit_While(self, node: ast.While):
        # The process logic for If and While is the same: check if node.test contains an offending variable, and create
        # corresponding random variables with
        return self.visit_If(node)

    def visit_If(self, node: Union[ast.If, ast.While]):
        inserted = []
        for component in extract_nodes(node.test, lambda n: isinstance(n, ast.Name) or isinstance(n, ast.Subscript)):
            inserted.extend(self._create_random_variable(component))

        node = self.generic_visit(node)
        inserted.append(node)
        return inserted

    def visit_Expr(self, node: ast.Expr):
        if not isinstance(node.value, ast.Call):
            return node

        # We basically disallow any forms of function calls except for "output", which is reflected in the preprocessor.
        # Hence, here we assert that the function call is a call to "output".
        assert isinstance(node.value.func, ast.Name) and node.value.func.id == symbols.OUTPUT

        inserted = []
        for arg in node.value.args:
            inserted.extend(self._create_random_variable(arg))
        node = self.generic_visit(node)
        inserted.append(node)
        return inserted

    def visit_Assign(self, node: ast.Assign):
        # Assignments could declared new variables, so here we put the newly-declared variables to the live variable
        # set.
        for target in node.targets:
            variables = extract_nodes(target, lambda n: isinstance(n, ast.Subscript) or isinstance(n, ast.Name))
            for var in variables:
                if isinstance(var, ast.Subscript):
                    # Assigning a value to a subscript does not declare new variables, so we simply skip it.
                    pass
                elif isinstance(var, ast.Name):
                    # add the newly-declared variable to the live variable set
                    self._live_variables.add(var.id)
                else:
                    # not possible, just to be safe
                    raise TypeError(_variable_type_error(var))

        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # We disallow 
        assert isinstance(node.value, ast.Name)
        # replace the uses of offending variable with the replaced variable
        if node in self._offending_variables:
            return ast.Name(id=f'{symbols.REPLACED_VARIABLE}_{node.value.id}')
        return node

    def visit_Name(self, node: ast.Name):
        # replace the uses of offending variables with the replaced variable.
        if node in self._offending_variables:
            return ast.Name(id=f'{symbols.REPLACED_VARIABLE}_{node.id}')
        return node


def generate_sketch(func: ast.FunctionDef, privates: set[str], bound: str) -> ast.FunctionDef:
    # first do a deep copy of the ast to be modified in-place
    tree = deepcopy(func)

    # find the offending variables
    finder = OffendingVariableFinder(privates)
    finder.visit(tree)

    # use found offending variables to generate proper noise in the function (the AST will be modified in-place, but
    # here we are passing the copy of the original tree, so we will not do any harm on the input).
    variables = finder.offending_variables()
    generator = SketchGenerator(offending_variables=variables, privates=privates, bound=bound)
    generator.visit(tree)

    return tree
