import ast
import logging
from copy import deepcopy

import dpgen.frontend.symbols as symbols
from dpgen.frontend.typesystem import TypeSystem
from dpgen.frontend.utils import get_variable_name, is_divergent, ExpressionReplacer, DistanceGenerator

logger = logging.getLogger(__name__)


class _ShadowBranchGenerator(ast.NodeVisitor):
    """ this class generates the shadow branch statement"""

    def __init__(self, shadow_variables, type_system):
        """
        :param shadow_variables: the variable list whose shadow distances should be updated
        """
        self._shadow_variables = shadow_variables
        self._expression_replacer = ExpressionReplacer(type_system, False)
        self._type_system = type_system

    def _shadow_update(self, statements: list[ast.stmt]):
        # only generate shadow execution for dynamically tracked variables
        generated = []

        for assign in filter(lambda n: isinstance(n, ast.Assign), statements):
            assert isinstance(assign, ast.Assign)
            if len(assign.targets) > 1:
                raise NotImplementedError(
                    f'using multiple targets for assignments inside branch is currently not supported'
                )
            target = assign.targets[0]
            assert isinstance(target, ast.Name)
            if target.id not in self._type_system:
                raise NotImplementedError(
                    f'declaring new variables inside branch is currently not supported'
                )
            if target.id in self._shadow_variables:
                generated.append(deepcopy(assign))

        for node in generated:
            assert len(node.targets) == 1
            target = node.targets[0]
            assert isinstance(target, ast.Name)
            node.value = ast.BinOp(
                op='-',
                left=self._expression_replacer.visit(node.value),
                right=target
            )
            # change the assignment variable name to shadow distance variable
            node.targets = [f'{symbols.SHADOW_DISTANCE}_{target.id}']

        return generated

    def visit_If(self, node: ast.If):
        # TODO: currently doesn't support Subscript
        shadow_cond = ExpressionReplacer(self._type_system, False).visit(deepcopy(node.test))
        shadow_branch = ast.If(
            test=shadow_cond, body=self._shadow_update(node.body), orelse=self._shadow_update(node.orelse)
        )
        return shadow_branch


class Transformer(ast.NodeTransformer):
    """Traverse the AST and do necessary transformations on the AST according to the typing rules."""

    def __init__(self, type_system: TypeSystem = TypeSystem(), enable_shadow: bool = False):
        self._type_system = type_system
        self._parameters = []
        self._random_variables = set()
        # indicate the level of loop statements, this is needed in While statement, since transformation
        # shouldn't be done until types have converged
        self._loop_level = 0
        # pc corresponds to the pc value in paper, which means if the shadow execution diverges or not, and controls
        # the generation of shadow branch
        self._pc = False
        # this indicates if shadow execution should be used or not
        self._enable_shadow = enable_shadow

    def _update_pc(self, pc: bool, types: TypeSystem, condition: ast.expr) -> bool:
        if not self._enable_shadow:
            return False
        if pc:
            return True
        _, is_shadow_divergent = is_divergent(types, condition)
        return is_shadow_divergent

    # Instrumentation rule
    def _instrument(self, type_system_1: TypeSystem, type_system_2: TypeSystem, pc: bool) -> list[ast.Assign]:
        inserted_statement = []

        for name in set(type_system_1.names()).intersection(type_system_2.names()):
            for version, distance_1, distance_2 in zip(
                    (symbols.ALIGNED_DISTANCE, symbols.SHADOW_DISTANCE),
                    type_system_1.get_types(name),
                    type_system_2.get_types(name)
            ):
                if distance_1 is None or distance_2 is None:
                    continue
                # do not instrument shadow statements if pc = True or enable_shadow is not specified
                if not self._enable_shadow or (version == symbols.SHADOW_DISTANCE and pc):
                    continue
                if distance_1 != '*' and distance_2 == '*':
                    # insert "{version}_{name} = {distance_1}"
                    inserted_statement.append(ast.Assign(targets=[f'{version}_{name}'], value=distance_1))

        return inserted_statement

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # the start of the transformation
        logger.info(f'Start transforming function {node.name} ...')

        # first store the parameters of the function for later uses
        self._parameters = tuple(arg.arg for arg in node.args.args)
        logger.debug(f'Params: {self._parameters}')

        # visit children
        node = self.generic_visit(node)
        assert isinstance(node, ast.FunctionDef)

        # insert initialization of inserted variables
        insert_statements = [
            # insert v_epsilon = 0
            ast.parse(f'{symbols.V_EPSILON} = 0'),
            # insert sample_index = 0
            ast.parse(f'{symbols.SAMPLE_INDEX} = 0')
        ]

        # initialize distance variables for dynamically tracked local variables
        for name, *distances, _, _ in filter(
                lambda variable: variable[0] not in self._parameters, self._type_system.variables()):
            for version, distance in zip((symbols.ALIGNED_DISTANCE, symbols.SHADOW_DISTANCE), distances):
                # skip shadow generation if enable_shadow is not specified
                if version == symbols.SHADOW_DISTANCE and not self._enable_shadow:
                    continue
                if distance == '*' or distance == f'{version}_{name}':
                    insert_statements.append(ast.parse(f'{version}_{name} = 0'))

        # prepend the inserted statements
        node.body[:0] = insert_statements
        return node

    def visit_Call(self, node: ast.Call):
        """T-Return rule, which adds assertion after the OUTPUT command."""
        assert isinstance(node.func, ast.Name)

        insert = []
        if self._loop_level == 0 and node.func.id == symbols.OUTPUT:
            distance_generator = DistanceGenerator(self._type_system)
            # add assertion of the output distance == 0
            aligned_distance = distance_generator.visit(node.args[0])[0]
            # there is no need to add assertion if the distance is obviously 0
            if aligned_distance != '0':
                insert.append(ast.parse(f'{symbols.ASSERT}({aligned_distance} == 0)'))

        insert.append(node)
        return insert

    def _normal_assign(self, node: ast.Assign):
        """T-Asgn rule for handling normal assignments"""
        assert not isinstance(node.value, ast.Call)

        insert_before, insert_after = [], []

        # first get distance of the value
        value_aligned, value_shadow = DistanceGenerator(self._type_system).visit(node.value)

        for target in node.targets:
            assert isinstance(target, ast.Name)

            # update the distance of the target in the type system
            var_name = get_variable_name(target)
            # If the variable has not been registered, the distances are defaulted to <'0', '0'>. This is fine since we
            # will immediately merge it with the value distances.
            if var_name not in self._type_system:
                # TODO: put base types to the type system if type hints are given
                var_aligned, var_shadow = '0', '0'
            else:
                var_aligned, var_shadow, *_ = self._type_system.get_types(var_name)
            # Then, merge the value distances with the existing distances.
            var_aligned = '0' if var_aligned == '0' and value_aligned == '0' else '*'
            var_shadow = '0' if not self._pc and var_aligned == '0' and value_aligned == '0' else '*'
            self._type_system.update_distance(var_name, var_aligned, var_shadow)
            logger.debug(f'types: {self._type_system}')

            # Next, we insert distance variable update statements. Note that we only insert distance updates when we are
            # out of loops of finding fixed points in T-While.
            if self._loop_level != 0:
                return

            # insert x^align = n^align if x^aligned is *
            if var_aligned == '*':
                insert_after.append(ast.parse(f'{symbols.ALIGNED_DISTANCE}_{var_name} = {value_aligned}'))

            if self._enable_shadow:
                # generate x^shadow = x + x^shadow - e according to (T-Asgn)
                if self._pc:
                    # Currently we do not support assigning to a subscript, so the target must be a ast.Name node.
                    assert isinstance(target, ast.Name)
                    # insert x^shadow = x + x^shadow - e;
                    insert_before.append(
                        ast.parse(
                            f'{symbols.SHADOW_DISTANCE}_{var_name} = '
                            f'{var_name} + {symbols.SHADOW_DISTANCE}_{var_name} - ({node.value})'
                        )
                    )
                # insert x^shadow = n^shadow if n^shadow is not 0
                elif var_shadow == '*':
                    insert_after.append(ast.parse(f'{symbols.SHADOW_DISTANCE}_{var_name} = {value_shadow}'))

        return insert_before + [node] + insert_after

    def _random_assign(self, node: ast.Assign):
        """T-Laplace rule for handling random variable assignments"""
        assert isinstance(node.value, ast.Call)

        if self._enable_shadow and self._pc:
            raise ValueError('cannot have random variable assignment in shadow-diverging branches')

        if len(node.targets) != 1:
            raise ValueError(f'cannot assign random variable distribution to multiple variables: {ast.unparse(node)}')

        target = node.targets[0]
        assert isinstance(target, ast.Name)

        # add the random variable to the set for later uses
        self._random_variables.add(target.id)
        logger.debug(f'Random variables: {self._random_variables}')

        # first update the distance for the random variable
        self._type_system.update_distance(target.id, '*', '0')

        if self._enable_shadow:
            # Since we have to dynamically switch (the aligned distances) to shadow version, we have to guard the switch
            # with the selector.
            shadow_type_system = deepcopy(self._type_system)
            for name, _, shadow_distance, _, _ in shadow_type_system.variables():
                shadow_type_system.update_distance(name, shadow_distance, shadow_distance)
            self._type_system.merge(shadow_type_system)

        # Next, we insert statements for updating the distance for the random variable. Note that we only insert
        # distance updates when we are out of loops of finding fixed points in T-While.
        if self._loop_level != 0:
            return

        insert = []
        if self._enable_shadow:
            # insert distance updates for normal variables
            distance_update_statements = []
            for name, align, shadow, _, _ in self._type_system.variables():
                if align == '*' and name not in self._parameters and name != target.id:
                    shadow_distance = f'{symbols.SHADOW_DISTANCE}_{name}' if shadow == '*' else shadow
                    distance_update_statements.append(
                        ast.parse(f'{symbols.ALIGNED_DISTANCE}_{name} = {shadow_distance}')
                    )
            distance_update = ast.If(
                test=ast.parse(f'{symbols.SELECTOR}_{target.id} == {symbols.SELECT_SHADOW}'),
                body=distance_update_statements,
                orelse=[],
            )
            insert.append(distance_update)

        # insert distance template for the variable
        distance = ast.parse(
            f'{symbols.ALIGNED_DISTANCE}_{target.id} = {symbols.RANDOM_DISTANCE}_{target.id}'
        )
        insert.append(distance)

        # insert cost variable update statement
        scale = ast.unparse(node.value.args[0])
        cost = f'(abs({symbols.ALIGNED_DISTANCE}_{target.id}) * (1 / ({scale})))'
        # calculate v_epsilon by combining normal cost and sampling cost
        if self._enable_shadow:
            previous_cost = \
                f'(({symbols.SELECTOR}_{target.id} == {symbols.SELECT_ALIGNED}) ? {symbols.V_EPSILON} : 0)'
        else:
            previous_cost = symbols.V_EPSILON
        v_epsilon = ast.parse(f'{symbols.V_EPSILON} = {previous_cost} + {cost}')
        insert.append(v_epsilon)

        # transform sampling command to havoc command
        node.value = ast.parse(f'{symbols.SAMPLE_ARRAY}[{symbols.SAMPLE_INDEX}]')
        insert.append(ast.parse(f'{symbols.SAMPLE_INDEX} = {symbols.SAMPLE_INDEX} + 1'))

        # prepend the node to the insert list
        insert.insert(0, node)
        return insert

    def visit_Assign(self, node: ast.Assign):
        # Assignments have two categories: (1) normal assignments, and (2) random variable assignments. Here we simply
        # dispatches the processing task to _normal_assign or _random_assign helper functions based on the value.
        logger.debug(f'Line {str(node.lineno)}: {ast.unparse(node)}')

        for target in node.targets:
            # TODO: Add support for assigning to a subscript
            if not isinstance(target, ast.Name):
                raise NotImplementedError(f'currently only support assigning to a simple variable: {ast.unparse(node)}')

        # The value of random variable assignments will always be a simple function call to a random variable
        # distribution.
        if isinstance(node.value, ast.Call):
            transformed = self._random_assign(node)
        else:
            transformed = self._normal_assign(node)

        logger.debug(f'types: {self._type_system}')
        return transformed

    def visit_If(self, node: ast.If):
        logger.debug(f'types(before branch): {self._type_system}')
        logger.debug(f'Line {node.lineno}: if({ast.unparse(node.test)})')

        # update pc value updPC
        before_pc = self._pc
        self._pc = self._update_pc(self._pc, self._type_system, node.test)

        # backup the current types before entering the true or false branch
        before_types = deepcopy(self._type_system)

        # to be used in if branch transformation assert(e^aligned);
        aligned_true_cond = ExpressionReplacer(self._type_system, True).visit(deepcopy(node.test))
        for stmt in node.body:
            self.visit(stmt)
        true_types = self._type_system
        logger.debug(f'types(true branch): {true_types}')

        # revert current types back to enter the false branch
        self._type_system = before_types

        logger.debug(f'Line: {node.lineno} else')
        for stmt in node.orelse:
            self.visit(stmt)

        # to be used in else branch transformation assert(not (e^aligned));
        aligned_false_cond = ExpressionReplacer(self._type_system, True).visit(deepcopy(node.test))
        logger.debug(f'types(false branch): {self._type_system}')
        false_types = deepcopy(self._type_system)
        self._type_system.merge(true_types)
        logger.debug(f'types(after merge): {self._type_system}')

        insert = []
        if self._loop_level == 0:
            if self._enable_shadow and self._pc and not before_pc:
                # insert c_shadow
                shadow_branch_generator = _ShadowBranchGenerator(
                    {name for name, _, shadow, *_ in self._type_system.variables() if shadow == '*'},
                    self._type_system
                )
                c_shadow = shadow_branch_generator.visit(node)
                insert.append(c_shadow)

            # insert assert functions to corresponding branch
            is_aligned_divergent, _ = is_divergent(self._type_system, node.test)
            for aligned_cond, block_items in zip((aligned_true_cond, aligned_false_cond),
                                                 (node.body, node.orelse)):
                # insert the assertion
                assert_body = aligned_cond if aligned_cond is aligned_true_cond else \
                    ast.UnaryOp(op='!', operand=aligned_cond)
                if is_aligned_divergent:
                    block_items.insert(0, ast.Expr(
                        value=ast.Call(func=ast.Name(symbols.ASSERT), args=[ast.arg(arg=assert_body)])
                    ))

            # instrument statements for updating aligned or shadow distance variables (Instrumentation rule)
            for types in (true_types, false_types):
                block_items = node.body if types is true_types else node.orelse
                inserts = self._instrument(types, self._type_system, self._pc)
                block_items.extend(inserts)

        self._pc = before_pc
        # prepend the node itself
        insert.insert(0, node)
        return insert

    def visit_While(self, node: ast.While):
        before_pc = self._pc
        self._pc = self._update_pc(self._pc, self._type_system, node.test)

        before_types = deepcopy(self._type_system)

        fixed_types = None
        # don't output logs while doing iterations
        logger.disabled = True
        self._loop_level += 1
        while fixed_types != self._type_system:
            fixed_types = deepcopy(self._type_system)
            self.generic_visit(node)
            self._type_system.merge(fixed_types)
        logger.disabled = False
        self._loop_level -= 1

        insert_before, insert_body = [], []

        if self._loop_level == 0:
            logger.debug(f'Line {node.lineno}: while({ast.unparse(node.test)})')
            logger.debug(f'types(fixed point): {self._type_system}')

            # generate assertion under While if aligned distance is not zero
            is_aligned_divergent, _ = is_divergent(self._type_system, node.test)
            if is_aligned_divergent:
                aligned_cond = ExpressionReplacer(self._type_system, True).visit(deepcopy(node.test))
                assertion = ast.Call(func=ast.Name(id=symbols.ASSERT), args=[ast.arg(arg=aligned_cond)])
                insert_body.append(ast.Expr(value=assertion))

            self.generic_visit(node)
            after_visit = deepcopy(self._type_system)
            self._type_system = deepcopy(before_types)
            self._type_system.merge(fixed_types)

            # instrument c_s part
            c_s = self._instrument(before_types, self._type_system, self._pc)
            insert_before.append(c_s)

            # instrument c'' part
            update_statements = self._instrument(after_visit, self._type_system, self._pc)
            insert_body.append(update_statements)

            # TODO: while shadow branch
            if self._enable_shadow and self._pc and not before_pc:
                pass

        self._pc = before_pc

        node.body = insert_body + node.body
        return insert_before + [node]


def initialize_type_system(func: ast.FunctionDef, type_system: TypeSystem, privates: set[str]):
    for arg in func.args.args:
        if isinstance(arg.annotation, ast.Subscript):
            # checked in preprocessor
            assert isinstance(arg.annotation.slice, ast.Name)
            type_system.update_base_type(arg.arg, arg.annotation.slice.id, is_array=True)
        elif isinstance(arg.annotation, ast.Name):
            type_system.update_base_type(arg.arg, arg.annotation.id, is_array=False)
        else:
            raise ValueError(f'unexpected type for annotation: {type(arg.annotation)} in {ast.unparse(arg)}')

        if arg.arg in privates:
            type_system.update_distance(arg.arg, '*', '*')
        else:
            type_system.update_distance(arg.arg, '0', '0')


def transform(node: ast.FunctionDef, privates: set[str]) -> ast.FunctionDef:
    # make a deep copy of the node so that we won't corrupt the input node
    tree = deepcopy(node)

    # initialize the type system
    type_system = TypeSystem()
    initialize_type_system(node, type_system, privates)

    # transformer modifies the tree in-place
    transformer = Transformer().visit(tree)

    # add returning the final cost variable
    tree.body.append(ast.Return(value=ast.Name(id=symbols.V_EPSILON)))
    return tree
