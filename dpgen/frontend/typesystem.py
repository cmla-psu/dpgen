from dataclasses import dataclass, astuple
from typing import Optional


@dataclass
class _VariableType:
    """Variable class that represents the (name, distance, type) record inside the type system"""
    # the distance types can only be '0' or '*'
    aligned_distance: Optional[str] = None
    shadow_distance: Optional[str] = None
    base_type: Optional[str] = None
    is_array: Optional[bool] = False

    def __str__(self):
        return f"<{self.aligned_distance}, {self.shadow_distance}, {self.base_type}, " \
               f"{'ARRAY' if self.is_array else 'VARIABLE'}>"


class TypeSystem:
    """ TypeSystem keeps track of the distances of each variable. The distance of each variable is internally
    represented by c_ast node, and gets simplified and casted to strings when get_distance method is called"""

    def __init__(self):
        self._variables: dict[str, _VariableType] = {}

    def __len__(self):
        return len(self._variables)

    def __contains__(self, item):
        return self._variables.__contains__(item)

    def __str__(self):
        variable_strings = []
        for variable, variable_type in self._variables.items():
            variable_strings.append(f'{variable}: {variable_type}')
        return '{' + ', '.join(variable_strings) + '}'

    def __eq__(self, other):
        if not isinstance(other, TypeSystem):
            return False
        return self._variables == other._variables

    def names(self):
        return self._variables.keys()

    def clear(self):
        self._variables.clear()

    def variables(self):
        for variable, variable_type in self._variables.items():
            yield variable, *astuple(variable_type)

    def merge(self, other):
        if not isinstance(other, TypeSystem):
            raise TypeError('The other variable is not of type TypeSystem')

        for name, *types in other.variables():
            if name not in self._variables:
                self._variables[name] = _VariableType(*types)
            else:
                current_type = self._variables[name]
                aligned_distance, shadow_distance, base_type, is_array = types
                if not (current_type.base_type == base_type and current_type.is_array == is_array):
                    raise ValueError(f'The base type or is_array of the variable {name} does not match,'
                                     f'current: ({current_type.base_type}, {current_type.is_array}),'
                                     f'to merge: ({base_type}, {is_array})')
                # here we will promote the types from the other type system
                if not (current_type.aligned_distance == aligned_distance == '0'):
                    current_type.aligned_distance = '*'
                if not (current_type.shadow_distance == shadow_distance == '0'):
                    current_type.shadow_distance = '*'

    def get_types(self, name):
        return astuple(self._variables[name])

    def update_base_type(self, name: str, base_type: str, is_array: bool):
        if name not in self._variables:
            self._variables[name] = _VariableType(base_type=base_type, is_array=is_array)
        else:
            variable_type = self._variables[name]
            variable_type.base_type = base_type
            variable_type.is_array = is_array

    def update_distance(self, name: str, aligned_distance: str, shadow_distance: str):
        if aligned_distance not in ('0', '*') or shadow_distance not in ('0', '*'):
            raise ValueError(f'Distance can only be 0 or *, got {(aligned_distance, shadow_distance)}')
        if name not in self._variables:
            self._variables[name] = _VariableType(aligned_distance=aligned_distance, shadow_distance=shadow_distance)
        else:
            variable_type = self._variables[name]
            variable_type.aligned_distance = aligned_distance
            variable_type.shadow_distance = shadow_distance
