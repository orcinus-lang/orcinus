# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
import contextlib
import weakref
from io import StringIO
from typing import cast, Sequence, Optional, Set

from more_itertools import first

from orcinus.collections import NamedScope
from orcinus.diagnostics import DiagnosticManager
from orcinus.locations import Location
from orcinus.signals import Signal
from orcinus.utils import cached_property

BUILTINS_MODULE = '__builtins__'

TYPE_INTEGER_NAME = 'int'
TYPE_BOOLEAN_NAME = 'bool'
TYPE_VOID_NAME = 'void'
TYPE_STRING_NAME = 'str'
TYPE_FLOAT_NAME = 'float'

VALUE_NONE_NAME = 'None'
VALUE_TRUE_NAME = 'True'
VALUE_FALSE_NAME = 'False'

INIT_NAME = '__init__'
NEW_NAME = '__new__'


class ModuleLoader(abc.ABC):
    """ This loader interface is used for load modules """

    @abc.abstractmethod
    def open(self, name: str) -> Module:
        raise RuntimeError


class SymbolContext:
    def __init__(self, loader: ModuleLoader, *, diagnostics: DiagnosticManager = None):
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
        self.loader = loader
        self.modules = {}

    @cached_property
    def builtins_module(self) -> Module:
        return self.load(BUILTINS_MODULE)

    @cached_property
    def boolean_type(self) -> BooleanType:
        founded_type = next((tpy for tpy in self.builtins_module.types if tpy.name == TYPE_BOOLEAN_NAME), None)
        assert isinstance(founded_type, BooleanType)
        return cast(BooleanType, founded_type)

    @cached_property
    def integer_type(self) -> IntegerType:
        founded_type = next((tpy for tpy in self.builtins_module.types if tpy.name == TYPE_INTEGER_NAME), None)
        assert isinstance(founded_type, IntegerType)
        return cast(IntegerType, founded_type)

    @cached_property
    def float_type(self) -> FloatType:
        founded_type = next((tpy for tpy in self.builtins_module.types if tpy.name == TYPE_FLOAT_NAME), None)
        assert isinstance(founded_type, FloatType)
        return cast(FloatType, founded_type)

    @cached_property
    def void_type(self) -> VoidType:
        founded_type = next((tpy for tpy in self.builtins_module.types if tpy.name == TYPE_VOID_NAME), None)
        assert isinstance(founded_type, VoidType)
        return cast(VoidType, founded_type)

    @cached_property
    def string_type(self) -> StringType:
        founded_type = next((tpy for tpy in self.builtins_module.types if tpy.name == TYPE_STRING_NAME), None)
        assert isinstance(founded_type, StringType)
        return cast(StringType, founded_type)

    def add_module(self, module: Module):
        self.modules[module.name] = module

    def load(self, name) -> Module:
        if name in self.modules:
            return self.modules[name]
        return self.loader.open(name)


class Symbol(abc.ABC):
    """ Abstract base for all symbols """

    @property
    @abc.abstractmethod
    def context(self) -> SymbolContext:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def location(self) -> Location:
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return f'<{type(self).__name__}: {self}>'


class Named(Symbol, abc.ABC):
    """ Abstract base for all named symbols """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return self.name


class Child(Named, abc.ABC):
    """ Abstract base for all owned symbols """

    @property
    @abc.abstractmethod
    def owner(self) -> Container:
        raise NotImplementedError

    @property
    def module(self) -> Module:
        if isinstance(self.owner, Module):
            return cast(Module, self.owner)
        return cast(Child, self.owner).module

    @property
    def context(self) -> SymbolContext:
        return self.module.context


class Container(Symbol, abc.ABC):
    """ Abstract base for container symbols """

    def __init__(self):
        self.__members = []
        self.on_add_member = Signal()

    @property
    def members(self) -> Sequence[Child]:
        return self.__members

    @property
    def types(self) -> Sequence[Type]:
        return [member for member in self.members if isinstance(member, Type)]

    @property
    def functions(self) -> Sequence[Function]:
        return [member for member in self.members if isinstance(member, Function)]

    def add_member(self, symbol: Child):
        if symbol in self.__members:
            return

        self.__members.append(symbol)
        self.on_add_member(self, symbol)

    def get_member(self, name: str) -> Optional[Child]:
        return first((member for member in self.members if member.name == name), None)


class ErrorSymbol(Container):
    __location: Location

    def __init__(self, module: Module, location: Location):
        super().__init__()

        self.__module = module
        self.__location = location

    @property
    def context(self) -> SymbolContext:
        return self.module.context

    @property
    def module(self) -> Module:
        return self.__module

    @property
    def location(self) -> Location:
        return self.__location

    def __str__(self):
        return '<error>'


class Value(Symbol, abc.ABC):
    """ Abstract base for all values """

    def __init__(self, value_type: Type, location: Location):
        self.__location = location
        self.__type = value_type

    @property
    def context(self) -> SymbolContext:
        return self.type.context

    @property
    def type(self) -> Type:
        return self.__type

    @property
    def location(self) -> Location:
        return self.__location

    @location.setter
    def location(self, value: Location):
        self.__location = value

    @abc.abstractmethod
    def as_val(self) -> str:
        raise NotImplementedError


class Attribute(Named):
    def __init__(self, module: Module, name: str, arguments: Sequence[Value], location: Location):
        self.__module = module
        self.__location = location
        self.__name = name
        self.__arguments = arguments

    @property
    def context(self) -> SymbolContext:
        return self.module.context

    @property
    def module(self) -> Module:
        return self.__module

    @property
    def name(self) -> str:
        return self.__name

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def arguments(self) -> Sequence[Value]:
        return self.__arguments

    def __str__(self):
        return self.name


class Module(Named, Container):
    def __init__(self, context: SymbolContext, name, location: Location):
        super(Module, self).__init__()

        self.__context = weakref.ref(context)
        self.__name = name
        self.__location = location
        self.__declared_functions = []
        self.__declared_types = []
        self.__dependencies = []

        self.context.add_module(self)
        if self is not self.context.builtins_module:
            self.add_dependency(self.context.builtins_module)

    @property
    def context(self) -> SymbolContext:
        return self.__context()

    @property
    def name(self) -> str:
        return self.__name

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def dependencies(self) -> Sequence[Module]:
        return self.__dependencies

    @property
    def declared_functions(self) -> Sequence[Function]:
        return self.__declared_functions

    @property
    def declared_types(self) -> Sequence[Type]:
        return self.__declared_types

    def add_function(self, member: Function):
        self.__declared_functions.append(member)

    def add_type(self, member: Type):
        self.__declared_types.append(member)

    def add_dependency(self, module: Module):
        self.__dependencies.append(module)


class GenericSymbol(Child, abc.ABC):
    @property
    def is_generic(self) -> bool:
        if self.generic_parameters:
            return True
        return any(arg.is_generic for arg in self.generic_arguments)

    @property
    @abc.abstractmethod
    def definition(self) -> Optional[GenericSymbol]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def generic_parameters(self) -> Sequence[GenericType]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def generic_arguments(self) -> Sequence[Type]:
        raise NotImplementedError

    @abc.abstractmethod
    def add_generic_parameter(self, generic: GenericType):
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location):
        raise NotImplementedError

    def __str__(self):
        arguments = None
        if self.generic_arguments:
            arguments = ', '.join(str(arg) for arg in self.generic_arguments)
        if self.generic_parameters:
            arguments = ', '.join(str(arg) for arg in self.generic_parameters)

        if arguments:
            return f'{self.name}[{arguments}]'
        return super(GenericSymbol, self).__str__()


class Type(GenericSymbol, Container, abc.ABC):
    """ Abstract base for all types """

    def __init__(self, owner: Container, name: str, location: Location):
        super(Type, self).__init__()

        self.__owner = owner
        self.__name = name
        self.__location = location
        self.__parents = []
        self.__attributes = []
        self.__definition = None
        self.__generic_parameters = []
        self.__generic_arguments = []

        self.module.add_type(self)

        self.is_builded = False
        self.on_build = Signal()

    @property
    def owner(self) -> Container:
        return self.__owner

    @property
    def name(self) -> str:
        return self.__name

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def attributes(self) -> Sequence[Attribute]:
        return self.__attributes

    @property
    def definition(self) -> Optional[Type]:
        return self.__definition

    @property
    def generic_parameters(self) -> Sequence[GenericType]:
        return self.__generic_parameters

    @property
    def generic_arguments(self) -> Sequence[Type]:
        return self.__generic_arguments

    @property
    def parents(self) -> Sequence[Type]:
        return self.__parents

    @property
    def ascendants(self) -> Set[Type]:
        ascendants = set(self.parents)
        for parent in self.parents:
            ascendants.update(parent.ascendants)
        return ascendants

    @property
    def methods(self) -> Sequence[Function]:
        return [member for member in self.members if isinstance(member, Function)]

    @property
    def fields(self) -> Sequence[Field]:
        return tuple(member for member in self.members if isinstance(member, Field))

    @property
    def properties(self) -> Sequence[Property]:
        return tuple(member for member in self.members if isinstance(member, Property))

    def add_parent(self, parent: Type):
        if parent is self or self in parent.ascendants:
            raise RuntimeError(u'Can not add parent: Circular dependency involving')
        self.__parents.append(parent)

    def add_generic_parameter(self, generic: GenericType):
        self.__generic_parameters.append(generic)

    def add_attribute(self, attr: Attribute):
        self.__attributes.append(attr)

    def build(self):
        if self.is_builded:
            return

        if not self.definition:
            for parent in self.parents:
                builder = InheritanceBuilder(parent, self)
                for member in parent.members:
                    builder.inherit_member(member)

        if not self.definition:
            for method in self.methods:
                if method.name == INIT_NAME:
                    init_func = cast(Function, method)
                    init_type = init_func.function_type
                    parameters = init_type.parameters[1:]

                    new_type = FunctionType(init_type.owner, parameters, self, init_type.location)
                    new_func = Function(self, NEW_NAME, new_type, init_type.location)
                    builder = IRBuilder(new_func.append_basic_block('entry'))
                    result = builder.new(init_func, new_func.parameters, location=init_type.location)
                    builder.ret(result, location=init_type.location)

                    super().add_member(new_func)

        self.is_builded = True
        self.on_build(self)

    def instantiate_type(self, module: Module) -> Type:
        raise RuntimeError('Can not instantiate non generic type')

    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location) -> Type:
        if len(self.generic_parameters) != len(generic_arguments):
            self.context.diagnostics.error(location, 'Can not instantiate type: mismatch count of arguments')
            return self

        if not self.generic_parameters:
            self.context.diagnostics.error(location, 'Can not instantiate type: type is not generic')
            return self

        # self instantiate
        if all(param is arg for param, arg in zip(self.generic_parameters, generic_arguments)):
            return self

        instance = self.instantiate_type(module)
        setattr(instance, '_Type__definition', self)
        setattr(instance, '_Type__generic_arguments', generic_arguments)
        if self.is_builded:
            instance.__instantiate_members(location)
        else:
            self.on_build.connect(lambda x: instance.__instantiate_members(location), False)
        return instance

    def __instantiate_members(self, location: Location):
        builder = InstantiateBuilder(self.module, self.definition.generic_parameters, self.generic_arguments)
        builder.add(self.definition, self)
        for attr in self.attributes:
            self.add_attribute(attr)
        for parent in self.definition.parents:
            self.add_parent(cast(Type, builder.instantiate(parent, location)))
        for member in self.definition.members:
            self.add_member(cast(Child, builder.instantiate(member, location)))
        self.build()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return not (self == other)


class ErrorType(Type):
    """ Instance of this type is represented errors in semantic analyze """

    def __init__(self, module: Module, location: Location):
        super(ErrorType, self).__init__(module, '<error>', location)

    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location) -> Type:
        return self


class GenericType(Type):
    def is_generic(self) -> bool:
        return True


class VoidType(Type):
    def __init__(self, owner: Container, location: Location):
        super(VoidType, self).__init__(owner, TYPE_VOID_NAME, location)


class BooleanType(Type):
    def __init__(self, owner: Container, location: Location):
        super(BooleanType, self).__init__(owner, TYPE_BOOLEAN_NAME, location)


class StringType(Type):
    def __init__(self, owner: Container, location: Location):
        super(StringType, self).__init__(owner, TYPE_STRING_NAME, location)


class IntegerType(Type):
    def __init__(self, owner: Container, location: Location):
        super(IntegerType, self).__init__(owner, TYPE_INTEGER_NAME, location)


class FloatType(Type):
    def __init__(self, owner: Container, location: Location):
        super(FloatType, self).__init__(owner, TYPE_FLOAT_NAME, location)


class ClassType(Type):
    def instantiate_type(self, module: Module) -> ClassType:
        return ClassType(module, self.name, self.location)


class StructType(Type):
    def instantiate_type(self, module: Module) -> StructType:
        return StructType(module, self.name, self.location)


class InterfaceType(Type):
    def instantiate_type(self, module: Module) -> InterfaceType:
        return InterfaceType(module, self.name, self.location)


class EnumType(Type):
    @property
    def values(self) -> Sequence[EnumConstant]:
        return tuple(member for member in self.members if isinstance(member, EnumConstant))

    def build(self):
        location = self.location

        # Properties
        name_type = FunctionType(self.module, [self], self.context.string_type, location)
        name_func = Function(self, 'name', name_type, location)
        name_func.add_attribute(create_native_attribute(self.module, 'orx_enum_name', location=location))
        name_prop = Property(self, name_func.name, name_func, location=location)
        self.add_member(name_prop)

        value_type = FunctionType(self.module, [self], self.context.integer_type, location)
        value_func = Function(self, 'value', value_type, location)
        value_func.add_attribute(create_native_attribute(self.module, 'orx_enum_value', location=location))
        value_prop = Property(self, value_func.name, value_func, location=location)
        self.add_member(value_prop)

        # Comparators
        compare_type = FunctionType(self.module, [self, self], self.context.boolean_type, location)
        for name, native_name in (
                ('__eq__', 'orx_integer_eq'),
                ('__ne__', 'orx_integer_ne'),
        ):
            compare_func = Function(self, name, compare_type, location)
            compare_func.add_attribute(create_native_attribute(self.module, native_name, location=location))
            self.add_member(compare_func)

        return super().build()


class FunctionType(Type):
    def __init__(self, owner: Container, parameters: Sequence[Type], return_type: Type, location: Location):
        super(FunctionType, self).__init__(owner, "Function", location)

        assert isinstance(return_type, Type)
        assert all(isinstance(param_type, Type) for param_type in parameters)

        self.__return_type = return_type
        self.__parameters = parameters

    @property
    def return_type(self) -> Type:
        return self.__return_type

    @property
    def parameters(self) -> Sequence[Type]:
        return self.__parameters

    def __eq__(self, other):
        if not isinstance(other, FunctionType):
            return False
        is_return_equal = self.return_type == other.return_type
        return is_return_equal and all(
            param == other_param for param, other_param in zip(self.parameters, other.parameters))

    def __hash__(self):
        return id(self)

    def __str__(self):
        parameters = ', '.join(str(param_type) for param_type in self.parameters)
        return f"({parameters}) -> {self.return_type}"


class Property(Child):
    def __init__(self, owner: Type, name: str, getter: Function = None, setter: Function = None, *, location: Location):
        self.__owner = owner
        self.__name = name
        self.__type = type
        self.__location = location
        self.__getter = getter
        self.__setter = setter

    @property
    def owner(self) -> Type:
        return self.__owner

    @property
    def type(self) -> Type:
        return self.__getter.return_type

    @property
    def name(self) -> str:
        return self.__name

    @property
    def getter(self) -> Optional[Function]:
        return self.__getter

    @getter.setter
    def getter(self, value: Optional[Function]):
        self.__getter = value

    @property
    def setter(self) -> Optional[Function]:
        return self.__setter

    @setter.setter
    def setter(self, value: Optional[Function]):
        self.__setter = value

    @property
    def location(self) -> Location:
        return self.__location


class Field(Child):
    def __init__(self, owner: Type, name: str, field_type: Type, location: Location):
        assert isinstance(owner, (ClassType, StructType))
        self.__owner = owner
        self.__name = name
        self.__type = field_type
        self.__location = location

    @property
    def owner(self) -> Type:
        return self.__owner

    @property
    def type(self) -> Type:
        return self.__type

    @property
    def name(self) -> str:
        return self.__name

    @property
    def location(self) -> Location:
        return self.__location


class EnumValue(Child, Value):
    def __init__(self, owner: EnumType, name: str, value: int, location: Location):
        super(EnumValue, self).__init__(owner, location)

        self.__name = name
        self.__value = value

    @property
    def owner(self) -> EnumType:
        return cast(EnumType, self.type)

    @property
    def name(self) -> str:
        return self.__name

    @property
    def value(self) -> int:
        return self.__value

    def as_val(self) -> str:
        return str(self.value)


class ErrorValue(Value):
    """ Instance of this class is represented errors in semantic analyze """

    def __init__(self, module: Module, location: Location):
        super(ErrorValue, self).__init__(ErrorType(module, location), location)

    def __str__(self):
        return '<error>'

    def as_val(self) -> str:
        return str(self)


class Parameter(Child, Value):
    def __init__(self, owner: Function, name: str, param_type: Type):
        super(Parameter, self).__init__(param_type, owner.location)

        self.__owner = owner
        self.__name = name

    @property
    def owner(self) -> Function:
        return self.__owner

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str):
        self.__name = self.owner.scope.add(value, self.__name)

    def __str__(self):
        return f'{self.name}: {self.type}'

    def as_val(self) -> str:
        return str(self)


class Function(GenericSymbol, Value):
    def __init__(self, owner: Container, name: str, func_type: FunctionType, location: Location):
        super(Function, self).__init__(func_type, location)
        self.__owner = owner
        self.__name = name
        self.__blocks = []
        self.__scope = NamedScope()
        self.__attributes = []
        self.__definition = None
        self.__generic_parameters = []
        self.__generic_arguments = []
        self.__parameters = tuple(
            Parameter(self, self.scope.add(), param_type) for idx, param_type in enumerate(func_type.parameters)
        )

        self.module.add_function(self)

    @property
    def owner(self) -> Container:
        return self.__owner

    @property
    def name(self) -> str:
        return self.__name

    @property
    def definition(self) -> Optional[Function]:
        return self.__definition

    @property
    def generic_parameters(self) -> Sequence[GenericType]:
        return self.__generic_parameters

    @property
    def generic_arguments(self) -> Sequence[Type]:
        return self.__generic_arguments

    @property
    def attributes(self) -> Sequence[Attribute]:
        return self.__attributes

    @property
    def is_abstract(self) -> bool:
        return not bool(self.blocks)

    @cached_property
    def is_native(self) -> bool:
        return any(attr.name == 'native' for attr in self.attributes)

    @cached_property
    def native_name(self) -> Optional[str]:
        native_attr = next((attr for attr in self.attributes if attr.name == 'native'), None)
        if native_attr:
            if len(native_attr.arguments) == 1:
                assert isinstance(native_attr.arguments[0], StringConstant)
                return cast(StringConstant, native_attr.arguments[0]).value

    @property
    def function_type(self) -> FunctionType:
        return cast(FunctionType, self.type)

    @property
    def parameters(self) -> Sequence[Parameter]:
        return self.__parameters

    @property
    def return_type(self) -> Type:
        return self.function_type.return_type

    @property
    def blocks(self) -> Sequence[BasicBlock]:
        return self.__blocks

    @property
    def entry_block(self) -> Optional[BasicBlock]:
        return self.__blocks[0] if self.__blocks else None

    @property
    def scope(self) -> NamedScope:
        return self.__scope

    def add_generic_parameter(self, generic: GenericType):
        self.__generic_parameters.append(generic)

    def add_attribute(self, attr: Attribute):
        self.__attributes.append(attr)

    def append_basic_block(self, name: str) -> BasicBlock:
        block = BasicBlock(self, self.scope.add(name))
        self.__blocks.append(block)
        return block

    def remove_basic_block(self, block: BasicBlock):
        assert self == block.owner
        self.__blocks.remove(block)

    def move_basic_block(self, index: int, block: BasicBlock):
        assert self == block.owner
        self.__blocks.remove(block)
        self.__blocks.insert(index, block)

    def as_val(self) -> str:
        return str(self)

    def as_blocks(self) -> str:
        buffer = StringIO()
        buffer.write('def ')
        buffer.write(str(self))
        buffer.write(' {\n')
        for idx, block in enumerate(self.blocks):
            if idx:
                buffer.write("\n")
            buffer.write(block.as_block())
        buffer.write('}\n')
        return buffer.getvalue()

    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location) -> Function:
        if not self.generic_parameters:
            raise RuntimeError('Can not instantiate non generic function')

        if len(self.generic_parameters) != len(generic_arguments):
            raise RuntimeError('Can not instantiate generic function')

        # self instantiate
        if all(param is arg for param, arg in zip(self.generic_parameters, generic_arguments)):
            return self

        builder = InstantiateBuilder(module, self.generic_parameters, generic_arguments)
        instance: Function = builder.instantiate(self, location)
        setattr(instance, '_FUnction__definition', self)
        setattr(instance, '_FUnction__generic_arguments', generic_arguments)
        for attr in self.attributes:
            self.add_attribute(attr)
        return instance

    def __str__(self):
        parameters = ', '.join('{}: {}'.format(param.name, param.type) for param in self.parameters)
        return f'{self.name}({parameters}) -> {self.return_type}'


class Overload(Named):
    def __init__(self, module: Module, name: str, functions: Sequence[Function], location: Location):
        self.__module = module
        self.__name = name
        self.__functions = functions
        self.__location = location

    @property
    def name(self) -> str:
        return self.__name

    @property
    def context(self) -> SymbolContext:
        return self.module.context

    @property
    def module(self) -> Module:
        return self.__module

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def functions(self) -> Sequence[Function]:
        return self.__functions


class IntegerConstant(Value):
    def __init__(self, context: SymbolContext, value: int, location: Location):
        super(IntegerConstant, self).__init__(context.integer_type, location)

        self.value = value

    def __str__(self):
        return str(self.value)

    def as_val(self) -> str:
        return str(self)


class BooleanConstant(Value):
    def __init__(self, context: SymbolContext, value: bool, location: Location):
        super(BooleanConstant, self).__init__(context.boolean_type, location)

        self.value = value

    def __str__(self):
        return VALUE_TRUE_NAME if self.value else VALUE_FALSE_NAME

    def as_val(self) -> str:
        return str(self)


class StringConstant(Value):
    def __init__(self, context: SymbolContext, value: str, location: Location):
        super(StringConstant, self).__init__(context.string_type, location)

        self.value = value

    def __str__(self):
        value = self.value.replace('"', '\\"')
        return f'"{value}"'

    def as_val(self) -> str:
        return str(self)


class NoneConstant(Value):
    def __init__(self, context: SymbolContext, location: Location):
        super(NoneConstant, self).__init__(context.void_type, location)

    def __str__(self):
        return VALUE_NONE_NAME

    def as_val(self) -> str:
        return str(self)


class EnumConstant(Child, Value):
    def __init__(self, owner: EnumType, name: str, value: Optional[int], location: Location):
        super(EnumConstant, self).__init__(owner, location)

        self.__name = name
        self.__value = value

    @property
    def name(self) -> str:
        return self.__name

    @property
    def owner(self) -> Container:
        raise self.owner

    @property
    def value(self):
        return self.__value

    def as_val(self) -> str:
        return f'{self.type}.{self.name}'


class BasicBlock:
    def __init__(self, owner: Function, name: str):
        self.__owner = owner
        self.__name = name
        self.__instructions = []
        self.__predecessors = set()
        self.__successors = set()

    @property
    def owner(self) -> Function:
        return self.__owner

    @property
    def context(self) -> SymbolContext:
        return self.owner.context

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str):
        self.__name = self.owner.scope.add(value, self.__name)

    @property
    def predecessors(self) -> Set[BasicBlock]:
        return self.__predecessors

    @property
    def successors(self) -> Set[BasicBlock]:
        return self.__successors

    @property
    def instructions(self) -> Sequence[Instruction]:
        return self.__instructions

    @property
    def terminator(self) -> Optional[TerminatorInstruction]:
        if not self.instructions:
            return None

        inst = self.instructions[-1]
        return inst if inst.is_terminator else None

    @property
    def is_terminated(self) -> bool:
        return bool(self.terminator)

    def append(self, inst: Instruction):
        if self.is_terminated:
            raise RuntimeError(u'Can not append instruction to terminated block')

        elif inst.parent:
            raise RuntimeError(u'Can not move instruction from another block')

        self.__instructions.append(inst)
        self.__inject(inst)

    def insert(self, idx: int, inst: Instruction):
        if self.is_terminated and isinstance(inst, TerminatorInstruction) and idx == len(self.instructions):
            raise RuntimeError(u'Can not insert terminator instruction to terminated block')

        elif inst.parent:
            raise RuntimeError(u'Can not move instruction from another block')

        self.__instructions.insert(idx, inst)
        self.__inject(inst)

    def __inject(self, inst: Instruction):
        # inject instruction
        inst.parent = self

        # successors and predecessor
        for block in inst.successors:
            self.__successors.add(block)
            block.__predecessors.add(self)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<{type(self).__name__}: {self}>'

    def as_block(self) -> str:
        buffer = StringIO()
        buffer.write(f"%{self.name}:")

        if self.predecessors:
            buffer.write(' ; ')
            for idx, previous in enumerate(self.predecessors):
                if idx:
                    buffer.write(', ')
                buffer.write(f'%{previous}')
        buffer.write('\n')

        for inst in self.instructions:
            buffer.write("    {}    ; {}\n".format(inst.as_inst(), inst.location))
        return buffer.getvalue()

    @property
    def debug_string(self) -> str:
        return self.as_block()


class Instruction(Value, abc.ABC):
    def __init__(self, value_type: Type, *, name: str = None, location: Location):
        super().__init__(value_type, location)

        assert isinstance(value_type, Type)

        self.__name = name or ''
        self.__block = None

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str):
        self.__name = self.owner.scope.add(value, self.__name) if self.owner else value

    @property
    def parent(self) -> Optional[BasicBlock]:
        return self.__block

    @parent.setter
    def parent(self, value: Optional[BasicBlock]):
        self.__block = value
        self.__name = self.owner.scope.add(self.__name) if self.owner else self.__name

    @property
    def owner(self) -> Optional[Function]:
        return self.parent.owner if self.parent else None

    @property
    def is_terminator(self) -> bool:
        return False

    @property
    def successors(self) -> Set[BasicBlock]:
        return set()

    @abc.abstractmethod
    def as_inst(self) -> str:
        raise NotImplementedError

    def as_val(self) -> str:
        return '${}'.format(self.name)

    def __str__(self):
        return self.as_inst()


class TerminatorInstruction(Instruction, abc.ABC):
    @property
    def is_terminator(self) -> bool:
        return True


class ReturnInstruction(TerminatorInstruction):
    def __init__(self, value: Value, *, location: Location):
        super(ReturnInstruction, self).__init__(value.context.void_type, location=location)

        assert isinstance(value, Value)

        self.value = value

    def as_inst(self) -> str:
        if isinstance(self.value, NoneConstant):
            return 'ret'
        return 'ret {}'.format(self.value.as_val())


class BranchInstruction(TerminatorInstruction):
    def __init__(self, condition: Optional[Value], then_block: BasicBlock, else_block: Optional[BasicBlock], *,
                 location: Location):
        context = then_block.context
        super(BranchInstruction, self).__init__(context.void_type, location=location)

        assert isinstance(condition, (Value, type(None)))
        assert isinstance(then_block, BasicBlock)
        assert isinstance(else_block, (BasicBlock, type(None)))

        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

    @cached_property
    def successors(self) -> Set[BasicBlock]:
        return {self.then_block, self.else_block} if self.else_block else {self.then_block}

    def as_inst(self) -> str:
        if self.condition:
            return 'br {} %{} %{}'.format(self.condition.as_val(), self.then_block, self.else_block)
        return 'br %{}'.format(self.then_block)


class AllocaInstruction(Instruction):
    def as_inst(self) -> str:
        return '{} = alloca {}'.format(self.as_val(), self.type)


class LoadInstruction(Instruction):
    def __init__(self, source: Value, *, name: str = None, location: Location):
        super().__init__(source.type, name=name, location=location)

        assert isinstance(source, AllocaInstruction)

        self.source = source

    def as_inst(self) -> str:
        return '{} = load {}'.format(self.as_val(), self.source.as_val())


class StoreInstruction(Instruction):
    def __init__(self, target: Value, source: Value, *, name: str = None, location: Location):
        context = target.context
        super().__init__(context.void_type, name=name, location=location)

        assert isinstance(source, Value)
        assert isinstance(target, AllocaInstruction)

        self.source = source
        self.target = target

    def as_inst(self) -> str:
        return 'store {} {}'.format(self.source.as_val(), self.target.as_val())


class CallInstruction(Instruction):
    def __init__(self, func: Function, arguments: Sequence[Value], *, name: str = None, location: Location):
        super(CallInstruction, self).__init__(func.return_type, name=name, location=location)

        assert isinstance(func, Function)
        assert all(isinstance(arg, Value) for arg in arguments)
        assert len(func.parameters) == len(arguments)

        self.function = func
        self.arguments = arguments

    def as_inst(self) -> str:
        arguments = ', '.join(arg.as_val() for arg in self.arguments)
        return '{} = call {}({})'.format(self.as_val(), self.function.name, arguments)


class NewInstruction(Instruction):
    def __init__(self, func: Function, arguments: Sequence[Value], *, name: str = None, location: Location):
        super(NewInstruction, self).__init__(func.parameters[0].type, name=name, location=location)

        assert isinstance(func, Function)
        assert all(isinstance(arg, Value) for arg in arguments)
        assert len(func.parameters) - 1 == len(arguments)

        self.function = func
        self.arguments = arguments

    def as_inst(self) -> str:
        arguments = ', '.join(arg.as_val() for arg in self.arguments)
        return '{} = new {}({})'.format(self.as_val(), self.type.name, arguments)


class ExtractValueInstruction(Instruction):
    def __init__(self, instance: Value, field: Field, *, name: str = None, location: Location):
        super(ExtractValueInstruction, self).__init__(field.type, name=name, location=location)

        self.instance = instance
        self.field = field

    def as_inst(self) -> str:
        return '{} = extractvalue {}.{}'.format(self.as_val(), self.instance.as_val(), self.field.name)


class InsertValueInstruction(Instruction):
    def __init__(self, instance: Value, field: Field, source: Value, location: Location):
        context = instance.context
        super(InsertValueInstruction, self).__init__(context.void_type, location=location)

        self.instance = instance
        self.field = field
        self.source = source

    def as_inst(self) -> str:
        return 'insertvalue {}.{} {}'.format(self.instance.as_val(), self.field.name, self.source.as_val())


class IRBuilder:
    def __init__(self, block: BasicBlock):
        self.__block = block
        self.__anchor = len(block.instructions)

    @property
    def context(self) -> SymbolContext:
        return self.block.context

    @property
    def module(self) -> Module:
        return self.function.module

    @property
    def function(self) -> Function:
        return self.block.owner

    @property
    def block(self) -> BasicBlock:
        return self.__block

    @property
    def is_terminated(self) -> bool:
        return self.block.is_terminated

    @contextlib.contextmanager
    def goto_block(self, block: BasicBlock):
        old_block = self.block
        if block.terminator:
            self.position_before(block.terminator)
        else:
            self.position_at_end(block)
        yield
        self.position_at_end(old_block)

    @contextlib.contextmanager
    def goto_entry_block(self):
        with self.goto_block(self.function.entry_block):
            yield

    def get_default_value(self, value_type: Type, *, location: Location):
        if isinstance(value_type, BooleanType):
            return BooleanConstant(self.context, False, location)
        elif isinstance(value_type, IntegerType):
            return IntegerConstant(self.context, 0, location)
        elif isinstance(value_type, VoidType):
            return NoneConstant(self.context, location)

        self.context.diagnostics.error(location, "Type `{}` doesn't have a default values".format(value_type))
        return ErrorValue(self.module, location)

    def append_basic_block(self, name: str) -> BasicBlock:
        return self.function.append_basic_block(name)

    def position_before(self, inst: Instruction):
        self.__block = inst.parent
        self.__anchor = self.__block.instructions.index(inst)

    def position_after(self, inst: Instruction):
        """
        Position immediately after the given instruction.  The current block
        is also changed to the instruction's basic block.
        """
        assert isinstance(inst, Instruction)
        self.__block = inst.parent
        self.__anchor = self.__block.instructions.index(inst) + 1

    def position_at_start(self, block):
        """
        Position at the start of the basic *block*.
        """
        assert isinstance(block, BasicBlock)
        self.__block = block
        self.__anchor = 0

    def position_at_end(self, block: BasicBlock):
        """
        Position at the end of the basic *block*.
        """
        assert isinstance(block, BasicBlock)
        self.__block = block
        self.__anchor = len(block.instructions)

    def __append(self, inst: Instruction) -> Instruction:
        self.__block.insert(self.__anchor, inst)
        self.__anchor += 1
        return inst

    def alloca(self, value_type: Type, *, name: str = None, location: Location) -> Instruction:
        return self.__append(AllocaInstruction(value_type, name=name, location=location))

    def load(self, value: Value, *, name: str = None, location: Location) -> Instruction:
        return self.__append(LoadInstruction(value, name=name, location=location))

    def store(self, target: Value, source: Value, *, location: Location) -> Instruction:
        return self.__append(StoreInstruction(target, source, location=location))

    def ret(self, value: Value, *, location: Location) -> Instruction:
        return self.__append(ReturnInstruction(value, location=location))

    def cbranch(self, condition: Optional[Value], then_block: BasicBlock, else_block: Optional[BasicBlock], *,
                location: Location) -> Instruction:
        return self.__append(BranchInstruction(condition, then_block, else_block, location=location))

    def branch(self, then_block: BasicBlock, *, location: Location) -> Instruction:
        return self.__append(BranchInstruction(None, then_block, None, location=location))

    def call(self, func: Function, arguments: Sequence[Value], *, name: str = None, location: Location) -> Instruction:
        return self.__append(CallInstruction(func, arguments, name=name, location=location))

    def new(self, func: Function, arguments: Sequence[Value], *, name: str = None, location: Location) -> Instruction:
        return self.__append(NewInstruction(func, arguments, name=name, location=location))

    def insert_value(self, instance: Value, field: Field, source: Value, *, location):
        return self.__append(InsertValueInstruction(instance, field, source, location=location))

    def extract_value(self, instance, field, *, name: str = None, location: Location):
        return self.__append(ExtractValueInstruction(instance, field, name=name, location=location))


class InheritanceBuilder:
    def __init__(self, definition: Type, instance: Type):
        self.definition = definition
        self.instance = instance
        self.mapping = {}

    @property
    def module(self) -> Module:
        return self.instance.module

    def add(self, original: Symbol, instance: Symbol):
        self.mapping[original] = instance
        self.instance.add_member(cast(Child, instance))

    def inherit_member(self, original: Symbol) -> Symbol:
        if original in self.mapping:
            return self.mapping[original]

        if isinstance(original, Function):
            return self.inherit_function(original)
        elif isinstance(original, Field):
            return self.inherit_field(original)
        elif isinstance(original, Property):
            return self.inherit_property(original)

        raise NotImplementedError

    def inherit_function(self, original: Function) -> Function:
        if original.parameters and original.parameters[0].type == self.definition:
            parameters = [self.instance]
            parameters.extend(original.function_type.parameters[1:])
        else:
            parameters = original.function_type.parameters
        inherit_type = FunctionType(self.module, parameters, original.return_type, original.function_type.location)
        instance = Function(self.instance, original.name, inherit_type, original.location)
        for original_param, instance_param in zip(original.parameters, instance.parameters):
            instance_param.name = original_param.name
            instance_param.location = original_param.location

        self.add(original, instance)
        return instance

    def inherit_property(self, original: Property) -> Property:
        getter = self.inherit_member(original.getter) if original.getter else None
        setter = self.inherit_member(original.setter) if original.getter else None
        instance = Property(self.instance, original.name, getter, setter, location=original.location)
        self.add(original, instance)
        return instance

    def inherit_field(self, original: Field) -> Field:
        instance = Field(self.instance, original.name, original.type, location=original.location)
        self.add(original, instance)
        return instance


class InstantiateBuilder:
    def __init__(self, module: Module, parameters, arguments):
        self.module = module
        self.mapping = {param: arg for param, arg in zip(parameters, arguments)}

    def add(self, original: Symbol, instance: Symbol):
        self.mapping[original] = instance

    def instantiate(self, original: Symbol, location: Location) -> Symbol:
        if original in self.mapping:
            return self.mapping[original]

        if isinstance(original, Function):
            return self.instantiate_function(original, location)
        elif isinstance(original, FunctionType):
            return self.instantiate_function_type(original, location)
        elif isinstance(original, BooleanType):
            return original
        elif isinstance(original, VoidType):
            return original
        elif isinstance(original, IntegerType):
            return original
        elif isinstance(original, StringType):
            return original
        elif isinstance(original, ClassType):
            return original
        elif isinstance(original, InterfaceType):
            return original

        raise NotImplementedError

    def instantiate_function_type(self, original: FunctionType, location: Location) -> FunctionType:
        parameters = [cast(Type, self.instantiate(param, location)) for param in original.parameters]
        return_type = cast(Type, self.instantiate(original.return_type, location))
        instance = FunctionType(self.module, parameters, return_type, original.location)
        self.add(original, instance)
        return instance

    def instantiate_function(self, original: Function, location: Location) -> Symbol:
        owner = cast(Container, self.instantiate(original.owner, location))
        instance_type: FunctionType = self.instantiate(original.type, location)
        instance = Function(owner, original.name, instance_type, original.location)
        for original_param, instance_param in zip(original.parameters, instance.parameters):
            instance_param.name = original_param.name
            instance_param.location = original_param.location
        for attr in original.attributes:
            instance.add_attribute(attr)
        self.add(original, instance)
        return instance


def create_native_attribute(module: Module, name: str = None, *, location: Location) -> Attribute:
    if name:
        compare_name = StringConstant(module.context, name, location)
        return Attribute(module, 'native', [compare_name], location)
    return Attribute(module, 'native', [], location)
