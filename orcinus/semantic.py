# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import heapq
import itertools
import logging
from typing import Deque, List, Tuple
from typing import MutableMapping

from orcinus.collections import LazyDictionary
from orcinus.exceptions import OrcinusError
from orcinus.symbols import *
from orcinus.syntax import *
from orcinus.utils import cached_property

logger = logging.getLogger('orcinus')

Arguments = Sequence['SemanticSymbol']
Keywords = Mapping[str, 'SemanticSymbol']

UNARY_NAMES = {
    UnaryID.Pos: '__pos__',
    UnaryID.Neg: '__neg__',
    UnaryID.Inv: '__inv__',
    # UnaryID.Not is logical and is not implemented as call
}
BINARY_NAMES = {
    BinaryID.Add: '__add__',
    BinaryID.Sub: '__sub__',
    BinaryID.Mul: '__mul__',
    BinaryID.Mod: '__mod__',
    BinaryID.Pow: '__pow__',
    BinaryID.And: '__and__',
    BinaryID.Or: '__or__',
    BinaryID.Xor: '__xor__',
    BinaryID.LeftShift: '__lshift__',
    BinaryID.RightShift: '__rshift__',
}
COMPARE_NAMES = {
    CompareID.Equal: '__eq__',
    CompareID.NotEqual: '__ne__',
    CompareID.Less: '__lt__',
    CompareID.LessEqual: '__le__',
    CompareID.Great: '__gt__',
    CompareID.GreatEqual: '__ge__',
}


class SemanticModuleLoader(ModuleLoader):
    def __init__(self, context: SemanticContext):
        self.__context = weakref.ref(context)

    @property
    def context(self) -> SemanticContext:
        return self.__context()

    def open(self, name: str) -> Module:
        model = self.context.open(name)
        return model.module


class SyntaxTreeLoader(abc.ABC):
    """ This loader interface is used for load syntax tree """

    @abc.abstractmethod
    def load(self, filename: str) -> SyntaxTree:
        """ Load syntax tree by filename """
        raise NotImplementedError

    @abc.abstractmethod
    def open(self, name: str) -> SyntaxTree:
        """ Load syntax tree by module name"""
        raise NotImplementedError


class SemanticContext:
    def __init__(self, loader: SyntaxTreeLoader, *, diagnostics: DiagnosticManager = None):
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
        self.symbol_context = SymbolContext(SemanticModuleLoader(self), diagnostics=self.diagnostics)
        self.syntax_context = SyntaxContext(self.diagnostics)

        self.__loader = loader
        self.__models = {}
        self.__filenames = {}

    def load(self, filename: str) -> SemanticModel:
        """
        Open file with module

        :param filename: relative path to module source file
        :param name:     module name
        """
        if filename in self.__filenames:
            return self.__filenames[filename]

        tree = self.__loader.load(filename)
        return self.__register(tree)

    def open(self, name: str) -> SemanticModel:
        """
        Load module from packages
        """
        if name in self.__models:
            return self.__models[name]

        tree = self.__loader.open(name)
        return self.__register(tree)

    def __register(self, tree: SyntaxTree) -> SemanticModel:
        model = SemanticModel(self, tree)
        self.__models[tree.name] = model
        self.__models[tree.filename] = model
        model.analyze()
        return model


class SemanticModel:
    tree: SyntaxTree
    symbols: LazyDictionary[SyntaxNode, SemanticSymbol]
    scopes: LazyDictionary[SyntaxNode, SemanticScope]
    environments: LazyDictionary[SyntaxNode, SemanticEnvironment]

    def __init__(self, context: SemanticContext, tree: SyntaxTree):
        self.context = context
        self.tree = tree
        self.is_analyzed = False  # True, if analysis is complete

        self.__globals = GlobalEnvironment(self)
        self.__initializer = SymbolInitializer(self.globals)
        self.environments = LazyDictionary[SyntaxNode, SemanticScope](EnvironmentAnnotator(self.globals))
        self.scopes = LazyDictionary[SyntaxNode, SemanticScope](ScopeAnnotator(self.globals))
        self.symbols = LazyDictionary[SyntaxNode, SemanticSymbol](SymbolAnnotator(self.globals), self.__initializer)

    @property
    def symbol_context(self):
        return self.context.symbol_context

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.context.diagnostics

    @cached_property
    def module(self) -> Module:
        return self.semantic_module.module

    @cached_property
    def semantic_module(self) -> SemanticModule:
        return cast(SemanticModule, self.symbols[self.tree])

    @property
    def globals(self) -> GlobalEnvironment:
        return self.__globals

    def analyze(self):
        # check if analysis is completed
        if self.is_analyzed:
            return
        self.is_analyzed = True

        # # import symbols
        annotator = ImportAnnotator(self.globals)
        if self.module is not self.symbol_context.builtins_module:
            annotator.import_all(self.symbol_context.builtins_module)

        for node in self.tree.imports:
            annotator.visit(node)

        # initialize all symbols
        self.__initializer.analyze()

        # emit functions
        functions: Sequence[FunctionNode] = list(
            self.tree.find_descendants(lambda node: isinstance(node, FunctionNode))
        )
        for func in functions:
            if not func.is_abstract:
                emitter = LocalEnvironment(self.environments[func.parent], func)
                emitter.analyze()

    def initialize(self, symbol: SemanticSymbol):
        self.__initializer.analyze_symbol(symbol)

    def __repr__(self):
        return f'<SemanticModel: {self.module}>'


class SemanticEnvironment(abc.ABC):
    """
    Environment stored information about declared symbols and their types in current environment.

    Environment changed
    """

    def __init__(self, model: SemanticModel):
        self.__model = weakref.ref(model)

    @property
    def model(self) -> SemanticModel:
        return self.__model()

    @property
    def module(self) -> Module:
        """ Returns analyzing module """
        return self.model.module

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.model.diagnostics

    @property
    def symbol_context(self) -> SymbolContext:
        return self.model.symbol_context

    @property
    @abc.abstractmethod
    def scope(self) -> SemanticScope:
        """ Top scope of environment """
        raise NotImplementedError

    def define(self, name: str, symbol: SemanticSymbol):
        """ Define symbol in top scope of environment """
        return self.scope.define(name, symbol)

    def resolve(self, name: str) -> Optional[SemanticSymbol]:
        """ Define symbol from top scope of environment """
        return self.scope.resolve(name)

    def emit(self, node: SyntaxNode) -> SemanticSymbol:
        symbol = self.model.symbols[node]
        if not symbol:
            self.diagnostics.error(node.location, f"Not found symbol for node: {type(symbol).__name__}")
            return SemanticSymbol.from_symbol(self, ErrorValue(self.module, node.location))
        return symbol

    def as_symbol(self, node: SyntaxNode) -> Symbol:
        semantic_symbol = self.emit(node)
        if isinstance(semantic_symbol, SemanticSymbol):
            return semantic_symbol.as_symbol(node.location)
        return ErrorSymbol(self.module, location=node.location)

    def as_type(self, node: SyntaxNode) -> Type:
        """ Convert semantic symbol to type """
        semantic_symbol = self.emit(node)
        if isinstance(semantic_symbol, SemanticSymbol):
            return semantic_symbol.as_type(node.location)
        return ErrorType(self.module, location=node.location)

    def as_value(self, node: SyntaxNode) -> Value:
        """ Convert semantic symbol to value """
        semantic_symbol = self.emit(node)
        if isinstance(semantic_symbol, SemanticSymbol):
            return semantic_symbol.as_value(node.location)
        return ErrorValue(self.module, location=node.location)

    def search_named_symbol(self, name: str, location: Location) -> Optional[SemanticSymbol]:
        if name == TYPE_VOID_NAME:
            result = self.symbol_context.void_type
        elif name == TYPE_BOOLEAN_NAME:
            result = self.symbol_context.boolean_type
        elif name == TYPE_INTEGER_NAME:
            result = self.symbol_context.integer_type
        elif name == TYPE_STRING_NAME:
            result = self.symbol_context.string_type
        elif name == TYPE_FLOAT_NAME:
            result = self.symbol_context.float_type
        elif name == VALUE_TRUE_NAME:
            result = BooleanConstant(self.symbol_context, True, location=location)
        elif name == VALUE_FALSE_NAME:
            result = BooleanConstant(self.symbol_context, False, location=location)
        elif name == VALUE_NONE_NAME:
            result = NoneConstant(self.symbol_context, location=location)
        else:
            return self.resolve(name)

        return SemanticSymbol.from_symbol(self, result)


class GlobalEnvironment(SemanticEnvironment):
    """ This environment is used for manage symbols of module """

    def __init__(self, model: SemanticModel):
        super(GlobalEnvironment, self).__init__(model)
        self.__scope = SemanticScope()

    @property
    def scope(self) -> SemanticScope:
        return self.__scope


class StaticEnvironment(SemanticEnvironment):
    """ This environment is used for manage symbols on function and type declaration level """

    def __init__(self, parent: SemanticEnvironment):
        super().__init__(parent.model)

        self.__parent = weakref.ref(parent)
        self.__scope = SemanticScope()

    @property
    def scope(self) -> SemanticScope:
        return self.__scope

    @property
    def parent(self) -> SemanticEnvironment:
        return self.__parent()

    def resolve(self, name: str) -> Optional[Symbol]:
        """ Define symbol from top scope of environment """
        return self.scope.resolve(name) or self.parent.resolve(name)


class SemanticScope:
    def __init__(self, parent: SemanticScope = None):
        self.__parent: SemanticScope = parent
        self.__declared: MutableMapping[str, SemanticSymbol] = {}

    @property
    def parent(self) -> SemanticScope:
        return self.__parent

    def keys(self):
        return self.__declared.keys()

    def define(self, name: str, symbol: SemanticSymbol):
        assert isinstance(symbol, SemanticSymbol)
        self.__declared[name] = symbol

    def resolve(self, name: str) -> Optional[SemanticSymbol]:
        symbol = self.__declared.get(name)
        if symbol:
            return symbol
        elif self.parent:
            return self.parent.resolve(name)


class LocalScope(SemanticScope):
    def __init__(self, enter_block: BasicBlock, parent: SemanticScope = None):
        super(LocalScope, self).__init__(parent)

        self.enter_block = enter_block
        self.exit_block = enter_block


class SemanticSymbol(abc.ABC):
    def __init__(self, environment: SemanticEnvironment):
        self.__environment = weakref.ref(environment)

    @property
    def model(self) -> SemanticModel:
        return self.environment.model

    @property
    def environment(self) -> SemanticEnvironment:
        return self.__environment()

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.environment.diagnostics

    @property
    @abc.abstractmethod
    def location(self) -> Location:
        raise NotImplementedError

    @classmethod
    def from_symbol(cls, environment: SemanticEnvironment, symbol: Symbol):
        assert isinstance(symbol, Symbol)

        if isinstance(symbol, Module):
            return SemanticModule(environment, symbol)
        elif isinstance(symbol, Type):
            return SemanticType(environment, symbol)
        elif isinstance(symbol, Function):
            return SemanticFunction(environment, symbol)
        elif isinstance(symbol, Value):
            return SemanticValue(environment, symbol)
        elif isinstance(symbol, Field):
            return SemanticField(environment, symbol)
        elif isinstance(symbol, Property):
            return SemanticProperty(environment, symbol)
        elif isinstance(symbol, Attribute):
            return SemanticAttribute(environment, symbol)

        raise DiagnosticError(symbol.location, "Can not convert symbol to semantic symbol")

    @abc.abstractmethod
    def as_symbol(self, location: Location) -> Symbol:
        raise NotImplementedError

    @abc.abstractmethod
    def as_type(self, location: Location) -> Type:
        raise NotImplementedError

    @abc.abstractmethod
    def as_value(self, location: Location) -> Value:
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate(self, arguments: Sequence[Type], location: Location) -> SemanticSymbol:
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self):
        return f'<{type(self).__name__}: {self}>'


class SemanticError(SemanticSymbol):
    def __init__(self, environment: SemanticEnvironment, location: Location):
        super().__init__(environment)

        self.__location = location

    @property
    def location(self) -> Location:
        return self.__location

    def as_symbol(self, location: Location) -> Symbol:
        return ErrorSymbol(self.model.module, self.location)

    def as_type(self, location: Location) -> Type:
        return ErrorType(self.model.module, self.location)

    def as_value(self, location: Location) -> Value:
        return ErrorValue(self.model.module, self.location)

    def instantiate(self, arguments: Sequence[Type], location: Location) -> SemanticSymbol:
        return self

    def __str__(self) -> str:
        return '<error>'


class SemanticModule(SemanticSymbol):
    def __init__(self, environment: SemanticEnvironment, module: Module):
        super().__init__(environment)

        self.__module = module

    @property
    def module(self) -> Module:
        return self.__module

    @property
    def name(self) -> str:
        return self.module.name

    @property
    def location(self) -> Location:
        return self.module.location

    def as_symbol(self, location: Location) -> Symbol:
        return self.module

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f"Can not use model ‘{self.module.name}’ as type")
        return ErrorType(self.environment.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f"Can not use model ‘{self.module.name}’ as value")
        return ErrorValue(self.environment.module, location)

    def instantiate(self, arguments: Sequence[Type], location: Location) -> SemanticError:
        self.diagnostics.error(location, f"Can not instantiate module ‘{self.module.name}’")
        return SemanticError(self.environment, location)

    def add_member(self, symbol: SemanticSymbol):
        member: Child = symbol.as_symbol(symbol.location)
        self.module.add_member(member)
        self.environment.define(member.name, symbol)

    def __str__(self) -> str:
        return str(self.module)


class SemanticType(SemanticSymbol):
    def __init__(self, environment: SemanticEnvironment, symbol: Type):
        super().__init__(environment)

        self.__type = symbol

    @property
    def type(self) -> Type:
        return self.__type

    @property
    def name(self) -> str:
        return self.type.name

    @property
    def location(self) -> Location:
        return self.type.location

    def as_symbol(self, location: Location) -> Symbol:
        return self.type

    def as_type(self, location: Location) -> Type:
        self.model.initialize(self)  # force initialization
        return self.type

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f"Can not use type ‘{self.type.name}’ as value")
        return ErrorValue(self.environment.module, location)

    def add_member(self, symbol: SemanticSymbol):
        member: Child = symbol.as_symbol(symbol.location)
        self.type.add_member(member)
        self.environment.define(member.name, symbol)

    def add_generic(self, member: SemanticType):
        param = cast(GenericType, member.as_type(member.location))
        self.type.add_generic_parameter(param)

    def add_attribute(self, attribute: Attribute):
        self.type.add_attribute(attribute)

    def instantiate(self, arguments: Sequence[Type], location: Location) -> SemanticType:
        self.model.initialize(self)  # force initialization

        instance = self.type.instantiate(self.environment.module, arguments, location)
        return SemanticType(self.environment, instance)

    def __str__(self) -> str:
        return str(self.type)


class SemanticFunction(SemanticSymbol):
    def __init__(self, environment: SemanticEnvironment, function: Function):
        super().__init__(environment)

        self.__function = function

    @property
    def function(self) -> Function:
        return self.__function

    @property
    def name(self) -> str:
        return self.field.name

    @property
    def location(self) -> Location:
        return self.function.location

    @property
    def type(self) -> FunctionType:
        return self.function.function_type

    def as_symbol(self, location: Location) -> Symbol:
        return self.function

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f"Can not use function ‘{self.function}’ as type")
        return ErrorType(self.environment.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f"Can not use function ‘{self.function}’ as value")
        return ErrorValue(self.environment.module, location)

    def add_generic(self, member: SemanticType):
        param = cast(GenericType, member.as_type(member.location))
        self.function.add_generic_parameter(param)

    def add_attribute(self, attribute: Attribute):
        self.function.add_attribute(attribute)

    def instantiate(self, arguments: Sequence[Type], location: Location) -> SemanticFunction:
        instance = self.function.instantiate(self.environment.module, arguments, location)
        return SemanticFunction(self.environment, instance)

    def __str__(self) -> str:
        return str(self.function)


class SemanticOverload(SemanticSymbol):
    @property
    def functions(self) -> Sequence[Function]:
        return self.__functions


class SemanticValue(SemanticSymbol):
    def __init__(self, environment: SemanticEnvironment, value: Value):
        super().__init__(environment)

        self.__value = value

    @property
    def value(self) -> Value:
        return self.__value

    @property
    def type(self) -> Type:
        return self.value.type

    @property
    def location(self) -> Location:
        return self.value.location

    def as_symbol(self, location: Location) -> Symbol:
        return self.value

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f"Can not use value ‘{self.value}’ as type")
        return ErrorType(self.environment.module, location)

    def as_value(self, location: Location) -> Value:
        return self.value

    def instantiate(self, arguments: Sequence[Type], location: Location) -> SemanticError:
        self.diagnostics.error(location, f"Can not instantiate value")
        return SemanticError(self.environment, location)

    def __str__(self) -> str:
        return str(self.value)


class SemanticField(SemanticSymbol):
    def __init__(self, environment: SemanticEnvironment, field: Field):
        super().__init__(environment)

        self.__field = field

    @property
    def field(self) -> Field:
        return self.__field

    @property
    def name(self) -> str:
        return self.field.name

    @property
    def type(self) -> Type:
        return self.field.type

    @property
    def location(self) -> Location:
        return self.field.location

    def as_symbol(self, location: Location) -> Symbol:
        return self.field

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f"Can not use field ‘{self.name}’ as type")
        return ErrorType(self.environment.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f"Can not use field ‘{self.name}’ as value")
        return ErrorValue(self.environment.module, location)

    def instantiate(self, arguments: Sequence[Type], location: Location) -> SemanticError:
        self.diagnostics.error(location, f"Can not instantiate field ‘{self.field.name}’")
        return SemanticError(self.environment, location)

    def __str__(self) -> str:
        return str(self.field)


class SemanticProperty(SemanticSymbol):
    py_property = property

    def __init__(self, environment: SemanticEnvironment, symbol: Property):
        super().__init__(environment)

        self.__property = symbol

    @property
    def property(self) -> Property:
        return self.__property

    @py_property
    def name(self) -> str:
        return self.property.name

    @py_property
    def type(self) -> Type:
        return self.property.type

    @py_property
    def location(self) -> Location:
        return self.property.location

    del py_property

    def as_symbol(self, location: Location) -> Symbol:
        return self.property

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f"Can not use property ‘{self.name}’ as type")
        return ErrorType(self.environment.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f"Can not use property ‘{self.name}’ as type")
        return ErrorValue(self.environment.module, location)

    def instantiate(self, arguments: Sequence[Type], location: Location) -> SemanticError:
        self.diagnostics.error(location, f"Can not instantiate property ‘{self.property.name}’")
        return SemanticError(self.environment, location)

    def __str__(self) -> str:
        return str(self.property)


class SemanticAttribute(SemanticSymbol):
    def __init__(self, environment: SemanticEnvironment, attribute: Attribute):
        super().__init__(environment)

        self.__attribute = attribute

    @property
    def attribute(self) -> Attribute:
        return self.__attribute

    @property
    def name(self) -> str:
        return self.attribute.name

    @property
    def location(self) -> Location:
        return self.attribute.location

    def as_symbol(self, location: Location) -> Symbol:
        return self.attribute

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f"Can not use decorator ‘{self.name}’ as type")
        return ErrorType(self.environment.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f"Can not use decorator ‘{self.name}’ as type")
        return ErrorValue(self.environment.module, location)

    def instantiate(self, arguments: Sequence[Type], location: Location) -> SemanticError:
        self.diagnostics.error(location, f"Can not instantiate decorator ‘{self.attribute.name}’")
        return SemanticError(self.environment, location)

    def __str__(self) -> str:
        return str(self.attribute)


class SemanticMixin(abc.ABC):
    def __init__(self, environment: SemanticEnvironment):
        assert isinstance(environment, SemanticEnvironment)

        self.__environment = weakref.ref(environment)

    @property
    def model(self) -> SemanticModel:
        return self.environment.model

    @property
    def symbol_context(self) -> SymbolContext:
        return self.model.symbol_context

    @property
    def semantic_module(self) -> SemanticModule:
        return self.model.semantic_module

    @property
    def module(self) -> Module:
        return self.model.module

    @property
    def environment(self) -> SemanticEnvironment:
        return self.__environment()

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.model.diagnostics

    def as_symbol(self, node: SyntaxNode) -> Symbol:
        """ Convert node to type """
        return self.environment.as_symbol(node)

    def as_type(self, node: SyntaxNode) -> Type:
        """ Convert node to type """
        return self.environment.as_type(node)

    def as_value(self, node: SyntaxNode) -> Value:
        """ Convert node to value """
        return self.environment.as_value(node)


class EnvironmentAnnotator(SemanticMixin, NodeVisitor[SemanticEnvironment]):
    def __call__(self, node: SyntaxNode) -> SemanticEnvironment:
        return self.visit(node)

    def visit_node(self, node: SyntaxNode) -> SemanticEnvironment:
        return self.model.environments[node.parent]

    def visit_tree(self, _: SyntaxTree) -> SemanticEnvironment:
        return self.model.globals

    def visit_member(self, node: MemberNode) -> SemanticEnvironment:
        # TODO: environment for nested function must inherited from parent environment and scope
        return StaticEnvironment(self.model.environments[node.parent])


class ImportAnnotator(SemanticMixin, ImportVisitor[None]):
    def import_symbol(self, module: Module, name: str, alias: Optional[str] = None) -> bool:
        member = module.get_member(name)
        if not member:
            return False
        self.environment.define(alias or name, SemanticSymbol.from_symbol(self.environment, member))
        return True

    def import_all(self, module: Module):
        for name in {member.name for member in module.members}:
            self.import_symbol(module, name)

    def visit_import(self, node: ImportNode):
        for alias in node.aliases:
            try:
                module = self.model.context.open(alias.name).module
            except OrcinusError as ex:
                self.diagnostics.error(alias.location, str(ex))
            else:
                self.diagnostics.error(alias.location, 'Not implemented import')

    def visit_from_import(self, node: ImportFromNode):
        # Attempt to open module
        try:
            module = self.model.context.open(node.module).module
        except OrcinusError as ex:
            self.diagnostics.error(node.location, str(ex))
            return

            # Load aliases from nodes
        for alias in node.aliases:
            if not self.import_symbol(module, alias.name, alias.alias):
                self.diagnostics.error(alias.location, f'Cannot import name ‘{alias.name}’ from ‘{module.name}’')


class ScopeAnnotator(SemanticMixin, NodeVisitor[SemanticScope]):
    def __call__(self, node: SyntaxNode) -> SemanticScope:
        return self.visit(node)

    def visit_node(self, node: SyntaxNode) -> SemanticScope:
        return self.model.scopes[node.parent]

    def visit_tree(self, _: SyntaxTree) -> SemanticScope:
        return self.model.globals.scope

    def visit_statement(self, node: StatementNode) -> SemanticScope:
        raise DiagnosticError(node.location, "Can not retrieve scope for not analyzed function")


class SymbolAnnotator(SemanticMixin, NodeVisitor[Optional[Symbol]]):
    def __call__(self, node: SyntaxNode) -> Optional[SemanticSymbol]:
        symbol = self.visit(node)
        if isinstance(symbol, Symbol):
            return SemanticSymbol.from_symbol(self.model.environments[node], symbol)
        assert not symbol or isinstance(symbol, SemanticSymbol)
        return symbol

    @staticmethod
    def visit_node(_: SyntaxNode):
        return None

    def visit_tree(self, node: SyntaxTree) -> Module:
        return Module(self.model.symbol_context, node.name, node.location)

    def visit_member(self, node: MemberNode) -> Optional[Symbol]:
        parent = self.as_symbol(node.parent)
        environment = self.model.environments[node.parent]

        if isinstance(parent, Module):
            return ModuleMemberAnnotator(environment, parent).visit(node)
        elif isinstance(parent, ClassType):
            return ClassMemberAnnotator(environment, parent).visit(node)
        elif isinstance(parent, StructType):
            return StructMemberAnnotator(environment, parent).visit(node)
        elif isinstance(parent, InterfaceType):
            return InterfaceMemberAnnotator(environment, parent).visit(node)
        elif isinstance(parent, EnumType):
            return EnumMemberAnnotator(environment, parent).visit(node)
        elif isinstance(parent, Type):
            return TypeMemberAnnotator(environment, parent).visit(node)

        raise DiagnosticError(node.location, u'Not implemented annotation of member')

    def visit_generic_parameter(self, node: GenericParameterNode) -> Symbol:
        parent: Container = self.as_symbol(node.parent)
        return GenericType(parent, node.name, node.location)

    def visit_named_type(self, node: NamedTypeNode) -> SemanticSymbol:
        symbol = self.model.environments[node].search_named_symbol(node.name, node.location)
        if not symbol:
            self.diagnostics.error(node.location, f"Not found ‘{node.name}’ in current scope")
            return SemanticError(self, node.location)
        return symbol

    def visit_parameterized_type(self, node: ParameterizedTypeNode) -> SemanticSymbol:
        symbol: SemanticSymbol = self.visit(node.type)
        return symbol.instantiate([self.as_type(arg) for arg in node.arguments], node.location)

    def visit_auto_type(self, node: AutoTypeNode) -> Symbol:
        self.diagnostics.error("Not implemented type inference")
        return ErrorType(self.module, node.location)

    def visit_decorator(self, node: DecoratorNode) -> Symbol:
        arguments = cast(Sequence[Value], [self.as_value(arg) for arg in node.arguments])
        return Attribute(self.module, node.name, arguments, node.location)

    def visit_expression(self, node: ExpressionNode) -> Symbol:
        return ConstantEmitter(self.environment).visit(node)


class SymbolInitializer(SemanticMixin, NodeVisitor[None]):
    queue: Deque[SyntaxNode]  # Queue with not annotated or initialized nodes
    initialized: Set[SyntaxNode]

    def __init__(self, environment: SemanticEnvironment):
        super().__init__(environment)

        self.queue = Deque[SyntaxNode]()
        self.initialized = set()
        self.mapping = {}

    def __call__(self, node: SyntaxNode):
        if node in self.initialized:
            return

        self.queue.append(node)
        self.mapping[self.model.symbols[node]] = node

    def analyze(self):
        while self.queue:
            self.analyze_node(self.queue.popleft())

    def analyze_symbol(self, symbol: SemanticSymbol):
        if symbol not in self.mapping:
            return

        self.analyze_node(self.mapping[symbol])
        del self.mapping[symbol]  # remove from mapping

    def analyze_node(self, node: SyntaxNode):
        if node in self.initialized:
            return
        self.initialized.add(node)
        self.visit(node)

    def annotate_members(self, symbol: SemanticSymbol, members: Sequence[MemberNode]):
        types: List[TypeDeclarationNode] = []
        functions: List[FunctionNode] = []
        other: List[MemberNode] = []

        for node in members:
            if isinstance(node, TypeDeclarationNode):
                types.append(node)
            elif isinstance(node, FunctionNode):
                functions.append(node)
            else:
                other.append(node)

        for node in itertools.chain(types, functions, other):
            member: Optional[SemanticSymbol] = self.model.symbols[node]
            if member:
                symbol.add_member(member)

    def annotate_decorators(self, symbol: SemanticSymbol, decorators: Sequence[DecoratorNode]):
        # insert attributes
        for decorator in decorators:
            attr = self.model.symbols[decorator]
            symbol.add_attribute(attr.as_symbol(decorator.location))

    def annotate_parents(self, symbol: SemanticType, parents: Sequence[TypeNode]):
        for node in parents:
            parent = self.as_type(node)
            if parent is symbol.type or symbol.type in parent.ascendants:
                # TODO: Can find all types involving in circular dependency
                self.diagnostics.error(node.location, "Circular dependency")
            else:
                symbol.type.add_parent(parent)

    def annotate_generic_parameters(self, symbol: SemanticSymbol, generic_parameters: Sequence[TypeNode]):
        generic_symbol: GenericSymbol = symbol.as_symbol(symbol.location)
        assert isinstance(generic_symbol, GenericSymbol)

        for node in generic_parameters:
            semantic_param: SemanticType = self.model.symbols[node]
            generic_param: GenericType = semantic_param.as_type(node.location)

            symbol.environment.define(semantic_param.name, semantic_param)
            generic_symbol.add_generic_parameter(generic_param)

    def visit_node(self, node: SyntaxNode):
        pass

    def visit_tree(self, node: SyntaxTree):
        semantic_module = cast(SemanticModule, self.model.symbols[node])
        self.annotate_members(semantic_module, node.members)

    def visit_type_declaration(self, node: TypeDeclarationNode):
        semantic_type = cast(SemanticType, self.model.symbols[node])
        self.annotate_generic_parameters(semantic_type, node.generic_parameters)
        self.annotate_parents(semantic_type, node.parents)
        self.annotate_members(semantic_type, node.members)
        semantic_type.type.build()

    def visit_function(self, node: FunctionNode):
        function = cast(SemanticFunction, self.model.symbols[node])

        self.annotate_generic_parameters(function, node.generic_parameters)
        self.annotate_decorators(function, node.decorators)


class ParentMemberAnnotator(SemanticMixin, MemberVisitor[Optional[Symbol]]):
    def __init__(self, environment: SemanticEnvironment, parent: Container):
        super(ParentMemberAnnotator, self).__init__(environment)
        self.__parent = parent

    @property
    def parent(self) -> Container:
        return self.__parent

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.environment.diagnostics

    def visit_member(self, node: MemberNode):
        message = f"Not implemented member emitting: {type(node).__name__}"
        self.diagnostics.error(node.location, message)

    def visit_pass_member(self, node: PassMemberNode):
        pass

    def visit_function(self, node: FunctionNode) -> Function:
        # if function is method of struct/class/interface and first arguments type is auto, then it type can
        # be inferred to owner type
        if isinstance(self.parent, Type) and node.parameters and isinstance(node.parameters[0].type, AutoTypeNode):
            parameters = [self.parent]
            parameters.extend(self.as_type(param.type) for param in node.parameters[1:])
        else:
            parameters = [self.as_type(param.type) for param in node.parameters]

        # if return type of function is auto it can be inferred to void
        if isinstance(node.return_type, AutoTypeNode):
            return_type = self.symbol_context.void_type
        else:
            return_type = self.as_type(node.return_type)

        # create function symbol
        func_type = FunctionType(self.module, parameters, return_type, location=node.location)
        func = Function(self.parent, node.name, func_type, location=node.location)

        for node_param, func_param in zip(node.parameters, func.parameters):
            func_param.name = node_param.name
            func_param.location = node_param.location

            self.model.symbols[node_param] = SemanticSymbol.from_symbol(self.environment, func_param)

        # assert node not in self.symbols
        return func

    def visit_class(self, node: ClassNode) -> ClassType:
        return ClassType(self.parent, node.name, location=node.location)

    def visit_struct(self, node: StructNode) -> StructType:
        return StructType(self.parent, node.name, location=node.location)

    def visit_enum(self, node: EnumNode) -> EnumType:
        if node.generic_parameters:
            self.diagnostics.error(node.location, 'Generic types is not implemented')
        return EnumType(self.parent, node.name, location=node.location)

    def visit_interface(self, node: InterfaceNode) -> InterfaceType:
        return InterfaceType(self.parent, node.name, location=node.location)


class ModuleMemberAnnotator(ParentMemberAnnotator):
    @property
    def parent(self) -> Module:
        return cast(Module, super().parent)

    def visit_class(self, node: ClassNode) -> Type:
        if self.environment.module == self.environment.symbol_context.builtins_module:
            if node.name == TYPE_STRING_NAME:
                return StringType(self.parent, location=node.location)
            elif node.name == TYPE_ARRAY_NAME:
                return ArrayType(self.parent, location=node.location)

        return super().visit_class(node)

    def visit_struct(self, node: StructNode) -> Type:
        if self.environment.module == self.environment.symbol_context.builtins_module:
            if node.name == TYPE_INTEGER_NAME:
                return IntegerType(self.parent, location=node.location)
            elif node.name == TYPE_BOOLEAN_NAME:
                return BooleanType(self.parent, location=node.location)
            elif node.name == TYPE_VOID_NAME:
                return VoidType(self.parent, location=node.location)
            elif node.name == TYPE_FLOAT_NAME:
                return FloatType(self.parent, location=node.location)

        return super().visit_struct(node)


class TypeMemberAnnotator(ParentMemberAnnotator):
    @property
    def parent(self) -> Type:
        return cast(Type, super().parent)

    def create_property_field(self, node: FieldNode) -> Field:
        field_type = self.as_type(node.type)
        field = Field(self.parent, node.name, field_type, node.location)

        self.parent.add_member(field)
        return field

    def create_property_getter(self, node: FieldNode, _: Field = None) -> Function:
        prop_type = self.as_type(node.type)
        getter_type = FunctionType(self.module, [self.parent], prop_type, node.location)
        getter_func = Function(self.parent, node.name, getter_type, node.location)
        getter_func.parameters[0].name = 'self'

        self.parent.add_member(getter_func)
        return getter_func

    def create_property_setter(self, node: FieldNode, _: Field = None) -> Function:
        prop_type = self.as_type(node.type)
        setter_type = FunctionType(self.module, [self.parent, prop_type], self.symbol_context.void_type,
                                   node.location)
        setter_func = Function(self.parent, node.name, setter_type, node.location)
        setter_func.parameters[0].name = 'self'
        setter_func.parameters[1].name = 'value'

        self.parent.add_member(setter_func)
        return setter_func


class StructMemberAnnotator(TypeMemberAnnotator):
    @property
    def parent(self) -> StructType:
        return cast(StructType, super().parent)

    def visit_field(self, node: FieldNode) -> Property:
        field = self.create_property_field(node)
        getter_func = self.create_property_getter(node, field)
        setter_func = self.create_property_setter(node, field)
        return Property(self.parent, node.name, getter_func, setter_func, location=node.location)


class ClassMemberAnnotator(TypeMemberAnnotator):
    @property
    def parent(self) -> ClassType:
        return cast(ClassType, super().parent)

    def visit_field(self, node: FieldNode) -> Property:
        field = self.create_property_field(node)
        getter_func = self.create_property_getter(node, field)
        setter_func = self.create_property_setter(node, field)
        return Property(self.parent, node.name, getter_func, setter_func, location=node.location)


class InterfaceMemberAnnotator(TypeMemberAnnotator):
    @property
    def parent(self) -> InterfaceType:
        return cast(InterfaceType, super().parent)

    def visit_field(self, node: FieldNode) -> Property:
        getter_func = self.create_property_getter(node)
        setter_func = self.create_property_setter(node)
        return Property(self.parent, node.name, getter_func, setter_func, location=node.location)


class EnumMemberAnnotator(TypeMemberAnnotator):
    @property
    def parent(self) -> EnumType:
        return cast(EnumType, super().parent)

    def visit_enum_value(self, node: EnumValueNode) -> Optional[Symbol]:
        value = node.parent.members.index(node)
        if node.value:
            # expr = self.model.symbols[node.value]
            # if not isinstance(expr, IntegerConstant):
            #     value = expr.as_value()
            # else:
            self.diagnostics.error(node.location, "Enumeration values must have integer value or ‘...’")

        return EnumValue(self.parent, node.name, value, node.location)


class ConstantEmitter(SemanticMixin, ExpressionVisitor[Value]):
    def visit_expression(self, node: ExpressionNode) -> Value:
        self.diagnostics.error(node.location, "Required constant expression")
        return ErrorValue(self.module, node.location)

    def visit_integer_expression(self, node: IntegerExpressionNode) -> Value:
        return IntegerConstant(self.symbol_context, int(node.value), location=node.location)

    def visit_string_expression(self, node: StringExpressionNode) -> Value:
        return StringConstant(self.symbol_context, node.value, location=node.location)


class FunctionResolver(SemanticMixin):
    def __init__(self,
                 environment: SemanticEnvironment,
                 arguments: Sequence[Value] = None,
                 keywords: Mapping[str, Value] = None,
                 *, location: Location):
        super(FunctionResolver, self).__init__(environment)

        self.location = location

        self.arguments = arguments or []
        self.keywords = keywords or {}
        self.candidates = []

    def add_scope_functions(self, name: str):
        symbol = self.environment.resolve(name)
        if isinstance(symbol, SemanticFunction):
            self.candidates.append(symbol.function)
        elif isinstance(symbol, SemanticOverload):
            self.candidates.extend(symbol.functions)

    def add_type_functions(self, clazz: Type, name: str):
        functions = (member for member in clazz.members if isinstance(member, Function) and member.name == name)
        self.candidates.extend(functions)

    def add_self_functions(self, name: str):
        if self.arguments:
            self.add_type_functions(self.arguments[0].type, name)

    def check_function(self, func: Function) -> Optional[Tuple[int, Function, Sequence[Value]]]:
        """
        Returns:

            - None              - if function can not be called with arguments
            - int               - if function can be called with arguments. Returns priority

        :param func:
        :return:
        """
        arguments = self.arguments[:]
        arguments.extend(None for _ in range(0, max(0, len(func.parameters) - len(arguments))))

        for name, value in self.keywords.items():
            idx = next((idx for idx, param in enumerate(func.parameters) if param.name == name), -1)
            if idx == -1:
                return None  # Fast exit, because one of keywords is not filled, e.g. must check default value
            arguments[idx] = value

        if len(func.parameters) != len(arguments):
            return None  # Fast exit, because not all arguments is not filled

        if any(arg is None for arg in arguments):
            return None  # Fast exit, because not all arguments is not filled

        priority: int = 0
        for param, arg in zip(func.parameters, arguments):
            if arg.type != param.type:
                return None
            priority += 2
        return priority, func, arguments

    def find_function(self) -> Optional[Tuple[Function, Sequence[Value]]]:
        # check candidates
        counter = itertools.count()
        candidates = []
        for func in self.candidates:
            result = self.check_function(func)
            if result is not None:
                priority, instance, arguments = result
                heapq.heappush(candidates, (priority, next(counter), instance, arguments))

        # pop all function with minimal priority
        functions = []
        current_priority = None
        while candidates:
            priority, _, func, arguments = heapq.heappop(candidates)
            if current_priority is not None and current_priority != priority:
                break

            current_priority = priority
            functions.append((func, arguments))

        if functions:
            return functions[0]
        return None


class LocalEnvironment(SemanticEnvironment):
    """ This environment is used for emitting symbols on function level """

    def __init__(self, parent: SemanticEnvironment, node: FunctionNode):
        super(LocalEnvironment, self).__init__(parent.model)

        self.__parent = weakref.ref(parent)
        self.__node = node
        self.__builder = IRBuilder(self.function.append_basic_block('entry'))
        self.__scopes = [LocalScope(self.builder.block)]

        for parameter in self.function.parameters:
            self.define(parameter.name, SemanticSymbol.from_symbol(self, parameter))

    @property
    def scope(self) -> LocalScope:
        return self.__scopes[-1]

    @scope.setter
    def scope(self, scope: LocalScope):
        self.__scopes[-1] = scope

    @property
    def parent(self) -> SemanticEnvironment:
        return self.__parent()

    @property
    def builder(self) -> IRBuilder:
        return self.__builder

    @property
    def node(self) -> FunctionNode:
        return self.__node

    @cached_property
    def semantic_function(self) -> SemanticFunction:
        return cast(SemanticFunction, self.model.symbols[self.node])

    @cached_property
    def function(self) -> Function:
        return self.semantic_function.function

    def push(self, block: BasicBlock = None) -> LocalScope:
        """ Create new scope inherited from current and push to stack """
        scope = LocalScope(block or self.builder.block, self.scope)
        self.__scopes.append(scope)
        return scope

    def pop(self) -> LocalScope:
        """ Pop scope from stack """
        scope = self.__scopes.pop()
        return scope

    @contextlib.contextmanager
    def usage(self, block: BasicBlock) -> LocalScope:
        scope = self.push(block)
        yield scope
        scope.exit_block = self.builder.block
        self.pop()

    def resolve(self, name: str) -> Optional[Symbol]:
        return super().resolve(name) or self.parent.resolve(name)

    def search_named_call(self, name, arguments=None, keywords=None, *, location):
        function = FunctionResolver(self, arguments, keywords, location=location)
        function.add_self_functions(name)
        function.add_scope_functions(name)
        return function.find_function()

    def emit_error_call(self, name, arguments=None, keywords=None, *, location: Location):
        arguments = (f"{arg.type}" for arg in arguments) if arguments else ()
        keywords = (f"{key}: {arg.type}" for key, arg in keywords.items()) if keywords else ()
        arguments = '. '.join(itertools.chain(arguments, keywords))
        self.diagnostics.error(location, f"Not found function to call ‘{name}({arguments})’")
        return ErrorValue(self.module, location)

    def emit_resolver_call(self, result, name, arguments=None, keywords=None, *, location: Location):
        if not result:
            return self.emit_error_call(name, arguments, keywords, location=location)
        function, arguments = result
        return self.builder.call(function, arguments, location=location)

    def emit_named_call(self,
                        name: str,
                        arguments: Sequence[Value] = None,
                        keywords: Mapping[str, Value] = None,
                        *, location: Location):
        result = self.search_named_call(name, arguments, keywords, location=location)
        return self.emit_resolver_call(result, name, arguments, keywords, location=location)

    def emit_constructor_call(self,
                              clazz: Type,
                              arguments: Sequence[Value] = None,
                              keywords: Mapping[str, Value] = None,
                              *, location: Location):
        resolver = FunctionResolver(self, arguments, keywords, location=location)
        resolver.add_type_functions(clazz, NEW_NAME)
        result = resolver.find_function()
        return self.emit_resolver_call(result, str(clazz), arguments, keywords, location=location)

    def emit_return(self, value: Value, location: Location):
        return_type = self.function.return_type
        value_type = value.type if value else self.symbol_context.void_type

        if isinstance(return_type, ErrorType) or isinstance(value_type, ErrorType):
            pass  # Skip error propagation
        elif value and value_type != return_type:
            message = f"Function must return value of ‘{return_type}’ type, but got ‘{value_type}’"
            self.diagnostics.error(location, message)

        self.builder.ret(value, location=location)

    def emit_boolean(self, value: Value, location: Location):
        boolean_type = self.symbol_context.boolean_type
        if isinstance(value.type, BooleanType):
            return value

        result = self.search_named_call('__bool__', [value], location=location)
        if result:
            return self.emit_resolver_call(result, '__bool_', [value], location=location)

        message = f"Logical value must be value of ‘{boolean_type}’ type, but got ‘{value.type}’"
        self.diagnostics.error(location, message)
        return ErrorValue(self.module, location)

    def emit_statement(self, node: StatementNode):
        emitter = StatementEmitter(self)
        return emitter.emit_statement(node)

    def emit_expression(self, node: ExpressionNode) -> Value:
        emitter = ExpressionEmitter(self)
        return emitter.emit_expression(node)

    def emit(self, node: SyntaxNode) -> SemanticSymbol:
        if isinstance(node, StatementNode):
            return self.emit_statement(node)
        elif isinstance(node, ExpressionNode):
            value = self.emit_expression(node)
            return SemanticSymbol.from_symbol(self, value)

        return super().emit(node)

    def analyze(self):
        self.emit(self.node.statement)
        if not self.builder.is_terminated:
            if isinstance(self.function.return_type, VoidType):
                self.emit_return(NoneConstant(self.symbol_context, self.node.location), self.node.location)
            else:
                self.diagnostics.error(self.node.location, 'Function must return value')
        return self.function


class FunctionMixin(SemanticMixin):
    # 1. flow graph (basic block)

    def __init__(self, parent: LocalEnvironment):
        super(FunctionMixin, self).__init__(parent)

    @property
    def environment(self) -> LocalEnvironment:
        return cast(LocalEnvironment, super().environment)

    @property
    def function(self) -> Function:
        return self.environment.function

    @property
    def builder(self) -> IRBuilder:
        return self.environment.builder

    @property
    def block(self) -> BasicBlock:
        return self.builder.block

    def merge(self, scopes: Set[SemanticScope], location: Location) -> SemanticScope:
        """ Merge scopes in next scope """
        scope = self.environment.push()

        names = [set(previous.keys()) for previous in scopes]
        names = set.intersection(*map(set, names))

        for name in names:
            bases: Mapping[LocalScope, SemanticValue] = {previous: previous.resolve(name) for previous in scopes}
            if not bases:
                continue

            # merge only semantic values
            if not all(isinstance(base, SemanticValue) for base in bases.values()):
                continue

            first_name: SemanticValue = first(iter(bases.values()))

            # merge only values with equal type
            if not all(base.type == first_name.type for base in bases.values()):
                continue

            # merge
            with self.builder.goto_entry_block():
                variable = self.builder.alloca(first_name.type, name=name, location=first_name.location)

            for previous, base in bases.items():
                value = base.as_value(location)
                with self.builder.goto_block(previous.exit_block):
                    if isinstance(value, AllocaInstruction):
                        value = self.builder.load(value, location=first_name.location)
                    self.builder.store(variable, value, location=first_name.location)

            scope.define(name, SemanticValue(self.environment, variable))

        return scope

    def emit_boolean(self, value: Value, location: Location) -> Value:
        return self.environment.emit_boolean(value, location)

    def emit_statement(self, node: StatementNode):
        return self.environment.emit_statement(node)

    def emit_expression(self, node: ExpressionNode) -> Value:
        return self.environment.emit_expression(node)


class StatementEmitter(FunctionMixin, StatementVisitor[None]):
    def emit_statement(self, node: StatementNode):
        """
        Emit environment and IR for statement.
        """
        if self.builder.is_terminated:
            return self.diagnostics.error(node.location, f"Unreachable code")

        return self.visit(node)

    def visit_statement(self, node: StatementNode):
        message = f"Not implemented statement emitting: {type(node).__name__}"
        self.diagnostics.error(node.location, message)

    def visit_block_statement(self, node: BlockStatementNode):
        for statement in node.statements:
            self.emit_statement(statement)

    def visit_pass_statement(self, node: PassStatementNode):
        pass

    def visit_return_statement(self, node: ReturnStatementNode):
        value = self.emit_expression(node.value) if node.value else None
        value = value or NoneConstant(self.symbol_context, node.location)
        self.environment.emit_return(value, value.location)

    def visit_expression_statement(self, node: ExpressionStatementNode):
        self.emit_expression(node.value)

    def visit_condition_statement(self, node: ConditionStatementNode):
        condition = self.emit_expression(node.condition)
        condition = self.emit_boolean(condition, node.condition.location)

        begin_scope = self.environment.scope
        begin_block = self.builder.block

        continue_blocks = set()  # Block that continue execution

        # then block and scope
        then_block = self.builder.append_basic_block('if.then')
        with self.environment.usage(then_block) as then_scope:  # new scope
            self.builder.position_at_end(then_block)
            self.emit_statement(node.then_statement)
            if not self.builder.is_terminated:
                continue_blocks.add(self.builder.block)

        # else block and scope
        if node.else_statement:
            else_block = self.builder.append_basic_block('if.else')
            with self.environment.usage(else_block) as else_scope:  # new scope
                self.builder.position_at_end(else_block)
                self.emit_statement(node.else_statement)
                if not self.builder.is_terminated:
                    continue_blocks.add(self.builder.block)
            next_block = None

        # begin scope
        else:
            else_block = self.builder.append_basic_block('if.next')
            next_block = else_block
            else_scope = begin_scope

        # branch from begin block to then and else blocks
        with self.builder.goto_block(begin_block):
            self.builder.cbranch(condition, then_block, else_block, location=node.location)

        # check if next blocks is unreachable
        if not next_block and not continue_blocks:
            return

        if not next_block:
            next_block = self.builder.append_basic_block('if.next')

        for block in filter(None, continue_blocks):
            with self.builder.goto_block(block):
                self.builder.branch(next_block, location=node.location)

        # mark as continue
        self.builder.position_at_end(next_block)

        # merge scopes
        self.merge({then_scope, else_scope}, node.location)

    def visit_else_statement(self, node: ElseStatementNode):
        return self.emit_statement(node.statement)

    def visit_while_statement(self, node: WhileStatementNode):
        cond_block = self.builder.append_basic_block('while.cond')
        self.builder.branch(cond_block, location=node.location)
        self.builder.position_at_end(cond_block)

        condition = self.emit_expression(node.condition)
        condition = self.emit_boolean(condition, node.condition.location)

        begin_scope = self.environment.scope
        begin_block = self.builder.block

        continue_blocks = set()  # Block that continue execution

        # then block and scope
        then_block = self.builder.append_basic_block('while.body')
        with self.environment.usage(then_block) as then_scope:  # new scope
            self.builder.position_at_end(then_block)
            self.emit_statement(node.then_statement)
            if not self.builder.is_terminated:
                self.builder.branch(cond_block, location=node.location)

        # else block and scope
        if node.else_statement:
            else_block = self.builder.append_basic_block('while.else')
            with self.environment.usage(else_block) as else_scope:  # new scope
                self.builder.position_at_end(else_block)
                self.emit_statement(node.else_statement)
                if not self.builder.is_terminated:
                    continue_blocks.add(self.builder.block)
            next_block = None

        # begin scope
        else:
            else_block = self.builder.append_basic_block('while.next')
            next_block = else_block
            else_scope = begin_scope

        # branch from begin block to then and else blocks
        with self.builder.goto_block(begin_block):
            self.builder.cbranch(condition, then_block, else_block, location=node.location)

        # check if next blocks is unreachable
        if not next_block and not continue_blocks:
            return

        if not next_block:
            next_block = self.builder.append_basic_block('while.next')

        for block in filter(None, continue_blocks):
            with self.builder.goto_block(block):
                self.builder.branch(next_block, location=node.location)

        # mark as continue
        self.builder.position_at_end(next_block)

        # merge scope
        self.merge({then_scope, else_scope}, node.location)

    def visit_with_statement(self, node: WithStatementNode):
        # Current implementation
        #
        # with <expression> [ as <target> ] :
        #   <condition>
        #
        # $1 = <expression>
        # $source = call __enter__ $1
        # [ store <target> $source ]
        # <condition> ...
        # call __exit__ $1
        contexts = []
        for item in node.items:
            context = self.emit_expression(item.expression)
            contexts.append((context, item.expression.location))
            source = self.environment.emit_named_call('__enter__', [context], location=item.expression.location)
            if item.target:
                emitter = AssignmentEmitter(self.environment, source, node.location)
                emitter.visit(item.target)

        self.emit_statement(node.statement)  # Can be unreachabled!

        for context, location in reversed(contexts):
            self.environment.emit_named_call('__exit__', [context], location=location)

    def visit_variable_statement(self, node: VariableStatementNode):
        # allocate register for variable
        var_type = self.as_type(node.type)
        with self.builder.goto_entry_block():
            variable = self.builder.alloca(var_type, name=node.name, location=node.location)

        # initialize variable with initial value
        if node.initial_value:
            initial_value = self.emit_expression(node.initial_value)
        else:
            initial_value = self.builder.get_default_value(var_type, location=node.location)
        self.builder.store(variable, initial_value, location=node.location)

        # store in environment
        symbol = SemanticSymbol.from_symbol(self.environment, variable)
        self.environment.define(node.name, symbol)

    def visit_assign_statement(self, node: AssignStatementNode):
        source = self.emit_expression(node.source)
        emitter = AssignmentEmitter(self.environment, source, node.location)
        emitter.visit(node.target)


class ExpressionEmitter(FunctionMixin, ExpressionVisitor[Symbol]):
    def emit_expression(self, node: ExpressionNode):
        return self.visit(node)

    def visit_expression(self, node: ExpressionNode) -> Symbol:
        self.diagnostics.error(node.location, "Not implemented expression")
        return ErrorValue(self.module, node.location)

    def visit_integer_expression(self, node: IntegerExpressionNode) -> Value:
        return IntegerConstant(self.symbol_context, int(node.value), location=node.location)

    def visit_string_expression(self, node: StringExpressionNode) -> Value:
        return StringConstant(self.symbol_context, node.value, location=node.location)

    def visit_named_expression(self, node: NamedExpressionNode) -> Symbol:
        symbol = self.environment.search_named_symbol(node.name, node.location)
        if isinstance(symbol, SemanticValue):
            value = symbol.as_value(node.location)
            if isinstance(value, AllocaInstruction):
                return self.builder.load(value, location=node.location)
        elif not symbol:
            self.diagnostics.error(node.location, f"Not found ‘{node.name}’ in current scope")
            return ErrorValue(self.module, node.location)
        return symbol.as_symbol(node.location)

    def visit_attribute_expression(self, node: AttributeExpressionNode) -> Symbol:
        # Can assign to: fields
        instance = self.emit_expression(node.instance)
        if isinstance(instance, Value):
            member = instance.type.get_member(node.name)
            if isinstance(member, Field):
                return self.builder.extract_value(instance, member, name=node.name, location=node.location)
            else:
                self.diagnostics.error(node.location, 'Can not load value to expression')
        else:
            self.diagnostics.error(node.location, 'Can not load value to expression')
        return ErrorValue(self.module, node.location)

    def visit_compare_expression(self, node: CompareExpressionNode) -> Symbol:
        left_argument = self.emit_expression(node.left_argument)

        if len(node.comparators) > 1:
            self.diagnostics.error(node.location, "Multiple comparators is not implemented")
            return BooleanConstant(self.symbol_context, False, node.location)
        elif not node.comparators:
            self.diagnostics.error(node.location, "Compare expression must contains at least one on comparator")
            return BooleanConstant(self.symbol_context, False, node.location)

        for comparator in node.comparators:
            right_argument = self.emit_expression(comparator.right_argument)

            # TODO: Special cases: `in` and `is`
            name = COMPARE_NAMES.get(comparator.opcode)
            if not name:
                self.diagnostics.error(comparator.location, "Not implemented comparison operator")
                return BooleanConstant(self.symbol_context, False, comparator.location)

            function = FunctionResolver(self.environment, [left_argument, right_argument],
                                        location=comparator.location)
            function.add_self_functions(name)
            result = function.find_function()
            if not result:
                self.diagnostics.error(comparator.location, f"Not found method ‘{name}’")  # TODO: Normal message
                return BooleanConstant(self.symbol_context, False, node.location)
            else:
                function, arguments = result
                return self.builder.call(function, arguments, location=comparator.location)

        # Unreachable
        return BooleanConstant(self.symbol_context, False, node.location)

    def visit_unary_expression(self, node: UnaryExpressionNode) -> Symbol:
        argument = self.emit_expression(node.argument)

        name = UNARY_NAMES.get(node.opcode)
        if not name:
            self.diagnostics.error(node.location, "Not implemented binary operator")
            return ErrorValue(self.module, node.location)

        return self.environment.emit_named_call(name, [argument], location=node.location)

    def visit_binary_expression(self, node: BinaryExpressionNode) -> Symbol:
        left_argument = self.emit_expression(node.left_argument)
        right_argument = self.emit_expression(node.right_argument)

        name = BINARY_NAMES.get(node.opcode)
        if not name:
            self.diagnostics.error(node.location, "Not implemented binary operator")
            return ErrorValue(self.module, node.location)

        return self.environment.emit_named_call(name, [left_argument, right_argument], location=node.location)

    def visit_call_expression(self, node: CallExpressionNode) -> Symbol:
        arguments = [self.emit_expression(arg) for arg in node.arguments]
        keywords = {name: self.emit_expression(arg) for name, arg in node.keywords.items()}

        emitter = CallEmitter(self.environment, arguments, keywords, node.location)
        return emitter.visit(node.instance)

    def visit_tuple_expression(self, node: TupleExpressionNode):
        if len(node.arguments) != 1:
            self.diagnostics.error(node.location, f"Tuple value is not implemented")  # TODO: Normal message
            return ErrorValue(self.module, node.location)
        return self.emit_expression(node.arguments[0])


class AssignmentEmitter(FunctionMixin, ExpressionVisitor[None]):
    def __init__(self, environment: LocalEnvironment, source: Value, location: Location):
        super().__init__(environment)

        self.source = source
        self.location = location

    def visit_expression(self, node: ExpressionNode):
        self.diagnostics.error(node.location, 'Can not store value')

    def visit_named_expression(self, node: NamedExpressionNode):
        variable = self.environment.search_named_symbol(node.name, node.location)

        if not variable:
            # define new implicit variable
            with self.builder.goto_entry_block():
                target = self.builder.alloca(self.source.type, name=node.name, location=node.location)
            self.environment.define(node.name, SemanticValue(self.environment, target))
        else:
            target = variable.as_value(node.location)

        if isinstance(target, AllocaInstruction):
            # store to variable
            self.builder.store(target, self.source, location=self.location)

        elif not isinstance(target, ErrorValue):
            self.diagnostics.error(self.location, 'Can not store value')

    def visit_attribute_expression(self, node: AttributeExpressionNode):
        # Can assign to: fields
        instance = self.emit_expression(node.instance)
        if isinstance(instance, Value):
            member = instance.type.get_member(node.name)
            if isinstance(member, Field):
                self.builder.insert_value(instance, member, self.source, location=self.location)
            else:
                self.diagnostics.error(node.location, 'Can not store value')
        else:
            self.diagnostics.error(node.location, 'Can not store value')


class CallEmitter(FunctionMixin, ExpressionVisitor[Value]):
    def __init__(self,
                 environment: LocalEnvironment,
                 arguments: Sequence[Value],
                 keywords: Mapping[str, Value],
                 location: Location):

        super(CallEmitter, self).__init__(environment)

        self.arguments = list(arguments)
        self.keywords = keywords
        self.location = location

    def emit_named(self, name: str):
        return self.environment.emit_named_call(name, self.arguments, self.keywords, location=self.location)

    def emit_constructor(self, clazz: Type):
        return self.environment.emit_constructor_call(clazz, self.arguments, self.keywords, location=self.location)

    def visit_expression(self, node: ExpressionNode):
        self.arguments.insert(0, self.emit_expression(node))
        return self.emit_named('__call__')

    def visit_named_expression(self, node: NamedExpressionNode):
        symbol = self.environment.search_named_symbol(node.name, node.location)
        if isinstance(symbol, SemanticType):
            return self.emit_constructor(symbol.as_type(node.location))
        return self.emit_named(node.name)

    def visit_attribute_expression(self, node: AttributeExpressionNode):
        value = self.emit_expression(node.instance)
        if isinstance(value, Value):
            member = value.type.get_member(node.name)
            if not member or isinstance(member, Function):
                self.arguments.insert(0, value)
                return self.emit_named(node.name)

            self.arguments.insert(0, self.emit_expression(node))
            return self.emit_named('__call__')

        self.diagnostics.error(self.location, f"Not implemented call’")  # TODO: Normal message
        return ErrorValue(self.module, self.location)
