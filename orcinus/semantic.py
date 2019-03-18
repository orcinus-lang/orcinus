# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import heapq
import logging
import os
from typing import MutableMapping, Tuple, Mapping

from orcinus.diagnostics import DiagnosticError
from orcinus.parser import Parser
from orcinus.scanner import Scanner
from orcinus.symbols import *
from orcinus.syntax import *
from orcinus.utils import cached_property

logger = logging.getLogger('orcinus')

Arguments = Sequence['SemanticSymbol']
Keywords = Mapping[str, 'SemanticSymbol']
SymbolConstructor = Callable[[SyntaxNode], 'SemanticSymbol']


class SemanticModuleLoader(ModuleLoader):
    def __init__(self, context: SemanticContext):
        self.__context = weakref.ref(context)

    @property
    def context(self) -> SemanticContext:
        return self.__context()

    def open(self, name: str) -> Module:
        model = self.context.open(name)
        return model.module


class SemanticContext:
    def __init__(self, paths=None, *, diagnostics: DiagnosticManager = None):
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
        self.symbol_context = SymbolContext(SemanticModuleLoader(self), diagnostics=self.diagnostics)
        self.syntax_context = SyntaxContext(self.diagnostics)

        self.paths = []
        self.paths.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'packages/stdlib'))
        self.paths.extend(paths)
        self.paths.append(os.path.join(os.getcwd()))

        self.__models = {}
        self.__filenames = {}

    @staticmethod
    def get_module_name(path, filename) -> Optional[str]:
        """ Find module name for filename """
        module_name, _ = os.path.splitext(os.path.relpath(filename, path))
        module_name = module_name.strip('/')
        module_name = module_name.replace(os.path.sep, '.')
        return module_name

    def load(self, filename: str, name: str = None) -> SemanticModel:
        """
        Open file with module

        :param filename: relative path to module source file
        :param name:     module name
        """
        # iterate over paths
        for path in self.paths:
            # check, if file already opened
            fullname = os.path.join(path, filename)
            if fullname in self.__filenames:
                return self.__filenames[fullname]

            # try open file from path
            name = self.get_module_name(path, fullname)
            model = self.__try_open_file(fullname, name)
            if model:
                self.__models[name] = model
                self.__filenames[fullname] = model
                model.analyze()
                model.emit()
                return model

        paths = '\n  - ' + ('\n  - '.join(map(str, self.paths)))
        raise DiagnosticError(Location(filename), f"Not found module ’{name}’ from ’{filename}’ in: {paths}")

    def __try_open_file(self, fullname: str, name: str):
        """
        This method is used for attempt to load AST module from file and create
        semantic model for him.
        """
        try:
            with Scanner(fullname) as scanner:
                logger.debug(f"Load module ’{name}’ from file ’{fullname}’")

                parser = Parser(scanner, self.syntax_context)
                tree = parser.parse()
        except IOError:
            logger.debug(f"Module ’{name}’ not found file ’{fullname}’")
            return None
        else:
            # Create semantic model
            return SemanticModel(self, tree, name)

    def open(self, name: str) -> SemanticModel:
        """
        Load module from packages
        """
        if name in self.__models:
            return self.__models[name]

        filename = name.replace('.', os.path.sep) + '.orx'
        return self.load(filename, name)


class SemanticModel:
    tree: SyntaxTree

    def __init__(self, context: SemanticContext, tree: SyntaxTree, name: str):
        self.context = context
        self.tree = tree
        self.name = name
        self.imports = {}

        annotator = StaticAnnotator(self)
        self.symbols = SemanticScope(self, constructor=lambda n: annotator.annotate(n))

    @property
    def symbol_context(self):
        return self.context.symbol_context

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.context.diagnostics

    @cached_property
    def module(self) -> Module:
        return cast(SemanticModule, self.symbols[self.tree]).module

    def as_symbol(self, symbol: Symbol) -> SemanticSymbol:
        if isinstance(symbol, Function):
            return SemanticFunction(self, symbol)
        elif isinstance(symbol, Value):
            return SemanticValue(self, symbol)
        elif isinstance(symbol, Type):
            return SemanticType(self, symbol)
        elif isinstance(symbol, Module):
            return SemanticModule(self, symbol)
        elif isinstance(symbol, Field):
            return SemanticField(self, symbol)

        raise NotImplementedError

    def initialize(self, node: SyntaxNode):
        initializer = SemanticInitializer(self)
        initializer.initialize(node)

    def import_symbol(self, module: Module, name: str, alias: Optional[str] = None) -> bool:
        members = [member for member in module.members if member.name == name]

        if members and all(isinstance(member, Function) for member in members):
            functions = cast(Sequence[Function], members)
            member = Overload(module, alias or name, functions, members[0].location)
        elif len(members) != 1:
            return False
        else:
            member = members[0]
        self.imports[alias or name] = member
        return True

    def import_all(self, module: Module):
        for name in {member.name for member in module.members}:
            self.import_symbol(module, name)

    def import_symbols(self):
        if self.name != BUILTINS_MODULE:
            self.import_all(self.symbol_context.builtins_module)

        for node in self.tree.imports:
            if isinstance(node, ImportFromNode):
                module = self.context.open(node.module).module
                for alias in node.aliases:  # type: AliasNode
                    if not self.import_symbol(module, alias.name, alias.alias):
                        self.diagnostics.error(node.location, f'Cannot import name ‘{alias.name}’ from ‘{module.name}’')
            else:
                self.diagnostics.error(node.location, 'Not implemented member')

    def analyze(self):
        self.import_symbols()
        self.symbols.get(self.tree)

    def emit_function(self, node: FunctionNode):
        annotator = FunctionAnnotator(self, self.symbols, node)
        annotator.annotate_statement(node.statement)

    def emit(self):
        functions: Sequence[FunctionNode] = list(
            self.tree.find_descendants(lambda node: isinstance(node, FunctionNode))
        )
        for func in functions:
            if func.statement:
                self.emit_function(func)


class SemanticSymbol(abc.ABC):
    """ Abstract base for all semantic symbols """

    def __init__(self, model: SemanticModel, symbol: Symbol):
        self.model = model
        self.symbol = symbol

    @property
    def location(self) -> Location:
        return self.symbol.location

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.model.diagnostics

    @abc.abstractmethod
    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f'Operation is not implemented')
        return ErrorType(self.model, location)

    @abc.abstractmethod
    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f'Operation is not implemented')
        return ErrorValue(self.model, location)

    def as_functions(self, location: Location) -> Sequence[Function]:
        self.diagnostics.error(location, f'Operation is not implemented')
        return []

    def find_call(self, arguments: Arguments, keywords: Keywords, location: Location) -> Optional[SemanticSymbol]:
        self.diagnostics.error(location, f'Operation is not implemented')
        return None

    def find_attribute(self, name: str, location: Location) -> Optional[SemanticSymbol]:
        self.diagnostics.error(location, f'Operation is not implemented')
        return None

    def subscript(self, arguments: Arguments, location: Location) -> SemanticSymbol:
        self.diagnostics.error(location, f'Operation is not implemented')
        return SemanticError(self.model, location)

    def __str__(self) -> str:
        return str(self.symbol)


class SemanticModule(SemanticSymbol):
    @property
    def module(self) -> Module:
        return cast(Module, self.symbol)

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f'Can not use module ‘{self.module}’ as type')
        return ErrorType(self.model.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f'Can not use module ‘{self.module}’ as value')
        return ErrorValue(self.model.module, location)


class SemanticAttribute(SemanticSymbol):
    @property
    def attribute(self) -> Attribute:
        return cast(Attribute, self.symbol)

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f'Can not use module ‘{self.attribute}’ as type')
        return ErrorType(self.model.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f'Can not use module ‘{self.attribute}’ as value')
        return ErrorValue(self.model.module, location)


class SemanticError(SemanticSymbol):
    def __init__(self, model, location: Location):
        super(SemanticError, self).__init__(model, ErrorValue(model.module, location))

    def as_type(self, location: Location) -> Type:
        return cast(ErrorValue, self.symbol).type

    def as_value(self, location: Location) -> Value:
        return cast(ErrorValue, self.symbol)

    def subscript(self, arguments: Sequence[SemanticSymbol], location: Location) -> SemanticSymbol:
        return self


class SemanticType(SemanticSymbol):
    # TODO: SemanticType and SemanticValue must inherit from common parent type
    # TODO: SemanticFunction must inherit from SemanticValue

    @property
    def type(self) -> Type:
        return cast(Type, self.symbol)

    def as_type(self, location: Location) -> Type:
        return self.type

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f'Can not use type ‘{self.type}’ as value')
        return ErrorValue(self.model.module, location)

    def find_call(self, arguments: Arguments, keywords: Keywords, location: Location):
        arguments = [arg.as_value(location) for arg in arguments]
        keywords = {name: arg.as_value(location) for name, arg in keywords.values()}

        resolver = FunctionResolver(self.model, arguments, keywords, location=location)
        resolver.add_type_functions(self.type, '__new__')

        candidate = resolver.find_function()
        if candidate:
            return candidate

    def subscript(self, arguments: Sequence[SemanticValue], location: Location) -> SemanticSymbol:
        arguments = [arg.as_type(arg.location) for arg in arguments]
        result = self.type.instantiate(self.model.module, arguments, location)
        return SemanticType(self.model, result)


class SemanticValue(SemanticSymbol):
    @property
    def value(self) -> Value:
        return cast(Value, self.symbol)

    @property
    def type(self) -> Type:
        return self.value.type

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f'Can not use value ‘{self.value}’ as type')
        return ErrorType(self.model.module, location)

    def as_value(self, location: Location) -> Value:
        return self.value

    def find_attribute(self, name: str, location: Location) -> Optional[SemanticSymbol]:
        members = [member for member in self.type.members if member.name == name]
        if len(members) == 1:
            return self.model.as_symbol(members[0])
        elif not members:
            return None

        self.diagnostics.error(location, 'Not implemented attribute')
        return None


class SemanticFunction(SemanticSymbol):
    @property
    def function(self) -> Function:
        return cast(Function, self.symbol)

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f'Can not use function ‘{self.function}’ as type')
        return ErrorType(self.model.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f'Can not use function ‘{self.function}’ as value')
        return ErrorValue(self.model.module, location)

    def as_functions(self, location: Location) -> Sequence[Function]:
        return [self.function]


class SemanticOverload(SemanticSymbol):
    @property
    def overload(self) -> Overload:
        return cast(Overload, self.symbol)

    @property
    def functions(self) -> Sequence[Function]:
        return self.overload.functions

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f'Can not use function ‘{self.overload}’ as type')
        return ErrorType(self.model.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f'Can not use function ‘{self.overload}’ as value')
        return ErrorValue(self.model.module, location)

    def as_functions(self, location: Location) -> Sequence[Function]:
        return self.functions


class SemanticField(SemanticSymbol):
    @property
    def field(self) -> Field:
        return cast(Field, self.symbol)

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f'Can not use field ‘{self.field}’ as type')
        return ErrorType(self.model.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f'Can not use field ‘{self.field}’ as value')
        return ErrorValue(self.model.module, location)


class SemanticProperty(SemanticSymbol):
    @property
    def property(self) -> Property:
        return cast(Property, self.symbol)

    def as_type(self, location: Location) -> Type:
        self.diagnostics.error(location, f'Can not use property ‘{self.property}’ as type')
        return ErrorType(self.model.module, location)

    def as_value(self, location: Location) -> Value:
        self.diagnostics.error(location, f'Can not use property ‘{self.property}’ as value')
        return ErrorValue(self.model.module, location)


class SemanticScope(MutableMapping[SyntaxNode, SemanticSymbol]):
    def __init__(self, model: SemanticModel, *, parent: SemanticScope = None, constructor: SymbolConstructor = None):
        if not parent and not constructor:
            raise ValueError(u'Required any of this arguments: parent or constructor')
        self.__model = model
        self.__items = dict()
        self.__parent = parent
        self.__constructor = constructor

    def __len__(self) -> int:
        return len(self.__items)

    def __iter__(self) -> Iterator[SyntaxNode]:
        return iter(self.__items)

    def __getitem__(self, key: SyntaxNode) -> Optional[SemanticSymbol]:
        # TODO: Can we cache result of lookup in parent scope?
        try:
            return self.__items[key]
        except KeyError:
            # bypass construction, if key already existed in ancestors scopes
            if self.__parent is not None and key in self.__parent:
                return self.__parent[key]

            # if constructor is existed: construct new symbol for node
            if self.__constructor:
                value = self.__constructor(key)
                if value is not None:
                    self.__items[key] = value
                    self.__model.initialize(key)
                    return value

            # otherwise return it's from parent
            return self.__parent[key] if self.__parent else None

    def __setitem__(self, key: SyntaxNode, value: SemanticSymbol):
        self.__items[key] = value

    def __delitem__(self, key: SyntaxNode):
        raise NotImplementedError  # Not mutable

    def __contains__(self, key: SyntaxNode) -> bool:
        if key in self.__items:
            return True
        elif self.__parent is not None:
            return key in self.__parent
        return False


# symbol processing:
#   - annotate      : annotate symbol
#   - initialize    : prepare symbol, e.g. add generic parameters, execute prepare decorators
#   - finalize      : build symbol, e.g. inherit and instantiate members


class SemanticMixin:
    def __init__(self, model: SemanticModel):
        self.__model = model

    @property
    def symbol_context(self) -> SymbolContext:
        return self.model.symbol_context

    @property
    def model(self) -> SemanticModel:
        return self.__model

    @property
    def symbols(self) -> SemanticScope:
        return self.model.symbols

    @property
    def module(self) -> Module:
        return self.model.module

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.model.diagnostics

    def as_symbol(self, symbol: Symbol) -> SemanticSymbol:
        return self.model.as_symbol(symbol)


class SemanticInitializer(SemanticMixin):
    def initialize(self, node: SyntaxNode):
        if isinstance(node, TypeDeclarationNode):
            self.initialize_type(node)
        elif isinstance(node, FunctionNode):
            self.initialize_function(node)
        elif isinstance(node, SyntaxTree):
            self.initialize_module(node)

    def initialize_module(self, node: SyntaxTree):
        symbol = cast(SemanticModule, self.symbols[node])
        for child in node.members:
            member: Optional[SemanticSymbol] = self.symbols[child]
            if member:
                symbol.module.add_member(cast(OwnedSymbol, member.symbol))

    def initialize_type(self, node: TypeDeclarationNode):
        symbol = cast(SemanticType, self.symbols[node])

        for parent_node in node.parents:
            parent = self.symbols[parent_node].as_type(parent_node.location)
            if parent is symbol.type or symbol.type in parent.ascendants:
                # TODO: Can find all types involving in circular dependency
                self.diagnostics.error(parent_node.location, "Circular dependency")
            else:
                symbol.type.add_parent(parent)

        for generic in self.annotate_generics(node.generic_parameters):
            symbol.type.add_generic(generic)

        # insert attributes
        for attr in [self.symbols[decorator] for decorator in node.decorators]:
            symbol.type.add_attribute(cast(SemanticAttribute, attr).attribute)

        for child in node.members:
            member: Optional[SemanticSymbol] = self.symbols[child]
            if member:
                symbol.type.add_member(cast(OwnedSymbol, member.symbol))

        symbol.type.build()

    def initialize_function(self, node: FunctionNode):
        symbol = cast(SemanticFunction, self.symbols[node])

        for generic in self.annotate_generics(node.generic_parameters):
            symbol.function.add_generic(generic)

        # insert attributes
        for attr in [self.symbols[decorator] for decorator in node.decorators]:
            symbol.function.add_attribute(cast(SemanticAttribute, attr).attribute)

    def annotate_generics(self, parents: Sequence[GenericParameterNode]) -> Sequence[GenericType]:
        return [cast(GenericType, self.symbols[param].as_type(param.location)) for param in parents]


class SemanticAnnotator(SemanticMixin, abc.ABC):
    def __init__(self, model: SemanticModel, symbols: SemanticScope = None):
        super(SemanticAnnotator, self).__init__(model)

        self.__symbols = symbols

    @property
    def symbols(self) -> SemanticScope:
        return self.__symbols if self.__symbols is not None else self.model.symbols

    def as_type(self, node: SyntaxNode) -> Type:
        """ Convert node to type """
        symbol = self.symbols[node]
        if not symbol:
            return ErrorType(self.module, node.location)
        return symbol.as_type(node.location)

    def as_value(self, node: SyntaxNode) -> Value:
        """ Convert node to value """
        symbol = self.symbols[node]
        if not symbol:
            return ErrorValue(self.module, node.location)
        return symbol.as_value(node.location)

    def annotate(self, node: SyntaxNode) -> Optional[SemanticSymbol]:
        return None

    def search_named_symbol(self, scope: SyntaxScope, name: str, location: Location):
        if name == TYPE_VOID_NAME:
            return SemanticType(self.model, self.symbol_context.void_type)
        elif name == TYPE_BOOLEAN_NAME:
            return SemanticType(self.model, self.symbol_context.boolean_type)
        elif name == TYPE_INTEGER_NAME:
            return SemanticType(self.model, self.symbol_context.integer_type)
        elif name == TYPE_STRING_NAME:
            return SemanticType(self.model, self.symbol_context.string_type)
        elif name == VALUE_TRUE_NAME:
            return SemanticValue(self.model, BooleanConstant(self.symbol_context, True, location))
        elif name == VALUE_FALSE_NAME:
            return SemanticValue(self.model, BooleanConstant(self.symbol_context, False, location))
        elif name == VALUE_NONE_NAME:
            return SemanticValue(self.model, NoneConstant(self.symbol_context, location))

        # resolve static symbol
        symbol = scope.resolve(name)
        if symbol:
            return self.symbols[symbol]

        # resolve import symbol
        symbol = self.model.imports.get(name)
        if symbol:
            return self.as_symbol(symbol)
        return None


class ExpressionAnnotator(SemanticAnnotator):
    def annotate(self, node: SyntaxNode) -> Optional[SemanticSymbol]:
        if isinstance(node, ExpressionNode):
            return self.annotate_expression(node)
        return None

    def annotate_expression(self, node: ExpressionNode) -> SemanticSymbol:
        if isinstance(node, IntegerExpressionNode):
            return self.annotate_integer_expression(node)
        elif isinstance(node, StringExpressionNode):
            return self.annotate_string_expression(node)
        elif isinstance(node, NamedExpressionNode):
            return self.annotate_named_expression(node)
        elif isinstance(node, AttributeExpressionNode):
            return self.annotate_attribute_expression(node)
        elif isinstance(node, ConditionExpressionNode):
            return self.annotate_condition_expression(node)
        elif isinstance(node, UnaryExpressionNode):
            return self.annotate_unary_expression(node)
        elif isinstance(node, BinaryExpressionNode):
            return self.annotate_binary_expression(node)
        elif isinstance(node, LogicExpressionNode):
            return self.annotate_logic_expression(node)
        elif isinstance(node, CompareExpressionNode):
            return self.annotate_compare_expression(node)
        elif isinstance(node, CallExpressionNode):
            return self.annotate_call_expression(node)
        elif isinstance(node, SubscribeExpressionNode):
            return self.annotate_subscribe_expression(node)

        self.diagnostics.error(node.location, 'Not implemented expression')
        return SemanticError(self.model, node.location)

    def annotate_integer_expression(self, node: IntegerExpressionNode) -> SemanticSymbol:
        return SemanticValue(self.model, IntegerConstant(self.symbol_context, int(node.value), node.location))

    def annotate_string_expression(self, node: StringExpressionNode) -> SemanticSymbol:
        return SemanticValue(self.model, StringConstant(self.symbol_context, node.value[1:-1], node.location))

    def annotate_named_expression(self, node: NamedExpressionNode) -> SemanticSymbol:
        symbol = self.search_named_symbol(node.scope, node.name, node.location)
        if not symbol:
            self.diagnostics.error(node.location, f'Name ‘{node.name}’ not found in current scope')
            return SemanticError(self.model, node.location)
        return symbol

    def annotate_attribute_expression(self, node: AttributeExpressionNode) -> SemanticSymbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return SemanticError(self.model, node.location)

    def annotate_condition_expression(self, node: ConditionExpressionNode) -> SemanticSymbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return SemanticError(self.model, node.location)

    def annotate_unary_expression(self, node: UnaryExpressionNode) -> SemanticSymbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return SemanticError(self.model, node.location)

    def annotate_binary_expression(self, node: BinaryExpressionNode) -> SemanticSymbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return SemanticError(self.model, node.location)

    def annotate_logic_expression(self, node: LogicExpressionNode) -> SemanticSymbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return SemanticError(self.model, node.location)

    def annotate_compare_expression(self, node: CompareExpressionNode) -> SemanticSymbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return SemanticError(self.model, node.location)

    def annotate_call_expression(self, node: CallExpressionNode) -> SemanticSymbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return SemanticError(self.model, node.location)

    def annotate_subscribe_expression(self, node: SubscribeExpressionNode) -> SemanticSymbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return SemanticError(self.model, node.location)


class StaticAnnotator(ExpressionAnnotator):
    def annotate(self, node: SyntaxNode) -> Optional[SemanticSymbol]:
        if isinstance(node, SyntaxTree):
            return self.annotate_module(node)
        elif isinstance(node, TypeNode):
            return self.annotate_type(node)
        elif isinstance(node, MemberNode):
            return self.annotate_member(node)
        elif isinstance(node, ExpressionNode):
            return self.annotate_expression(node)
        elif isinstance(node, OverloadNode):
            return self.annotate_overload(node)
        elif isinstance(node, DecoratorNode):
            return self.annotate_decorator(node)

        self.diagnostics.error(node.location, 'Not implemented member')
        return None

    def annotate_module(self, node: SyntaxTree) -> SemanticModule:
        module = Module(self.symbol_context, self.model.name, node.location)
        return SemanticModule(self.model, module)

    def annotate_decorator(self, node: DecoratorNode) -> SemanticAttribute:
        arguments = cast(Sequence[Value], [self.as_value(arg) for arg in node.arguments])
        return SemanticAttribute(self.model, Attribute(self.module, node.name, arguments, node.location))

    def annotate_type(self, node: TypeNode) -> SemanticSymbol:
        if isinstance(node, NamedTypeNode):
            return self.annotate_named_type(node)
        elif isinstance(node, ParameterizedTypeNode):
            return self.annotate_parameterized_type(node)
        elif isinstance(node, GenericParameterNode):
            return self.annotate_generic_type(node)

        self.diagnostics.error(node.location, 'Not implemented type')
        return SemanticError(self.model, node.location)

    def annotate_generic_type(self, node: GenericParameterNode) -> SemanticType:
        return SemanticType(self.model, GenericType(self.model.module, node.name, node.location))

    def annotate_named_type(self, node: NamedTypeNode) -> SemanticSymbol:
        symbol = self.search_named_symbol(node.scope, node.name, node.location)
        if not symbol:
            self.diagnostics.error(node.location, f'Type ‘{node.name}’ not found in current scope')
            return SemanticError(self.model, node.location)
        return symbol

    def annotate_parameterized_type(self, node: ParameterizedTypeNode) -> SemanticSymbol:
        generic = self.symbols[node.type]
        arguments = [self.symbols[arg] for arg in node.arguments]
        return generic.subscript(arguments, node.location)

    def annotate_member(self, node: MemberNode) -> Optional[SemanticSymbol]:
        parent: SemanticSymbol = self.symbols[node.parent]
        if node in self.symbols:
            return self.symbols[node]

        if isinstance(parent.symbol, Module):
            return ModuleMemberAnnotator(self.model, parent.symbol).annotate_member(node)
        elif isinstance(parent.symbol, ClassType):
            return ClassMemberAnnotator(self.model, parent.symbol).annotate_member(node)
        elif isinstance(parent.symbol, StructType):
            return StructMemberAnnotator(self.model, parent.symbol).annotate_member(node)
        elif isinstance(parent.symbol, InterfaceType):
            return InterfaceMemberAnnotator(self.model, parent.symbol).annotate_member(node)
        elif isinstance(parent.symbol, EnumType):
            return EnumMemberAnnotator(self.model, parent.symbol).annotate_member(node)
        elif isinstance(parent.symbol, Type):
            return TypeMemberAnnotator(self.model, parent.symbol).annotate_member(node)

        self.diagnostics.error(parent.location, 'Not implemented parent')
        return None

    def annotate_overload(self, node: OverloadNode) -> SemanticOverload:
        functions = [self.symbols[func].function for func in node.functions]
        overload = Overload(self.module, node.name, cast(Sequence[Function], functions), node.location)
        return SemanticOverload(self.model, overload)


class ParentMemberAnnotator(SemanticAnnotator):
    def __init__(self, model: SemanticModel, parent: ContainerSymbol):
        super().__init__(model)

        self.__parent = parent

    @property
    def parent(self) -> ContainerSymbol:
        return self.__parent

    def annotate_member(self, node: MemberNode) -> Optional[SemanticSymbol]:
        # symbols that don't have a parent
        if isinstance(node, FunctionNode):
            return self.annotate_function(node)
        elif isinstance(node, ClassNode):
            return self.annotate_class(node)
        elif isinstance(node, StructNode):
            return self.annotate_struct(node)
        elif isinstance(node, InterfaceNode):
            return self.annotate_interface(node)
        elif isinstance(node, EnumNode):
            return self.annotate_enum(node)
        elif isinstance(node, FieldNode):
            return self.annotate_field(node)
        elif isinstance(node, EnumMemberNode):
            return self.annotate_enum_value(node)
        elif isinstance(node, PassMemberNode):
            return None

    def annotate_function(self, node: FunctionNode) -> SemanticSymbol:
        # if function is method of struct/class/interface and first arguments type is auto, then it type can
        # be inferred to owner type
        if isinstance(self.parent, Type) and node.parameters and isinstance(node.parameters[0].type, AutoTypeNode):
            parameters = [self.parent]
            parameters.extend(self.symbols[param.type].as_type(param.location) for param in node.parameters[1:])
        else:
            parameters = [self.symbols[param.type].as_type(param.location) for param in node.parameters]

        # if return type of function is auto it can be inferred to void
        if isinstance(node.return_type, AutoTypeNode):
            return_type = self.symbol_context.void_type
        else:
            return_type = self.symbols[node.return_type].as_type(node.return_type.location)

        # create function symbol
        func_type = FunctionType(self.module, parameters, return_type, node.location)
        func = Function(self.parent, node.name, func_type, node.location)

        for node_param, func_param in zip(node.parameters, func.parameters):
            func_param.name = node_param.name
            func_param.location = node_param.location

            self.symbols[node_param] = SemanticValue(self.model, func_param)

        # assert node not in self.symbols
        return SemanticFunction(self.model, func)

    def annotate_class(self, node: ClassNode) -> SemanticType:
        if self.module == self.symbol_context.builtins_module:
            if node.name == TYPE_STRING_NAME:
                return SemanticType(self.model, StringType(self.parent, node.location))

        return SemanticType(self.model, ClassType(self.parent, node.name, node.location))

    def annotate_struct(self, node: StructNode) -> SemanticType:
        if self.module == self.symbol_context.builtins_module:
            if node.name == TYPE_INTEGER_NAME:
                return SemanticType(self.model, IntegerType(self.parent, node.location))
            elif node.name == TYPE_BOOLEAN_NAME:
                return SemanticType(self.model, BooleanType(self.parent, node.location))
            elif node.name == TYPE_VOID_NAME:
                return SemanticType(self.model, VoidType(self.parent, node.location))

        return SemanticType(self.model, StructType(self.parent, node.name, node.location))

    def annotate_enum(self, node: EnumNode) -> SemanticType:
        if node.generic_parameters:
            self.diagnostics.error(node.location, 'Generic types is not implemented')
        return SemanticType(self.model, EnumType(self.parent, node.name, node.location))

    def annotate_interface(self, node: InterfaceNode) -> SemanticType:
        return SemanticType(self.model, InterfaceType(self.parent, node.name, node.location))

    def annotate_field(self, node: FieldNode) -> Optional[SemanticSymbol]:
        self.diagnostics.error(node.location, 'Not implemented in current context')
        return None

    def annotate_enum_value(self, node: EnumMemberNode) -> Optional[SemanticSymbol]:
        self.diagnostics.error(node.location, 'Not implemented in current context')
        return None


class ModuleMemberAnnotator(ParentMemberAnnotator):
    @property
    def parent(self) -> Module:
        return cast(Module, super().parent)


class TypeMemberAnnotator(ParentMemberAnnotator):
    @property
    def parent(self) -> Type:
        return cast(Type, super().parent)

    def create_property_field(self, node: FieldNode) -> Field:
        field_type = self.symbols[node.type].as_type(node.type.location)
        field = Field(self.parent, node.name, field_type, node.location)

        self.parent.add_member(field)
        return field

    def create_property_getter(self, node: FieldNode, field: Field = None) -> Function:
        prop_type = self.symbols[node.type].as_type(node.type.location)
        getter_type = FunctionType(self.module, [self.parent], prop_type, node.location)
        getter_func = Function(self.parent, node.name, getter_type, node.location)
        getter_func.parameters[0].name = 'self'

        self.parent.add_member(getter_func)
        return getter_func

    def create_property_setter(self, node: FieldNode, field: Field = None) -> Function:
        prop_type = self.symbols[node.type].as_type(node.type.location)
        setter_type = FunctionType(self.module, [self.parent, prop_type], self.symbol_context.void_type, node.location)
        setter_func = Function(self.parent, node.name, setter_type, node.location)
        setter_func.parameters[0].name = 'self'
        setter_func.parameters[1].name = 'value'

        self.parent.add_member(setter_func)
        return setter_func


class StructMemberAnnotator(TypeMemberAnnotator):
    @property
    def parent(self) -> StructType:
        return cast(StructType, super().parent)

    def annotate_field(self, node: FieldNode) -> Optional[SemanticSymbol]:
        field = self.create_property_field(node)
        getter_func = self.create_property_getter(node, field)
        setter_func = self.create_property_setter(node, field)
        prop = Property(self.parent, node.name, getter_func, setter_func, location=node.location)
        return SemanticProperty(self.model, prop)


class ClassMemberAnnotator(TypeMemberAnnotator):
    @property
    def parent(self) -> ClassType:
        return cast(ClassType, super().parent)

    def annotate_field(self, node: FieldNode) -> Optional[SemanticSymbol]:
        field = self.create_property_field(node)
        getter_func = self.create_property_getter(node, field)
        setter_func = self.create_property_setter(node, field)
        prop = Property(self.parent, node.name, getter_func, setter_func, location=node.location)
        return SemanticProperty(self.model, prop)


class InterfaceMemberAnnotator(TypeMemberAnnotator):
    @property
    def parent(self) -> InterfaceType:
        return cast(InterfaceType, super().parent)

    def annotate_field(self, node: FieldNode) -> Optional[SemanticSymbol]:
        getter_func = self.create_property_getter(node)
        setter_func = self.create_property_setter(node)
        prop = Property(self.parent, node.name, getter_func, setter_func, location=node.location)
        return SemanticProperty(self.model, prop)


class EnumMemberAnnotator(TypeMemberAnnotator):
    @property
    def parent(self) -> EnumType:
        return cast(EnumType, super().parent)

    def annotate_enum_value(self, node: EnumMemberNode) -> Optional[SemanticSymbol]:
        value = node.parent.members.index(node)
        if node.value:
            expr = self.symbols[node.value]
            if isinstance(expr, IntegerConstant):
                value = expr.value
            else:
                self.diagnostics.error(node.location, "Enumeration values must have integer value or ‘...’")

        return SemanticValue(self.model, EnumValue(self.parent, node.name, value, node.location))


class FunctionResolver(SemanticMixin):
    def __init__(self, model: SemanticModel, arguments: Sequence[Value] = None, keywords: Mapping[str, Value] = None,
                 *, location: Location):
        super(FunctionResolver, self).__init__(model)

        self.location = location

        self.arguments = arguments or []
        self.keywords = keywords or {}
        self.candidates = []

    def add_scope_functions(self, scope: SyntaxScope, name: str):
        node = scope.resolve(name)
        if isinstance(node, (FunctionNode, OverloadNode)):
            self.candidates.extend(self.symbols[node].as_functions(self.location))

    def add_type_functions(self, type: Type, name: str):
        functions = (member for member in type.members if isinstance(member, Function) and member.name == name)
        self.candidates.extend(functions)

    def add_self_functions(self, name: str):
        if self.arguments:
            self.add_type_functions(self.arguments[0].type, name)

    def add_import_functions(self, name: str):
        symbol = self.model.imports.get(name, None)
        if isinstance(symbol, Function):
            self.candidates.append(symbol)
        elif isinstance(symbol, Overload):
            self.candidates.extend(child for child in symbol.functions)

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


class FunctionAnnotator(ExpressionAnnotator):
    def __init__(self, model: SemanticModel, symbols: SemanticScope[SyntaxNode, SemanticSymbol], node: FunctionNode):
        super().__init__(model, SemanticScope(model, parent=symbols, constructor=lambda n: self.annotate(n)))

        self.__node = node
        self.__builder = IRBuilder(self.function.append_basic_block('entry'))

    @property
    def node(self) -> FunctionNode:
        return self.__node

    @cached_property
    def function(self) -> Function:
        return cast(SemanticFunction, self.symbols[self.node]).function

    @property
    def block(self) -> BasicBlock:
        return self.builder.block

    @property
    def builder(self) -> IRBuilder:
        return self.__builder

    def emit_return(self, value: Value, location: Location):
        return_type = self.function.return_type
        value_type = value.type if value else self.symbol_context.void_type

        if isinstance(return_type, ErrorType) or isinstance(value_type, ErrorType):
            pass  # Skip error propagation
        elif value and value_type != return_type:
            message = f"Function must return value of ‘{return_type}’ type, but got ‘{value_type}’"
            self.diagnostics.error(location, message)

        self.builder.ret(value, location=location)

    def annotate_statement(self, node: StatementNode) -> bool:
        if isinstance(node, BlockStatementNode):
            return self.annotate_block_statement(node)
        elif isinstance(node, PassStatementNode):
            return self.annotate_pass_statement(node)
        elif isinstance(node, ReturnStatementNode):
            return self.annotate_return_statement(node)
        elif isinstance(node, ConditionStatementNode):
            return self.annotate_condition_statement(node)
        elif isinstance(node, WhileStatementNode):
            return self.annotate_while_statement(node)
        elif isinstance(node, ExpressionStatementNode):
            return self.annotate_expression_statement(node)
        elif isinstance(node, AssignStatementNode):
            return self.annotate_assign_statement(node)
        elif isinstance(node, BreakStatementNode):
            return self.annotate_break_statement(node)
        elif isinstance(node, ContinueStatementNode):
            return self.annotate_continue_statement(node)
        else:
            self.diagnostics.error(node.location, 'Not implemented statement')
            return False

    def annotate_block_statement(self, node: BlockStatementNode) -> bool:
        for statement in node.statements:
            if self.annotate_statement(statement):
                return True
        return False

    def annotate_pass_statement(self, _: PassStatementNode) -> bool:
        return False

    def annotate_return_statement(self, node: ReturnStatementNode) -> bool:
        location = node.location

        if node.value:
            value = self.as_value(node.value)
            location = node.value.location
        else:
            value = NoneConstant(self.symbol_context, location)

        self.emit_return(value, location)
        return True

    # def emit_condition_statement(self, node: ConditionStatementNode) -> bool:
    #     condition = self.emit_condition(node.scope, self.emit_expression(node.condition), node.condition.location)
    #     begin_block = self.block
    #
    #     # then branch
    #     then_block = self.builder.append_basic_block('if.then', node.then_statement.location)
    #     self.builder.position_at(then_block)
    #     then_terminated = self.emit_statement(node.then_statement)
    #     then_finish = self.block
    #
    #     # else branch
    #     if node.else_statement:
    #         else_block = self.builder.append_basic_block('if.else', node.else_statement.location)
    #         self.builder.position_at(else_block)
    #         else_terminated = self.emit_statement(node.else_statement)
    #         else_finish = self.block
    #     else:
    #         else_block = None
    #         else_terminated = False
    #         else_finish = None
    #
    #     # next branch
    #     next_block = self.builder.append_basic_block('if.next', node.location)
    #
    #     # condition branch
    #     self.builder.position_at(begin_block)
    #     self.builder.cbranch(condition, then_block, else_block or next_block, location=node.location)
    #
    #     # terminate then branch
    #     if not then_terminated:
    #         self.builder.position_at(then_finish)
    #         self.builder.branch(next_block, location=node.location)
    #
    #     # terminate else branch
    #     if not else_terminated and else_finish:
    #         self.builder.position_at(else_finish)
    #         self.builder.branch(next_block, location=node.location)
    #
    #     # rest
    #     self.builder.position_at(next_block)
    #     return then_terminated and else_terminated
    #
    # def emit_while_statement(self, node: WhileStatementNode) -> bool:
    #     cond_block = self.builder.append_basic_block('while.cond', location=node.location)
    #     self.builder.branch(cond_block, location=node.location)
    #     self.builder.position_at(cond_block)
    #     condition = self.emit_condition(node.scope, self.emit_expression(node.condition), node.condition.location)
    #     cond_end_block = self.builder.block
    #
    #     # next branch
    #     next_block = self.builder.append_basic_block('while.next', node.location)
    #
    #     # then branch
    #     then_block = self.builder.append_basic_block('while.then', node.location)
    #     self.builder.position_at(then_block)
    #     with self.emit_loop(next_block, cond_block) as loop_info:
    #         self.emit_statement(node.then_statement)
    #         loop_breaked = loop_info.is_breaked
    #     then_finish = self.block
    #
    #     # else branch
    #     if node.else_statement:
    #         else_block = self.builder.append_basic_block('while.else', node.location)
    #         self.builder.position_at(else_block)
    #         else_terminated = self.emit_statement(node.else_statement)
    #         if not else_terminated:
    #             self.builder.branch(next_block, location=node.location)
    #     else:
    #         else_block = None
    #         else_terminated = False
    #
    #     # condition branch
    #     self.builder.position_at(cond_end_block)
    #     self.builder.cbranch(condition, then_block, else_block or next_block, location=node.location)
    #
    #     # terminate then branch
    #     if not then_finish.is_terminated:
    #         self.builder.position_at(then_finish)
    #         self.builder.branch(cond_block, location=node.location)
    #
    #     self.builder.position_at(next_block or else_block)
    #     if else_terminated and not loop_breaked:
    #         self.function.remove_basic_block(then_block)
    #     else:
    #         self.function.move_basic_block(len(self.function.blocks), then_block)
    #     return else_terminated
    #
    def annotate_assign_statement(self, node: AssignStatementNode) -> bool:
        source = self.as_value(node.source)

        annotator = AssignAnnotator(self, source, node.location)
        annotator.annotate_assign(node.target)

        return False

    # def emit_continue_statement(self, node: ContinueStatementNode) -> bool:
    #     if not self.loop:
    #         self.diagnostics.error(node.location, f'Not found loop for continue')
    #     else:
    #         self.builder.branch(self.loop.continue_block, location=node.location)
    #     return True
    #
    # def emit_break_statement(self, node: BreakStatementNode) -> bool:
    #     if not self.loop:
    #         self.diagnostics.error(node.location, f'Not found loop for break')
    #     else:
    #         self.is_break = True
    #         self.builder.branch(self.loop.break_block, location=node.location)
    #     return True

    def annotate_expression_statement(self, node: ExpressionStatementNode) -> bool:
        self.symbols.get(node.value)
        return False

    # def emit_condition_expression(self, node: ConditionExpressionNode) -> Value:
    #     condition = self.emit_condition(node.scope, self.emit_expression(node.condition), node.condition.location)
    #     begin_block = self.builder.block
    #
    #     # then branch
    #     then_block = self.builder.append_basic_block('if.then', node.location)
    #     self.builder.position_at(then_block)
    #     then_value = self.emit_expression(node.then_value)
    #
    #     # else branch
    #     else_block = self.builder.append_basic_block('if.else', node.location)
    #     self.builder.position_at(then_block)
    #     else_value = self.emit_expression(node.else_value)
    #
    #     # condition branch
    #     self.builder.position_at(begin_block)
    #     result_type = self.find_union(then_value.type, else_value.type, location=node.location)
    #     result = self.builder.alloca(result_type, location=node.location)
    #     self.builder.cbranch(condition, then_block, else_block, location=node.location)
    #
    #     # next branch
    #     next_block = self.builder.append_basic_block('if.next', node.location)
    #     for block, value, location in [
    #         (then_block, then_value, node.then_value.location),
    #         (else_block, else_value, node.else_value.location)
    #     ]:
    #         self.builder.position_at(block)
    #         self.builder.store(result, value, location=location)
    #         self.builder.branch(next_block, location=location)
    #
    #     # rest
    #     self.builder.position_at(next_block)
    #     return self.builder.load(result, location=node.location)
    #
    # def emit_unary_expression(self, node: UnaryExpressionNode):
    #     if node.operator == UnaryID.Not:
    #         condition = self.emit_condition(node.scope, self.emit_expression(node.operand), node.operand.location)
    #         cst_check = BooleanConstant(self.symbol_context, False, node.location)
    #         return self.builder.is_(condition, cst_check, location=node.location)
    #
    #     name = '__{}__'.format(node.operator.name).lower()
    #     arguments = [self.emit_expression(node.operand)]
    #     return self.emit_named_call(node.scope, name, arguments, location=node.location)
    #
    # def emit_binary_expression(self, node: BinaryExpressionNode):
    #     name = '__{}__'.format(node.operator.name).lower()
    #     arguments = [self.emit_expression(node.left_operand), self.emit_expression(node.right_operand)]
    #     return self.emit_named_call(node.scope, name, arguments, location=node.location)
    #
    # def emit_logic_expression(self, node: LogicExpressionNode) -> Value:
    #     # emit first value
    #     first_value = self.emit_expression(node.left_operand)
    #     condition = self.emit_condition(node.scope, first_value, node.left_operand.location)
    #     begin_block = self.builder.block
    #
    #     # then branch
    #     first_block = self.builder.append_basic_block('cond.then', node.location)
    #
    #     # else branch
    #     second_block = self.builder.append_basic_block('cond.else', node.location)
    #     self.builder.position_at(second_block)
    #     second_value = self.emit_expression(node.right_operand)
    #     second_end = self.block
    #
    #     # condition branch
    #     self.builder.position_at(begin_block)
    #     result_type = self.find_union(first_value.type, second_value.type, location=node.location)
    #     result = self.builder.alloca(result_type, location=node.location)
    #
    #     if node.operator == LogicID.And:
    #         cst_check = BooleanConstant(self.symbol_context, False, node.location)
    #     else:
    #         cst_check = BooleanConstant(self.symbol_context, True, node.location)
    #     check = self.builder.is_(condition, cst_check, location=node.location)
    #     self.builder.cbranch(check, first_block, second_block, location=node.location)
    #
    #     # next branch
    #     next_block = self.builder.append_basic_block('cond.next', node.location)
    #     for block, value, location in [
    #         (first_block, first_value, node.left_operand.location),
    #         (second_end, second_value, node.right_operand.location)
    #     ]:
    #         self.builder.position_at(block)
    #         self.builder.store(result, value, location=location)
    #         self.builder.branch(next_block, location=location)
    #
    #     # rest
    #     self.builder.position_at(next_block)
    #     return self.builder.load(result, location=node.location)
    #
    # def emit_compare_expression(self, node: CompareExpressionNode) -> Value:
    #     result = self.builder.alloca(self.symbol_context.boolean_type, location=node.location)
    #     self.builder.store(result, BooleanConstant(self.symbol_context, False, node.location), location=node.location)
    #
    #     blocks = []
    #
    #     left_operand = self.emit_expression(node.left_operand)
    #     for idx, comparator in enumerate(node.comparators):  # type: ComparatorNode
    #         right_operand = self.emit_expression(comparator.right_operand)
    #
    #         next_block = self.builder.append_basic_block('cmp.{}.next'.format(idx), location=node.location)
    #
    #         name = '__{}__'.format(comparator.operator.name).lower()
    #         cmp_result = self.emit_named_call(node.scope, name, [left_operand, right_operand], location=node.location)
    #         blocks.append([cmp_result, self.builder.block, next_block])
    #
    #         self.builder.position_at(next_block)
    #         left_operand = right_operand
    #
    #     # true block
    #     self.builder.store(result, BooleanConstant(self.symbol_context, True, node.location), location=node.location)
    #
    #     false_block = self.builder.append_basic_block('cmp.end', location=node.location)
    #     self.builder.branch(false_block, location=node.location)
    #
    #     for cmp_result, current_block, next_block in blocks:
    #         self.builder.position_at(current_block)
    #         self.builder.cbranch(cmp_result, next_block, false_block, location=cmp_result.location)
    #
    #     self.builder.position_at(false_block)
    #     return self.builder.load(result, location=node.location)

    def annotate_call_expression(self, node: CallExpressionNode):
        arguments = [self.symbols[arg] for arg in node.arguments]
        keywords = {name: self.symbols[arg] for name, arg in node.keywords.items()}

        annotator = CallAnnotator(self, arguments, keywords, node.location)
        return annotator.annotate_call(node.instance)

    def annotate_subscribe_expression(self, node: SubscribeExpressionNode):
        arguments = [self.symbols[arg] for arg in node.arguments]
        instance = self.symbols[node.instance]
        return instance.subscript(arguments, node.location)


class FunctionMixin(SemanticAnnotator):
    def __init__(self, parent: FunctionAnnotator):
        super(FunctionMixin, self).__init__(parent.model, parent.symbols)

        self.parent = parent

    @property
    def function(self) -> Function:
        return self.parent.function

    @property
    def block(self) -> BasicBlock:
        return self.parent.block

    @property
    def builder(self) -> IRBuilder:
        return self.parent.builder


class CallAnnotator(FunctionMixin):
    def __init__(self, parent: FunctionAnnotator, arguments: Arguments, keywords: Keywords, location: Location):
        super(CallAnnotator, self).__init__(parent)

        self.arguments = arguments
        self.keywords = keywords
        self.location = location

    def get_arguments(self) -> str:
        arguments = [str(arg) for arg in self.arguments]
        arguments.extend(f'{name}: {arg}' for name, arg in self.keywords.items())
        return ', '.join(arguments)

    def resolve_named_call(self, scope: SyntaxScope, name: str) -> Optional[Tuple[Function, Sequence[Value]]]:
        arguments = [arg.as_value(self.location) for arg in self.arguments]
        keywords = {name: arg.as_value(self.location) for name, arg in self.keywords.values()}

        resolver = FunctionResolver(self.model, arguments, keywords, location=self.location)
        resolver.add_import_functions(name)
        resolver.add_scope_functions(scope, name)
        resolver.add_self_functions(name)
        return resolver.find_function()

    def emit_symbol_call(self, symbol: SemanticSymbol) -> SemanticSymbol:
        candidate = symbol.find_call(self.arguments, self.keywords, self.location)
        if candidate:
            func, arguments = candidate
            return SemanticValue(self.model, self.builder.call(func, arguments, location=self.location))

        arguments = self.get_arguments()
        self.diagnostics.error(self.location, 'Not found function for call ‘{}({})’'.format(symbol, arguments))
        return SemanticError(self.model, self.location)

    def emit_named_call(self, node: NamedExpressionNode) -> SemanticSymbol:
        candidate = self.resolve_named_call(node.scope, node.name)
        if candidate:
            func, arguments = candidate
            return SemanticValue(self.model, self.builder.call(func, arguments, location=self.location))

        arguments = self.get_arguments()
        self.diagnostics.error(self.location, 'Not found function for call ‘{}({})’'.format(node.name, arguments))
        return SemanticError(self.model, self.location)

    def annotate_call(self, node: ExpressionNode) -> SemanticSymbol:
        if isinstance(node, NamedExpressionNode):
            return self.annotate_named_expression(node)
        elif isinstance(node, AttributeExpressionNode):
            return self.annotate_attribute_expression(node)
        return self.annotate_another_expression(node)

    def annotate_named_expression(self, node: NamedExpressionNode) -> SemanticSymbol:
        instance = self.search_named_symbol(node.scope, node.name, node.location)

        # UFC:
        #   - <function>(...)   -> call function
        #   - <overload>(...)   -> call function
        #   - <none>(...)       -> call function
        if isinstance(instance, (SemanticFunction, SemanticOverload, type(None))):
            return self.emit_named_call(node)

        # Otherwise:
        #   - <type>(...)       -> call constructor
        #   - <another>(...)    -> call value
        return self.emit_symbol_call(instance)

    def annotate_attribute_expression(self, node: AttributeExpressionNode) -> SemanticSymbol:
        instance = self.symbols[node.instance]
        attribute = instance.find_attribute(node.name, node.location)

        # UFC:
        #   - instance.<function>(...)   -> call function
        #   - instance.<overload>(...)   -> call function
        #   - instance.<none>(...)       -> call function
        if isinstance(attribute, (SemanticFunction, SemanticOverload, type(None))):
            cast(list, self.arguments).insert(0, instance)
            return self.emit_named_call(node)

        # instance.name(...)
        self.diagnostics.error(node.location, 'NOT IMPLEMENTED!')
        return SemanticError(self.model, node.location)

    def annotate_another_expression(self, node: ExpressionNode) -> SemanticSymbol:
        return self.emit_symbol_call(self.symbols[node])


class AssignAnnotator(FunctionMixin):
    def __init__(self, parent: FunctionAnnotator, source: Value, location: Location):
        super(AssignAnnotator, self).__init__(parent)

        self.source = source
        self.location = location

    def annotate_assign(self, node: SyntaxNode):
        if isinstance(node, NamedExpressionNode):
            return self.annotate_named_expression(node)
        elif isinstance(node, AttributeExpressionNode):
            return self.annotate_attribute_expression(node)

        self.diagnostics.error(node.location, 'Not implemented statement')

    def annotate_named_expression(self, node: NamedExpressionNode):
        symbol = self.search_named_symbol(node.scope, node.name, node.location)
        if symbol is None:
            symbol = self.builder.alloca(self.source.type, location=node.location)
            self.symbols[node] = self.as_symbol(symbol)
            node.scope.declare(node.name, node)

        target = self.as_value(node)
        if isinstance(target, AllocaInstruction):
            self.builder.store(target, self.source, location=node.location)
        else:
            self.diagnostics.error(node.location, 'Can not assign value')
        return False

    # def annotate_attribute_expression(self, node: AttributeExpressionNode):
    #     pass

    #
    # elif isinstance(node.target, AttributeExpressionNode):
    #     instance = self.emit_symbol(node.target.instance)
    #     if isinstance(instance, Value):
    #         if isinstance(instance.type, (ClassType, StructType)):
    #             field = next((field for field in instance.type.fields if field.name == node.target.name), None)
    #             if field:
    #                 source = self.emit_cast(source, field.type, location=node.source.location)
    #                 self.builder.insert_value(instance, field, source, location=node.location)
    #                 return False
    #
    #             self.diagnostics.error(
    #                 node.target.location, f'Field ‘{node.target.name}’ not found in type ‘{instance.type}’')
    #         else:
    #             self.diagnostics.error(
    #                 node.target.location, 'Can change field only for instance of structure or class')
    #         return False
    #
    # self.diagnostics.error(node.location, 'Not implemented statement')