# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import heapq
import itertools
import logging
from typing import Deque
from typing import MutableMapping, Tuple

from orcinus.exceptions import OrcinusError
from orcinus.flow import FlowGraph, FlowBuilder
from orcinus.symbols import *
from orcinus.syntax import *
from orcinus.utils import cached_property

logger = logging.getLogger('orcinus')

Arguments = Sequence['SemanticSymbol']
Keywords = Mapping[str, 'SemanticSymbol']
SymbolConstructor = Callable[[SyntaxNode], 'SemanticSymbol']

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

    def __init__(self, context: SemanticContext, tree: SyntaxTree):
        self.context = context
        self.tree = tree
        self.imports = {}
        self.queue = Deque[SyntaxNode]()
        self.is_analyzed = False

        annotator = StaticAnnotator(self)
        self.symbols = SemanticMapping(self, constructor=lambda n: annotator.annotate(n))
        self.environment = GlobalEnvironment()

    @property
    def symbol_context(self):
        return self.context.symbol_context

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.context.diagnostics

    @cached_property
    def module(self) -> Module:
        return cast(Module, self.symbols[self.tree])

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
        if self.module is not self.symbol_context.builtins_module:
            self.import_all(self.symbol_context.builtins_module)

        for node in self.tree.imports:
            if isinstance(node, ImportFromNode):
                # Attempt to open module
                try:
                    module = self.context.open(node.module).module
                except OrcinusError as ex:
                    self.diagnostics.error(node.location, str(ex))
                    continue

                # Load aliases from nodes
                for alias in node.aliases:  # type: AliasNode
                    if not self.import_symbol(module, alias.name, alias.alias):
                        self.diagnostics.error(node.location, f'Cannot import name ‘{alias.name}’ from ‘{module.name}’')
            else:
                self.diagnostics.error(node.location, 'Not implemented member')

    def emit_function(self, node: FunctionNode):
        emitter = FunctionEmitter(self, self.symbols, node)
        function = emitter.emit()
        # print(function.as_blocks())

    def analyze(self):
        if self.is_analyzed:
            return
        self.is_analyzed = True

        self.import_symbols()

        self.queue.append(self.tree)
        self.symbols.get(self.tree)
        while self.queue:
            child = self.queue.popleft()
            self.initialize(child)

        functions: Sequence[FunctionNode] = list(
            self.tree.find_descendants(lambda node: isinstance(node, FunctionNode))
        )
        for func in functions:
            if not func.is_abstract:
                self.emit_function(func)

    def __repr__(self):
        return f'<SemanticModel: {self.module}>'


class SemanticMapping(MutableMapping[SyntaxNode, Symbol]):
    def __init__(self, model: SemanticModel, *, parent: SemanticMapping = None, constructor: SymbolConstructor = None):
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

    def __getitem__(self, key: SyntaxNode) -> Optional[Symbol]:
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
                    self.__model.queue.append(key)
                    return value

            # otherwise return it's from parent
            return self.__parent[key] if self.__parent else None

    def __setitem__(self, key: SyntaxNode, value: Symbol):
        self.__items[key] = value

    def __delitem__(self, key: SyntaxNode):
        raise NotImplementedError  # Not mutable

    def __contains__(self, key: SyntaxNode) -> bool:
        if key in self.__items:
            return True
        elif self.__parent is not None:
            return key in self.__parent
        return False


class SemanticEnvironment(abc.ABC):
    """
    Environment stored information about declared symbols and their types in current environment.

    Environment changed
    """

    @property
    @abc.abstractmethod
    def scope(self) -> SemanticScope:
        """ Current scope """
        raise NotImplementedError

    def define(self, name: str, symbol: Symbol):
        return self.scope.define(name, symbol)

    def resolve(self, name: str) -> Optional[Symbol]:
        return self.scope.resolve(name)


class GlobalEnvironment(SemanticEnvironment):
    def __init__(self):
        self.__scopes = [SemanticScope()]

    @property
    def scope(self) -> SemanticScope:
        return self.__scopes[-1]


class LocalEnvironment(SemanticEnvironment):
    def __init__(self, environment: GlobalEnvironment):
        self.__globals = environment
        self.__scopes = [SemanticScope()]

    @property
    def globals(self) -> GlobalEnvironment:
        return self.__globals

    @property
    def scope(self) -> SemanticScope:
        return self.__scopes[-1]

    @scope.setter
    def scope(self, scope: SemanticScope):
        self.__scopes[-1] = scope

    def push(self) -> SemanticScope:
        """ Create new scope inherited from current and push to stack """
        scope = SemanticScope(self.scope)
        self.__scopes.append(scope)
        return scope

    def pop(self) -> SemanticScope:
        """ Pop scope from stack """
        scope = self.__scopes.pop()
        return scope

    def merge(self, scopes: Set[SemanticScope]) -> SemanticScope:
        """ Merge current scope with another scopes and replace """
        self.scope = SemanticScope(self.scope)

        # Find intersection between received scopes and add to new scope

        # Return result scope
        return self.scope

    @contextlib.contextmanager
    def usage(self) -> SemanticScope:
        scope = self.push()
        yield scope
        self.pop()

    def resolve(self, name: str) -> Optional[Symbol]:
        return super().resolve(name) or self.globals.resolve(name)


class SemanticScope:
    def __init__(self, parent: SemanticScope = None):
        self.__parent: SemanticScope = parent
        self.__declared: MutableMapping[str, Symbol] = {}
        self.__typed: MutableMapping[str, Type] = {}

    @property
    def parent(self) -> SemanticScope:
        return self.__parent

    def define(self, name: str, symbol: Symbol):
        self.__declared[name] = symbol

    def resolve(self, name: str) -> Optional[Symbol]:
        symbol = self.__declared.get(name)
        if symbol:
            return symbol
        elif self.parent:
            return self.parent.resolve(name)


class SemanticMixin(abc.ABC):
    def __init__(self, model: SemanticModel):
        self.__model = model

    @property
    def environment(self) -> SemanticEnvironment:
        return self.model.environment

    @property
    def symbol_context(self) -> SymbolContext:
        return self.model.symbol_context

    @property
    def model(self) -> SemanticModel:
        return self.__model

    @property
    def symbols(self) -> SemanticMapping:
        return self.model.symbols

    @property
    def module(self) -> Module:
        return self.model.module

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.model.diagnostics

    def as_type(self, node: SyntaxNode) -> Type:
        """ Convert node to type """
        symbol = self.symbols[node]
        if isinstance(symbol, Type):
            return symbol
        return ErrorType(self.module, location=node.location)

    def as_value(self, node: SyntaxNode) -> Value:
        """ Convert node to value """
        symbol = self.symbols[node]
        if isinstance(symbol, Value):
            return symbol
        return ErrorValue(self.module, location=node.location)


class SemanticInitializer(SemanticMixin):
    def initialize(self, node: SyntaxNode):
        if isinstance(node, TypeDeclarationNode):
            self.initialize_type(node)
        elif isinstance(node, FunctionNode):
            self.initialize_function(node)
        elif isinstance(node, SyntaxTree):
            self.initialize_module(node)

    def initialize_module(self, node: SyntaxTree):
        module = cast(Module, self.symbols[node])
        for child in node.members:
            member: Optional[Symbol] = self.symbols[child]
            if isinstance(member, Child):
                module.add_member(member)

    def initialize_type(self, node: TypeDeclarationNode):
        type = cast(Type, self.symbols[node])

        for parent_node in node.parents:
            parent = self.symbols[parent_node].as_type(parent_node.location)
            if parent is type or type in parent.ascendants:
                # TODO: Can find all types involving in circular dependency
                self.diagnostics.error(parent_node.location, "Circular dependency")
            else:
                type.add_parent(parent)

        for generic in self.annotate_generics(node.generic_parameters):
            type.add_generic(generic)

        # insert attributes
        for attr in [self.symbols[decorator] for decorator in node.decorators]:
            type.add_attribute(cast(Attribute, attr))

        for child in node.members:
            member: Optional[Symbol] = self.symbols[child]
            if isinstance(member, Child):
                type.add_member(member)

        # symbol.type.build()

    def initialize_function(self, node: FunctionNode):
        function = cast(Function, self.symbols[node])

        for generic in self.annotate_generics(node.generic_parameters):
            function.add_generic(generic)

        # insert attributes
        for attr in [self.symbols[decorator] for decorator in node.decorators]:
            function.add_attribute(cast(Attribute, attr))

    def annotate_generics(self, parents: Sequence[GenericParameterNode]) -> Sequence[GenericType]:
        return [cast(GenericType, self.as_type(param)) for param in parents]


class SemanticAnnotator(SemanticMixin, abc.ABC):
    def __init__(self, model: SemanticModel, symbols: SemanticMapping = None):
        super(SemanticAnnotator, self).__init__(model)

        self.__symbols = symbols

    @property
    def symbols(self) -> SemanticMapping:
        return self.__symbols if self.__symbols is not None else self.model.symbols

    def annotate(self, node: SyntaxNode) -> Optional[Symbol]:
        return self.model.symbols[node]

    def search_named_symbol(self, name: str, location: Location) -> Optional[Symbol]:
        if name == TYPE_VOID_NAME:
            return self.symbol_context.void_type
        elif name == TYPE_BOOLEAN_NAME:
            return self.symbol_context.boolean_type
        elif name == TYPE_INTEGER_NAME:
            return self.symbol_context.integer_type
        elif name == TYPE_STRING_NAME:
            return self.symbol_context.string_type
        elif name == TYPE_FLOAT_NAME:
            return self.symbol_context.float_type
        elif name == VALUE_TRUE_NAME:
            return BooleanConstant(self.symbol_context, True, location=location)
        elif name == VALUE_FALSE_NAME:
            return BooleanConstant(self.symbol_context, False, location=location)
        elif name == VALUE_NONE_NAME:
            return NoneConstant(self.symbol_context, location=location)

        # resolve static symbol
        return self.environment.resolve(name)


class ExpressionAnnotator(SemanticAnnotator, ExpressionVisitor[Value], abc.ABC):
    def visit_expression(self, node: ExpressionNode) -> Value:
        self.diagnostics.error(node.location, f"Not implemented emitting symbols for statement: {type(node).__name__}")
        return ErrorValue(self.module, node.location)

    def visit_integer_expression(self, node: IntegerExpressionNode) -> Symbol:
        return IntegerConstant(self.symbol_context, int(node.value), location=node.location)

    def visit_string_expression(self, node: StringExpressionNode) -> Symbol:
        return StringConstant(self.symbol_context, node.value, location=node.location)

    def visit_named_expression(self, node: NamedExpressionNode) -> Symbol:
        symbol = self.search_named_symbol(node.name, node.location)
        if not symbol:
            self.diagnostics.error(node.location, f'Name ‘{node.name}’ not found in current scope')
            return ErrorValue(self.module, node.location)
        return symbol

    def visit_attribute_expression(self, node: AttributeExpressionNode) -> Symbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return ErrorValue(self.module, node.location)

    def visit_condition_expression(self, node: ConditionExpressionNode) -> Symbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return ErrorValue(self.module, node.location)

    def visit_unary_expression(self, node: UnaryExpressionNode) -> Symbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return ErrorValue(self.module, node.location)

    def visit_binary_expression(self, node: BinaryExpressionNode) -> Symbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return ErrorValue(self.module, node.location)

    def visit_logic_expression(self, node: LogicExpressionNode) -> Symbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return ErrorValue(self.module, node.location)

    def visit_compare_expression(self, node: CompareExpressionNode) -> Symbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return ErrorValue(self.module, node.location)

    def visit_call_expression(self, node: CallExpressionNode) -> Symbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return ErrorValue(self.module, node.location)

    def visit_subscribe_expression(self, node: SubscribeExpressionNode) -> Symbol:
        self.diagnostics.error(node.location, 'This expression can not used in this context')
        return ErrorValue(self.module, node.location)


class StaticAnnotator(SemanticAnnotator):
    @property
    def environment(self) -> SemanticEnvironment:
        return self.model.environment

    def annotate(self, node: SyntaxNode) -> Optional[Symbol]:
        if isinstance(node, SyntaxTree):
            return self.annotate_module(node)
        elif isinstance(node, TypeNode):
            return self.annotate_type(node)
        elif isinstance(node, MemberNode):
            return self.annotate_member(node)
        elif isinstance(node, ExpressionNode):
            return ExpressionAnnotator(self.model).visit(node)
        elif isinstance(node, OverloadNode):
            return self.annotate_overload(node)
        elif isinstance(node, DecoratorNode):
            return self.annotate_decorator(node)

        self.diagnostics.error(node.location, 'Not implemented member')
        return None

    def annotate_module(self, node: SyntaxTree) -> Module:
        return Module(self.symbol_context, node.name, node.location)

    def annotate_decorator(self, node: DecoratorNode) -> Attribute:
        arguments = cast(Sequence[Value], [self.as_value(arg) for arg in node.arguments])
        return Attribute(self.module, node.name, arguments, node.location)

    def annotate_type(self, node: TypeNode) -> Symbol:
        if isinstance(node, NamedTypeNode):
            return self.annotate_named_type(node)
        elif isinstance(node, ParameterizedTypeNode):
            return self.annotate_parameterized_type(node)
        elif isinstance(node, GenericParameterNode):
            return self.annotate_generic_type(node)

        self.diagnostics.error(node.location, 'Not implemented type')
        return ErrorType(self.module, location=node.location)

    def annotate_generic_type(self, node: GenericParameterNode) -> Type:
        return GenericType(self.model.module, node.name, node.location)

    def annotate_named_type(self, node: NamedTypeNode) -> Symbol:
        symbol = self.search_named_symbol(node.name, node.location)
        if not symbol:
            self.diagnostics.error(node.location, f'Type ‘{node.name}’ not found in current scope')
            return ErrorType(self.module, node.location)
        return symbol

    def annotate_parameterized_type(self, node: ParameterizedTypeNode) -> Symbol:
        generic = self.symbols[node.type]
        arguments = [self.symbols[arg] for arg in node.arguments]
        return generic.subscript(arguments, node.location)

    def annotate_member(self, node: MemberNode) -> Optional[Symbol]:
        parent: Symbol = self.symbols[node.parent]
        if node in self.symbols:
            return self.symbols[node]

        if isinstance(parent, Module):
            return ModuleMemberAnnotator(self.model, parent).annotate(node)
        elif isinstance(parent, ClassType):
            return ClassMemberAnnotator(self.model, parent).annotate(node)
        elif isinstance(parent, StructType):
            return StructMemberAnnotator(self.model, parent).annotate(node)
        elif isinstance(parent, InterfaceType):
            return InterfaceMemberAnnotator(self.model, parent).annotate(node)
        elif isinstance(parent, EnumType):
            return EnumMemberAnnotator(self.model, parent).annotate(node)
        elif isinstance(parent, Type):
            return TypeMemberAnnotator(self.model, parent).annotate(node)

        self.diagnostics.error(parent.location, 'Not implemented parent')
        return None

    def annotate_overload(self, node: OverloadNode) -> Overload:
        functions = [self.symbols[func] for func in node.functions]
        return Overload(self.module, node.name, cast(Sequence[Function], functions), location=node.location)


class ParentMemberAnnotator(SemanticAnnotator, MemberVisitor[Optional[Symbol]]):
    def __init__(self, model: SemanticModel, parent: Container):
        super().__init__(model)

        self.__parent = parent

    @property
    def parent(self) -> Container:
        return self.__parent

    def annotate(self, node: MemberNode) -> Optional[Symbol]:
        return self.visit(node)

    def visit_member(self, node: MemberNode):
        self.diagnostics.error(node.location, f"Not implemented emitting symbols for statement: {type(node).__name__}")

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

            self.symbols[node_param] = func_param

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

    def annotate(self, node: MemberNode) -> Optional[Symbol]:
        symbol = super().annotate(node)
        if isinstance(symbol, Named):
            self.environment.define(symbol.name, symbol)
        return symbol

    def visit_class(self, node: ClassNode) -> Type:
        if self.module == self.symbol_context.builtins_module:
            if node.name == TYPE_STRING_NAME:
                return StringType(self.parent, location=node.location)

        return super().visit_class(node)

    def visit_struct(self, node: StructNode) -> Type:
        if self.module == self.symbol_context.builtins_module:
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
            expr = self.symbols[node.value]
            if not isinstance(expr, IntegerConstant):
                value = expr.value
            else:
                self.diagnostics.error(node.location, "Enumeration values must have integer value or ‘...’")

        return EnumValue(self.parent, node.name, value, node.location)


class FunctionResolver(SemanticMixin):
    def __init__(self,
                 model: SemanticModel,
                 arguments: Sequence[Value] = None,
                 keywords: Mapping[str, Value] = None,
                 *, location: Location):
        super(FunctionResolver, self).__init__(model)

        self.location = location

        self.arguments = arguments or []
        self.keywords = keywords or {}
        self.candidates = []

    def add_scope_functions(self, environment: SemanticEnvironment, name: str):
        function = environment.resolve(name)
        if isinstance(function, Function):
            self.candidates.append(function)

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


class FlowAnnotator:
    def __init__(self, diagnostics):
        self.__diagnostics = diagnostics
        self.__graph = FlowGraph()
        self.__builder = FlowBuilder(self.__graph)

    @property
    def graph(self) -> FlowGraph:
        return self.__graph

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.__diagnostics

    @property
    def builder(self) -> FlowBuilder:
        return self.__builder

    def annotate(self, node: FunctionNode) -> FlowGraph:
        if not node.is_abstract:
            self.annotate_statement(node.statement)

        # last exit
        if self.builder.block and not self.builder.block.is_terminated:
            self.builder.append_link(self.builder.exit_block)

        graph = self.builder.graph
        return graph

    def annotate_statement(self, node: StatementNode):
        annotator = FlowStatementAnnotator(self)
        return annotator.annotate_statement(node)

    def annotate_expression(self, node: ExpressionNode):
        annotator = FlowExpressionAnnotator(self)
        return annotator.annotate_expression(node)


class FlowMixin:
    def __init__(self, parent: FlowAnnotator):
        self.parent = parent

    @property
    def graph(self) -> FlowGraph:
        return self.parent.graph

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.parent.diagnostics

    @property
    def builder(self) -> FlowBuilder:
        return self.parent.builder

    def annotate_statement(self, node: StatementNode):
        return self.parent.annotate_statement(node)

    def annotate_expression(self, node: ExpressionNode):
        return self.parent.annotate_expression(node)


class FlowStatementAnnotator(FlowMixin, StatementVisitor[None]):
    def annotate_statement(self, node: StatementNode):
        return self.visit(node)

    def visit_statement(self, node: StatementNode):
        self.diagnostics.error(node.location, 'Not implemented control flow annotation for statement')

    def visit_block_statement(self, node: BlockStatementNode):
        for child in node.statements:
            self.annotate_statement(child)

    def visit_while_statement(self, node: WhileStatementNode):
        terminated_blocks = []

        # loop start
        start_block = self.builder.append_block('while.start')
        self.builder.block.append_link(start_block)

        # loop condition
        self.builder.block = start_block
        with self.builder.block_helper('while.cond'):
            self.annotate_expression(node.condition)

        cond_block = self.builder.block
        assert cond_block, "Condition must continue execution of flow"

        # loop then block
        with self.builder.block_helper('while.then') as then_helper:
            with self.builder.loop_helper(cond_block) as loop_helper:
                self.annotate_statement(node.then_statement)

        if then_helper.exit_block:
            self.builder.append_link(cond_block)

        # loop else block
        if node.else_statement:
            self.builder.block = cond_block
            with self.builder.block_helper('while.else') as else_helper:
                self.annotate_statement(node.else_statement)
            terminated_blocks.append(else_helper.exit_block)
        else:
            terminated_blocks.append(cond_block)

        # TODO: break
        # TODO: continue

        # loop next block
        terminated_blocks = [block for block in terminated_blocks if block]
        if terminated_blocks:
            if loop_helper.break_block:
                loop_helper.break_block.name = 'while.next'
            next_block = loop_helper.break_block or self.builder.append_block('while.next')
            for block in terminated_blocks:
                block.append_link(next_block)
            self.builder.block = next_block
        else:
            # Unreachable code
            self.builder.unreachable()

    def visit_condition_statement(self, node: ConditionStatementNode):
        terminated_blocks = []

        # condition
        self.annotate_expression(node.condition)
        self.builder.append_instruction(node)
        enter_block = self.builder.block
        assert enter_block, "Condition must continue execution of flow"

        # then block
        self.builder.block = enter_block
        with self.builder.block_helper('if.then') as then_helper:
            self.annotate_statement(node.then_statement)
        terminated_blocks.append(then_helper.exit_block)

        # else block
        if node.else_statement:
            self.builder.block = enter_block
            with self.builder.block_helper('if.else') as else_helper:
                self.annotate_statement(node.else_statement)
            terminated_blocks.append(else_helper.exit_block)
        else:
            terminated_blocks.append(enter_block)

        # next block
        terminated_blocks = [block for block in terminated_blocks if block]
        if terminated_blocks:
            next_block = self.builder.append_block('if.next')
            for block in terminated_blocks:
                block.append_link(next_block)
            self.builder.block = next_block
        else:
            # Unreachable code
            self.builder.unreachable()

    def visit_else_statement(self, node: ElseStatementNode):
        return self.annotate_statement(node.statement)

    def visit_finally_statement(self, node: FinallyStatementNode):
        return self.annotate_statement(node.statement)

    def visit_expression_statement(self, node: ExpressionStatementNode):
        self.annotate_expression(node.value)
        self.builder.append_instruction(node)

    def visit_pass_statement(self, node: PassStatementNode):
        pass  # skip

    def visit_return_statement(self, node: ReturnStatementNode):
        if node.value:
            self.annotate_expression(node.value)

        self.builder.append_instruction(node)
        self.builder.append_link(self.builder.exit_block)
        self.builder.unreachable()

    def visit_continue_statement(self, node: ContinueStatementNode):
        if self.builder.continue_block:
            self.builder.append_link(self.builder.continue_block)
        else:
            self.diagnostics.error(node.location, "‘continue’ is outside of loop")

        self.builder.unreachable()

    def visit_break_statement(self, node: BreakStatementNode):
        if self.builder.break_block:
            self.builder.append_link(self.builder.break_block)
        else:
            self.diagnostics.error(node.location, "‘break’ is outside of loop")

        self.builder.unreachable()

    def visit_assign_statement(self, node: AssignStatementNode):
        self.annotate_expression(node.source)
        self.annotate_expression(node.target)
        self.builder.append_instruction(node)


class FlowExpressionAnnotator(FlowMixin, ExpressionVisitor[None]):
    def annotate_expression(self, node: ExpressionNode):
        return self.visit(node)

    def visit_expression(self, node: ExpressionNode):
        self.diagnostics.error(node.location, 'Not implemented control flow annotation for statement')

    def visit_integer_expression(self, node: IntegerExpressionNode):
        self.builder.append_instruction(node)

    def visit_string_expression(self, node: StringExpressionNode):
        self.builder.append_instruction(node)

    def visit_named_expression(self, node: NamedExpressionNode):
        self.builder.append_instruction(node)

    def visit_attribute_expression(self, node: AttributeExpressionNode):
        self.builder.append_instruction(node)

    def visit_call_expression(self, node: CallExpressionNode):
        self.builder.append_instruction(node)


class FunctionEmitter(ExpressionAnnotator):
    def __init__(self, model: SemanticModel, symbols: SemanticMapping[SyntaxNode, Symbol], node: FunctionNode):
        super().__init__(model, SemanticMapping(model, parent=symbols, constructor=lambda n: self.annotate(n)))

        self.__node = node
        self.__builder = IRBuilder(self.function.append_basic_block('entry'))
        self.__environment = LocalEnvironment(model.environment)

        for parameter in self.function.parameters:
            self.environment.define(parameter.name, parameter)

    @property
    def environment(self) -> LocalEnvironment:
        return self.__environment

    @property
    def node(self) -> FunctionNode:
        return self.__node

    @cached_property
    def function(self) -> Function:
        return cast(Function, self.symbols[self.node])

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

    def emit_statement(self, node: StatementNode):
        emitter = StatementEmitter(self)
        return emitter.emit_statement(node)

    def emit_expression(self, node: ExpressionNode) -> Value:
        emitter = ExpressionEmitter(self)
        return emitter.emit_expression(node)

    def emit(self):
        self.emit_statement(self.node.statement)
        if not self.builder.is_terminated:
            if isinstance(self.function.return_type, VoidType):
                self.emit_return(NoneConstant(self.symbol_context, self.node.location), self.node.location)
            else:
                self.diagnostics.error(self.node.location, 'Function must return value')
        return self.function


class FunctionMixin(SemanticAnnotator):
    # 1. flow graph (basic block)

    def __init__(self, parent: FunctionEmitter):
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

    @property
    def environment(self) -> LocalEnvironment:
        return self.parent.environment

    def emit_statement(self, node: StatementNode):
        return self.parent.emit_statement(node)

    def emit_expression(self, node: ExpressionNode) -> Value:
        return self.parent.emit_expression(node)


class StatementEmitter(FunctionMixin, StatementVisitor[None]):
    def emit_statement(self, node: StatementNode):
        """
        Emit environment and IR for statement.
        """
        if self.builder.is_terminated:
            return self.diagnostics.error(node.location, f"Unreachable code")

        return self.visit(node)

    def visit_statement(self, node: StatementNode):
        self.diagnostics.error(node.location, f"Not implemented emitting symbols for statement: {type(node).__name__}")

    def visit_block_statement(self, node: BlockStatementNode):
        for statement in node.statements:
            self.emit_statement(statement)

    def visit_pass_statement(self, node: PassStatementNode):
        pass

    def visit_return_statement(self, node: ReturnStatementNode):
        value = self.emit_expression(node.value) if node.value else None
        value = value or NoneConstant(self.symbol_context, node.location)
        self.parent.emit_return(value, value.location)

        # TODO: Unreachable code

    def visit_condition_statement(self, node: ConditionStatementNode):
        """
        if <condition>:
            <then_statement>
        [ else ':'
            <else_statement> ]
        <next_statement>    # can be unreachable
        """
        condition = self.emit_expression(node.condition)
        begin_scope = self.environment.scope
        begin_block = self.builder.block

        continue_blocks = set()  # Block that continue execution

        # then block and scope
        with self.environment.usage() as then_scope:  # new scope
            then_block = self.builder.append_basic_block('if.then')
            self.builder.position_at(then_block)
            self.emit_statement(node.then_statement)
            if not self.builder.is_terminated:
                continue_blocks.add(self.builder.block)

        # else block and scope
        if node.else_statement:
            with self.environment.usage() as else_scope:  # new scope
                else_block = self.builder.append_basic_block('if.else')
                self.builder.position_at(else_block)
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

        # merge scope
        self.environment.merge({then_scope, else_scope})

        # check if next blocks is unreachable
        if not next_block and not continue_blocks:
            return

        if not next_block:
            next_block = self.builder.append_basic_block('if.next')

        for block in filter(None, continue_blocks):
            with self.builder.goto_block(block):
                self.builder.branch(next_block, location=node.location)

        # mark as continue
        self.builder.position_at(next_block)

    def visit_else_statement(self, node: ElseStatementNode):
        return self.emit_statement(node.statement)

    def visit_while_statement(self, node: WhileStatementNode):
        cond_block = self.builder.append_basic_block('while.cond')
        self.builder.branch(cond_block, location=node.location)
        self.builder.position_at(cond_block)

        condition = self.emit_expression(node.condition)
        begin_scope = self.environment.scope
        begin_block = self.builder.block

        continue_blocks = set()  # Block that continue execution

        # then block and scope
        with self.environment.usage() as then_scope:  # new scope
            then_block = self.builder.append_basic_block('while.body')
            self.builder.position_at(then_block)
            self.emit_statement(node.then_statement)
            if not self.builder.is_terminated:
                self.builder.branch(cond_block, location=node.location)

        # else block and scope
        if node.else_statement:
            with self.environment.usage() as else_scope:  # new scope
                else_block = self.builder.append_basic_block('while.else')
                self.builder.position_at(else_block)
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

        # merge scope
        self.environment.merge({then_scope, else_scope})

        # check if next blocks is unreachable
        if not next_block and not continue_blocks:
            return

        if not next_block:
            next_block = self.builder.append_basic_block('while.next')

        for block in filter(None, continue_blocks):
            with self.builder.goto_block(block):
                self.builder.branch(next_block, location=node.location)

        # mark as continue
        self.builder.position_at(next_block)

    def visit_variable_statement(self, node: VariableStatementNode):
        var_type = self.as_type(node.type)
        with self.builder.goto_block(self.function.entry_block):
            variable = self.builder.alloca(var_type, name=node.name, location=node.location)

        if node.initial_value:
            initial_value = self.emit_expression(node.initial_value)
            self.builder.store(variable, initial_value, location=node.initial_value.location)

        self.environment.define(node.name, variable)

    def visit_assign_statement(self, node: AssignStatementNode):
        source = self.emit_expression(node.source)
        emitter = AssignmentEmitter(self.parent, source, node.location)
        emitter.visit(node.target)


class ExpressionEmitter(FunctionMixin, ExpressionAnnotator):
    def emit_expression(self, node: ExpressionNode):
        return self.visit(node)

    def visit_named_expression(self, node: NamedExpressionNode) -> Symbol:
        symbol = super().visit_named_expression(node)
        if isinstance(symbol, AllocaInstruction):
            return self.builder.load(symbol, location=node.location)
        return symbol

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

            function = FunctionResolver(self.model, [left_argument, right_argument], location=comparator.location)
            function.add_self_functions(name)
            result = function.find_function()
            if not result:
                self.diagnostics.error(comparator.location, f"Not found method ‘{name}’")  # TODO: Normal message
                return BooleanConstant(self.symbol_context, False, node.location)
            else:
                function, arguments = result
                return self.builder.call(function, arguments, location=comparator.location)

        # Unreachable
        raise NotImplementedError

    def visit_unary_expression(self, node: UnaryExpressionNode) -> Symbol:
        argument = self.emit_expression(node.argument)

        name = UNARY_NAMES.get(node.opcode)
        if not name:
            self.diagnostics.error(node.location, "Not implemented binary operator")
            return ErrorValue(self.module, node.location)

        function = FunctionResolver(self.model, [argument], location=node.location)
        function.add_self_functions(name)
        result = function.find_function()
        if not result:
            self.diagnostics.error(node.location, f"Not found method ‘{name}’")  # TODO: Normal message
            return BooleanConstant(self.symbol_context, False, node.location)
        else:
            function, arguments = result
            return self.builder.call(function, arguments, location=node.location)

    def visit_binary_expression(self, node: BinaryExpressionNode) -> Symbol:
        left_argument = self.emit_expression(node.left_argument)
        right_argument = self.emit_expression(node.right_argument)

        name = BINARY_NAMES.get(node.opcode)
        if not name:
            self.diagnostics.error(node.location, "Not implemented binary operator")
            return ErrorValue(self.module, node.location)

        function = FunctionResolver(self.model, [left_argument, right_argument], location=node.location)
        function.add_self_functions(name)
        result = function.find_function()
        if not result:
            self.diagnostics.error(node.location, f"Not found method ‘{name}’")  # TODO: Normal message
            return BooleanConstant(self.symbol_context, False, node.location)
        else:
            function, arguments = result
            return self.builder.call(function, arguments, location=node.location)

    def visit_call_expression(self, node: CallExpressionNode) -> Symbol:
        arguments = [self.emit_expression(arg) for arg in node.arguments]
        keywords = {name: self.emit_expression(arg) for name, arg in node.keywords.items()}

        emitter = CallEmitter(self.parent, arguments, keywords, node.location)
        return emitter.visit(node.instance)

    def visit_tuple_expression(self, node: TupleExpressionNode):
        if len(node.arguments) != 1:
            self.diagnostics.error(self.location, f"Tuple value is not implemented")  # TODO: Normal message
            return ErrorValue(self.module, self.location)
        return self.emit_expression(node.arguments[0])


class AssignmentEmitter(FunctionMixin, ExpressionVisitor[None]):
    def __init__(self, parent: FunctionEmitter, source: Value, location: Location):
        super().__init__(parent)

        self.source = source
        self.location = location

    def visit_expression(self, node: ExpressionNode):
        self.diagnostics.error(node.location, 'Can not assign value to expression')

    def visit_named_expression(self, node: NamedExpressionNode):
        symbol = self.search_named_symbol(node.name, node.location)
        if isinstance(symbol, AllocaInstruction):
            self.builder.store(symbol, self.source, location=self.location)
        else:
            self.diagnostics.error(self.location, 'Can not assign value')


class CallEmitter(FunctionMixin, ExpressionVisitor[Value]):
    def __init__(self,
                 parent: FunctionEmitter,
                 arguments: Sequence[Value],
                 keywords: Mapping[str, Value],
                 location: Location):

        super(CallEmitter, self).__init__(parent)

        self.arguments = list(arguments)
        self.keywords = keywords
        self.location = location

    def emit_named(self, name: str):
        resolver = FunctionResolver(self.model, self.arguments, self.keywords, location=self.location)
        resolver.add_scope_functions(self.environment, name)
        resolver.add_self_functions(name)
        result = resolver.find_function()

        if not result:
            self.diagnostics.error(self.location, f"Not found method ‘{name}’")  # TODO: Normal message
            return ErrorValue(self.module, self.location)

        function, arguments = result
        return self.builder.call(function, arguments, location=self.location)

    def emit_constructor(self, clazz: Type):
        resolver = FunctionResolver(self.model, self.arguments, self.keywords, location=self.location)
        resolver.add_type_functions(clazz, NEW_NAME)
        result = resolver.find_function()

        if not result:
            self.diagnostics.error(self.location, f"Not found method ‘{NEW_NAME}’")  # TODO: Normal message
            return ErrorValue(self.module, self.location)

        function, arguments = result
        return self.builder.call(function, arguments, location=self.location)

    def visit_expression(self, node: ExpressionNode):
        self.arguments.insert(0, self.emit_expression(node))
        return self.emit_named('__call__')

    def visit_named_expression(self, node: NamedExpressionNode):
        symbol = self.search_named_symbol(node.name, node.location)
        if isinstance(symbol, Type):
            return self.emit_constructor(symbol)
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
