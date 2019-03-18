# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
import dataclasses
import enum
import weakref
from typing import Sequence, Iterator, Optional, Union, Iterable, TypeVar, Callable, Mapping

from multidict import MultiDict

from orcinus.diagnostics import DiagnosticManager
from orcinus.locations import Location


@enum.unique
class TokenID(enum.IntEnum):
    Accent = enum.auto()
    Ampersand = enum.auto()
    And = enum.auto()
    As = enum.auto()
    At = enum.auto()
    Break = enum.auto()
    Class = enum.auto()
    Colon = enum.auto()
    Comma = enum.auto()
    Comment = enum.auto()
    Continue = enum.auto()
    Def = enum.auto()
    Dot = enum.auto()
    Double = enum.auto()
    DoubleSlash = enum.auto()
    DoubleSlashEqual = enum.auto()
    DoubleStar = enum.auto()
    Elif = enum.auto()
    Ellipsis = enum.auto()
    Else = enum.auto()
    EndOfFile = enum.auto()
    Enum = enum.auto()
    EqEqual = enum.auto()
    Equal = enum.auto()
    Error = enum.auto()
    Except = enum.auto()
    Finally = enum.auto()
    For = enum.auto()
    From = enum.auto()
    Great = enum.auto()
    GreatEqual = enum.auto()
    If = enum.auto()
    Import = enum.auto()
    In = enum.auto()
    Indent = enum.auto()
    Integer = enum.auto()
    Interface = enum.auto()
    Is = enum.auto()
    LeftCurly = enum.auto()
    LeftParenthesis = enum.auto()
    LeftSquare = enum.auto()
    Less = enum.auto()
    LessEqual = enum.auto()
    Minus = enum.auto()
    MinusEqual = enum.auto()
    Name = enum.auto()
    NewLine = enum.auto()
    Not = enum.auto()
    NotEqual = enum.auto()
    Or = enum.auto()
    Pass = enum.auto()
    Plus = enum.auto()
    PlusEqual = enum.auto()
    Raise = enum.auto()
    Return = enum.auto()
    RightCurly = enum.auto()
    RightParenthesis = enum.auto()
    RightSquare = enum.auto()
    Semicolon = enum.auto()
    Slash = enum.auto()
    SlashEqual = enum.auto()
    Star = enum.auto()
    StarEqual = enum.auto()
    String = enum.auto()
    Struct = enum.auto()
    Then = enum.auto()
    Tilde = enum.auto()
    Try = enum.auto()
    Undent = enum.auto()
    VeriticalLine = enum.auto()
    With = enum.auto()
    While = enum.auto()
    Whitespace = enum.auto()
    Yield = enum.auto()


@dataclasses.dataclass
class SyntaxToken:
    id: TokenID
    value: str
    location: Location


class SyntaxScope:
    def __init__(self, context: SyntaxContext, node: SyntaxNode, parent: SyntaxScope = None):
        self.__context = weakref.ref(context)
        self.__parent = parent
        self.__node = weakref.ref(node)
        self.__defines = MultiDict()
        self.__resolves = {}

    @property
    def context(self) -> SyntaxContext:
        context = self.__context()
        if context is None:
            raise RuntimeError('Syntax context is disposed')
        return context

    @property
    def node(self) -> SyntaxNode:
        node = self.__node()
        if node is None:
            raise RuntimeError('Syntax node is disposed')
        return node

    @property
    def parent(self) -> Optional[SyntaxScope]:
        return self.__parent

    def declare(self, name, node: SyntaxNode):
        self.__defines[name] = weakref.ref(node)

    def resolve(self, name: str) -> Optional[SyntaxNode]:
        # """
        # Resolve symbol by name in current scope.
        #
        # If symbol is defined in current scope and:
        #
        #     - has type `Overload` it must extended with functions from parent scope
        #
        # If symbol is not defined in current scope, check if it can be resolved in parent scope.
        # """

        # If symbol already resolved then returns it.
        if name in self.__resolves:
            return self.__resolves[name]

        # Resolve symbol in current scope
        symbol = None
        symbols = [ref() for ref in self.__defines.getall(name, ())]
        is_populate = False

        # Found single symbol: import overload function
        if len(symbols) == 1:
            symbol = symbols[0]
            if isinstance(symbol, FunctionNode):
                symbol = OverloadNode(self.context, name, SyntaxCollection([symbol]), symbol.location)

        # Found multiple symbols: import overload function
        elif symbols:
            is_other = any(not isinstance(symbol, FunctionNode) for symbol in symbols)
            if is_other:
                for symbol in symbols:
                    self.context.diagnostics.error(symbol.location, f"Already declared symbol `{symbol.name}`")

            is_populate = True

            symbols = (symbol for symbol in symbols if isinstance(symbol, FunctionNode))
            symbol = OverloadNode(self.context, name, SyntaxCollection(symbols), symbol.location)

        # Resolve symbol in parent scope
        elif self.parent:
            is_populate = False
            symbol = self.parent.resolve(name)
            if isinstance(symbol, OverloadNode):
                symbol = OverloadNode(self.context, name, symbol.functions, symbol.location)

        # # Import nodes from root scope
        # elif name in self.__imported:
        #     is_populate = False
        #     symbol = self.__imported[name]

        # Return None, if symbol is not found in current and ascendant scopes
        if not symbol:
            return None

        # Populate overload from parent scopes
        elif is_populate and isinstance(symbol, OverloadNode):
            overload = self.parent.resolve(name)
            if isinstance(overload, OverloadNode):
                symbol = symbol.extend(overload)

            # overload = self.__imported.get(name)
            # if isinstance(overload, Overload):
            #     symbol = symbol.extend(overload)

        # Save resolved symbol
        self.__resolves[name] = symbol
        return symbol


class SyntaxContext:
    def __init__(self, diagnostics: DiagnosticManager):
        self.__diagnostics = diagnostics
        self.__scopes = {}
        self.__parents = {}

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.__diagnostics

    def get_scope(self, node: SyntaxNode) -> SyntaxScope:
        return self.__scopes[node]

    def get_parent(self, node: SyntaxNode) -> Optional[SyntaxNode]:
        return self.__parents[node]

    def annotate(self, parent: SyntaxNode):
        if not isinstance(parent, SyntaxNode):
            raise RuntimeError('Required syntax node')

        self.__scopes[parent] = self.annotate_scope(parent)
        self.declare_symbol(parent)

        for child in parent.children:
            if not isinstance(child, SyntaxNode):
                raise RuntimeError('Required syntax node')
            self.__parents[child] = parent
            self.annotate(child)

    def annotate_scope(self, node: SyntaxNode) -> SyntaxScope:
        if isinstance(node, SyntaxTree):
            return SyntaxScope(self, node)

        parent = node.parent.scope
        if isinstance(node, (FunctionNode, TypeDeclarationNode, BlockStatementNode)):
            return SyntaxScope(self, node, parent)
        return parent

    def declare_symbol(self, node: SyntaxNode):
        if not isinstance(node, (FunctionNode, TypeDeclarationNode, ParameterNode)):
            return

        node.declare_scope.declare(node.name, node)


class SyntaxNode(abc.ABC):
    __context: weakref.ReferenceType
    location: Location

    def __init__(self, context: SyntaxContext, location: Location):
        self.__context = weakref.ref(context)
        self.location = location

    @property
    def context(self) -> SyntaxContext:
        context = self.__context()
        if context is None:
            raise RuntimeError('Syntax context is disposed')
        return context

    @property
    def parent(self) -> Optional[SyntaxNode]:
        return self.context.get_parent(self)

    @property
    def declare_scope(self) -> SyntaxScope:
        if not self.parent:
            return self.scope
        return self.parent.scope

    @property
    def scope(self) -> SyntaxScope:
        return self.context.get_scope(self)

    @property
    @abc.abstractmethod
    def children(self) -> Sequence[SyntaxNode]:
        raise NotImplementedError

    def find_ascendants(self, predicate: Callable[[SyntaxNode], bool]) -> Iterator[SyntaxNode]:
        if not self.parent:
            return

        if predicate(self.parent):
            yield self.parent
        yield from self.parent.find_ascendants(predicate)

    def find_descendants(self, predicate: Callable[[SyntaxNode], bool]) -> Iterator[SyntaxNode]:
        for child in self.children:
            if predicate(child):
                yield child
            yield from child.find_descendants(predicate)

    def __iter__(self) -> Iterator[SyntaxNode]:
        return iter(self.children)


K = TypeVar('K')
T = TypeVar('T')


class SyntaxCollection(Sequence[T]):
    def __init__(self, items: Iterable[T] = None):
        items = tuple(items) if items else ()
        for item in items:
            if not isinstance(item, SyntaxNode):
                raise RuntimeError('Collection must contains only syntax nodes')
        self.__items = items

    def __getitem__(self, i: int) -> T:
        return self.__items[i]

    def __len__(self) -> int:
        return len(self.__items)


class SyntaxDictionary(Mapping[K, T]):
    def __init__(self, items: Mapping[K, T]):
        self.__items = items

    def __getitem__(self, k: K) -> T:
        return self.__items[k]

    def __len__(self) -> int:
        return len(self.__items)

    def __iter__(self) -> Iterator[K]:
        return iter(self.__items)


class SyntaxTree(SyntaxNode):
    name: str
    imports: SyntaxCollection[ImportNode]
    members: SyntaxCollection[MemberNode]

    def __init__(self, context: SyntaxContext, name: str, imports: SyntaxCollection[ImportNode],
                 members: SyntaxCollection[MemberNode], location: Location):
        super(SyntaxTree, self).__init__(context, location)

        self.name = name
        self.members = members
        self.imports = imports

    @property
    def filename(self) -> str:
        return self.location.filename

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.imports, self.members])


class OverloadNode(SyntaxNode):
    def __init__(self, context: SyntaxContext, name: str, functions: SyntaxCollection[FunctionNode],
                 location: Location):
        super(OverloadNode, self).__init__(context, location)

        self.name = name
        self.functions = functions

        assert all(isinstance(node, FunctionNode) for node in functions)

    def extend(self, other: OverloadNode) -> OverloadNode:
        functions = SyntaxCollection[FunctionNode](list(self.functions) + list(other.functions))
        return OverloadNode(self.context, self.name, functions, self.location)

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []  # What??


class ImportNode(SyntaxNode):
    aliases: SyntaxCollection[AliasNode]

    def __init__(self, context: SyntaxContext, aliases: SyntaxCollection[AliasNode], location: Location):
        super(ImportNode, self).__init__(context, location)

        self.aliases = aliases

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.aliases])


class ImportFromNode(ImportNode):
    module: str

    def __init__(self, context: SyntaxContext, module: str, aliases: SyntaxCollection[AliasNode], location: Location):
        super(ImportFromNode, self).__init__(context, aliases, location)

        self.module = module


class AliasNode(SyntaxNode):
    name: str
    alias: Optional[str]

    def __init__(self, context: SyntaxContext, name: str, alias: Optional[str], location: Location):
        super(AliasNode, self).__init__(context, location)

        self.name = name
        self.alias = alias

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class DecoratorNode(SyntaxNode):
    name: str
    arguments: SyntaxCollection[ArgumentNode]

    def __init__(self, context: SyntaxContext, name: str, arguments: SyntaxCollection[ArgumentNode],
                 location: Location):
        super(DecoratorNode, self).__init__(context, location)

        self.name = name
        self.arguments = arguments

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.arguments])


class MemberNode(SyntaxNode, abc.ABC):
    pass


class PassMemberNode(MemberNode, abc.ABC):
    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class ParameterNode(SyntaxNode):
    name: str
    type: TypeNode
    default_value: Optional[ExpressionNode]

    def __init__(self, context: SyntaxContext, name: str, type: TypeNode, default_value: Optional[ExpressionNode],
                 location: Location):
        super(ParameterNode, self).__init__(context, location)

        self.name = name
        self.type = type
        self.default_value = default_value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.type, self.default_value])


class EnumMemberNode(MemberNode):
    name: str
    value: ExpressionNode

    def __init__(self, context: SyntaxContext, name: str, value: ExpressionNode, location: Location):
        super(EnumMemberNode, self).__init__(context, location)

        self.name = name
        self.value = value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.value])


class FieldNode(MemberNode):
    name: str
    type: TypeNode
    default_value: Optional[ExpressionNode]

    def __init__(self, context: SyntaxContext, name: str, type: TypeNode, default_value: Optional[ExpressionNode],
                 location: Location):
        super(FieldNode, self).__init__(context, location)
        self.name = name
        self.type = type
        self.default_value = default_value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.type, self.default_value])


class FunctionNode(MemberNode):
    decorators: SyntaxCollection[DecoratorNode]
    name: str
    generic_parameters: SyntaxCollection[GenericParameterNode]
    parameters: SyntaxCollection[ParameterNode]
    return_type: TypeNode
    statement: Optional[StatementNode]

    def __init__(self, context: SyntaxContext, decorators: SyntaxCollection[DecoratorNode], name: str,
                 generic_parameters: SyntaxCollection[GenericParameterNode],
                 parameters: SyntaxCollection[ParameterNode],
                 return_type: TypeNode, statement: Optional[StatementNode], location: Location):
        super(FunctionNode, self).__init__(context, location)

        self.decorators = decorators
        self.name = name
        self.generic_parameters = generic_parameters
        self.parameters = parameters
        self.return_type = return_type
        self.statement = statement

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([
            self.decorators,
            self.generic_parameters,
            self.parameters,
            self.return_type,
            self.statement
        ])


class TypeDeclarationNode(MemberNode):
    decorators: SyntaxCollection[DecoratorNode]
    name: str
    generic_parameters: SyntaxCollection[GenericParameterNode]
    parents: SyntaxCollection[TypeNode]
    members: SyntaxCollection[MemberNode]

    def __init__(self, context: SyntaxContext, decorators: SyntaxCollection[DecoratorNode], name: str,
                 generic_parameters: SyntaxCollection[GenericParameterNode], parents: SyntaxCollection[TypeNode],
                 members: SyntaxCollection[MemberNode], location: Location):
        super(TypeDeclarationNode, self).__init__(context, location)

        self.decorators = decorators
        self.name = name
        self.generic_parameters = generic_parameters
        self.parents = parents
        self.members = members

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([
            self.decorators,
            self.generic_parameters,
            self.parents,
            self.members
        ])


class ClassNode(TypeDeclarationNode):
    pass


class StructNode(TypeDeclarationNode):
    pass


class InterfaceNode(TypeDeclarationNode):
    pass


class EnumNode(TypeDeclarationNode):
    pass


class TypeNode(SyntaxNode, abc.ABC):
    pass


class AutoTypeNode(TypeNode):
    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class NamedTypeNode(TypeNode):
    name: str

    def __init__(self, context: SyntaxContext, name: str, location: Location):
        super(NamedTypeNode, self).__init__(context, location)

        self.name = name

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class GenericParameterNode(TypeNode):
    name: str

    def __init__(self, context: SyntaxContext, name: str, location: Location):
        super(GenericParameterNode, self).__init__(context, location)

        self.name = name

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class ParameterizedTypeNode(TypeNode):
    type: TypeNode
    arguments: SyntaxCollection[TypeNode]

    def __init__(self, context: SyntaxContext, basis: TypeNode, arguments: SyntaxCollection[TypeNode],
                 location: Location):
        super(ParameterizedTypeNode, self).__init__(context, location)

        self.type = basis
        self.arguments = arguments

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.type, self.arguments])


class StatementNode(SyntaxNode, abc.ABC):
    pass


class BlockStatementNode(StatementNode):
    statements: SyntaxCollection[StatementNode]

    def __init__(self, context: SyntaxContext, statements: SyntaxCollection[StatementNode], location: Location):
        super(BlockStatementNode, self).__init__(context, location)

        self.statements = statements

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.statements])


class PassStatementNode(StatementNode):
    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class BreakStatementNode(StatementNode):
    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class ContinueStatementNode(StatementNode):
    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class ReturnStatementNode(StatementNode):
    value: Optional[ExpressionNode]

    def __init__(self, context: SyntaxContext, value: Optional[ExpressionNode], location: Location):
        super(ReturnStatementNode, self).__init__(context, location)

        self.value = value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.value])


class YieldStatementNode(StatementNode):
    value: Optional[ExpressionNode]

    def __init__(self, context: SyntaxContext, value: Optional[ExpressionNode], location: Location):
        super(YieldStatementNode, self).__init__(context, location)

        self.value = value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.value])


class AssignStatementNode(StatementNode):
    target: ExpressionNode
    source: ExpressionNode

    def __init__(self, context: SyntaxContext, target: ExpressionNode, source: ExpressionNode, location: Location):
        super(AssignStatementNode, self).__init__(context, location)

        self.target = target
        self.source = source

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.target, self.source])


class AugmentedAssignStatementNode(StatementNode):
    target: ExpressionNode
    opcode: BinaryID
    source: ExpressionNode

    def __init__(self, context: SyntaxContext, target: ExpressionNode, opcode: BinaryID, source: ExpressionNode,
                 location: Location):
        super(AugmentedAssignStatementNode, self).__init__(context, location)

        self.target = target
        self.opcode = opcode
        self.source = source

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.target, self.source])


class ConditionStatementNode(StatementNode):
    condition: ExpressionNode
    then_statement: StatementNode
    else_statement: Optional[StatementNode]

    def __init__(self, context: SyntaxContext, condition: ExpressionNode, then_statement: StatementNode,
                 else_statement: Optional[StatementNode], location: Location):
        super(ConditionStatementNode, self).__init__(context, location)

        self.condition = condition
        self.then_statement = then_statement
        self.else_statement = else_statement

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.condition, self.then_statement, self.else_statement])


class WhileStatementNode(StatementNode):
    condition: ExpressionNode
    then_statement: StatementNode
    else_statement: Optional[StatementNode]

    def __init__(self, context: SyntaxContext, condition: ExpressionNode, then_statement: StatementNode,
                 else_statement: Optional[StatementNode], location: Location):
        super(WhileStatementNode, self).__init__(context, location)

        self.condition = condition
        self.then_statement = then_statement
        self.else_statement = else_statement

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.condition, self.then_statement, self.else_statement])


class ForStatementNode(StatementNode):
    target: ExpressionNode
    source: ExpressionNode
    then_statement: StatementNode
    else_statement: Optional[StatementNode]

    def __init__(self, context: SyntaxContext, target: ExpressionNode, source: ExpressionNode,
                 then_statement: StatementNode, else_statement: Optional[StatementNode], location: Location):
        super(ForStatementNode, self).__init__(context, location)

        self.target = target
        self.source = source
        self.then_statement = then_statement
        self.else_statement = else_statement

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.target, self.source, self.then_statement, self.else_statement])


class RaiseStatementNode(StatementNode):
    exception: Optional[ExpressionNode]
    cause_exception: Optional[ExpressionNode]

    def __init__(self, context: SyntaxContext, exception: Optional[ExpressionNode],
                 cause_exception: Optional[ExpressionNode], location: Location):
        super(RaiseStatementNode, self).__init__(context, location)

        self.exception = exception
        self.cause_exception = cause_exception

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.exception, self.cause_exception])


class ExceptHandlerNode(SyntaxNode):
    expression: Optional[ExpressionNode]
    name: Optional[str]
    statement: StatementNode

    def __init__(self, context: SyntaxContext, expression: Optional[ExpressionNode], name: Optional[str],
                 statement: StatementNode, location: Location):
        super(ExceptHandlerNode, self).__init__(context, location)

        self.expression = expression
        self.name = name
        self.statement = statement

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.expression, self.statement])


class TryStatementNode(StatementNode):
    try_statement: StatementNode
    handlers: SyntaxCollection[ExceptHandlerNode]
    else_statement: Optional[StatementNode]
    finally_statement: Optional[StatementNode]

    def __init__(self, context: SyntaxContext, try_statement: StatementNode,
                 handlers: SyntaxCollection[ExceptHandlerNode], else_statement: Optional[StatementNode],
                 finally_statement: Optional[StatementNode],
                 location: Location):
        super(TryStatementNode, self).__init__(context, location)

        self.try_statement = try_statement
        self.handlers = handlers
        self.else_statement = else_statement
        self.finally_statement = finally_statement

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.try_statement, self.handlers, self.else_statement, self.finally_statement])


class WithItemNode(SyntaxNode):
    expression: ExpressionNode
    target: Optional[ExpressionNode]

    def __init__(self, context: SyntaxContext, expression: ExpressionNode, target: Optional[ExpressionNode],
                 location: Location):
        super(WithItemNode, self).__init__(context, location)

        self.expression = expression
        self.target = target

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.expression, self.target])


class WithStatementNode(StatementNode):
    items: SyntaxCollection[WithItemNode]
    statement: StatementNode

    def __init__(self, context: SyntaxContext, items: SyntaxCollection[WithItemNode], statement: StatementNode,
                 location: Location):
        super(WithStatementNode, self).__init__(context, location)

        self.items = items
        self.statement = statement

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.items, self.statement])


class ExpressionStatementNode(StatementNode):
    value: ExpressionNode

    def __init__(self, context: SyntaxContext, value: ExpressionNode, location: Location):
        super(ExpressionStatementNode, self).__init__(context, location)

        self.value = value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.value])


class ExpressionNode(SyntaxNode, abc.ABC):
    pass


class IntegerExpressionNode(ExpressionNode):
    value: int

    def __init__(self, context: SyntaxContext, value: int, location: Location):
        super(IntegerExpressionNode, self).__init__(context, location)

        self.value = value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class StringExpressionNode(ExpressionNode):
    value: str

    def __init__(self, context: SyntaxContext, value: str, location: Location):
        super(StringExpressionNode, self).__init__(context, location)

        self.value = value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class NamedExpressionNode(ExpressionNode):
    name: str

    def __init__(self, context: SyntaxContext, name: str, location: Location):
        super(NamedExpressionNode, self).__init__(context, location)

        self.name = name

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return []


class AttributeExpressionNode(ExpressionNode):
    instance: ExpressionNode
    name: str

    def __init__(self, context: SyntaxContext, instance: ExpressionNode, name: str, location: Location):
        super(AttributeExpressionNode, self).__init__(context, location)

        self.instance = instance
        self.name = name

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return [self.instance]


@enum.unique
class UnaryID(enum.IntEnum):
    Not = enum.auto()
    Pos = enum.auto()
    Neg = enum.auto()
    Inv = enum.auto()


class UnaryExpressionNode(ExpressionNode):
    operator: UnaryID
    operand: ExpressionNode

    def __init__(self, context: SyntaxContext, operator: UnaryID, operand: ExpressionNode, location: Location):
        super(UnaryExpressionNode, self).__init__(context, location)

        self.operator = operator
        self.operand = operand

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return [self.operand]


@enum.unique
class BinaryID(enum.IntEnum):
    Add = enum.auto()
    Sub = enum.auto()
    Mul = enum.auto()
    TrueDiv = enum.auto()
    FloorDiv = enum.auto()
    Pow = enum.auto()
    And = enum.auto()
    Or = enum.auto()
    Xor = enum.auto()


class BinaryExpressionNode(ExpressionNode):
    operator: BinaryID
    left_operand: ExpressionNode
    right_operand: ExpressionNode

    def __init__(self, context: SyntaxContext, operator: BinaryID, left_operand: ExpressionNode,
                 right_operand: ExpressionNode, location: Location):
        super(BinaryExpressionNode, self).__init__(context, location)

        self.operator = operator
        self.left_operand = left_operand
        self.right_operand = right_operand

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.left_operand, self.right_operand])


class CompareExpressionNode(ExpressionNode):
    left_operand: ExpressionNode
    comparators: SyntaxCollection[ComparatorNode]

    def __init__(self, context: SyntaxContext, left_operand: ExpressionNode,
                 comparators: SyntaxCollection[ComparatorNode], location: Location):
        super(CompareExpressionNode, self).__init__(context, location)

        self.left_operand = left_operand
        self.comparators = comparators

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.left_operand, self.comparators])


@enum.unique
class CompareID(enum.IntEnum):
    In = enum.auto()
    Is = enum.auto()
    NotIn = enum.auto()
    IsNot = enum.auto()

    Eq = enum.auto()
    Ne = enum.auto()
    Lt = enum.auto()
    Le = enum.auto()
    Gt = enum.auto()
    Ge = enum.auto()


class ComparatorNode(SyntaxNode):
    operator: CompareID
    right_operand: ExpressionNode

    def __init__(self, context: SyntaxContext, opcode: CompareID, right_operand: ExpressionNode, location: Location):
        super(ComparatorNode, self).__init__(context, location)

        self.operator = opcode
        self.right_operand = right_operand

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return [self.right_operand]


class ConditionExpressionNode(ExpressionNode):
    condition: ExpressionNode
    then_value: ExpressionNode
    else_value: ExpressionNode

    def __init__(self, context: SyntaxContext, then_value: ExpressionNode, condition: ExpressionNode,
                 else_value: ExpressionNode, location: Location):
        super(ConditionExpressionNode, self).__init__(context, location)

        self.then_value = then_value
        self.condition = condition
        self.else_value = else_value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return [self.then_value, self.condition, self.else_value]


@enum.unique
class LogicID(enum.IntEnum):
    And = enum.auto()
    Or = enum.auto()


class LogicExpressionNode(ExpressionNode):
    operator: LogicID
    left_operand: ExpressionNode
    right_operand: ExpressionNode

    def __init__(self, context: SyntaxContext, operator: LogicID, left_operand: ExpressionNode,
                 right_operand: ExpressionNode, location: Location):
        super(LogicExpressionNode, self).__init__(context, location)

        self.operator = operator
        self.left_operand = left_operand
        self.right_operand = right_operand

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return [self.left_operand, self.right_operand]


class CallExpressionNode(ExpressionNode):
    instance: ExpressionNode
    arguments: SyntaxCollection[ExpressionNode]
    keywords: SyntaxDictionary[str, ExpressionNode]

    def __init__(self, context: SyntaxContext, instance: ExpressionNode, arguments: SyntaxCollection[ArgumentNode],
                 location: Location):
        super(CallExpressionNode, self).__init__(context, location)

        self.instance = instance
        self.arguments = SyntaxCollection[ExpressionNode](
            [arg.value for arg in arguments if isinstance(arg, PositionArgumentNode)])
        self.keywords = SyntaxDictionary[str, ExpressionNode]({
            arg.name: arg.value for arg in arguments if isinstance(arg, KeywordArgumentNode)
        })

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.instance, self.arguments])


class SubscribeExpressionNode(ExpressionNode):
    instance: ExpressionNode
    arguments: SyntaxCollection[Union[ExpressionNode, SliceArgumentNode]]

    def __init__(self, context: SyntaxContext, instance: ExpressionNode,
                 arguments: SyntaxCollection[Union[ExpressionNode, SliceArgumentNode]], location: Location):
        super(SubscribeExpressionNode, self).__init__(context, location)

        self.instance = instance
        self.arguments = arguments

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.instance, self.arguments])


class ArrayExpressionNode(ExpressionNode):
    arguments: SyntaxCollection[ExpressionNode]

    def __init__(self, context: SyntaxContext, arguments: SyntaxCollection[ExpressionNode], location: Location):
        super(ArrayExpressionNode, self).__init__(context, location)

        self.arguments = arguments

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.arguments])


class SetExpressionNode(ExpressionNode):
    arguments: SyntaxCollection[ExpressionNode]

    def __init__(self, context: SyntaxContext, arguments: SyntaxCollection[ExpressionNode], location: Location):
        super(SetExpressionNode, self).__init__(context, location)

        self.arguments = arguments

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.arguments])


class TupleExpressionNode(ExpressionNode):
    arguments: SyntaxCollection[ExpressionNode]

    def __init__(self, context: SyntaxContext, arguments: SyntaxCollection[ExpressionNode], location: Location):
        super(TupleExpressionNode, self).__init__(context, location)

        self.arguments = arguments

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.arguments])


class DictExpressionNode(ExpressionNode):
    arguments: SyntaxCollection[DictArgumentNode]

    def __init__(self, context: SyntaxContext, arguments: SyntaxCollection[DictArgumentNode], location: Location):
        super(DictExpressionNode, self).__init__(context, location)

        self.arguments = arguments

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.arguments])


class DictArgumentNode(SyntaxNode):
    key: ExpressionNode
    value: ExpressionNode

    def __init__(self, context: SyntaxContext, key: ExpressionNode, value: ExpressionNode, location: Location):
        super(DictArgumentNode, self).__init__(context, location)

        self.key = key
        self.value = value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.key, self.value])


class ArgumentNode(SyntaxNode, abc.ABC):
    value: ExpressionNode

    def __init__(self, context: SyntaxContext, value: ExpressionNode, location: Location):
        super(ArgumentNode, self).__init__(context, location)

        self.value = value

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return [self.value]


class PositionArgumentNode(ArgumentNode):
    pass


class KeywordArgumentNode(ArgumentNode):
    name: str

    def __init__(self, context: SyntaxContext, name: str, value: ExpressionNode, location: Location):
        super(KeywordArgumentNode, self).__init__(context, value, location)

        self.name = name


class SliceArgumentNode(SyntaxNode, abc.ABC):
    lower_bound: Optional[ExpressionNode]
    upper_bound: Optional[ExpressionNode]
    stride: Optional[ExpressionNode]

    def __init__(self, context: SyntaxContext, lower_bound: Optional[ExpressionNode],
                 upper_bound: Optional[ExpressionNode], stride: Optional[ExpressionNode], location: Location):
        super(SliceArgumentNode, self).__init__(context, location)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stride = stride

    @property
    def children(self) -> Sequence[SyntaxNode]:
        return make_sequence([self.lower_bound, self.upper_bound, self.stride])


def make_sequence(sequence: Iterable[object]) -> Sequence[SyntaxNode]:
    result = []
    for item in sequence:
        if item is None:
            continue

        elif isinstance(item, SyntaxNode):
            result.append(item)

        elif isinstance(item, Sequence):
            for child in item:
                result.append(child)

        else:
            raise NotImplementedError
    return result
