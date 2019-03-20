# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
import enum
import weakref
from typing import Sequence, Iterator, Optional, Union, Iterable, TypeVar, Callable, Mapping

from multidict import MultiDict

from orcinus.diagnostics import DiagnosticManager
from orcinus.locations import Location, Position
from orcinus.utils import cached_property


@enum.unique
class TokenID(enum.IntEnum):
    Accent = enum.auto()
    Ampersand = enum.auto()
    AmpersandEqual = enum.auto()
    And = enum.auto()
    As = enum.auto()
    At = enum.auto()
    AtEqual = enum.auto()
    Break = enum.auto()
    Circumflex = enum.auto()
    CircumflexEqual = enum.auto()
    Class = enum.auto()
    Colon = enum.auto()
    Comma = enum.auto()
    Comment = enum.auto()
    Continue = enum.auto()
    Def = enum.auto()
    Dot = enum.auto()
    Double = enum.auto()
    DoubleCircumflex = enum.auto()
    DoubleSlash = enum.auto()
    DoubleSlashEqual = enum.auto()
    DoubleStar = enum.auto()
    DoubleStarEqual = enum.auto()
    Elif = enum.auto()
    Ellipsis = enum.auto()
    Else = enum.auto()
    EndOfFile = enum.auto()
    Enum = enum.auto()
    DoubleEqual = enum.auto()
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
    LeftShift = enum.auto()
    LeftShiftEqual = enum.auto()
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
    Percent = enum.auto()
    PercentEqual = enum.auto()
    Plus = enum.auto()
    PlusEqual = enum.auto()
    Raise = enum.auto()
    Return = enum.auto()
    RightCurly = enum.auto()
    RightParenthesis = enum.auto()
    RightSquare = enum.auto()
    RightShift = enum.auto()
    RightShiftEqual = enum.auto()
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
    VerticalLine = enum.auto()
    VerticalLineEqual = enum.auto()
    While = enum.auto()
    Whitespace = enum.auto()
    With = enum.auto()
    Yield = enum.auto()


class SyntaxSymbol(abc.ABC):
    @property
    @abc.abstractmethod
    def location(self) -> Location:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parent(self) -> SyntaxNode:
        raise NotImplementedError

    @parent.setter
    @abc.abstractmethod
    def parent(self, value: SyntaxNode):
        raise NotImplementedError

    @property
    def begin_position(self) -> Position:
        return self.location.begin

    @property
    def end_position(self) -> Position:
        return self.location.end


class SyntaxToken(SyntaxSymbol):
    __id: TokenID
    __value: str
    __location: Location

    def __init__(self, token_id: TokenID, value: str, location: Location):
        self.__id = token_id
        self.__value = value
        self.__location = location
        self.__parent = None

    @property
    def id(self) -> TokenID:
        return self.__id

    @property
    def value(self) -> str:
        return self.__value

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def parent(self) -> SyntaxNode:
        return self.__parent and self.__parent()

    @parent.setter
    def parent(self, parent: Optional[SyntaxNode]):
        if parent:
            self.__parent = weakref.ref(parent)
        else:
            self.__parent = None

    def __eq__(self, other: SyntaxToken):
        if not isinstance(other, SyntaxToken):
            return False

        return self.id == other.id and self.value == other.value

    def __str__(self):
        return f'{self.id.name}: {self.value} [{self.location}]'

    def __repr__(self):
        class_name = type(self).__name__
        return f'<{class_name}: {self}>'


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

    def __iter__(self):
        return iter(self.__defines)


class SyntaxContext:
    def __init__(self, diagnostics: DiagnosticManager = None):
        self.__diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
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
            if isinstance(child, SyntaxSymbol):
                child.parent = parent
                if isinstance(child, SyntaxNode):
                    self.annotate(child)
            else:
                raise RuntimeError('Required syntax node')

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


class SyntaxNode(SyntaxSymbol, abc.ABC):
    __context: weakref.ReferenceType
    __parent: Optional[weakref.ReferenceType]

    def __init__(self, context: SyntaxContext):
        self.__context = weakref.ref(context)
        self.__parent = None

    @property
    def context(self) -> SyntaxContext:
        context = self.__context()
        if context is None:
            raise RuntimeError('Syntax context is disposed')
        return context

    @property
    def parent(self) -> Optional[SyntaxNode]:
        return self.__parent and self.__parent()

    @parent.setter
    def parent(self, parent: Optional[SyntaxNode]):
        if parent:
            self.__parent = weakref.ref(parent)
        else:
            self.__parent = None

    @property
    def begin_position(self) -> Position:
        if not self.children:
            return self.location.begin
        return self.children[0].location.begin

    @property
    def end_position(self) -> Position:
        if not self.children:
            return self.location.end
        return self.children[-1].location.end

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
    def children(self) -> Sequence[SyntaxSymbol]:
        raise NotImplementedError

    @cached_property
    def nodes(self) -> Sequence[SyntaxNode]:
        return tuple(child for child in self.children if isinstance(child, SyntaxNode))

    @cached_property
    def tokens(self) -> Sequence[SyntaxToken]:
        return tuple(child for child in self.children if isinstance(child, SyntaxToken))

    def find_closest(self, predicate: Callable[[SyntaxNode], bool]) -> Optional[SyntaxNode]:
        return next(self.find_ascendants(predicate), None)

    def find_ascendants(self, predicate: Callable[[SyntaxNode], bool]) -> Iterator[SyntaxNode]:
        parent = self.parent
        while parent:
            if predicate(parent):
                yield parent
            parent = parent.parent

    def find_descendants(self, predicate: Callable[[SyntaxNode], bool]) -> Iterator[SyntaxNode]:
        for child in self:
            if predicate(child):
                yield child
            yield from child.find_descendants(predicate)

    def find_contains(self, position: Position) -> Optional[SyntaxSymbol]:
        """ Find syntax symbol contains position """
        for child in self.children:
            if child.begin_position <= position <= child.end_position:
                if isinstance(child, SyntaxNode):
                    return child.find_contains(position)
                return child

        if self.begin_position <= position <= self.end_position:
            return self
        return None

    def __iter__(self) -> Iterator[SyntaxNode]:
        return iter(self.nodes)

    def __hash__(self) -> int:
        return id(self)

    def __str__(self):
        return f'{type(self).__name__} [{self.location}]'

    def __repr__(self):
        return f'<{type(self).__name__}: {self} [{self.location}]>'


K = TypeVar('K')
T = TypeVar('T')


class SyntaxCollection(Sequence[T]):
    __children: Sequence[SyntaxSymbol]

    def __init__(self, items: Iterable[SyntaxSymbol] = None):
        self.__children = tuple(item for item in items if isinstance(item, SyntaxSymbol)) if items else ()

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self.__children

    @cached_property
    def nodes(self) -> Sequence[T]:
        return tuple(child for child in self.children if isinstance(child, SyntaxNode))

    @cached_property
    def tokens(self) -> Sequence[SyntaxToken]:
        return tuple(child for child in self.children if isinstance(child, SyntaxToken))

    def __getitem__(self, i: int) -> T:
        return self.nodes[i]

    def __len__(self) -> int:
        return len(self.nodes)


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
                 members: SyntaxCollection[MemberNode], token_eof: SyntaxToken):
        super(SyntaxTree, self).__init__(context)

        self.name = name
        self.members = members
        self.imports = imports
        self.token_eof = token_eof

    @cached_property
    def location(self) -> Location:
        return Location(self.token_eof.location.filename)

    @property
    def filename(self) -> str:
        return self.location.filename

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.imports, self.members])


class OverloadNode(SyntaxNode):
    def __init__(self,
                 context: SyntaxContext,
                 name: str,
                 functions: SyntaxCollection[FunctionNode],
                 location: Location):
        super(OverloadNode, self).__init__(context)

        self.name = name
        self.functions = functions
        self.__location = location

        assert all(isinstance(node, FunctionNode) for node in functions)

    def extend(self, other: OverloadNode) -> OverloadNode:
        functions = SyntaxCollection[FunctionNode](list(self.functions) + list(other.functions))
        return OverloadNode(self.context, self.name, functions, self.location)

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return []  # What??


class ImportNode(SyntaxNode):
    token_import: SyntaxToken
    aliases: SyntaxCollection[AliasNode]
    token_newline: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_import: SyntaxToken,
                 aliases: SyntaxCollection[AliasNode],
                 token_newline: SyntaxToken
                 ):
        super(ImportNode, self).__init__(context)

        self.token_import = token_import
        self.aliases = aliases
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        return self.token_import.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_import, self.aliases, self.token_newline])


class ImportFromNode(ImportNode):
    token_from: SyntaxToken
    token_name: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_from: SyntaxToken,
                 token_name: SyntaxToken,
                 token_import: SyntaxToken,
                 aliases: SyntaxCollection[AliasNode],
                 token_newline: SyntaxToken
                 ):
        super(ImportFromNode, self).__init__(context, token_import, aliases, token_newline)

        self.token_from = token_from
        self.token_name = token_name

    @property
    def module(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_from, self.token_name, self.token_import, self.aliases, self.token_newline])


class AliasNode(SyntaxNode):
    token_name: SyntaxToken
    token_as: Optional[SyntaxToken]
    token_alias: Optional[SyntaxToken]

    def __init__(self,
                 context: SyntaxContext,
                 token_name: SyntaxToken,
                 token_as: Optional[SyntaxToken],
                 token_alias: Optional[SyntaxToken]):
        super(AliasNode, self).__init__(context)

        self.token_name = token_name
        self.token_as = token_as
        self.token_alias = token_alias

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def alias(self) -> Optional[str]:
        return self.token_alias and self.token_alias.name

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_name, self.token_as, self.token_alias])


class DecoratorNode(SyntaxNode):
    token_at: SyntaxToken
    token_name: SyntaxToken
    token_open: Optional[SyntaxToken]
    arguments: SyntaxCollection[ArgumentNode]
    token_close: Optional[SyntaxToken]
    token_newline: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_at: SyntaxToken,
                 token_name: SyntaxToken,
                 token_open: Optional[SyntaxToken],
                 arguments: SyntaxCollection[ArgumentNode],
                 token_close: Optional[SyntaxToken],
                 token_newline: SyntaxToken):
        super(DecoratorNode, self).__init__(context)

        self.token_at = token_at
        self.token_name = token_name
        self.token_open = token_open
        self.arguments = arguments
        self.token_close = token_close
        self.token_newline = token_newline

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.token_at,
            self.token_name,
            self.token_open,
            self.arguments,
            self.token_close,
            self.token_newline
        ])


class MemberNode(SyntaxNode, abc.ABC):
    pass


class PassMemberNode(MemberNode, abc.ABC):
    token_pass: SyntaxToken
    token_newline: SyntaxToken

    def __init__(self, context: SyntaxContext, token_pass: SyntaxToken, token_newline: SyntaxToken):
        super().__init__(context)

        self.token_pass = token_pass
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        return self.token_pass.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_pass, self.token_newline]


class ParameterNode(SyntaxNode):
    token_name: SyntaxToken
    token_colon: Optional[SyntaxToken]
    type: TypeNode
    token_equal: Optional[SyntaxToken]
    default_value: Optional[ExpressionNode]

    def __init__(self,
                 context: SyntaxContext,
                 token_name: SyntaxToken,
                 token_colon: Optional[SyntaxToken],
                 param_type: TypeNode,
                 token_equal: Optional[SyntaxToken],
                 default_value: Optional[ExpressionNode]):
        super(ParameterNode, self).__init__(context)

        self.token_name = token_name
        self.token_colon = token_colon
        self.type = param_type
        self.token_equal = token_equal
        self.default_value = default_value

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.token_name,
            self.token_colon,
            self.type,
            self.token_equal,
            self.default_value
        ])

    def __str__(self):
        return f'{self.name}: {self.type} [{self.location}]'


class EnumMemberNode(MemberNode):
    token_name: SyntaxToken
    token_equal: SyntaxToken
    token_ellipsis: Optional[SyntaxToken]
    value: Optional[ExpressionNode]
    token_newline: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_name: SyntaxToken,
                 token_equal: SyntaxToken,
                 token_ellipsis: Optional[SyntaxToken],
                 value: Optional[ExpressionNode],
                 token_newline: SyntaxToken):
        super(EnumMemberNode, self).__init__(context)

        self.token_name = token_name
        self.token_equal = token_equal
        self.token_ellipsis = token_ellipsis
        self.value = value
        self.token_newline = token_newline

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_name, self.token_equal, self.token_ellipsis, self.value, self.token_newline])


class FieldNode(MemberNode):
    token_name: SyntaxToken
    token_colon: SyntaxToken
    type: TypeNode
    token_equal: Optional[SyntaxToken]
    default_value: Optional[ExpressionNode]
    token_newline: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_name: SyntaxToken,
                 token_colon: SyntaxToken,
                 filed_type: TypeNode,
                 token_equal: SyntaxToken,
                 default_value: Optional[ExpressionNode],
                 token_newline: SyntaxToken):
        super(FieldNode, self).__init__(context)

        self.token_name = token_name
        self.token_colon = token_colon
        self.type = filed_type
        self.token_equal = token_equal
        self.default_value = default_value
        self.token_newline = token_newline

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.token_name,
            self.token_colon,
            self.type,
            self.token_equal,
            self.default_value,
            self.token_name
        ])


class FunctionNode(MemberNode):
    decorators: SyntaxCollection[DecoratorNode]
    token_def: SyntaxToken
    token_name: SyntaxToken
    generic_parameters: SyntaxCollection[GenericParameterNode]
    parameters: SyntaxCollection[ParameterNode]
    token_then: SyntaxToken
    return_type: TypeNode
    token_colon: SyntaxToken
    statement: StatementNode

    def __init__(self,
                 context: SyntaxContext,
                 decorators: SyntaxCollection[DecoratorNode],
                 token_def: SyntaxToken,
                 token_name: SyntaxToken,
                 generic_parameters: SyntaxCollection[GenericParameterNode],
                 parameters: SyntaxCollection[ParameterNode],
                 token_then: SyntaxToken,
                 return_type: TypeNode,
                 token_colon: SyntaxToken,
                 statement: StatementNode):
        super(FunctionNode, self).__init__(context)

        self.decorators = decorators
        self.token_def = token_def
        self.token_name = token_name
        self.generic_parameters = generic_parameters
        self.parameters = parameters
        self.token_then = token_then
        self.return_type = return_type
        self.token_colon = token_colon
        self.statement = statement

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.decorators,
            self.token_def,
            self.token_name,
            self.generic_parameters,
            self.parameters,
            self.token_then,
            self.return_type,
            self.token_colon,
            self.statement
        ])

    @property
    def is_abstract(self) -> bool:
        return isinstance(self.statement, EllipsisStatementNode)


class TypeDeclarationNode(MemberNode, abc.ABC):
    decorators: SyntaxCollection[DecoratorNode]
    token_name: SyntaxToken
    generic_parameters: SyntaxCollection[GenericParameterNode]
    parents: SyntaxCollection[TypeNode]
    token_colon: SyntaxToken
    token_newline: SyntaxToken
    token_indent: SyntaxToken
    members: SyntaxCollection[MemberNode]
    token_undent: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 decorators: SyntaxCollection[DecoratorNode],
                 token_name: SyntaxToken,
                 generic_parameters: SyntaxCollection[GenericParameterNode],
                 parents: SyntaxCollection[TypeNode],
                 token_colon: SyntaxToken,
                 token_newline: SyntaxToken,
                 token_indent: SyntaxToken,
                 members: SyntaxCollection[MemberNode],
                 token_undent: SyntaxToken):
        super(TypeDeclarationNode, self).__init__(context)

        self.decorators = decorators
        self.token_name = token_name
        self.generic_parameters = generic_parameters
        self.parents = parents
        self.token_colon = token_colon
        self.token_newline = token_newline
        self.token_indent = token_indent
        self.members = members
        self.token_undent = token_undent

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def name(self) -> str:
        return self.token_name.value


class ClassNode(TypeDeclarationNode):
    token_class: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 decorators: SyntaxCollection[DecoratorNode],
                 token_class: SyntaxToken,
                 token_name: SyntaxToken,
                 generic_parameters: SyntaxCollection[GenericParameterNode],
                 parents: SyntaxCollection[TypeNode],
                 token_colon: SyntaxToken,
                 token_newline: SyntaxToken,
                 token_indent: SyntaxToken,
                 members: SyntaxCollection[MemberNode],
                 token_undent: SyntaxToken):
        super().__init__(
            context,
            decorators,
            token_name,
            generic_parameters,
            parents,
            token_colon,
            token_newline,
            token_indent,
            members,
            token_undent
        )

        self.token_class = token_class

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.decorators,
            self.token_class,
            self.token_name,
            self.generic_parameters,
            self.parents,
            self.token_colon,
            self.token_newline,
            self.token_indent,
            self.members,
            self.token_undent
        ])


class StructNode(TypeDeclarationNode):
    token_struct: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 decorators: SyntaxCollection[DecoratorNode],
                 token_struct: SyntaxToken,
                 token_name: SyntaxToken,
                 generic_parameters: SyntaxCollection[GenericParameterNode],
                 parents: SyntaxCollection[TypeNode],
                 token_colon: SyntaxToken,
                 token_newline: SyntaxToken,
                 token_indent: SyntaxToken,
                 members: SyntaxCollection[MemberNode],
                 token_undent: SyntaxToken):
        super().__init__(
            context,
            decorators,
            token_name,
            generic_parameters,
            parents,
            token_colon,
            token_newline,
            token_indent,
            members,
            token_undent
        )

        self.token_struct = token_struct

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.decorators,
            self.token_struct,
            self.token_name,
            self.generic_parameters,
            self.parents,
            self.token_colon,
            self.token_newline,
            self.token_indent,
            self.members,
            self.token_undent
        ])


class InterfaceNode(TypeDeclarationNode):
    token_interface: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 decorators: SyntaxCollection[DecoratorNode],
                 token_interface: SyntaxToken,
                 token_name: SyntaxToken,
                 generic_parameters: SyntaxCollection[GenericParameterNode],
                 parents: SyntaxCollection[TypeNode],
                 token_colon: SyntaxToken,
                 token_newline: SyntaxToken,
                 token_indent: SyntaxToken,
                 members: SyntaxCollection[MemberNode],
                 token_undent: SyntaxToken):
        super().__init__(
            context,
            decorators,
            token_name,
            generic_parameters,
            parents,
            token_colon,
            token_newline,
            token_indent,
            members,
            token_undent
        )

        self.token_interface = token_interface

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.decorators,
            self.token_interface,
            self.token_name,
            self.generic_parameters,
            self.parents,
            self.token_colon,
            self.token_newline,
            self.token_indent,
            self.members,
            self.token_undent
        ])


class EnumNode(TypeDeclarationNode):
    token_enum: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 decorators: SyntaxCollection[DecoratorNode],
                 token_enum: SyntaxToken,
                 token_name: SyntaxToken,
                 generic_parameters: SyntaxCollection[GenericParameterNode],
                 parents: SyntaxCollection[TypeNode],
                 token_colon: SyntaxToken,
                 token_newline: SyntaxToken,
                 token_indent: SyntaxToken,
                 members: SyntaxCollection[MemberNode],
                 token_undent: SyntaxToken):
        super().__init__(
            context,
            decorators,
            token_name,
            generic_parameters,
            parents,
            token_colon,
            token_newline,
            token_indent,
            members,
            token_undent
        )

        self.token_enum = token_enum

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.decorators,
            self.token_enum,
            self.token_name,
            self.generic_parameters,
            self.parents,
            self.token_colon,
            self.token_newline,
            self.token_indent,
            self.members,
            self.token_undent
        ])


class TypeNode(SyntaxNode, abc.ABC):
    pass


class AutoTypeNode(TypeNode):
    token_auto: SyntaxToken

    def __init__(self, context: SyntaxContext, token_auto: SyntaxToken):
        super(AutoTypeNode, self).__init__(context)

        self.token_auto = token_auto

    @property
    def location(self) -> Location:
        return self.token_auto.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_auto]

    def __str__(self) -> str:
        return 'auto [{self.location}]'


class NamedTypeNode(TypeNode):
    token_name: SyntaxToken

    def __init__(self, context: SyntaxContext, token_name: SyntaxToken):
        super(NamedTypeNode, self).__init__(context)

        self.token_name = token_name

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_name]

    def __str__(self) -> str:
        return f'{self.name} [{self.location}]'


class GenericParameterNode(TypeNode):
    token_name: SyntaxToken

    def __init__(self, context: SyntaxContext, token_name: SyntaxToken):
        super(GenericParameterNode, self).__init__(context)

        self.token_name = token_name

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_name]

    def __str__(self) -> str:
        return f'{self.name} [{self.location}]'


class ParameterizedTypeNode(TypeNode):
    type: TypeNode
    arguments: SyntaxCollection[TypeNode]

    def __init__(self, context: SyntaxContext, basis: TypeNode, arguments: SyntaxCollection[TypeNode]):
        super(ParameterizedTypeNode, self).__init__(context)

        self.type = basis
        self.arguments = arguments

    @cached_property
    def location(self) -> Location:
        if self.arguments.children:
            return self.type.location + self.arguments.children[-1].location
        return self.type.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.type, self.arguments])

    def __str__(self) -> str:
        arguments = ', '.join(map(str, self.arguments))
        return f'{self.type}[{arguments}] [{self.location}]'


class StatementNode(SyntaxNode, abc.ABC):
    pass


class EllipsisStatementNode(StatementNode):
    token_ellipsis: SyntaxToken
    token_newline: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_ellipsis: SyntaxToken,
                 token_newline: SyntaxToken):
        super(EllipsisStatementNode, self).__init__(context)

        self.token_ellipsis = token_ellipsis
        self.token_newline = token_newline

    @cached_property
    def location(self) -> Location:
        return self.token_ellipsis.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_ellipsis, self.token_newline])


class BlockStatementNode(StatementNode):
    token_newline: SyntaxToken
    token_indent: SyntaxToken
    statements: SyntaxCollection[StatementNode]
    token_undent: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_newline: SyntaxToken,
                 token_indent: SyntaxToken,
                 statements: SyntaxCollection[StatementNode],
                 token_undent: SyntaxToken):
        super(BlockStatementNode, self).__init__(context)

        self.token_newline = token_newline
        self.token_indent = token_indent
        self.statements = statements
        self.token_undent = token_undent

    @cached_property
    def location(self) -> Location:
        return self.token_indent.location + self.token_undent.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.token_newline,
            self.token_indent,
            self.statements,
            self.token_undent
        ])


class ElseStatementNode(StatementNode):
    token_else: SyntaxToken
    token_colon: SyntaxToken
    statement: StatementNode

    def __init__(self,
                 context: SyntaxContext,
                 token_else: SyntaxToken,
                 token_colon: SyntaxToken,
                 statement: StatementNode):
        super().__init__(context)

        self.token_else = token_else
        self.token_colon = token_colon
        self.statement = statement

    @property
    def location(self) -> Location:
        return self.token_else.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_else, self.token_colon, self.statement])


class FinallyStatementNode(StatementNode):
    token_finally: SyntaxToken
    token_colon: SyntaxToken
    statement: StatementNode

    def __init__(self,
                 context: SyntaxContext,
                 token_finally: SyntaxToken,
                 token_colon: SyntaxToken,
                 statement: StatementNode):
        super().__init__(context)

        self.token_finally = token_finally
        self.token_colon = token_colon
        self.statement = statement

    @property
    def location(self) -> Location:
        return self.token_finally.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_finally, self.token_colon, self.statement])


class PassStatementNode(StatementNode):
    token_pass: SyntaxToken
    token_newline: SyntaxToken

    def __init__(self, context: SyntaxContext, token_pass: SyntaxToken, token_newline: SyntaxToken):
        super().__init__(context)

        self.token_pass = token_pass
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        return self.token_pass.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_pass, self.token_newline]


class BreakStatementNode(StatementNode):
    token_break: SyntaxToken
    token_newline: SyntaxToken

    def __init__(self, context: SyntaxContext, token_break: SyntaxToken, token_newline: SyntaxToken):
        super(BreakStatementNode, self).__init__(context)

        self.token_break = token_break
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        return self.token_break.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_break, self.token_newline]


class ContinueStatementNode(StatementNode):
    token_continue: SyntaxToken
    token_newline: SyntaxToken

    def __init__(self, context: SyntaxContext, token_continue: SyntaxToken, token_newline: SyntaxToken):
        super(ContinueStatementNode, self).__init__(context)

        self.token_continue = token_continue
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        return self.token_continue.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_continue, self.token_newline]


class ReturnStatementNode(StatementNode):
    token_return: SyntaxToken
    value: Optional[ExpressionNode]
    token_newline: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_return: SyntaxToken,
                 value: Optional[ExpressionNode],
                 token_newline: SyntaxToken):
        super(ReturnStatementNode, self).__init__(context)

        self.token_return = token_return
        self.value = value
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        return self.token_return.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_return, self.value, self.token_newline])


class YieldStatementNode(StatementNode):
    token_yield: SyntaxToken
    value: Optional[ExpressionNode]
    token_newline: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_yield: SyntaxToken,
                 value: Optional[ExpressionNode],
                 token_newline: SyntaxToken):
        super(YieldStatementNode, self).__init__(context)

        self.token_yield = token_yield
        self.value = value
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        yield self.token_yield.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_yield, self.value, self.token_newline])


class AssignStatementNode(StatementNode):
    target: ExpressionNode
    token_equal: SyntaxToken
    source: ExpressionNode
    token_newline: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 target: ExpressionNode,
                 token_equal: SyntaxToken,
                 source: ExpressionNode,
                 token_newline: SyntaxToken):
        super(AssignStatementNode, self).__init__(context)

        self.target = target
        self.token_equal = token_equal
        self.source = source
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        return self.token_equal.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.target, self.token_equal, self.source, self.token_newline])


class AugmentedAssignStatementNode(StatementNode):
    target: ExpressionNode
    token_operator: SyntaxToken
    source: ExpressionNode
    token_newline: SyntaxToken

    opcode: BinaryID

    def __init__(self,
                 context: SyntaxContext,
                 target: ExpressionNode,
                 token_operator: SyntaxToken,
                 opcode: BinaryID,
                 source: ExpressionNode,
                 token_newline: SyntaxToken):
        super(AugmentedAssignStatementNode, self).__init__(context)

        self.target = target
        self.token_operator = token_operator
        self.opcode = opcode
        self.source = source
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        return self.token_operator.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.target, self.token_operator, self.source, self.token_newline])


class ConditionStatementNode(StatementNode):
    token_if: SyntaxToken
    condition: ExpressionNode
    token_colon: SyntaxToken
    then_statement: StatementNode
    else_statement: Optional[StatementNode]

    def __init__(self,
                 context: SyntaxContext,
                 token_if: SyntaxToken,
                 condition: ExpressionNode,
                 token_colon: SyntaxToken,
                 then_statement: StatementNode,
                 else_statement: Optional[StatementNode]):
        super(ConditionStatementNode, self).__init__(context)

        self.token_if = token_if
        self.condition = condition
        self.token_colon = token_colon
        self.then_statement = then_statement
        self.else_statement = else_statement

    @property
    def location(self) -> Location:
        return self.token_if.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.token_if,
            self.condition,
            self.token_colon,
            self.then_statement,
            self.else_statement
        ])


class WhileStatementNode(StatementNode):
    token_while: SyntaxToken
    condition: ExpressionNode
    token_colon: SyntaxToken
    then_statement: StatementNode
    else_statement: Optional[StatementNode]

    def __init__(self,
                 context: SyntaxContext,
                 token_while: SyntaxToken,
                 condition: ExpressionNode,
                 token_colon: SyntaxToken,
                 then_statement: StatementNode,
                 else_statement: Optional[StatementNode]):
        super(WhileStatementNode, self).__init__(context)

        self.token_while = token_while
        self.condition = condition
        self.token_colon = token_colon
        self.then_statement = then_statement
        self.else_statement = else_statement

    @property
    def location(self) -> Location:
        return self.token_while.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.token_while,
            self.condition,
            self.token_colon,
            self.then_statement,
            self.else_statement
        ])


class ForStatementNode(StatementNode):
    token_for: SyntaxToken
    target: ExpressionNode
    token_in: SyntaxToken
    source: ExpressionNode
    token_colon: SyntaxToken
    then_statement: StatementNode
    else_statement: Optional[StatementNode]

    def __init__(self,
                 context: SyntaxContext,
                 token_for: SyntaxToken,
                 target: ExpressionNode,
                 token_in: SyntaxToken,
                 source: ExpressionNode,
                 token_colon: SyntaxToken,
                 then_statement: StatementNode,
                 else_statement: Optional[StatementNode]):
        super(ForStatementNode, self).__init__(context)

        self.token_for = token_for
        self.target = target
        self.token_in = token_in
        self.source = source
        self.token_colon = token_colon
        self.then_statement = then_statement
        self.else_statement = else_statement

    @property
    def location(self) -> Location:
        return self.token_for.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.token_for,
            self.target,
            self.token_in,
            self.source,
            self.token_colon,
            self.then_statement,
            self.else_statement
        ])


class RaiseStatementNode(StatementNode):
    token_raise: SyntaxToken
    exception: Optional[ExpressionNode]
    token_from: Optional[SyntaxToken]
    cause_exception: Optional[ExpressionNode]
    token_newline: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_raise: SyntaxToken,
                 exception: Optional[ExpressionNode],
                 token_from: Optional[SyntaxToken],
                 cause_exception: Optional[ExpressionNode],
                 token_newline: SyntaxToken):
        super(RaiseStatementNode, self).__init__(context)

        self.token_raise = token_raise
        self.exception = exception
        self.token_from = token_from
        self.cause_exception = cause_exception
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        return self.token_raise.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.token_raise,
            self.exception,
            self.token_from,
            self.cause_exception,
            self.token_newline
        ])


class ExceptHandlerNode(SyntaxNode):
    token_except: SyntaxToken
    expression: Optional[ExpressionNode]
    token_as: Optional[SyntaxToken]
    token_name: Optional[SyntaxToken]
    token_colon: SyntaxToken
    statement: StatementNode

    def __init__(self,
                 context: SyntaxContext,
                 token_except: SyntaxToken,
                 expression: Optional[ExpressionNode],
                 token_as: Optional[SyntaxToken],
                 token_name: Optional[SyntaxToken],
                 token_colon: SyntaxToken,
                 statement: StatementNode):
        super(ExceptHandlerNode, self).__init__(context)

        self.token_except = token_except
        self.expression = expression
        self.token_as = token_as
        self.token_name = token_name
        self.token_colon = token_colon
        self.statement = statement

    @property
    def location(self) -> Location:
        return self.token_except.location

    @property
    def name(self) -> Optional[str]:
        return self.token_name and self.token_name.value

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.token_except,
            self.expression,
            self.token_as,
            self.token_name,
            self.token_colon,
            self.statement
        ])


class TryStatementNode(StatementNode):
    token_try: SyntaxToken
    token_colon: SyntaxToken
    try_statement: StatementNode
    handlers: SyntaxCollection[ExceptHandlerNode]
    else_statement: Optional[StatementNode]
    finally_statement: Optional[StatementNode]

    def __init__(self,
                 context: SyntaxContext,
                 token_try: SyntaxToken,
                 token_colon: SyntaxToken,
                 try_statement: StatementNode,
                 handlers: SyntaxCollection[ExceptHandlerNode],
                 else_statement: Optional[StatementNode],
                 finally_statement: Optional[StatementNode]):
        super(TryStatementNode, self).__init__(context)

        self.token_try = token_try
        self.token_colon = token_colon
        self.try_statement = try_statement
        self.handlers = handlers
        self.else_statement = else_statement
        self.finally_statement = finally_statement

    @property
    def location(self) -> Location:
        return self.token_try.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.token_try,
            self.token_colon,
            self.try_statement,
            self.handlers,
            self.else_statement,
            self.finally_statement
        ])


class WithItemNode(SyntaxNode):
    expression: ExpressionNode
    token_as: Optional[SyntaxToken]
    target: Optional[ExpressionNode]

    def __init__(self,
                 context: SyntaxContext,
                 expression: ExpressionNode,
                 token_as: SyntaxToken,
                 target: Optional[ExpressionNode]):
        super(WithItemNode, self).__init__(context)

        self.expression = expression
        self.token_as = token_as
        self.target = target

    @property
    def location(self) -> Location:
        return self.expression.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.expression, self.token_as, self.target])


class WithStatementNode(StatementNode):
    token_with: SyntaxToken
    items: SyntaxCollection[WithItemNode]
    token_colon: SyntaxToken
    statement: StatementNode

    def __init__(self,
                 context: SyntaxContext,
                 token_with: SyntaxToken,
                 items: SyntaxCollection[WithItemNode],
                 token_colon: SyntaxToken,
                 statement: StatementNode):
        super(WithStatementNode, self).__init__(context)

        self.token_with = token_with
        self.items = items
        self.token_colon = token_colon
        self.statement = statement

    @property
    def location(self) -> Location:
        return self.token_with.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_with, self.items, self.token_colon, self.statement])


class ExpressionStatementNode(StatementNode):
    value: ExpressionNode
    token_newline: SyntaxToken

    def __init__(self, context: SyntaxContext, value: ExpressionNode, token_newline: SyntaxToken):
        super(ExpressionStatementNode, self).__init__(context)

        self.value = value
        self.token_newline = token_newline

    @property
    def location(self) -> Location:
        return self.value.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.value, self.token_newline])


class ExpressionNode(SyntaxNode, abc.ABC):
    pass


class IntegerExpressionNode(ExpressionNode):
    token_number: SyntaxToken

    def __init__(self, context: SyntaxContext, token_number: SyntaxToken):
        super(IntegerExpressionNode, self).__init__(context)

        self.token_number = token_number

    @property
    def location(self) -> Location:
        return self.token_number.location

    @property
    def value(self) -> int:
        return int(self.token_number.value)

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_number]


class StringExpressionNode(ExpressionNode):
    token_string: SyntaxToken

    def __init__(self, context: SyntaxContext, token_string: SyntaxToken):
        super(StringExpressionNode, self).__init__(context)

        self.token_string = token_string

    @property
    def location(self) -> Location:
        return self.token_string.location

    @property
    def value(self) -> str:
        return self.token_string.value[1:-1]

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_string]


class NamedExpressionNode(ExpressionNode):
    token_name: SyntaxToken

    def __init__(self, context: SyntaxContext, token_name: SyntaxToken):
        super(NamedExpressionNode, self).__init__(context)

        self.token_name = token_name

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_name]


class AttributeExpressionNode(ExpressionNode):
    instance: ExpressionNode
    token_dot: SyntaxToken
    token_name: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 instance: ExpressionNode,
                 token_dot: SyntaxToken,
                 token_name: SyntaxToken):
        super(AttributeExpressionNode, self).__init__(context)

        self.instance = instance
        self.token_dot = token_dot
        self.token_name = token_name

    @property
    def name(self) -> str:
        return self.token_name.value

    @cached_property
    def location(self) -> Location:
        return self.instance.location + self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.instance, self.token_dot, self.token_name]


@enum.unique
class UnaryID(enum.IntEnum):
    Not = enum.auto()
    Pos = enum.auto()
    Neg = enum.auto()
    Inv = enum.auto()


class UnaryExpressionNode(ExpressionNode):
    token_operator: SyntaxToken
    opcode: UnaryID
    argument: ExpressionNode

    def __init__(self, context: SyntaxContext, token_operator: SyntaxToken, opcode: UnaryID, argument: ExpressionNode):
        super(UnaryExpressionNode, self).__init__(context)

        self.token_operator = token_operator
        self.opcode = opcode
        self.argument = argument

    @property
    def location(self) -> Location:
        return self.token_operator.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_operator, self.argument]


@enum.unique
class BinaryID(enum.IntEnum):
    Add = enum.auto()
    Sub = enum.auto()
    Mul = enum.auto()
    MatrixMul = enum.auto()
    TrueDiv = enum.auto()
    FloorDiv = enum.auto()
    Mod = enum.auto()
    Pow = enum.auto()
    And = enum.auto()
    Or = enum.auto()
    Xor = enum.auto()
    LeftShift = enum.auto()
    RightShift = enum.auto()


class BinaryExpressionNode(ExpressionNode):
    left_argument: ExpressionNode
    token_operator: SyntaxToken
    opcode: BinaryID
    right_argument: ExpressionNode

    def __init__(self,
                 context: SyntaxContext,
                 left_argument: ExpressionNode,
                 token_operator: SyntaxToken,
                 opcode: BinaryID,
                 right_argument: ExpressionNode):
        super(BinaryExpressionNode, self).__init__(context)

        self.left_argument = left_argument
        self.token_operator = token_operator
        self.opcode = opcode
        self.right_argument = right_argument

    @property
    def location(self) -> Location:
        return self.token_operator.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.left_argument, self.token_operator, self.right_argument])


@enum.unique
class LogicID(enum.IntEnum):
    And = enum.auto()
    Or = enum.auto()


class LogicExpressionNode(ExpressionNode):
    left_argument: ExpressionNode
    token_operator: SyntaxToken
    opcode: LogicID
    right_argument: ExpressionNode

    def __init__(self,
                 context: SyntaxContext,
                 left_argument: ExpressionNode,
                 token_operator: SyntaxToken,
                 opcode: LogicID,
                 right_argument: ExpressionNode):
        super(LogicExpressionNode, self).__init__(context)

        self.left_argument = left_argument
        self.opcode = opcode
        self.token_operator = token_operator
        self.right_argument = right_argument

    @property
    def location(self) -> Location:
        return self.token_operator.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.left_argument, self.token_operator, self.right_argument]


class CompareExpressionNode(ExpressionNode):
    left_argument: ExpressionNode
    comparators: SyntaxCollection[ComparatorNode]

    def __init__(self,
                 context: SyntaxContext,
                 left_argument: ExpressionNode,
                 comparators: SyntaxCollection[ComparatorNode]):
        super(CompareExpressionNode, self).__init__(context)

        self.left_argument = left_argument
        self.comparators = comparators

    @property
    def location(self) -> Location:
        return self.comparators[0].location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.left_argument, self.comparators])


@enum.unique
class CompareID(enum.IntEnum):
    In = enum.auto()
    Is = enum.auto()
    NotIn = enum.auto()
    IsNot = enum.auto()
    Equal = enum.auto()
    NotEqual = enum.auto()
    Less = enum.auto()
    LessEqual = enum.auto()
    Great = enum.auto()
    GreatEqual = enum.auto()


class ComparatorNode(SyntaxNode):
    token_prefix: Optional[SyntaxToken]
    token_suffix: Optional[SyntaxToken]
    opcode: CompareID
    right_argument: ExpressionNode

    def __init__(self,
                 context: SyntaxContext,
                 token_prefix: Optional[SyntaxToken],
                 token_suffix: Optional[SyntaxToken],
                 opcode: CompareID,
                 right_argument: ExpressionNode):
        super(ComparatorNode, self).__init__(context)

        self.token_prefix = token_prefix
        self.token_suffix = token_suffix
        self.opcode = opcode
        self.right_argument = right_argument

    @cached_property
    def location(self) -> Location:
        tokens = list(filter(None, (self.token_prefix, self.token_suffix)))
        return tokens[0].location + tokens[-1].location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_prefix, self.token_suffix, self.right_argument])


class ConditionExpressionNode(ExpressionNode):
    condition: ExpressionNode
    token_if: SyntaxToken
    then_value: ExpressionNode
    token_else: SyntaxToken
    else_value: ExpressionNode

    def __init__(self,
                 context: SyntaxContext,
                 then_value: ExpressionNode,
                 token_if: SyntaxToken,
                 condition: ExpressionNode,
                 token_else: SyntaxToken,
                 else_value: ExpressionNode):
        super(ConditionExpressionNode, self).__init__(context)

        self.then_value = then_value
        self.token_if = token_if
        self.condition = condition
        self.token_else = token_else
        self.else_value = else_value

    @property
    def location(self) -> Location:
        return self.token_if.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.then_value, self.token_if, self.condition, self.token_else, self.else_value]


class CallExpressionNode(ExpressionNode):
    instance: ExpressionNode
    token_open: SyntaxToken
    arguments: SyntaxCollection[ExpressionNode]
    keywords: SyntaxDictionary[str, ExpressionNode]
    token_close: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 instance: ExpressionNode,
                 token_open: SyntaxToken,
                 arguments: SyntaxCollection[ArgumentNode],
                 token_close: SyntaxToken):
        super(CallExpressionNode, self).__init__(context)

        self.instance = instance
        self.token_open = token_open
        self.__arguments = arguments
        self.arguments = SyntaxCollection[ExpressionNode](
            [arg.value for arg in arguments if isinstance(arg, PositionArgumentNode)])
        self.keywords = SyntaxDictionary[str, ExpressionNode]({
            arg.name: arg.value for arg in arguments if isinstance(arg, KeywordArgumentNode)
        })
        self.token_close = token_close

    @cached_property
    def location(self) -> Location:
        return self.instance.location + self.token_close.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.instance, self.token_open, self.__arguments, self.token_close])


class SubscribeExpressionNode(ExpressionNode):
    instance: ExpressionNode
    arguments: SyntaxCollection[Union[ExpressionNode, SliceArgumentNode]]

    def __init__(self, context: SyntaxContext, instance: ExpressionNode,
                 arguments: SyntaxCollection[Union[ExpressionNode, SliceArgumentNode]], location: Location):
        super(SubscribeExpressionNode, self).__init__(context)

        self.instance = instance
        self.arguments = arguments

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.instance, self.arguments])


class ArrayExpressionNode(ExpressionNode):
    token_open: SyntaxToken
    arguments: SyntaxCollection[ExpressionNode]
    token_close: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_open: SyntaxToken,
                 arguments: SyntaxCollection[ExpressionNode],
                 token_close: SyntaxToken):
        super().__init__(context)

        self.token_open = token_open
        self.arguments = arguments
        self.token_close = token_close

    @cached_property
    def location(self) -> Location:
        return self.token_open.location + self.token_close.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_open, self.arguments, self.token_close])


class SetExpressionNode(ExpressionNode):
    token_open: SyntaxToken
    arguments: SyntaxCollection[ExpressionNode]
    token_close: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_open: SyntaxToken,
                 arguments: SyntaxCollection[ExpressionNode],
                 token_close: SyntaxToken):
        super().__init__(context)

        self.token_open = token_open
        self.arguments = arguments
        self.token_close = token_close

    @cached_property
    def location(self) -> Location:
        return self.token_open.location + self.token_close.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_open, self.arguments, self.token_close])


class TupleExpressionNode(ExpressionNode):
    token_open: Optional[SyntaxToken]
    arguments: SyntaxCollection[ExpressionNode]
    token_close: Optional[SyntaxToken]

    def __init__(self,
                 context: SyntaxContext,
                 token_open: Optional[SyntaxToken],
                 arguments: SyntaxCollection[ExpressionNode],
                 token_close: Optional[SyntaxToken]):
        super().__init__(context)

        self.token_open = token_open
        self.arguments = arguments
        self.token_close = token_close

    @cached_property
    def location(self) -> Location:
        if self.token_open:
            return self.token_open.location + self.token_close.location
        return self.arguments[0].location + self.arguments[-1].location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_open, self.arguments, self.token_close])


class DictExpressionNode(ExpressionNode):
    token_open: SyntaxToken
    arguments: SyntaxCollection[DictArgumentNode]
    token_close: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_open: SyntaxToken,
                 arguments: SyntaxCollection[DictArgumentNode],
                 token_close: SyntaxToken):
        super().__init__(context)

        self.token_open = token_open
        self.arguments = arguments
        self.token_close = token_close

    @cached_property
    def location(self) -> Location:
        return self.token_open.location + self.token_close.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_open, self.arguments, self.token_close])


class DictArgumentNode(SyntaxNode):
    key: ExpressionNode
    token_colon: SyntaxToken
    value: ExpressionNode

    def __init__(self, context: SyntaxContext, key: ExpressionNode, token_colon: SyntaxToken, value: ExpressionNode):
        super(DictArgumentNode, self).__init__(context)

        self.key = key
        self.token_colon = token_colon
        self.value = value

    @cached_property
    def location(self) -> Location:
        return self.key.location + self.value.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.key, self.token_colon, self.value])


class ArgumentNode(SyntaxNode, abc.ABC):
    value: ExpressionNode

    def __init__(self, context: SyntaxContext, value: ExpressionNode, location: Location):
        super(ArgumentNode, self).__init__(context)

        self.value = value

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
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
        super(SliceArgumentNode, self).__init__(context)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stride = stride

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.lower_bound, self.upper_bound, self.stride])


def make_sequence(sequence: Iterable[object]) -> Sequence[SyntaxNode]:
    result = []
    for item in sequence:
        if item is None:
            continue

        elif isinstance(item, SyntaxSymbol):
            result.append(item)

        elif isinstance(item, SyntaxCollection):
            for child in item.children:
                result.append(child)

        else:
            raise NotImplementedError
    return result
