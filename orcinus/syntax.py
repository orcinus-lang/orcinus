# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
import ast
import enum
import weakref
from typing import Sequence, Iterator, Optional, Union, Iterable, TypeVar, Callable, Mapping, Generic

from orcinus.diagnostics import DiagnosticManager
from orcinus.exceptions import DiagnosticError
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
    Float = enum.auto()
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


class SyntaxContext:
    def __init__(self, diagnostics: DiagnosticManager = None):
        self.__diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
        self.__parents = {}

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.__diagnostics

    def get_parent(self, node: SyntaxNode) -> Optional[SyntaxNode]:
        return self.__parents[node]

    def annotate(self, parent: SyntaxNode):
        if not isinstance(parent, SyntaxNode):
            raise RuntimeError('Required syntax node')

        for child in parent.children:
            if isinstance(child, SyntaxSymbol):
                child.parent = parent
                if isinstance(child, SyntaxNode):
                    self.annotate(child)
            else:
                raise RuntimeError('Required syntax node')


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
    arguments: SyntaxCollection[ExpressionNode]
    keywords: SyntaxDictionary[str, ExpressionNode]
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
        self.__arguments = arguments
        self.arguments = SyntaxCollection[ExpressionNode](
            [arg.value for arg in arguments if isinstance(arg, PositionArgumentNode)])
        self.keywords = SyntaxDictionary[str, ExpressionNode]({
            arg.name: arg.value for arg in arguments if isinstance(arg, KeywordArgumentNode)
        })
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
            self.__arguments,
            self.token_close,
            self.token_newline
        ])


class MemberNode(SyntaxNode, abc.ABC):
    pass


class PassMemberNode(MemberNode):
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


class EnumValueNode(MemberNode):
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
        super(EnumValueNode, self).__init__(context)

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
    token_colon: SyntaxToken
    concepts: SyntaxCollection[TypeNode]

    def __init__(self,
                 context: SyntaxContext,
                 token_name: SyntaxToken,
                 token_colon: SyntaxToken,
                 concepts: SyntaxCollection[TypeNode]):
        super(GenericParameterNode, self).__init__(context)

        self.token_name = token_name
        self.token_colon = token_colon
        self.concepts = concepts

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([self.token_name, self.token_colon, self.concepts])

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


class VariableStatementNode(StatementNode):
    node_name: ExpressionNode
    token_colon: SyntaxToken
    type: TypeNode
    token_equal: SyntaxToken
    initial_value: ExpressionNode
    token_newline: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 node_name: ExpressionNode,
                 token_colon: SyntaxToken,
                 var_type: TypeNode,
                 token_equal: SyntaxToken,
                 initial_value: ExpressionNode,
                 token_newline: SyntaxToken):
        super().__init__(context)

        self.node_name = node_name
        self.token_colon = token_colon
        self.type = var_type
        self.token_equal = token_equal
        self.initial_value = initial_value
        self.token_newline = token_newline

    @property
    def name(self) -> str:
        if isinstance(self.node_name, NamedExpressionNode):
            return self.node_name.name
        return ""

    @property
    def location(self) -> Location:
        return self.node_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return make_sequence([
            self.node_name,
            self.token_colon,
            self.type,
            self.token_equal,
            self.initial_value,
            self.token_newline
        ])


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


class FloatExpressionNode(ExpressionNode):
    token_number: SyntaxToken

    def __init__(self, context: SyntaxContext, token_number: SyntaxToken):
        super(FloatExpressionNode, self).__init__(context)

        self.token_number = token_number

    @property
    def location(self) -> Location:
        return self.token_number.location

    @property
    def value(self) -> float:
        return float(self.token_number.value)

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
    def raw_value(self) -> str:
        return self.token_string.value

    @property
    def value(self) -> str:
        return ast.literal_eval(self.raw_value)

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
    token_open: SyntaxToken
    arguments: SyntaxCollection[Union[ExpressionNode, SliceArgumentNode]]
    token_close: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 instance: ExpressionNode,
                 token_open: SyntaxToken,
                 arguments: SyntaxCollection[Union[ExpressionNode, SliceArgumentNode]],
                 token_close: SyntaxToken):
        super(SubscribeExpressionNode, self).__init__(context)

        self.instance = instance
        self.token_open = token_open
        self.arguments = arguments
        self.token_close = token_close

    @property
    def location(self) -> Location:
        return self.instance.location + self.token_close.location

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

    def __init__(self, context: SyntaxContext, value: ExpressionNode):
        super(ArgumentNode, self).__init__(context)

        self.value = value


class PositionArgumentNode(ArgumentNode):
    @property
    def location(self) -> Location:
        return self.value.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.value]


class KeywordArgumentNode(ArgumentNode):
    token_name: SyntaxToken
    token_equal: SyntaxToken

    def __init__(self,
                 context: SyntaxContext,
                 token_name: SyntaxToken,
                 token_equal: SyntaxToken,
                 value: ExpressionNode):
        super(KeywordArgumentNode, self).__init__(context, value)

        self.token_name = token_name
        self.token_equal = token_equal

    @property
    def name(self) -> str:
        return self.token_name.value

    @property
    def location(self) -> Location:
        return self.token_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.token_name, self.token_equal, self.value]


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


R = TypeVar('R')


class AbstractImportVisitor(Generic[R], abc.ABC):
    @abc.abstractmethod
    def visit_import(self, node: ImportNode) -> R:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_from_import(self, node: ImportFromNode) -> R:
        raise NotImplementedError


class ImportVisitor(AbstractImportVisitor[R], abc.ABC):
    def visit(self, node: ImportNode) -> R:
        if isinstance(node, ImportFromNode):
            return self.visit_from_import(node)
        elif isinstance(node, ImportNode):
            return self.visit_import(node)

        raise DiagnosticError(node.location, f"Not implemented visitor for {type(node).__name__}")


class AbstractTypeVisitor(Generic[R], abc.ABC):
    @abc.abstractmethod
    def visit_type(self, node: TypeNode) -> R:
        raise NotImplementedError

    def visit_named_type(self, node: NamedTypeNode) -> R:
        return self.visit_type(node)

    def visit_parameterized_type(self, node: ParameterizedTypeNode) -> R:
        return self.visit_type(node)

    def visit_auto_type(self, node: AutoTypeNode) -> R:
        return self.visit_type(node)


class TypeVisitor(AbstractTypeVisitor[R], abc.ABC):
    def visit(self, node: TypeNode) -> R:
        if isinstance(node, NamedTypeNode):
            return self.visit_named_type(node)
        elif isinstance(node, ParameterizedTypeNode):
            return self.visit_parameterized_type(node)
        elif isinstance(node, AutoTypeNode):
            return self.visit_auto_type(node)

        raise DiagnosticError(node.location, f"Not implemented visitor for {type(node).__name__}")


class AbstractMemberVisitor(Generic[R], abc.ABC):
    @abc.abstractmethod
    def visit_member(self, node: MemberNode) -> R:
        raise NotImplementedError

    def visit_pass_member(self, node: PassMemberNode) -> R:
        return self.visit_member(node)

    def visit_enum_value(self, node: EnumValueNode) -> R:
        return self.visit_member(node)

    def visit_field(self, node: FieldNode) -> R:
        return self.visit_member(node)

    def visit_function(self, node: FunctionNode) -> R:
        return self.visit_member(node)

    def visit_type_declaration(self, node: TypeDeclarationNode) -> R:
        return self.visit_member(node)

    def visit_class(self, node: ClassNode) -> R:
        return self.visit_type_declaration(node)

    def visit_struct(self, node: StructNode) -> R:
        return self.visit_type_declaration(node)

    def visit_interface(self, node: InterfaceNode) -> R:
        return self.visit_type_declaration(node)

    def visit_enum(self, node: EnumNode) -> R:
        return self.visit_type_declaration(node)


class MemberVisitor(AbstractMemberVisitor[R], abc.ABC):
    def visit(self, node: MemberNode) -> R:
        if isinstance(node, PassMemberNode):
            return self.visit_pass_member(node)
        elif isinstance(node, EnumValueNode):
            return self.visit_enum_value(node)
        elif isinstance(node, FieldNode):
            return self.visit_field(node)
        elif isinstance(node, FunctionNode):
            return self.visit_function(node)
        elif isinstance(node, ClassNode):
            return self.visit_class(node)
        elif isinstance(node, StructNode):
            return self.visit_struct(node)
        elif isinstance(node, InterfaceNode):
            return self.visit_interface(node)
        elif isinstance(node, EnumNode):
            return self.visit_enum(node)

        raise DiagnosticError(node.location, f"Not implemented visitor for {type(node).__name__}")


class AbstractStatementVisitor(Generic[R], abc.ABC):
    @abc.abstractmethod
    def visit_statement(self, node: StatementNode) -> R:
        raise NotImplementedError

    def visit_block_statement(self, node: BlockStatementNode) -> R:
        return self.visit_statement(node)

    def visit_pass_statement(self, node: PassStatementNode) -> R:
        return self.visit_statement(node)

    def visit_return_statement(self, node: ReturnStatementNode) -> R:
        return self.visit_statement(node)

    def visit_yield_statement(self, node: YieldStatementNode) -> R:
        return self.visit_statement(node)

    def visit_assign_statement(self, node: AssignStatementNode) -> R:
        return self.visit_statement(node)

    def visit_augmented_assign_statement(self, node: AugmentedAssignStatementNode) -> R:
        return self.visit_statement(node)

    def visit_expression_statement(self, node: ExpressionStatementNode) -> R:
        return self.visit_statement(node)

    def visit_break_statement(self, node: BreakStatementNode) -> R:
        return self.visit_statement(node)

    def visit_continue_statement(self, node: ContinueStatementNode) -> R:
        return self.visit_statement(node)

    def visit_condition_statement(self, node: ConditionStatementNode) -> R:
        return self.visit_statement(node)

    def visit_while_statement(self, node: WhileStatementNode) -> R:
        return self.visit_statement(node)

    def visit_for_statement(self, node: ForStatementNode) -> R:
        return self.visit_statement(node)

    def visit_ellipsis_statement(self, node: EllipsisStatementNode) -> R:
        return self.visit_statement(node)

    def visit_else_statement(self, node: ElseStatementNode) -> R:
        return self.visit_statement(node)

    def visit_finally_statement(self, node: FinallyStatementNode) -> R:
        return self.visit_statement(node)

    def visit_raise_statement(self, node: RaiseStatementNode) -> R:
        return self.visit_statement(node)

    def visit_try_statement(self, node: TryStatementNode) -> R:
        return self.visit_statement(node)

    def visit_with_statement(self, node: WithStatementNode) -> R:
        return self.visit_statement(node)

    def visit_variable_statement(self, node: VariableStatementNode) -> R:
        return self.visit_statement(node)


class StatementVisitor(AbstractStatementVisitor[R], abc.ABC):
    def visit(self, node: StatementNode) -> R:
        if isinstance(node, ReturnStatementNode):
            return self.visit_return_statement(node)
        elif isinstance(node, YieldStatementNode):
            return self.visit_yield_statement(node)
        elif isinstance(node, PassStatementNode):
            return self.visit_pass_statement(node)
        elif isinstance(node, BreakStatementNode):
            return self.visit_break_statement(node)
        elif isinstance(node, ContinueStatementNode):
            return self.visit_continue_statement(node)
        elif isinstance(node, ConditionStatementNode):
            return self.visit_condition_statement(node)
        elif isinstance(node, WhileStatementNode):
            return self.visit_while_statement(node)
        elif isinstance(node, ForStatementNode):
            return self.visit_for_statement(node)
        elif isinstance(node, AssignStatementNode):
            return self.visit_assign_statement(node)
        elif isinstance(node, AugmentedAssignStatementNode):
            return self.visit_augmented_assign_statement(node)
        elif isinstance(node, EllipsisStatementNode):
            return self.visit_ellipsis_statement(node)
        elif isinstance(node, ElseStatementNode):
            return self.visit_else_statement(node)
        elif isinstance(node, FinallyStatementNode):
            return self.visit_finally_statement(node)
        elif isinstance(node, RaiseStatementNode):
            return self.visit_raise_statement(node)
        elif isinstance(node, TryStatementNode):
            return self.visit_try_statement(node)
        elif isinstance(node, WithStatementNode):
            return self.visit_with_statement(node)
        elif isinstance(node, BlockStatementNode):
            return self.visit_block_statement(node)
        elif isinstance(node, ExpressionStatementNode):
            return self.visit_expression_statement(node)
        elif isinstance(node, VariableStatementNode):
            return self.visit_variable_statement(node)

        raise DiagnosticError(node.location, f"Not implemented visitor for {type(node).__name__}")


class AbstractExpressionVisitor(Generic[R], abc.ABC):
    @abc.abstractmethod
    def visit_expression(self, node: ExpressionNode) -> R:
        raise NotImplementedError

    def visit_integer_expression(self, node: IntegerExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_float_expression(self, node: FloatExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_string_expression(self, node: StringExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_named_expression(self, node: NamedExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_attribute_expression(self, node: AttributeExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_unary_expression(self, node: UnaryExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_binary_expression(self, node: BinaryExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_logic_expression(self, node: LogicExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_compare_expression(self, node: CompareExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_condition_expression(self, node: ConditionExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_call_expression(self, node: CallExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_subscribe_expression(self, node: SubscribeExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_array_expression(self, node: ArrayExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_set_expression(self, node: SetExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_tuple_expression(self, node: TupleExpressionNode) -> R:
        return self.visit_expression(node)

    def visit_dict_expression(self, node: DictExpressionNode) -> R:
        return self.visit_expression(node)


class ExpressionVisitor(AbstractExpressionVisitor[R], abc.ABC):
    def visit(self, node: ExpressionNode) -> R:
        if isinstance(node, IntegerExpressionNode):
            return self.visit_integer_expression(node)
        elif isinstance(node, FloatExpressionNode):
            return self.visit_float_expression(node)
        elif isinstance(node, StringExpressionNode):
            return self.visit_string_expression(node)
        elif isinstance(node, NamedExpressionNode):
            return self.visit_named_expression(node)
        elif isinstance(node, AttributeExpressionNode):
            return self.visit_attribute_expression(node)
        elif isinstance(node, UnaryExpressionNode):
            return self.visit_unary_expression(node)
        elif isinstance(node, BinaryExpressionNode):
            return self.visit_binary_expression(node)
        elif isinstance(node, LogicExpressionNode):
            return self.visit_logic_expression(node)
        elif isinstance(node, CompareExpressionNode):
            return self.visit_compare_expression(node)
        elif isinstance(node, ConditionExpressionNode):
            return self.visit_condition_expression(node)
        elif isinstance(node, CallExpressionNode):
            return self.visit_call_expression(node)
        elif isinstance(node, SubscribeExpressionNode):
            return self.visit_subscribe_expression(node)
        elif isinstance(node, ArrayExpressionNode):
            return self.visit_array_expression(node)
        elif isinstance(node, SetExpressionNode):
            return self.visit_set_expression(node)
        elif isinstance(node, TupleExpressionNode):
            return self.visit_tuple_expression(node)
        elif isinstance(node, DictExpressionNode):
            return self.visit_dict_expression(node)

        raise DiagnosticError(node.location, f"Not implemented visitor for {type(node).__name__}")


class NodeVisitor(Generic[R],
                  AbstractImportVisitor[R],
                  AbstractTypeVisitor[R],
                  AbstractMemberVisitor[R],
                  AbstractStatementVisitor[R],
                  AbstractExpressionVisitor[R],
                  abc.ABC):
    def visit(self, node: SyntaxNode) -> R:
        if isinstance(node, NamedTypeNode):
            return self.visit_named_type(node)
        elif isinstance(node, ParameterizedTypeNode):
            return self.visit_parameterized_type(node)
        elif isinstance(node, AutoTypeNode):
            return self.visit_auto_type(node)
        elif isinstance(node, ReturnStatementNode):
            return self.visit_return_statement(node)
        elif isinstance(node, YieldStatementNode):
            return self.visit_yield_statement(node)
        elif isinstance(node, PassStatementNode):
            return self.visit_pass_statement(node)
        elif isinstance(node, BreakStatementNode):
            return self.visit_break_statement(node)
        elif isinstance(node, ContinueStatementNode):
            return self.visit_continue_statement(node)
        elif isinstance(node, ConditionStatementNode):
            return self.visit_condition_statement(node)
        elif isinstance(node, WhileStatementNode):
            return self.visit_while_statement(node)
        elif isinstance(node, ForStatementNode):
            return self.visit_for_statement(node)
        elif isinstance(node, AssignStatementNode):
            return self.visit_assign_statement(node)
        elif isinstance(node, AugmentedAssignStatementNode):
            return self.visit_augmented_assign_statement(node)
        elif isinstance(node, EllipsisStatementNode):
            return self.visit_ellipsis_statement(node)
        elif isinstance(node, ElseStatementNode):
            return self.visit_else_statement(node)
        elif isinstance(node, FinallyStatementNode):
            return self.visit_finally_statement(node)
        elif isinstance(node, RaiseStatementNode):
            return self.visit_raise_statement(node)
        elif isinstance(node, TryStatementNode):
            return self.visit_try_statement(node)
        elif isinstance(node, WithStatementNode):
            return self.visit_with_statement(node)
        elif isinstance(node, BlockStatementNode):
            return self.visit_block_statement(node)
        elif isinstance(node, ExpressionStatementNode):
            return self.visit_expression_statement(node)
        elif isinstance(node, VariableStatementNode):
            return self.visit_variable_statement(node)
        elif isinstance(node, IntegerExpressionNode):
            return self.visit_integer_expression(node)
        elif isinstance(node, FloatExpressionNode):
            return self.visit_float_expression(node)
        elif isinstance(node, StringExpressionNode):
            return self.visit_string_expression(node)
        elif isinstance(node, NamedExpressionNode):
            return self.visit_named_expression(node)
        elif isinstance(node, AttributeExpressionNode):
            return self.visit_attribute_expression(node)
        elif isinstance(node, UnaryExpressionNode):
            return self.visit_unary_expression(node)
        elif isinstance(node, BinaryExpressionNode):
            return self.visit_binary_expression(node)
        elif isinstance(node, LogicExpressionNode):
            return self.visit_logic_expression(node)
        elif isinstance(node, CompareExpressionNode):
            return self.visit_compare_expression(node)
        elif isinstance(node, ConditionExpressionNode):
            return self.visit_condition_expression(node)
        elif isinstance(node, CallExpressionNode):
            return self.visit_call_expression(node)
        elif isinstance(node, SubscribeExpressionNode):
            return self.visit_subscribe_expression(node)
        elif isinstance(node, ArrayExpressionNode):
            return self.visit_array_expression(node)
        elif isinstance(node, SetExpressionNode):
            return self.visit_set_expression(node)
        elif isinstance(node, TupleExpressionNode):
            return self.visit_tuple_expression(node)
        elif isinstance(node, DictExpressionNode):
            return self.visit_dict_expression(node)
        elif isinstance(node, PassMemberNode):
            return self.visit_pass_member(node)
        elif isinstance(node, EnumValueNode):
            return self.visit_enum_value(node)
        elif isinstance(node, FieldNode):
            return self.visit_field(node)
        elif isinstance(node, FunctionNode):
            return self.visit_function(node)
        elif isinstance(node, ClassNode):
            return self.visit_class(node)
        elif isinstance(node, StructNode):
            return self.visit_struct(node)
        elif isinstance(node, InterfaceNode):
            return self.visit_interface(node)
        elif isinstance(node, EnumNode):
            return self.visit_enum(node)
        elif isinstance(node, ImportFromNode):
            return self.visit_from_import(node)
        elif isinstance(node, ImportNode):
            return self.visit_import(node)
        elif isinstance(node, AliasNode):
            return self.visit_alias(node)
        elif isinstance(node, SyntaxTree):
            return self.visit_tree(node)
        elif isinstance(node, DecoratorNode):
            return self.visit_decorator(node)
        elif isinstance(node, ParameterNode):
            return self.visit_parameter(node)
        elif isinstance(node, ExceptHandlerNode):
            return self.visit_except_handler(node)
        elif isinstance(node, WithItemNode):
            return self.visit_with_item(node)
        elif isinstance(node, ComparatorNode):
            return self.visit_comparator(node)
        elif isinstance(node, PositionArgumentNode):
            return self.visit_position_argument(node)
        elif isinstance(node, KeywordArgumentNode):
            return self.visit_keyword_argument(node)
        elif isinstance(node, DictArgumentNode):
            return self.visit_dict_argument(node)
        elif isinstance(node, GenericParameterNode):
            return self.visit_generic_parameter(node)

        raise DiagnosticError(node.location, f"Not implemented visitor for {type(node).__name__}")

    @abc.abstractmethod
    def visit_node(self, node: SyntaxNode) -> R:
        raise NotImplementedError

    def visit_tree(self, node: SyntaxTree) -> R:
        return self.visit_node(node)

    def visit_import(self, node: ImportNode) -> R:
        return self.visit_node(node)

    def visit_from_import(self, node: ImportFromNode) -> R:
        return self.visit_node(node)

    def visit_alias(self, node: AliasNode) -> R:
        return self.visit_node(node)

    def visit_type(self, node: TypeNode) -> R:
        return self.visit_node(node)

    def visit_member(self, node: MemberNode) -> R:
        return self.visit_node(node)

    def visit_statement(self, node: StatementNode) -> R:
        return self.visit_node(node)

    def visit_expression(self, node: ExpressionNode) -> R:
        return self.visit_node(node)

    def visit_decorator(self, node: DecoratorNode) -> R:
        return self.visit_node(node)

    def visit_parameter(self, node: ParameterNode) -> R:
        return self.visit_node(node)

    def visit_except_handler(self, node: ExceptHandlerNode) -> R:
        return self.visit_node(node)

    def visit_with_item(self, node: WithItemNode) -> R:
        return self.visit_node(node)

    def visit_comparator(self, node: ComparatorNode) -> R:
        return self.visit_node(node)

    def visit_argument(self, node: ArgumentNode) -> R:
        return self.visit_node(node)

    def visit_position_argument(self, node: PositionArgumentNode) -> R:
        return self.visit_argument(node)

    def visit_keyword_argument(self, node: KeywordArgumentNode) -> R:
        return self.visit_argument(node)

    def visit_dict_argument(self, node: DictArgumentNode) -> R:
        return self.visit_node(node)

    def visit_generic_parameter(self, node: GenericParameterNode) -> R:
        return self.visit_node(node)


def make_sequence(sequence: Iterable[object]) -> Sequence[SyntaxSymbol]:
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
