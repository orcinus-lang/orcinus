# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import contextlib
from collections import deque as Queue
from typing import Set, MutableSequence, Tuple

from orcinus.scanner import Scanner
from orcinus.syntax import *

IMPORT_STARTS: Set[TokenID] = {
    TokenID.Import,
    TokenID.From
}
MEMBER_STARTS: Set[TokenID] = {
    TokenID.Name,
    TokenID.Def,
    TokenID.Pass,
    TokenID.Class,
    TokenID.Interface,
    TokenID.Enum,
    TokenID.Struct,
    TokenID.At,
}
TARGET_STARTS: Set[TokenID] = {
    TokenID.Name,
    TokenID.Integer,
    TokenID.String,
    TokenID.Double,
    TokenID.LeftParenthesis,
    TokenID.LeftSquare,
    TokenID.Tilde,
    TokenID.Plus,
    TokenID.Minus,
}
EXPRESSION_STARTS: Set[TokenID] = TARGET_STARTS | {
    TokenID.Name,
    TokenID.Integer,
    TokenID.String,
    TokenID.Double,
    TokenID.LeftParenthesis,
    TokenID.LeftSquare,
    TokenID.LeftCurly,
    TokenID.Tilde,
    TokenID.Plus,
    TokenID.Minus,
    TokenID.Not,
}
SUBSCRIBE_ARGUMENT_STARTS = EXPRESSION_STARTS | {TokenID.Colon}
STATEMENT_STARTS: Set[TokenID] = EXPRESSION_STARTS | {
    TokenID.Break,
    TokenID.Continue,
    TokenID.For,
    TokenID.If,
    TokenID.Pass,
    TokenID.Raise,
    TokenID.Return,
    TokenID.Try,
    TokenID.While,
    TokenID.With,
    TokenID.Yield,
}
COMPARISON_STARTS: Set[TokenID] = {
    TokenID.DoubleEqual,
    TokenID.NotEqual,
    TokenID.Less,
    TokenID.LessEqual,
    TokenID.Great,
    TokenID.GreatEqual,
    TokenID.Not,
    TokenID.Is,
    TokenID.In,
}

DECORATED_STARTS: Set[TokenID] = {
    TokenID.Def,
    TokenID.Class,
    TokenID.Interface,
    TokenID.Enum,
    TokenID.Struct,
}
NAMED_MEMBER_STARTS: Set[TokenID] = {
    TokenID.Equal,
    TokenID.Colon,
}
PRIMARY_STARTS: Set[TokenID] = {
    TokenID.Integer,
    TokenID.String,
    TokenID.Name,
    TokenID.LeftParenthesis,
    TokenID.LeftSquare,
    TokenID.LeftCurly,
}
COMPARISON_IDS: Mapping[TokenID, CompareID] = {
    TokenID.DoubleEqual: CompareID.Equal,
    TokenID.NotEqual: CompareID.NotEqual,
    TokenID.Less: CompareID.Less,
    TokenID.LessEqual: CompareID.LessEqual,
    TokenID.Great: CompareID.Great,
    TokenID.GreatEqual: CompareID.GreatEqual
}
UNARY_IDS: Mapping[TokenID, UnaryID] = {
    TokenID.Minus: UnaryID.Neg,
    TokenID.Plus: UnaryID.Pos,
    TokenID.Tilde: UnaryID.Inv
}
BINARY_IDS: Mapping[TokenID, BinaryID] = {
    TokenID.Plus: BinaryID.Add,
    TokenID.Minus: BinaryID.Sub,
    TokenID.Star: BinaryID.Mul,
    TokenID.At: BinaryID.MatrixMul,
    TokenID.Slash: BinaryID.TrueDiv,
    TokenID.DoubleSlash: BinaryID.FloorDiv,
    TokenID.Percent: BinaryID.Mod,
    TokenID.VerticalLine: BinaryID.Or,
    TokenID.Circumflex: BinaryID.Xor,
    TokenID.Ampersand: BinaryID.And,
    TokenID.LeftShift: BinaryID.LeftShift,
    TokenID.RightShift: BinaryID.RightShift,
    TokenID.DoubleStar: BinaryID.Pow,
}
AUGMENTED_IDS: Mapping[TokenID, BinaryID] = {
    TokenID.PlusEqual: BinaryID.Add,
    TokenID.MinusEqual: BinaryID.Sub,
    TokenID.StarEqual: BinaryID.Mul,
    TokenID.AtEqual: BinaryID.MatrixMul,
    TokenID.SlashEqual: BinaryID.TrueDiv,
    TokenID.DoubleSlashEqual: BinaryID.FloorDiv,
    TokenID.PercentEqual: BinaryID.Mod,
    TokenID.CircumflexEqual: BinaryID.Xor,
    TokenID.VerticalLineEqual: BinaryID.Or,
    TokenID.AmpersandEqual: BinaryID.And,
    TokenID.DoubleStarEqual: BinaryID.Pow,
    TokenID.LeftShiftEqual: BinaryID.LeftShift,
    TokenID.RightShiftEqual: BinaryID.RightShift,
}
AUGMENTED_STATEMENT_STARTS: Set[TokenID] = set(AUGMENTED_IDS.keys())


class Parser:
    def __init__(self, name: str, scanner: Scanner, context: SyntaxContext):
        self.name = name
        self.context = context
        self.scanner = scanner
        self.current_token = self.scanner.consume_token()
        self.tokens = Queue()
        self.diagnostics = context.diagnostics
        self.errors = [False]  # Current error level
        self.skipped = [set()]  # Skipped
        self.last_position = self.current_token.location.end

    @property
    def is_error(self) -> bool:
        return any(self.errors)

    def match(self, index: TokenID) -> bool:
        # """
        # Match current token
        #
        # :param index:     Token identifier
        # :return: True, if current token is matched passed identifiers
        # """
        return self.match_any({index})

    def match_any(self, indexes: Set[TokenID]) -> bool:
        # """
        # Match current token
        #
        # :param indexes:     Token identifiers
        # :return: True, if current token is matched passed identifiers
        # """
        return self.current_token.id in indexes

    def consume(self, index: TokenID = None) -> SyntaxToken:
        # """
        # Consume current token
        #
        # :param index:     Token identifier
        # :return: Return consumed token
        # :raise Diagnostic if current token is not matched passed identifiers
        # """
        return self.consume_any({index} if index is not None else None)

    def consume_any(self, indexes: Set[TokenID]) -> SyntaxToken:
        # """
        # Consume current token
        #
        # :param indexes:     Token identifiers
        # :return: Return consumed token
        # :raise Diagnostic if current token is not matched passed identifiers
        # """
        if not indexes or self.match_any(indexes):
            # consume token
            token = self.current_token
            if self.tokens:
                self.current_token = self.tokens.popleft()
            else:
                self.current_token = self.scanner.consume_token()

            # stop error recovery mode
            if not self.skipped[-1] or token.id in self.skipped[-1]:
                self.errors[-1] = False

            # change last position for error tokens
            self.last_position = token.location.end
            return token

        if not self.is_error:
            self.errors[-1] = True
            self.error(indexes)

        return self.create_error_token(next(iter(indexes)))

    def create_error_token(self, token_id):
        return SyntaxToken(token_id, "", Location(
            self.current_token.location.filename,
            self.last_position,
            self.current_token.location.begin
        ))

    @contextlib.contextmanager
    def recovery(self, index: TokenID = None):
        """ Repair parser """
        with self.recovery_any({index} if index else {}):
            yield

    @contextlib.contextmanager
    def recovery_any(self, indexes: Set[TokenID]):
        """ Repair parser """
        self.errors.append(False)
        self.skipped.append(indexes)
        yield
        self.errors.pop()
        self.skipped.pop()

    def unput(self, token: SyntaxToken):
        self.tokens.append(self.current_token)
        self.current_token = token

    def error(self, indexes: Set[TokenID]):
        # generate exception message
        existed_name = self.current_token.id.name
        if len(indexes) > 1:
            required_names = []
            for x in indexes:
                required_names.append('‘{}’'.format(x.name))
            message = "Required one of {}, but got ‘{}’".format(', '.join(required_names), existed_name)
        else:
            required_name = next(iter(indexes), None).name
            message = "Required ‘{}’, but got ‘{}’".format(required_name, existed_name)

        self.diagnostics.error(self.current_token.location, message)
        return DiagnosticError(self.current_token.location, message)

    def parse(self) -> SyntaxTree:
        # """
        # module:
        #     imports members EOF
        # """
        with self.scanner:
            imports = self.parse_imports()
            members = self.parse_members()
            token_eof = self.consume(TokenID.EndOfFile)
            tree = SyntaxTree(self.context, name=self.name, imports=imports, members=members, token_eof=token_eof)
            self.context.annotate(tree)
        return tree

    def parse_imports(self) -> SyntaxCollection[ImportNode]:
        # """
        # imports:
        #     { import }
        # """
        imports = []
        while self.match_any(IMPORT_STARTS):
            imports.append(self.parse_import())

        return SyntaxCollection[ImportNode](imports)

    def parse_import(self) -> Optional[ImportNode]:
        # """
        # import:
        #     'from' module_name 'import' aliases
        #     'import' aliases
        # """
        with self.recovery(TokenID.NewLine):
            if self.match(TokenID.From):
                token_from = self.consume()
                token_name = self.parse_qualified_name()
                token_import = self.consume(TokenID.Import)
                aliases = self.parse_aliases()
                token_newline = self.consume(TokenID.NewLine)
                return ImportFromNode(self.context, token_from, token_name, token_import, aliases, token_newline)

            elif self.match(TokenID.Import):
                token_import = self.consume()
                aliases = self.parse_aliases()
                token_newline = self.consume(TokenID.NewLine)
                return ImportNode(self.context, token_import, aliases, token_newline)

        self.error(IMPORT_STARTS)
        return None

    def parse_aliases(self) -> SyntaxCollection[AliasNode]:
        # """
        # aliases:
        #     alias { ',' alias }
        #     '(' alias { ',' alias } ')'
        # """

        # '('
        if self.match(TokenID.LeftParenthesis):
            token_open = self.consume()
            token_skiped = {TokenID.RightParenthesis, TokenID.Comma}
        else:
            token_open = None
            token_skiped = {TokenID.Comma}

        with self.recovery_any(token_skiped):
            # alias { ',' alias }
            aliases = [token_open, self.parse_alias()]
            while self.match(TokenID.Comma):
                aliases.append(self.consume())
                aliases.append(self.parse_alias())

            # ')'
            if token_open:
                token_close = self.consume(TokenID.RightParenthesis)
            else:
                token_close = None
            aliases.append(token_close)

        return SyntaxCollection[AliasNode](aliases)

    def parse_alias(self) -> AliasNode:
        # """
        # alias:
        #     module_name [ 'as' Name ]
        # """
        token_name = self.parse_qualified_name()
        if self.match(TokenID.As):
            token_as = self.consume(TokenID.As)
            token_alias = self.consume(TokenID.Name)
        else:
            token_as = None
            token_alias = None

        return AliasNode(self.context, token_name, token_as, token_alias)

    def parse_qualified_name(self) -> SyntaxToken:
        # """
        # module_name:
        #     Name { '.' Name }
        # """
        # TODO: Capture whitespaces in token
        names = [self.consume(TokenID.Name)]
        while self.match(TokenID.Dot):
            self.consume()
            self.consume(TokenID.Name)

        items = []
        for t in names:
            items.append(t.value)
        name = '.'.join(items)
        location = names[0].location + names[-1].location
        return SyntaxToken(token_id=TokenID.Name, value=name, location=location)

    def parse_decorators(self) -> SyntaxCollection[DecoratorNode]:
        # """
        # decorators:
        #     decorator { decorator }
        # """
        decorators = [self.parse_decorator()]
        while self.match(TokenID.At):
            decorators.append(self.parse_decorator())
        return SyntaxCollection[DecoratorNode](decorators)

    def parse_decorator(self) -> DecoratorNode:
        # """
        # decorator:
        #     '@' full
        # """
        token_at = self.consume(TokenID.At)
        token_name = self.parse_qualified_name()
        if self.match(TokenID.LeftParenthesis):
            token_open = self.consume(TokenID.LeftParenthesis)
            arguments = self.parse_named_arguments()
            token_close = self.consume(TokenID.RightParenthesis)
        else:
            token_open = None
            arguments = SyntaxCollection[ArgumentNode]()
            token_close = None
        token_newline = self.consume(TokenID.NewLine)

        return DecoratorNode(self.context, token_at, token_name, token_open, arguments, token_close, token_newline)

    def parse_members(self) -> SyntaxCollection[MemberNode]:
        # """
        # members:
        #     { member }
        # """
        members = []
        while self.match_any(MEMBER_STARTS):
            members.append(self.parse_member())

        return SyntaxCollection[MemberNode](members)

    def parse_member(self) -> Optional[MemberNode]:
        # """
        # member:
        #     function
        #     class
        #     struct
        #     interface
        #     enum
        #     named_member
        #     pass
        # """
        if self.match(TokenID.At):
            return self.parse_decorated_member()
        elif self.match(TokenID.Enum):
            return self.parse_enum()
        elif self.match(TokenID.Interface):
            return self.parse_interface()
        elif self.match(TokenID.Class):
            return self.parse_class()
        elif self.match(TokenID.Struct):
            return self.parse_struct()
        elif self.match(TokenID.Name):
            return self.parse_named_member()
        elif self.match(TokenID.Def):
            return self.parse_function()
        elif self.match(TokenID.Pass):
            return self.parse_pass_member()

        self.error(MEMBER_STARTS)
        return None

    def parse_pass_member(self) -> PassMemberNode:
        # """
        # pass:
        #     "pass"
        # """
        token_pass = self.consume(TokenID.Pass)
        token_newline = self.consume(TokenID.NewLine)
        return PassMemberNode(self.context, token_pass, token_newline)

    def parse_decorated_member(self) -> Optional[MemberNode]:
        # """
        # decorated_member:
        #     decorators function
        # """
        decorators = self.parse_decorators()
        if self.match(TokenID.Enum):
            return self.parse_enum(decorators)
        elif self.match(TokenID.Interface):
            return self.parse_interface(decorators)
        elif self.match(TokenID.Class):
            return self.parse_class(decorators)
        elif self.match(TokenID.Struct):
            return self.parse_struct(decorators)
        elif self.match(TokenID.Def):
            return self.parse_function(decorators)

        self.error(DECORATED_STARTS)
        return None

    def parse_function(self, decorators=None) -> FunctionNode:
        # """
        # function:
        #     decorators 'def' Name generic_parameters arguments [ -> type ] ':' '...'
        # """
        token_def = self.consume(TokenID.Def)
        with self.recovery(TokenID.NewLine):
            token_name = self.consume(TokenID.Name)
            generic_parameters = self.parse_generic_parameters()
            parameters = self.parse_function_parameters()

            if self.match(TokenID.Then):
                token_then = self.consume(TokenID.Then)
                result_type = self.parse_type()
            else:
                token_then = None
                result_type = AutoTypeNode(self.context, token_name)

            token_colon = self.consume(TokenID.Colon)
            statement = self.parse_function_statement()

        return FunctionNode(
            self.context,
            decorators or SyntaxCollection[DecoratorNode](),
            token_def,
            token_name,
            generic_parameters,
            parameters,
            token_then,
            result_type,
            token_colon,
            statement
        )

    def parse_enum(self, decorators=None) -> EnumNode:
        # """
        # enum:
        #     "enum" Name parents ':' '\n' type_members
        # """
        token_enum = self.consume(TokenID.Enum)
        token_name = self.consume(TokenID.Name)
        generic_parameters = self.parse_generic_parameters()
        parents = self.parse_type_parents()
        token_colon = self.consume(TokenID.Colon)
        token_newline = self.consume(TokenID.NewLine)
        token_indent, members, token_undent = self.parse_type_members()

        return EnumNode(self.context,
                        decorators or SyntaxCollection[DecoratorNode](),
                        token_enum,
                        token_name,
                        generic_parameters,
                        parents,
                        token_colon,
                        token_newline,
                        token_indent,
                        members,
                        token_undent)

    def parse_interface(self, decorators=None) -> InterfaceNode:
        # """
        # interface:
        #     "interface" Name generic_parameters parents ':' '\n' type_members
        # """
        token_interface = self.consume(TokenID.Interface)
        token_name = self.consume(TokenID.Name)
        generic_parameters = self.parse_generic_parameters()
        parents = self.parse_type_parents()
        token_colon = self.consume(TokenID.Colon)
        token_newline = self.consume(TokenID.NewLine)
        token_indent, members, token_undent = self.parse_type_members()

        return InterfaceNode(self.context,
                             decorators or SyntaxCollection[DecoratorNode](),
                             token_interface,
                             token_name,
                             generic_parameters,
                             parents,
                             token_colon,
                             token_newline,
                             token_indent,
                             members,
                             token_undent)

    def parse_class(self, decorators=None) -> ClassNode:
        # """
        # class:
        #     "class" Name generic_parameters parents ':' '\n' type_members
        # """
        token_class = self.consume(TokenID.Class)
        token_name = self.consume(TokenID.Name)
        generic_parameters = self.parse_generic_parameters()
        parents = self.parse_type_parents()
        token_colon = self.consume(TokenID.Colon)
        token_newline = self.consume(TokenID.NewLine)
        token_indent, members, token_undent = self.parse_type_members()

        return ClassNode(self.context,
                         decorators or SyntaxCollection[DecoratorNode](),
                         token_class,
                         token_name,
                         generic_parameters,
                         parents,
                         token_colon,
                         token_newline,
                         token_indent,
                         members,
                         token_undent)

    def parse_struct(self, decorators=None) -> StructNode:
        # """
        # struct:
        #     "struct" Name generic_parameters parents ':' '\n' type_members
        # """
        token_struct = self.consume(TokenID.Struct)
        token_name = self.consume(TokenID.Name)
        generic_parameters = self.parse_generic_parameters()
        parents = self.parse_type_parents()
        token_colon = self.consume(TokenID.Colon)
        token_newline = self.consume(TokenID.NewLine)
        token_indent, members, token_undent = self.parse_type_members()

        return StructNode(self.context,
                          decorators or SyntaxCollection[DecoratorNode](),
                          token_struct,
                          token_name,
                          generic_parameters,
                          parents,
                          token_colon,
                          token_newline,
                          token_indent,
                          members,
                          token_undent)

    def parse_named_member(self) -> Optional[MemberNode]:
        # """
        # named_member:
        #     enum_member
        #     field_member
        # """
        token_name = self.consume(TokenID.Name)

        if self.match(TokenID.Equal):
            return self.parse_enum_member(token_name)

        elif self.match(TokenID.Colon):
            return self.parse_field_member(token_name)

        self.error(NAMED_MEMBER_STARTS)
        return None

    def parse_enum_member(self, token_name: SyntaxToken) -> EnumValueNode:
        # """
        # enum_member:
        #     Name '=' '...'
        #     Name '=' expression
        # """
        token_equal = self.consume(TokenID.Equal)
        if self.match(TokenID.Ellipsis):
            token_ellipsis = self.consume()
            value = None
        else:
            token_ellipsis = None
            value = self.parse_expression()
        token_newline = self.consume(TokenID.NewLine)
        return EnumValueNode(self.context, token_name, token_equal, token_ellipsis, value, token_newline)

    def parse_field_member(self, token_name: SyntaxToken) -> FieldNode:
        # """
        # field_member:
        #     Name ':' type
        # """
        token_colon = self.consume(TokenID.Colon)
        field_type = self.parse_type()
        if self.match(TokenID.Equal):
            token_equal = self.consume()
            default_value = self.parse_expression_list()
        else:
            token_equal = None
            default_value = None
        token_newline = self.consume(TokenID.NewLine)
        return FieldNode(self.context, token_name, token_colon, field_type, token_equal, default_value, token_newline)

    def parse_type_parents(self) -> SyntaxCollection[TypeNode]:
        # """
        # type_parents:
        #     [ '(' type { ',' type ) ') ]
        # """
        if not self.match(TokenID.LeftParenthesis):
            return SyntaxCollection[TypeNode]()

        parents = [self.consume(TokenID.LeftParenthesis)]
        with self.recovery(TokenID.RightParenthesis):
            parents.append(self.parse_type())
            while self.match(TokenID.Comma):
                parents.append(self.consume())
                parents.append(self.parse_type())
            parents.append(self.consume(TokenID.RightParenthesis))

        return SyntaxCollection[TypeNode](parents)

    def parse_type_members(self) -> Tuple[SyntaxToken, SyntaxCollection[MemberNode], SyntaxToken]:
        # """
        # type_members:
        #     Indent members Undent
        # """
        token_indent = self.consume(TokenID.Indent)
        members = self.parse_members()
        token_undent = self.consume(TokenID.Undent)
        return token_indent, members, token_undent

    def parse_generic_parameters(self) -> SyntaxCollection[GenericParameterNode]:
        # """
        # generic_parameters:
        #     [ '[' generic_parameter { ',' generic_parameter } '] ]
        # """
        if not self.match(TokenID.LeftSquare):
            return SyntaxCollection[GenericParameterNode]()

        generic_parameters = [self.consume(TokenID.LeftSquare)]
        with self.recovery(TokenID.RightSquare):
            generic_parameters.append(self.parse_generic_parameter())
            while self.match(TokenID.Comma):
                generic_parameters.append(self.consume(TokenID.Comma))
                generic_parameters.append(self.parse_generic_parameter())

            generic_parameters.append(self.consume(TokenID.RightSquare))
        return SyntaxCollection[GenericParameterNode](generic_parameters)

    def parse_generic_parameter(self) -> GenericParameterNode:
        # """
        # generic_parameter:
        #     Name
        # """
        token_name = self.consume(TokenID.Name)
        return GenericParameterNode(self.context, token_name)

    def parse_function_parameters(self) -> SyntaxCollection[ParameterNode]:
        # """
        # function_parameters:
        #     '(' [ function_parameter { ',' function_parameter } ] ')'
        # """
        parameters = [self.consume(TokenID.LeftParenthesis)]
        with self.recovery(TokenID.RightSquare):
            if self.match(TokenID.Name):
                parameters.append(self.parse_function_parameter())
                with self.recovery(TokenID.Comma):
                    while self.match(TokenID.Comma):
                        parameters.append(self.consume(TokenID.Comma))
                        parameters.append(self.parse_function_parameter())
            parameters.append(self.consume(TokenID.RightParenthesis))
        return SyntaxCollection[ParameterNode](parameters)

    def parse_function_parameter(self) -> ParameterNode:
        # """
        # function_parameter:
        #     Name [ ':' type ]
        # """
        token_name = self.consume(TokenID.Name)
        if self.match(TokenID.Colon):
            token_colon = self.consume(TokenID.Colon)
            param_type = self.parse_type()
        else:
            token_colon = None
            param_type = AutoTypeNode(self.context, token_name)

        if self.match(TokenID.Equal):
            token_equal = self.consume()
            default_value = self.parse_expression()
        else:
            token_equal = None
            default_value = None

        return ParameterNode(self.context, token_name, token_colon, param_type, token_equal, default_value)

    def parse_type(self) -> TypeNode:
        # """
        # type:
        #     Name
        #     Name '[' type_arguments ']'
        # """
        token_name = self.parse_qualified_name()
        result_type = NamedTypeNode(self.context, token_name)

        while self.match(TokenID.LeftSquare):
            arguments = self.parse_generic_arguments()
            result_type = ParameterizedTypeNode(self.context, result_type, arguments)

        return result_type

    def parse_generic_arguments(self) -> SyntaxCollection[TypeNode]:
        # """
        # generic_arguments:
        #     '[' type { ',' type} ']'
        # """

        arguments = [self.consume(TokenID.LeftSquare)]
        with self.recovery_any({TokenID.RightSquare, TokenID.Comma}):
            arguments.append(self.parse_type())
            while self.match(TokenID.Comma):
                arguments.append(self.consume(TokenID.Comma))
                arguments.append(self.parse_type())
            arguments.append(self.consume(TokenID.RightSquare))
        return SyntaxCollection[TypeNode](arguments)

    def parse_function_statement(self) -> StatementNode:
        # """
        # function_statement:
        #     '...' EndFile
        #     NewLine block_statement
        # """
        if self.match(TokenID.Ellipsis):
            token_ellipsis = self.consume(TokenID.Ellipsis)
            token_newline = self.consume(TokenID.NewLine)
            return EllipsisStatementNode(self.context, token_ellipsis, token_newline)

        return self.parse_block_statement()

    def parse_block_statement(self) -> BlockStatementNode:
        # """
        # block_statement:
        #     '\n' Indent statement { statement } Undent
        # """
        token_newline = self.consume(TokenID.NewLine)
        token_indent = self.consume(TokenID.Indent)
        statements = [self.parse_statement()]
        while self.match_any(STATEMENT_STARTS):
            statements.append(self.parse_statement())
        token_undent = self.consume(TokenID.Undent)

        # noinspection PyArgumentList
        return BlockStatementNode(self.context, token_newline, token_indent, SyntaxCollection(statements), token_undent)

    def parse_statement(self) -> Optional[StatementNode]:
        # """
        # statement:
        #     pass_statement
        #     return_statement
        #     yield_statement
        #     condition_statement
        #     while_statement
        #     for_statement
        #     raise_statement
        #     expression_statement
        # """
        if self.match(TokenID.Pass):
            return self.parse_pass_statement()
        elif self.match(TokenID.Return):
            return self.parse_return_statement()
        elif self.match(TokenID.Yield):
            return self.parse_yield_statement()
        elif self.match(TokenID.Continue):
            return self.parse_continue_statement()
        elif self.match(TokenID.Break):
            return self.parse_break_statement()
        elif self.match(TokenID.If):
            return self.parse_condition_statement()
        elif self.match(TokenID.While):
            return self.parse_while_statement()
        elif self.match(TokenID.For):
            return self.parse_for_statement()
        elif self.match(TokenID.Raise):
            return self.parse_raise_statement()
        elif self.match(TokenID.Try):
            return self.parse_try_statement()
        elif self.match(TokenID.With):
            return self.parse_with_statement()
        elif self.match_any(EXPRESSION_STARTS):
            return self.parse_expression_statement()

        self.error(STATEMENT_STARTS)
        return None

    def parse_pass_statement(self):
        # """ pass_statement: pass """
        token_pass = self.consume(TokenID.Pass)
        token_newline = self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return PassStatementNode(self.context, token_pass, token_newline)

    def parse_return_statement(self):
        # """
        # return_statement
        #     'return' [ expression_list ]
        # """
        token_return = self.consume(TokenID.Return)
        value = self.parse_expression_list() if self.match_any(EXPRESSION_STARTS) else None
        token_newline = self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return ReturnStatementNode(self.context, token_return, value, token_newline)

    def parse_yield_statement(self):
        # """
        # yield_statement
        #     'yield' [ expression_list ]
        # """
        token_yield = self.consume(TokenID.Yield)
        value = self.parse_expression_list() if self.match_any(EXPRESSION_STARTS) else None
        token_newline = self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return YieldStatementNode(self.context, token_yield, value, token_newline)

    def parse_break_statement(self):
        # """
        # break_statement
        #     'break'
        # """
        token_break = self.consume(TokenID.Break)
        token_newline = self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return BreakStatementNode(self.context, token_break, token_newline)

    def parse_continue_statement(self):
        # """
        # continue_statement
        #     'continue'
        # """
        token_continue = self.consume(TokenID.Continue)
        token_newline = self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return ContinueStatementNode(self.context, token_continue, token_newline)

    def parse_else_statement(self) -> ElseStatementNode:
        # """
        # else_statement:
        #     'else' ':' block_statement
        # """
        token_else = self.consume(TokenID.Else)
        with self.recovery(TokenID.NewLine):
            token_colon = self.consume(TokenID.Colon)
            statement = self.parse_block_statement()
            return ElseStatementNode(self.context, token_else, token_colon, statement)

    def parse_finally_statement(self) -> FinallyStatementNode:
        # """
        # finally_statement:
        #     'finally' ':' block_statement
        # """
        token_finally = self.consume(TokenID.Finally)
        with self.recovery(TokenID.NewLine):
            token_colon = self.consume(TokenID.Colon)
            statement = self.parse_block_statement()
            return FinallyStatementNode(self.context, token_finally, token_colon, statement)

    def parse_condition_statement(self) -> ConditionStatementNode:
        # """
        # condition_statement:
        #     'if' expression ':' block_statement            ⏎
        #         { 'elif' expression ':' block_statement }  ⏎
        #         [ else_statement ]
        # """
        token_if = self.consume()
        condition = self.parse_expression()
        token_colon = self.consume(TokenID.Colon)
        then_statement = self.parse_block_statement()

        else_statement = None
        if self.match(TokenID.Else):
            else_statement = self.parse_else_statement()
        elif self.match(TokenID.Elif):
            else_statement = self.parse_condition_statement()

        # noinspection PyArgumentList
        return ConditionStatementNode(self.context, token_if, condition, token_colon, then_statement, else_statement)

    def parse_while_statement(self) -> WhileStatementNode:
        # """
        # while_statement:
        #     'while' expression ':' block_statement     ⏎
        #         [ else_statement ]
        # """
        token_while = self.consume(TokenID.While)
        condition = self.parse_expression()
        token_colon = self.consume(TokenID.Colon)
        then_statement = self.parse_block_statement()
        else_statement = self.parse_else_statement() if self.match(TokenID.Else) else None

        # noinspection PyArgumentList
        return WhileStatementNode(self.context, token_while, condition, token_colon, then_statement, else_statement)

    def parse_for_statement(self) -> ForStatementNode:
        # """
        # for_statement:
        #     'for' target_list 'in' expression_list ':' '\n' block_statement     ⏎
        #         [ 'else' ':' '\n' block_statement ]
        # """
        token_for = self.consume(TokenID.For)
        target = self.parse_target_list()
        token_in = self.consume(TokenID.In)
        source = self.parse_expression_list()
        token_colon = self.consume(TokenID.Colon)
        then_statement = self.parse_block_statement()
        else_statement = self.parse_else_statement() if self.match(TokenID.Else) else None

        # noinspection PyArgumentList
        return ForStatementNode(
            self.context,
            token_for,
            target,
            token_in,
            source,
            token_colon,
            then_statement,
            else_statement,
            token_for.location
        )

    def parse_raise_statement(self) -> RaiseStatementNode:
        # """
        # raise_statement:
        #     'raise' [ expression [ "from" expression ] ] '\n'
        # """
        token_raise = self.consume(TokenID.Raise)
        exception = None
        cause_exception = None
        token_from = None
        if self.match_any(EXPRESSION_STARTS):
            exception = self.parse_expression()
            if self.match(TokenID.From):
                token_from = self.consume()
                cause_exception = self.parse_expression()

        token_newline = self.consume(TokenID.NewLine)
        return RaiseStatementNode(self.context, token_raise, exception, token_from, cause_exception, token_newline)

    def parse_try_statement(self):
        # """
        # try_statement:
        #     'try' ':' block_statement
        #         { except_handler }              ⏎
        #         [ else_statement ]              ⏎
        #         [ finally_statement ]
        #     'try' ':' block_statement finally_statement
        # """
        token_try = self.consume(TokenID.Try)
        token_colon = self.consume(TokenID.Colon)
        try_statement = self.parse_block_statement()

        handlers = []
        else_statement = None

        if self.match(TokenID.Except):
            while self.match(TokenID.Except):
                handlers.append(self.parse_except_handler())

            else_statement = self.parse_else_statement() if self.match(TokenID.Else) else None
            finally_statement = self.parse_finally_statement() if self.match(TokenID.Finally) else None
        else:
            finally_statement = self.parse_finally_statement()

        handlers = SyntaxCollection(handlers)

        return TryStatementNode(
            self.context,
            token_try,
            token_colon,
            try_statement,
            handlers,
            else_statement,
            finally_statement
        )

    def parse_except_handler(self) -> ExceptHandlerNode:
        # """
        # except_handler:
        #     'except' [ expression [ 'as' Name ] ':' block_statement
        # """
        token_except = self.consume(TokenID.Except)
        expression = None
        token_as = None
        token_name = None
        if self.match_any(EXPRESSION_STARTS):
            expression = self.parse_expression()
            if self.match(TokenID.As):
                token_as = self.consume()
                token_name = self.consume(TokenID.Name)
        token_colon = self.consume(TokenID.Colon)
        statement = self.parse_block_statement()

        return ExceptHandlerNode(self.context, token_except, expression, token_as, token_name, token_colon, statement)

    def parse_with_statement(self) -> WithStatementNode:
        # """
        # with_statement:
        #     'with' with_item { ',' with_item } ':'  block_statement
        # """
        token_with = self.consume(TokenID.With)
        items = [self.parse_with_item()]
        while self.match(TokenID.Comma):
            items.append(self.consume())
            items.append(self.parse_with_item())

        items = SyntaxCollection(items)
        token_colon = self.consume(TokenID.Colon)
        statement = self.parse_block_statement()

        return WithStatementNode(self.context, token_with, items, token_colon, statement)

    def parse_with_item(self):
        # """
        # with_item:
        #     expression [ 'as' target ]
        # """
        expression = self.parse_expression()
        if self.match(TokenID.As):
            token_as = self.consume()
            target = self.parse_target()
        else:
            token_as = None
            target = None

        return WithItemNode(self.context, expression, token_as, target)

    def parse_expression_statement(self):
        # """
        # expression_statement
        #     expression
        #     assign_expression
        # """
        expression = self.parse_expression_list()

        if self.match(TokenID.Equal):
            return self.parse_assign_statement(expression)
        elif self.match(TokenID.Colon):
            return self.parse_variable_statement(expression)
        elif self.match_any(AUGMENTED_STATEMENT_STARTS):
            return self.parse_augmented_assign_statement(expression)
        token_newline = self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return ExpressionStatementNode(self.context, expression, token_newline)

    def parse_assign_statement(self, target: ExpressionNode):
        # """
        # assign_expression
        #     targets_list '=' expression
        #
        # TODO: https://docs.python.org/3/reference/simple_stmts.html#grammar-token-assignment-stmt
        # """
        token_equal = self.consume(TokenID.Equal)
        source = self.parse_expression_list()
        token_newline = self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return AssignStatementNode(self.context, target, token_equal, source, token_newline)

    def parse_augmented_assign_statement(self, target: ExpressionNode):
        # """
        # assign_expression
        #     targets_list '=' expression
        #
        # TODO: https://docs.python.org/3/reference/simple_stmts.html#grammar-token-assignment-stmt
        # """
        token_operator = self.consume_any(AUGMENTED_STATEMENT_STARTS)
        opcode = AUGMENTED_IDS[token_operator.id]

        source = self.parse_expression_list()
        token_newline = self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return AugmentedAssignStatementNode(self.context, target, token_operator, opcode, source, token_newline)

    def parse_variable_statement(self, named: ExpressionNode):
        # """
        # assign_expression
        #     targets_list '=' expression
        #
        # TODO: https://docs.python.org/3/reference/simple_stmts.html#grammar-token-assignment-stmt
        # """
        if not isinstance(named, NamedExpressionNode):
            self.diagnostics.error(named.location, "Required ‘{}’, but got ‘{}’".format('name', 'expression'))

        token_colon = self.consume(TokenID.Colon)
        var_type = self.parse_type()

        if self.match(TokenID.Equal):
            token_equal = self.consume(TokenID.Equal)
            initial_value = self.parse_expression_list()
        else:
            token_equal = None
            initial_value = None

        token_newline = self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return VariableStatementNode(
            self.context,
            named,
            token_colon,
            var_type,
            token_equal,
            initial_value,
            token_newline
        )

    def parse_arguments(self, arguments: Sequence[ExpressionNode] = None) -> SyntaxCollection[ExpressionNode]:
        # """
        # arguments:
        #     [ expression { ',' expression } [','] ]
        # """
        if not arguments and not self.match_any(EXPRESSION_STARTS):
            return SyntaxCollection[ExpressionNode]()

        arguments = list(arguments or [])
        if not arguments:
            arguments.append(self.parse_expression())

        while self.match(TokenID.Comma):
            arguments.append(self.consume(TokenID.Comma))
            if self.match_any(EXPRESSION_STARTS):
                arguments.append(self.parse_expression())
            else:
                break
        return SyntaxCollection[ExpressionNode](arguments)

    def parse_dict_arguments(self, arguments: MutableSequence[DictArgumentNode] = None) \
            -> SyntaxCollection[DictArgumentNode]:
        # """
        # dict_arguments:
        #     [ dict_argument { ',' dict_argument } [','] ]
        # """
        if not arguments and not self.match_any(EXPRESSION_STARTS):
            return SyntaxCollection[ExpressionNode]()

        arguments = arguments or []
        if not arguments:
            arguments.append(self.parse_dict_argument())

        while self.match(TokenID.Comma):
            arguments.append(self.consume(TokenID.Comma))
            if self.match_any(EXPRESSION_STARTS):
                arguments.append(self.parse_dict_argument())
            else:
                break

        return SyntaxCollection[ExpressionNode](arguments)

    def parse_dict_argument(self) -> DictArgumentNode:
        # """
        # dict_argument:
        #     expression ':' expression
        # """
        key = self.parse_expression()
        token_colon = self.consume(TokenID.Colon)
        value = self.parse_expression()
        return DictArgumentNode(self.context, key, token_colon, value)

    def parse_named_arguments(self) -> SyntaxCollection[ArgumentNode]:
        # """
        # named_arguments:
        #     [ named_argument { ',' named_argument } [','] ]
        # """
        if not self.match_any(EXPRESSION_STARTS):
            return SyntaxCollection[ArgumentNode]()

        arguments = [self.parse_name_argument()]
        while self.match(TokenID.Comma):
            self.consume(TokenID.Comma)
            if self.match_any(EXPRESSION_STARTS):
                arguments.append(self.parse_name_argument())
            else:
                break

        return SyntaxCollection[ArgumentNode](arguments)

    def parse_name_argument(self) -> ArgumentNode:
        # """
        # argument:
        #     Name '=' expression
        #     expression
        # """
        if self.match(TokenID.Name):
            token_name = self.consume()
            if self.match(TokenID.Equal):
                token_equal = self.consume()
                value = self.parse_expression()
                return KeywordArgumentNode(self.context, token_name, token_equal, value)
            self.unput(token_name)

        value = self.parse_expression()
        return PositionArgumentNode(self.context, value)

    def parse_target_list(self) -> ExpressionNode:
        # """
        # target_list:
        #     target { ',' target }
        # """

        arguments = [
            self.parse_target()
        ]
        while self.match(TokenID.Comma):
            arguments.append(self.consume())
            arguments.append(self.parse_target())

        if len(arguments) == 1:
            return arguments[0]
        return TupleExpressionNode(self.context, None, SyntaxCollection(arguments), None)

    def parse_target(self):
        return self.parse_primary_expression()

    def parse_expression_list(self) -> ExpressionNode:
        # """
        # expression_list:
        #     expression { ',' expression } [',']
        # """
        arguments = [self.parse_expression()]
        while self.match(TokenID.Comma):
            arguments.append(self.consume())
            if self.match_any(EXPRESSION_STARTS):
                arguments.append(self.parse_expression())
            else:
                break

        if len(arguments) == 1:
            return arguments[0]
        return TupleExpressionNode(self.context, None, SyntaxCollection(arguments), None)

    def parse_expression(self) -> ExpressionNode:
        # """
        # expression:
        #     conditional_expression
        # """
        return self.parse_conditional_expression()

    def parse_conditional_expression(self) -> ExpressionNode:
        # """
        # conditional_expression:
        #     logical_or_expression [ 'if' logical_or_expression 'else' expression ]
        # """
        expression = self.parse_logical_or_expression()
        if self.match(TokenID.If):
            token_if = self.consume()
            condition = self.parse_logical_or_expression()
            token_else = self.consume(TokenID.Else)
            else_expression = self.parse_expression()
            expression = ConditionExpressionNode(
                self.context,
                expression,
                token_if,
                condition,
                token_else,
                else_expression
            )
        return expression

    def parse_logical_or_expression(self) -> ExpressionNode:
        # """
        # logical_or_expression:
        #     logical_and_expression
        #     logical_or_expression 'or' logical_and_expression
        # """
        expression = self.parse_logical_and_expression()
        while self.match(TokenID.Or):
            token_or = self.consume()
            right_operand = self.parse_logical_and_expression()
            expression = LogicExpressionNode(self.context, expression, token_or, LogicID.Or, right_operand)
        return expression

    def parse_logical_and_expression(self) -> ExpressionNode:
        # """
        # logical_and_expression:
        #     logical_not_expression
        #     logical_and_expression 'and' logical_not_expression
        # """
        expression = self.parse_logical_not_expression()
        while self.match(TokenID.And):
            token_and = self.consume()
            right_operand = self.parse_logical_not_expression()
            expression = LogicExpressionNode(self.context, expression, token_and, LogicID.And, right_operand)
        return expression

    def parse_logical_not_expression(self) -> ExpressionNode:
        # """
        # logical_not_expression:
        #     comparison_expression
        #     "not" logical_not_expression
        # """
        while self.match(TokenID.Not):
            token_not = self.consume(TokenID.Not)
            expression = self.parse_logical_not_expression()
            return UnaryExpressionNode(self.context, token_not, UnaryID.Not, expression)
        return self.parse_comparison_expression()

    def parse_comparison_expression(self) -> ExpressionNode:
        # """
        # comparison_expression:
        #     or_expression { comparison_parts }
        #
        # comparison_parts:
        #                   '<'  or_expression
        #                   '>'  or_expression
        #                   '==' or_expression
        #                   '!=' or_expression
        #                   '<=' or_expression
        #                   '>=' or_expression
        #                   'in' or_expression
        #                   'not' 'in' or_expression
        #                   'is' or_expression
        #                   'is' 'not' or_expression
        # """
        expression = self.parse_or_expression()
        comparators = []

        while self.match_any(COMPARISON_STARTS):
            if self.match(TokenID.Not):
                token_prefix = self.consume()
                token_suffix = self.consume(TokenID.In)
                opcode = CompareID.NotIn

            elif self.match(TokenID.Is):
                token_prefix = self.consume()
                if self.match(TokenID.Not):
                    token_suffix = self.consume()
                    opcode = CompareID.IsNot
                else:
                    token_suffix = None
                    opcode = CompareID.Is

            elif self.match(TokenID.In):
                token_prefix = self.consume()
                token_suffix = None
                opcode = CompareID.In

            else:
                token_prefix = self.consume_any(COMPARISON_STARTS)
                token_suffix = None
                opcode = COMPARISON_IDS[token_prefix.id]

            right_operand = self.parse_and_expression()
            comparators.append(ComparatorNode(self.context, token_prefix, token_suffix, opcode, right_operand))

        if comparators:
            return CompareExpressionNode(self.context, expression, SyntaxCollection[ComparatorNode](comparators))
        return expression

    def parse_or_expression(self) -> ExpressionNode:
        # """
        # or_expression:
        #     xor_expression
        #     or_expression '|' xor_expression
        # """
        expression = self.parse_xor_expression()
        while self.match(TokenID.VerticalLine):
            token_operator = self.consume(TokenID.VerticalLine)
            right_operand = self.parse_xor_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionNode(
                self.context,
                expression,
                token_operator,
                BINARY_IDS[token_operator.id],
                right_operand
            )
        return expression

    def parse_xor_expression(self) -> ExpressionNode:
        # """
        # xor_expression:
        #     and_expression
        #     xor_expression '^' and_expression
        # """
        expression = self.parse_and_expression()
        while self.match(TokenID.Circumflex):
            token_operator = self.consume(TokenID.Circumflex)
            right_operand = self.parse_and_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionNode(
                self.context,
                expression,
                token_operator,
                BINARY_IDS[token_operator.id],
                right_operand
            )
        return expression

    def parse_and_expression(self) -> ExpressionNode:
        # """
        # and_expression:
        #     shift_expression
        #     and_expression '&' shift_expression
        # """
        expression = self.parse_shift_expression()
        while self.match(TokenID.Ampersand):
            token_operator = self.consume(TokenID.Ampersand)
            right_operand = self.parse_shift_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionNode(
                self.context,
                expression,
                token_operator,
                BINARY_IDS[token_operator.id],
                right_operand
            )
        return expression

    def parse_shift_expression(self) -> ExpressionNode:
        # """
        # shift_expression:
        #     addition_expression
        #     shift_expression '<<' addition_expression
        #     shift_expression '>>' addition_expression
        # """
        expression = self.parse_addition_expression()
        while self.match_any({TokenID.LeftShift, TokenID.RightShift}):
            token_operator = self.consume()
            right_operand = self.parse_addition_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionNode(
                self.context,
                expression,
                token_operator,
                BINARY_IDS[token_operator.id],
                right_operand
            )
        return expression

    def parse_addition_expression(self) -> ExpressionNode:
        # """
        # addition_expression:
        #     multiplication_expression
        #     addition_expression '+' multiplication_expression
        #     addition_expression '-' multiplication_expression
        # """
        expression = self.parse_multiplication_expression()
        while self.match_any({TokenID.Plus, TokenID.Minus}):
            token_operator = self.consume()
            right_operand = self.parse_unary_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionNode(
                self.context,
                expression,
                token_operator,
                BINARY_IDS[token_operator.id],
                right_operand
            )
        return expression

    def parse_multiplication_expression(self) -> ExpressionNode:
        # """
        # multiplication_expression:
        #     unary_expression
        #     multiplication_expression '*' unary_expression
        #     multiplication_expression '@' multiplication_expression
        #     multiplication_expression '//' unary_expression
        #     multiplication_expression '/' unary_expression
        #     multiplication_expression '%' unary_expression
        # """
        expression = self.parse_unary_expression()
        while self.match_any({TokenID.Star, TokenID.At, TokenID.Slash, TokenID.DoubleSlash, TokenID.Percent}):
            token_operator = self.consume()
            right_operand = self.parse_unary_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionNode(
                self.context,
                expression,
                token_operator,
                BINARY_IDS[token_operator.id],
                right_operand)

        return expression

    def parse_unary_expression(self) -> ExpressionNode:
        # """
        # unary_expression:
        #     power_expression
        #     "-" unary_expression
        #     "+" unary_expression
        #     "~" unary_expression
        # """
        while self.match_any({TokenID.Minus, TokenID.Plus, TokenID.Tilde}):
            token_operator = self.consume()
            operand = self.parse_unary_expression()

            # noinspection PyArgumentList
            return UnaryExpressionNode(self.context, token_operator, UNARY_IDS[token_operator.id], operand)

        return self.parse_power_expression()

    def parse_power_expression(self) -> ExpressionNode:
        # """
        # power_expression:
        #     primary_expression ["**" unary_expression]
        # """
        expression = self.parse_primary_expression()
        if self.match(TokenID.DoubleStar):
            token_operator = self.consume(TokenID.DoubleStar)
            right_operand = self.parse_unary_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionNode(
                self.context,
                expression,
                token_operator,
                BINARY_IDS[token_operator.id],
                right_operand
            )
        return expression

    def parse_primary_expression(self) -> Optional[ExpressionNode]:
        # """
        # primary:
        #      number_expression
        #      name_expression
        #      parenthesis_expression
        #      array_expression
        #      set_expression
        #      dict_expression
        #      ending_expression
        # """
        if self.match(TokenID.Integer):
            expression = self.parse_integer_expression()
        elif self.match(TokenID.String):
            expression = self.parse_string_expression()
        elif self.match(TokenID.Name):
            expression = self.parse_named_expression()
        elif self.match(TokenID.LeftParenthesis):
            expression = self.parse_parenthesis_expression()
        elif self.match(TokenID.LeftSquare):
            expression = self.parse_array_expression()
        elif self.match(TokenID.LeftCurly):
            expression = self.parse_set_or_dict_expression()
        else:
            self.error(PRIMARY_STARTS)
            return IntegerExpressionNode(self.context, self.create_error_token(TokenID.Integer))

        return self.parse_ending_expression(expression)

    def parse_integer_expression(self) -> ExpressionNode:
        # """
        # number:
        #     Number
        # """
        token_number = self.consume(TokenID.Integer)

        # noinspection PyArgumentList
        return IntegerExpressionNode(self.context, token_number)

    def parse_string_expression(self) -> ExpressionNode:
        # """
        # number:
        #     Number
        # """
        token_string = self.consume(TokenID.String)

        # noinspection PyArgumentList
        return StringExpressionNode(self.context, token_string)

    def parse_named_expression(self) -> ExpressionNode:
        # """
        # name:
        #     Name
        # """
        token_name = self.consume(TokenID.Name)

        # noinspection PyArgumentList
        return NamedExpressionNode(self.context, token_name)

    def parse_ending_expression(self, expression: ExpressionNode) -> ExpressionNode:
        # """
        # ending_expression:
        #     call_expression
        #     slice_expression
        #     subscribe_expression
        #     attribute_expression
        # """
        while self.match_any({TokenID.LeftParenthesis, TokenID.LeftSquare, TokenID.Dot}):
            if self.match(TokenID.LeftParenthesis):
                expression = self.parse_call_expression(expression)
            elif self.match(TokenID.LeftSquare):
                expression = self.parse_subscribe_expression(expression)
            elif self.match(TokenID.Dot):
                expression = self.parse_attribute_expression(expression)
        return expression

    def parse_call_expression(self, expression: ExpressionNode) -> ExpressionNode:
        # """
        # call_expression
        #     atom '(' arguments ')'
        # """
        token_open = self.consume(TokenID.LeftParenthesis)
        arguments = self.parse_named_arguments()
        token_close = self.consume(TokenID.RightParenthesis)

        # noinspection PyArgumentList
        return CallExpressionNode(self.context, expression, token_open, arguments, token_close)

    def parse_subscribe_expression(self, expression: ExpressionNode) -> ExpressionNode:
        # """
        # subscribe_expression
        #     atom '[' slice_arguments ']'
        # """
        token_open = self.consume(TokenID.LeftSquare)
        arguments = self.parse_subscribe_arguments()
        token_close = self.consume(TokenID.RightSquare)

        # noinspection PyArgumentList
        return SubscribeExpressionNode(self.context, expression, token_open, arguments, token_close)

    def parse_attribute_expression(self, expression: ExpressionNode) -> ExpressionNode:
        # """
        # attribute_expression:
        #     atom '.' Name
        # """
        token_dot = self.consume(TokenID.Dot)
        token_name = self.consume(TokenID.Name)

        # noinspection PyArgumentList
        return AttributeExpressionNode(self.context, expression, token_dot, token_name)

    def parse_parenthesis_expression(self) -> ExpressionNode:
        # """
        # parenthesis_expression:
        #     '(' expression ')'
        # """
        token_open = self.consume(TokenID.LeftParenthesis)
        arguments = self.parse_arguments()
        token_close = self.consume(TokenID.RightParenthesis)

        return TupleExpressionNode(self.context, token_open, arguments, token_close)

    def parse_array_expression(self):
        # """
        # array_expression:
        #     '(' expression ')'
        # """
        token_open = self.consume(TokenID.LeftSquare)
        arguments = self.parse_arguments()
        token_close = self.consume(TokenID.RightSquare)
        return ArrayExpressionNode(self.context, token_open, arguments, token_close)

    def parse_set_or_dict_expression(self):
        # """
        # set_or_dict_expression:
        #     set_expression
        #     dict_expression
        # """
        token_open = self.consume(TokenID.LeftCurly)
        if not self.match_any(EXPRESSION_STARTS):
            token_close = self.consume(TokenID.RightCurly)
            return DictExpressionNode(self.context, token_open, SyntaxCollection(), token_close)

        key = self.parse_expression()
        if self.match(TokenID.Colon):
            token_colon = self.consume()
            value = self.parse_expression()
            return self.parse_dict_expression(token_open, [DictArgumentNode(self.context, key, token_colon, value)])
        return self.parse_set_expression(token_open, [key])

    def parse_set_expression(self, token_open: SyntaxToken, arguments: MutableSequence[ExpressionNode]):
        # """
        # set_expression:
        #     '{' arguments '}'
        # """
        arguments = self.parse_arguments(arguments)
        token_close = self.consume(TokenID.RightCurly)
        return SetExpressionNode(self.context, token_open, arguments, token_close)

    def parse_dict_expression(self, token_open: SyntaxToken, arguments: MutableSequence[DictArgumentNode]):
        # """
        # dict_expression:
        #     '{' dict_arguments '}'
        # """
        arguments = self.parse_dict_arguments(arguments)
        token_close = self.consume(TokenID.RightCurly)
        return DictExpressionNode(self.context, token_open, arguments, token_close)

    def parse_subscribe_arguments(self):
        # """
        # subscribe_arguments:
        #     [ subscribe_argument { ',' subscribe_argument [','] }  ]
        # """
        if not self.match_any(SUBSCRIBE_ARGUMENT_STARTS):
            return tuple()

        arguments = [self.parse_subscribe_argument()]
        while self.match(TokenID.Comma):
            self.consume(TokenID.Comma)
            if self.match_any(SUBSCRIBE_ARGUMENT_STARTS):
                arguments.append(self.parse_subscribe_argument())
            else:
                break
        return SyntaxCollection(arguments)

    def parse_subscribe_argument(self) -> Union[ExpressionNode, SliceArgumentNode]:
        # """
        # subscribe_argument:
        #     expression
        #     [ expression ] ':' [ expression ] [ ":" [ expression ] ]
        # """
        lower_bound = None
        upper_bound = None
        stride = None

        if self.match_any(EXPRESSION_STARTS):
            lower_bound = self.parse_expression()

            if not self.match(TokenID.Colon):
                return lower_bound

        self.consume(TokenID.Colon)
        if self.match_any(EXPRESSION_STARTS):
            upper_bound = self.parse_expression()

        if self.match(TokenID.Colon):
            self.consume()

            if self.match_any(EXPRESSION_STARTS):
                stride = self.parse_expression()

        token = lower_bound or upper_bound or stride or self.current_token
        return SliceArgumentNode(self.context, lower_bound, upper_bound, stride, token.location)
