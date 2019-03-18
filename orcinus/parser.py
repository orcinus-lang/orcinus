# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from collections import deque as Queue
from typing import Set, MutableSequence

from orcinus.exceptions import DiagnosticError
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
}
SUBSCRIBE_ARGUMENT_STARTS = EXPRESSION_STARTS | {TokenID.Colon}
STATEMENT_STARTS: Set[TokenID] = EXPRESSION_STARTS | {
    TokenID.Break,
    TokenID.Continue,
    TokenID.For,
    TokenID.If,
    TokenID.Raise,
    TokenID.Return,
    TokenID.Try,
    TokenID.While,
    TokenID.With,
    TokenID.Yield,
}
COMPARISON_STARTS: Set[TokenID] = {
    TokenID.EqEqual,
    TokenID.NotEqual,
    TokenID.Less,
    TokenID.LessEqual,
    TokenID.Great,
    TokenID.GreatEqual,
    TokenID.Not,
    TokenID.Is,
    TokenID.In,
}
AUGMENTED_STATEMENT_STARTS: Set[TokenID] = {TokenID.PlusEqual, TokenID.MinusEqual}
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
    TokenID.EqEqual: CompareID.Eq,
    TokenID.NotEqual: CompareID.Ne,
    TokenID.Less: CompareID.Lt,
    TokenID.LessEqual: CompareID.Le,
    TokenID.Great: CompareID.Gt,
    TokenID.GreatEqual: CompareID.Ge
}


class Parser:
    def __init__(self, name: str, scanner: Scanner, context: SyntaxContext):
        self.name = name
        self.context = context
        self.scanner = scanner
        self.current_token = self.scanner.consume_token()
        self.tokens = Queue()
        self.diagnostics = context.diagnostics

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
            token = self.current_token
            if self.tokens:
                self.current_token = self.tokens.popleft()
            else:
                self.current_token = self.scanner.consume_token()
            return token

        raise self.error(indexes)

    def unput(self, token: SyntaxToken):
        self.tokens.append(self.current_token)
        self.current_token = token

    def error(self, indexes: Set[TokenID]):
        # generate exception message
        existed_name = self.current_token.id.name
        if len(indexes) > 1:
            required_names = []
            for x in indexes:
                required_names.append('`{}`'.format(x.name))
            message = "Required one of {}, but got `{}`".format(', '.join(required_names), existed_name)
        else:
            required_name = next(iter(indexes), None).name
            message = "Required `{}`, but got `{}`".format(required_name, existed_name)

        self.diagnostics.error(self.current_token.location, message)
        raise DiagnosticError(self.current_token.location, message)

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

    def parse_import(self) -> ImportNode:
        # """
        # import:
        #     'from' module_name 'import' aliases
        #     'import' aliases
        # """
        if self.match(TokenID.From):
            tok_from = self.consume()
            token_name = self.parse_qualified_name()
            self.consume(TokenID.Import)
            aliases = self.parse_aliases()
            self.consume(TokenID.NewLine)
            return ImportFromNode(self.context, token_name.value, aliases, tok_from.location)

        elif self.match(TokenID.Import):
            tok_import = self.consume()
            aliases = self.parse_aliases()
            self.consume(TokenID.NewLine)
            return ImportNode(self.context, aliases, tok_import.location)

        raise self.error(IMPORT_STARTS)

    def parse_aliases(self) -> SyntaxCollection[AliasNode]:
        # """
        # aliases:
        #     alias { ',' alias }
        #     '(' alias { ',' alias } ')'
        # """
        is_parenthesis = self.match(TokenID.LeftParenthesis)
        if is_parenthesis:
            self.consume()
        aliases = [self.parse_alias()]
        while self.match(TokenID.Comma):
            self.consume()
            aliases.append(self.parse_alias())
        if is_parenthesis:
            self.consume(TokenID.RightParenthesis)
        return SyntaxCollection[AliasNode](aliases)

    def parse_alias(self) -> AliasNode:
        # """
        # alias:
        #     module_name [ 'as' Name ]
        # """
        token_name = self.parse_qualified_name()
        if self.match(TokenID.As):
            self.consume(TokenID.As)
            token_alias = self.consume(TokenID.Name)
            return AliasNode(self.context, token_name.value, token_alias.value, token_name.location)
        return AliasNode(self.context, token_name.value, None, token_name.location)

    def parse_qualified_name(self) -> SyntaxToken:
        # """
        # module_name:
        #     Name { '.' Name }
        # """
        names = [self.consume(TokenID.Name)]
        while self.match(TokenID.Dot):
            self.consume()
            self.consume(TokenID.Name)

        items = []
        for t in names:
            items.append(t.value)
        name = '.'.join(items)
        location = names[0].location + names[-1].location
        return SyntaxToken(id=TokenID.Name, value=name, location=location)

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
        self.consume(TokenID.At)
        token_name = self.parse_qualified_name()
        arguments = SyntaxCollection[ArgumentNode]()
        if self.match(TokenID.LeftParenthesis):
            self.consume(TokenID.LeftParenthesis)
            arguments = self.parse_named_arguments()
            self.consume(TokenID.RightParenthesis)
        self.consume(TokenID.NewLine)

        return DecoratorNode(self.context, token_name.value, arguments, token_name.location)

    def parse_members(self) -> SyntaxCollection[MemberNode]:
        # """
        # members:
        #     { member }
        # """
        members = []
        while self.match_any(MEMBER_STARTS):
            members.append(self.parse_member())

        return SyntaxCollection[MemberNode](members)

    def parse_member(self):
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
        return self.parse_pass_member()

    def parse_pass_member(self):
        # """
        # pass:
        #     "pass"
        # """
        token_pass = self.consume(TokenID.Pass)
        self.consume(TokenID.NewLine)
        return PassMemberNode(self.context, token_pass.location)

    def parse_decorated_member(self) -> MemberNode:
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

        raise self.error(DECORATED_STARTS)

    def parse_function(self, decorators=None):
        # """
        # function:
        #     decorators 'def' Name generic_parameters arguments [ -> type ] ':' '...'
        # """
        token_def = self.consume(TokenID.Def)
        token_name = self.consume(TokenID.Name)
        generic_parameters = self.parse_generic_parameters()
        parameters = self.parse_function_parameters()

        if self.match(TokenID.Then):
            token_then = self.consume()
            result_type = self.parse_type()
        else:
            token_then = None
            result_type = AutoTypeNode(self.context, token_name.location)

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

    def parse_enum(self, decorators=None):
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
        members = self.parse_type_members()

        return EnumNode(self.context,
                        decorators or SyntaxCollection[DecoratorNode](),
                        token_enum,
                        token_name,
                        generic_parameters,
                        parents,
                        token_colon,
                        token_newline,
                        members)

    def parse_interface(self, decorators=None):
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
        members = self.parse_type_members()

        return InterfaceNode(self.context,
                             decorators or SyntaxCollection[DecoratorNode](),
                             token_interface,
                             token_name,
                             generic_parameters,
                             parents,
                             token_colon,
                             token_newline,
                             members)

    def parse_class(self, decorators=None):
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
        members = self.parse_type_members()

        return ClassNode(self.context,
                         decorators or SyntaxCollection[DecoratorNode](),
                         token_class,
                         token_name,
                         generic_parameters,
                         parents,
                         token_colon,
                         token_newline,
                         members)

    def parse_struct(self, decorators=None):
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
        members = self.parse_type_members()

        return StructNode(self.context,
                          decorators or SyntaxCollection[DecoratorNode](),
                          token_struct,
                          token_name,
                          generic_parameters,
                          parents,
                          token_colon,
                          token_newline,
                          members)

    def parse_named_member(self):
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

        raise self.error(NAMED_MEMBER_STARTS)

    def parse_enum_member(self, token_name: SyntaxToken):
        # """
        # enum_member:
        #     Name '=' '...'
        #     Name '=' expression
        # """
        self.consume(TokenID.Equal)
        if self.match(TokenID.Ellipsis):
            self.consume()
            value = None
        else:
            value = self.parse_expression()
        self.consume(TokenID.NewLine)
        return EnumMemberNode(self.context, token_name.value, value, token_name.location)

    def parse_field_member(self, token_name: SyntaxToken):
        # """
        # field_member:
        #     Name ':' type
        # """
        self.consume(TokenID.Colon)
        field_type = self.parse_type()
        default_value = None
        if self.match(TokenID.Equal):
            self.consume()
            default_value = self.parse_expression_list()
        self.consume(TokenID.NewLine)
        return FieldNode(self.context, token_name.value, field_type, default_value, token_name.location)

    def parse_type_parents(self) -> SyntaxCollection[TypeNode]:
        # """
        # type_parents:
        #     [ '(' type { ',' type ) ') ]
        # """
        if not self.match(TokenID.LeftParenthesis):
            return SyntaxCollection[TypeNode]()

        self.consume(TokenID.LeftParenthesis)
        parents = [self.parse_type()]
        while self.match(TokenID.Comma):
            self.consume()
            parents.append(self.parse_type())
        self.consume(TokenID.RightParenthesis)

        return SyntaxCollection[TypeNode](parents)

    def parse_type_members(self) -> SyntaxCollection[MemberNode]:
        # """
        # type_members:
        #     Indent members Undent
        # """
        self.consume(TokenID.Indent)
        members = self.parse_members()
        self.consume(TokenID.Undent)
        return members

    def parse_generic_parameters(self) -> SyntaxCollection[GenericParameterNode]:
        # """
        # generic_parameters:
        #     [ '[' generic_parameter { ',' generic_parameter } '] ]
        # """
        if not self.match(TokenID.LeftSquare):
            return SyntaxCollection[GenericParameterNode]()

        self.consume(TokenID.LeftSquare)
        generic_parameters = [

            self.parse_generic_parameter()
        ]

        while self.match(TokenID.Comma):
            self.consume(TokenID.Comma)
            generic_parameters.append(self.parse_generic_parameter())

        self.consume(TokenID.RightSquare)
        return SyntaxCollection[GenericParameterNode](generic_parameters)

    def parse_generic_parameter(self):
        # """
        # generic_parameter:
        #     Name
        # """
        token_name = self.consume(TokenID.Name)
        return GenericParameterNode(self.context, token_name.value, token_name.location)

    def parse_function_parameters(self) -> SyntaxCollection[ParameterNode]:
        # """
        # function_parameters:
        #     '(' [ function_parameter { ',' function_parameter } ] ')'
        # """
        self.consume(TokenID.LeftParenthesis)
        parameters = []
        if self.match(TokenID.Name):
            parameters.append(self.parse_function_parameter())
            while self.match(TokenID.Comma):
                self.consume(TokenID.Comma)
                parameters.append(self.parse_function_parameter())
        self.consume(TokenID.RightParenthesis)
        return SyntaxCollection[ParameterNode](parameters)

    def parse_function_parameter(self):
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
            param_type = AutoTypeNode(self.context, token_name.location)

        if self.match(TokenID.Equal):
            token_equal = self.consume()
            default_value = self.parse_expression()
        else:
            token_equal = None
            default_value = None

        return ParameterNode(self.context, token_name, token_colon, param_type, token_equal, default_value)

    def parse_type(self):
        # """
        # type:
        #     Name
        #     Name '[' type_arguments ']'
        # """
        token_name = self.parse_qualified_name()
        result_type = NamedTypeNode(self.context, token_name)

        while self.match(TokenID.LeftSquare):
            arguments = self.parse_generic_arguments()
            result_type = ParameterizedTypeNode(self.context, result_type, arguments, token_name.location)

        return result_type

    def parse_generic_arguments(self) -> SyntaxCollection[TypeNode]:
        # """
        # generic_arguments:
        #     '[' type { ',' type} ']'
        # """
        self.consume(TokenID.LeftSquare)
        arguments = [self.parse_type()]
        while self.match(TokenID.Comma):
            self.consume(TokenID.Comma)
            arguments.append(self.parse_type())
        self.consume(TokenID.RightSquare)
        return SyntaxCollection[TypeNode](arguments)

    def parse_function_statement(self) -> Optional[StatementNode]:
        # """
        # function_statement:
        #     '...' EndFile
        #     NewLine block_statement
        # """
        if self.match(TokenID.Ellipsis):
            self.consume(TokenID.Ellipsis)
            self.consume(TokenID.NewLine)
            return None

        self.consume(TokenID.NewLine)
        return self.parse_block_statement()

    def parse_block_statement(self):
        # """
        # block_statement:
        #     Indent statement { statement } Undent
        # """
        token_indent = self.consume(TokenID.Indent)
        statements = [
            self.parse_statement()
        ]
        while self.match_any(STATEMENT_STARTS):
            statements.append(self.parse_statement())
        token_undent = self.consume(TokenID.Undent)

        # noinspection PyArgumentList
        return BlockStatementNode(self.context, token_indent, SyntaxCollection(statements), token_undent)

    def parse_statement(self):
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

        raise self.error(STATEMENT_STARTS)

    def parse_pass_statement(self):
        # """ pass_statement: pass """
        token_pass = self.consume(TokenID.Pass)
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return PassStatementNode(self.context, token_pass.location)

    def parse_return_statement(self):
        # """
        # return_statement
        #     'return' [ expression_list ]
        # """
        token_return = self.consume(TokenID.Return)
        value = self.parse_expression_list() if self.match_any(EXPRESSION_STARTS) else None
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return ReturnStatementNode(self.context, token_return, value)

    def parse_yield_statement(self):
        # """
        # yield_statement
        #     'yield' [ expression_list ]
        # """
        token_return = self.consume(TokenID.Yield)
        value = self.parse_expression_list() if self.match_any(EXPRESSION_STARTS) else None
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return YieldStatementNode(self.context, value, token_return.location)

    def parse_break_statement(self):
        # """
        # break_statement
        #     'break'
        # """
        token_break = self.consume(TokenID.Break)
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return BreakStatementNode(self.context, token_break.location)

    def parse_continue_statement(self):
        # """
        # continue_statement
        #     'continue'
        # """
        token_continue = self.consume(TokenID.Break)
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return ContinueStatementNode(self.context, token_continue.location)

    def parse_else_statement(self) -> StatementNode:
        # """
        # else_statement:
        #     'else' ':' '\n' block_statement
        # """
        self.consume(TokenID.Else)
        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        return self.parse_block_statement()

    def parse_finally_statement(self) -> StatementNode:
        # """
        # finally_statement:
        #     'finally' ':' '\n' block_statement
        # """
        self.consume(TokenID.Finally)
        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        return self.parse_block_statement()

    def parse_condition_statement(self, token_id: TokenID = TokenID.If):
        # """
        # condition_statement:
        #     'if' expression ':' '\n' block_statement            ⏎
        #         { 'elif' expression ':' '\n' block_statement }  ⏎
        #         [ else_statement ]
        # """
        token_if = self.consume(token_id)
        condition = self.parse_expression()
        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        then_statement = self.parse_block_statement()

        else_statement = None
        if self.match(TokenID.Else):
            else_statement = self.parse_else_statement()
        elif self.match(TokenID.Elif):
            else_statement = self.parse_condition_statement(TokenID.Elif)

        # noinspection PyArgumentList
        return ConditionStatementNode(self.context, condition, then_statement, else_statement, token_if.location)

    def parse_while_statement(self):
        # """
        # while_statement:
        #     'while' expression ':' '\n' block_statement     ⏎
        #         [ 'else' ':' '\n' block_statement ]
        # """
        token_while = self.consume(TokenID.While)
        condition = self.parse_expression()
        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        then_statement = self.parse_block_statement()
        else_statement = self.parse_else_statement() if self.match(TokenID.Else) else None

        # noinspection PyArgumentList
        return WhileStatementNode(self.context, condition, then_statement, else_statement, token_while.location)

    def parse_for_statement(self):
        # """
        # for_statement:
        #     'for' target_list 'in' expression_list ':' '\n' block_statement     ⏎
        #         [ 'else' ':' '\n' block_statement ]
        # """
        token_for = self.consume(TokenID.For)
        target = self.parse_target_list()
        self.consume(TokenID.In)
        source = self.parse_expression_list()
        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        then_statement = self.parse_block_statement()
        else_statement = self.parse_else_statement() if self.match(TokenID.Else) else None

        # noinspection PyArgumentList
        return ForStatementNode(
            self.context,
            target,
            source,
            then_statement,
            else_statement,
            token_for.location
        )

    def parse_raise_statement(self):
        # """
        # raise_statement:
        #     'raise' [ expression [ "from" expression ] ]
        # """
        token_raise = self.consume(TokenID.Raise)
        exception = None
        cause_exception = None
        if self.match_any(EXPRESSION_STARTS):
            exception = self.parse_expression()
            if self.match(TokenID.From):
                self.consume()
                cause_exception = self.parse_expression()

        self.consume(TokenID.NewLine)
        return RaiseStatementNode(self.context, exception, cause_exception, token_raise.location)

    def parse_try_statement(self):
        # """
        # try_statement:
        #     'try' ':' '\n' block_statement
        #         { except_handler }              ⏎
        #         [ else_statement ]              ⏎
        #         [ finally_statement ]
        #     'try' ':' '\n' block_statement finally_statement
        # """
        token_try = self.consume(TokenID.Try)
        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        try_statement = self.parse_block_statement()

        handlers = []
        else_statement = None

        if self.match(TokenID.Except):
            while self.match(TokenID.Except):
                handlers.append(self.parse_except_handler())

            else_statement = self.parse_else_statement() if self.match(TokenID.Else) else None
            finally_statement = self.parse_finally_statement() if self.match(TokenID.Finally) else None
        elif self.match(TokenID.Finally):
            finally_statement = self.parse_finally_statement()
        else:
            raise self.error({TokenID.Except, TokenID.Finally})

        return TryStatementNode(self.context, try_statement, SyntaxCollection(handlers), else_statement,
                                finally_statement, token_try.location)

    def parse_except_handler(self) -> ExceptHandlerNode:
        # """
        # except_handler:
        #     'except' [ expression [ 'as' Name ] ':' '\n' block_statement
        # """
        token_except = self.consume(TokenID.Except)
        expression = None
        name = None
        if self.match_any(EXPRESSION_STARTS):
            expression = self.parse_expression()
            if self.match(TokenID.As):
                self.consume()
                name = self.consume(TokenID.Name).value
        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        statement = self.parse_block_statement()

        return ExceptHandlerNode(self.context, expression, name, statement, token_except.location)

    def parse_with_statement(self):
        # """
        # with_statement:
        #     'with' with_item { ',' with_item } ':' '\n' block_statement
        # """
        token_with = self.consume(TokenID.With)
        items = [self.parse_with_item()]
        while self.match(TokenID.Comma):
            self.consume()
            items.append(self.parse_with_item())

        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        statement = self.parse_block_statement()

        return WithStatementNode(self.context, SyntaxCollection(items), statement, token_with.location)

    def parse_with_item(self):
        # """
        # with_item:
        #     expression [ 'as' target ]
        # """
        expression = self.parse_expression()
        target = None
        if self.match(TokenID.As):
            self.consume()
            target = self.parse_target()

        return WithItemNode(self.context, expression, target, expression.location)

    def parse_expression_statement(self):
        # """
        # expression_statement
        #     expression
        #     assign_expression
        # """
        expression = self.parse_expression_list()
        statement = None

        if self.match(TokenID.Equal):
            statement = self.parse_assign_statement(expression)
        elif self.match_any(AUGMENTED_STATEMENT_STARTS):
            statement = self.parse_augmented_assign_statement(expression)

        self.consume(TokenID.NewLine)
        if not statement:
            # noinspection PyArgumentList
            statement = ExpressionStatementNode(self.context, expression, expression.location)
        return statement

    def parse_assign_statement(self, target: ExpressionNode):
        # """
        # assign_expression
        #     targets_list '=' expression
        #
        # TODO: https://docs.python.org/3/reference/simple_stmts.html#grammar-token-assignment-stmt
        # """
        token_equals = self.consume(TokenID.Equal)
        source = self.parse_expression_list()

        # noinspection PyArgumentList
        return AssignStatementNode(self.context, target, source, token_equals.location)

    def parse_augmented_assign_statement(self, target: ExpressionNode):
        # """
        # assign_expression
        #     targets_list '=' expression
        #
        # TODO: https://docs.python.org/3/reference/simple_stmts.html#grammar-token-assignment-stmt
        # """
        if self.match(TokenID.PlusEqual):
            token_operator = self.consume()
            opcode = BinaryID.Add
        elif self.match(TokenID.MinusEqual):
            token_operator = self.consume()
            opcode = BinaryID.Sub
        else:
            raise self.error(AUGMENTED_STATEMENT_STARTS)
        source = self.parse_expression_list()

        # noinspection PyArgumentList
        return AugmentedAssignStatementNode(self.context, target, opcode, source, token_operator.location)

    def parse_arguments(self, arguments: MutableSequence[ExpressionNode] = None) -> SyntaxCollection[ExpressionNode]:
        # """
        # arguments:
        #     [ expression { ',' expression } [','] ]
        # """
        if not arguments and not self.match_any(EXPRESSION_STARTS):
            return SyntaxCollection[ExpressionNode]()

        arguments = arguments or []
        if not arguments:
            arguments.append(self.parse_expression())

        while self.match(TokenID.Comma):
            self.consume(TokenID.Comma)
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
            self.consume(TokenID.Comma)
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
        self.consume(TokenID.Colon)
        value = self.parse_expression()
        return DictArgumentNode(self.context, key, value, key.location)

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
                self.consume()
                value = self.parse_expression()
                return KeywordArgumentNode(self.context, token_name.value, value, token_name.location)
            self.unput(token_name)

        value = self.parse_expression()
        return PositionArgumentNode(self.context, value, value.location)

    def parse_target_list(self) -> ExpressionNode:
        # """
        # target_list:
        #     target { ',' target }
        # """

        targets = [
            self.parse_target()
        ]
        while self.match(TokenID.Comma):
            self.consume()
            targets.append(self.parse_target())

        if len(targets) == 1:
            return targets[0]
        return TupleExpressionNode(self.context, SyntaxCollection(targets), targets[0].location)

    def parse_target(self):
        return self.parse_primary_expression()

    def parse_expression_list(self) -> ExpressionNode:
        # """
        # expression_list:
        #     expression { ',' expression } [',']
        # """
        expressions = [self.parse_expression()]
        while self.match(TokenID.Comma):
            self.consume()
            if self.match_any(EXPRESSION_STARTS):
                expressions.append(self.parse_expression())
            else:
                break

        if len(expressions) == 1:
            return expressions[0]

        location = expressions[0].location + expressions[-1].location
        return TupleExpressionNode(self.context, SyntaxCollection(expressions), location)

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
            self.consume(TokenID.Else)
            else_expression = self.parse_expression()
            expression = ConditionExpressionNode(self.context, expression, condition, else_expression,
                                                 token_if.location)
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
            expression = LogicExpressionNode(self.context, LogicID.Or, expression, right_operand, token_or.location)
        return expression

    def parse_logical_and_expression(self) -> ExpressionNode:
        # """
        # logical_and_expression:
        #     logical_not_expression
        #     logical_and_expression 'and' logical_not_expression
        # """
        expression = self.parse_logical_not_expression()
        while self.match(TokenID.And):
            token_or = self.consume()
            right_operand = self.parse_logical_not_expression()
            expression = LogicExpressionNode(self.context, LogicID.And, expression, right_operand, token_or.location)
        return expression

    def parse_logical_not_expression(self) -> ExpressionNode:
        # """
        # logical_not_expression:
        #     comparison_expression
        #     "not" logical_not_expression
        # """
        while self.match(TokenID.Not):
            tok_not = self.consume(TokenID.Not)
            expression = self.parse_logical_not_expression()
            return UnaryExpressionNode(self.context, UnaryID.Not, expression, tok_not.location)
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
                self.consume()
                token_operator = self.consume(TokenID.In)
                opcode = CompareID.In

            elif self.match(TokenID.Is):
                token_operator = self.consume()
                if self.match(TokenID.Not):
                    self.consume()
                    opcode = CompareID.IsNot
                else:
                    opcode = CompareID.Is

            elif self.match(TokenID.In):
                token_operator = self.consume()
                opcode = CompareID.In

            else:
                token_operator = self.consume_any(COMPARISON_STARTS)
                opcode = COMPARISON_IDS[token_operator.id]

            right_operand = self.parse_and_expression()
            comparators.append(ComparatorNode(self.context, opcode, right_operand, token_operator.location))

        if comparators:
            return CompareExpressionNode(
                self.context, expression, SyntaxCollection[ComparatorNode](comparators), comparators[0].location)
        return expression

    def parse_or_expression(self) -> ExpressionNode:
        # """
        # or_expression:
        #     xor_expression
        #     or_expression '|' xor_expression
        # """
        expression = self.parse_and_expression()
        while self.match(TokenID.VerticalLine):
            token_operator = self.consume(TokenID.VerticalLine)
            right_operand = self.parse_and_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionNode(
                self.context,
                BinaryID.Or,
                expression,
                right_operand,
                token_operator.location
            )
        return expression

    def parse_xor_expression(self) -> ExpressionNode:
        # """
        # xor_expression:
        #     and_expression
        #     xor_expression '&' and_expression
        # """
        expression = self.parse_and_expression()
        while self.match(TokenID.Xor):
            token_operator = self.consume(TokenID.Xor)
            right_operand = self.parse_and_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionNode(
                operator=BinaryID.Xor,
                left_operand=expression,
                right_operand=right_operand,
                token_operator=token_operator
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
                operator=BinaryID.And,
                left_operand=expression,
                right_operand=right_operand,
                token_operator=token_operator
            )
        return expression

    def parse_shift_expression(self) -> ExpressionNode:
        # """
        # shift_expression:
        #     addition_expression
        # """
        return self.parse_addition_expression()

    def parse_addition_expression(self) -> ExpressionNode:
        # """
        # addition_expression:
        #     multiplication_expression
        #     addition_expression '+' multiplication_expression
        #     addition_expression '-' multiplication_expression
        # """
        expression = self.parse_multiplication_expression()
        while self.match_any({TokenID.Plus, TokenID.Minus}):
            if self.match(TokenID.Plus):
                token_operator = self.consume(TokenID.Plus)
                right_operand = self.parse_unary_expression()

                # noinspection PyArgumentList
                expression = BinaryExpressionNode(self.context, BinaryID.Add, expression, right_operand,
                                                  token_operator.location)
            elif self.match(TokenID.Minus):
                token_operator = self.consume(TokenID.Minus)
                right_operand = self.parse_unary_expression()

                # noinspection PyArgumentList
                expression = BinaryExpressionNode(self.context, BinaryID.Sub, expression, right_operand,
                                                  token_operator.location)
        return expression

    def parse_multiplication_expression(self) -> ExpressionNode:
        # """
        # multiplication_expression:
        #     unary_expression
        #     multiplication_expression '*' unary_expression
        #     # multiplication_expression '@' multiplication_expression
        #     multiplication_expression '//' unary_expression
        #     multiplication_expression '/' unary_expression
        #     # multiplication_expression '%' unary_expression
        # """
        expression = self.parse_unary_expression()
        while self.match_any({TokenID.Star, TokenID.Slash, TokenID.DoubleSlash}):
            if self.match(TokenID.Star):
                token_operator = self.consume(TokenID.Star)
                right_operand = self.parse_unary_expression()

                # noinspection PyArgumentList
                expression = BinaryExpressionNode(self.context, BinaryID.Mul, expression, right_operand,
                                                  token_operator.location)

            elif self.match(TokenID.Slash):
                token_operator = self.consume(TokenID.Slash)
                right_operand = self.parse_unary_expression()

                # noinspection PyArgumentList
                expression = BinaryExpressionNode(self.context, BinaryID.TrueDiv, expression, right_operand,
                                                  token_operator.location)

            elif self.match(TokenID.DoubleSlash):
                token_operator = self.consume(TokenID.DoubleSlash)
                right_operand = self.parse_unary_expression()

                # noinspection PyArgumentList
                expression = BinaryExpressionNode(self.context, BinaryID.FloorDiv, expression, right_operand,
                                                  token_operator.location)

        return expression

    def parse_unary_expression(self) -> ExpressionNode:
        # """
        # u_expr:
        #     power
        #     "-" u_expr
        #     "+" u_expr
        #     "~" u_expr
        # """
        if self.match(TokenID.Minus):
            token_operator = self.consume(TokenID.Minus)
            operand = self.parse_unary_expression()

            # noinspection PyArgumentList
            return UnaryExpressionNode(self.context, UnaryID.Neg, operand, token_operator.location)

        elif self.match(TokenID.Plus):
            token_operator = self.consume(TokenID.Plus)
            operand = self.parse_unary_expression()

            # noinspection PyArgumentList
            return UnaryExpressionNode(self.context, UnaryID.Pos, operand, token_operator.location)

        elif self.match(TokenID.Tilde):
            token_operator = self.consume(TokenID.Tilde)
            operand = self.parse_unary_expression()

            # noinspection PyArgumentList
            return UnaryExpressionNode(self.context, UnaryID.Inv, operand, token_operator.location)

        return self.parse_power_expression()

    def parse_power_expression(self) -> ExpressionNode:
        # """
        # power:
        #     primary ["**" u_expr]
        # """
        expression = self.parse_primary_expression()
        if self.match(TokenID.DoubleStar):
            token_operator = self.consume(TokenID.DoubleStar)
            unary_expression = self.parse_unary_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionNode(
                operator=BinaryID.Pow,
                left_operand=expression,
                right_operand=unary_expression,
                token_operator=token_operator
            )
        return expression

    def parse_primary_expression(self) -> ExpressionNode:
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
            raise self.error(PRIMARY_STARTS)

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
        return StringExpressionNode(self.context, token_string.value, token_string.location)

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
        self.consume(TokenID.LeftParenthesis)
        arguments = self.parse_named_arguments()
        self.consume(TokenID.RightParenthesis)

        # noinspection PyArgumentList
        return CallExpressionNode(self.context, expression, arguments, expression.location)

    def parse_subscribe_expression(self, expression: ExpressionNode) -> ExpressionNode:
        # """
        # subscribe_expression
        #     atom '[' slice_arguments ']'
        # """
        self.consume(TokenID.LeftSquare)
        arguments = self.parse_subscribe_arguments()
        self.consume(TokenID.RightSquare)

        # noinspection PyArgumentList
        return SubscribeExpressionNode(self.context, expression, arguments, expression.location)

    def parse_attribute_expression(self, expression: ExpressionNode) -> ExpressionNode:
        # """
        # attribute_expression:
        #     atom '.' Name
        # """
        token_dot = self.consume(TokenID.Dot)
        token_name = self.consume(TokenID.Name)

        # noinspection PyArgumentList
        return AttributeExpressionNode(self.context, expression, token_name.value, token_dot.location)

    def parse_parenthesis_expression(self) -> ExpressionNode:
        # """
        # parenthesis_expression:
        #     '(' expression ')'
        # """
        token_open = self.consume(TokenID.LeftParenthesis)
        arguments = self.parse_arguments()
        token_close = self.consume(TokenID.RightParenthesis)

        if len(arguments) == 1:
            return arguments[0]

        location = token_open.location + token_close.location
        return TupleExpressionNode(self.context, SyntaxCollection(arguments), location)

    def parse_array_expression(self):
        # """
        # array_expression:
        #     '(' expression ')'
        # """
        tok_array = self.consume(TokenID.LeftSquare)
        arguments = self.parse_arguments()
        self.consume(TokenID.RightSquare)
        return ArrayExpressionNode(self.context, arguments, tok_array.location)

    def parse_set_or_dict_expression(self):
        # """
        # set_or_dict_expression:
        #     set_expression
        #     dict_expression
        # """
        tok_open = self.consume(TokenID.LeftCurly)
        if not self.match_any(EXPRESSION_STARTS):
            tok_close = self.consume(TokenID.RightCurly)
            return DictExpressionNode(self.context, SyntaxCollection(), tok_open.location + tok_close.location)

        key = self.parse_expression()
        if self.match(TokenID.Colon):
            self.consume()
            value = self.parse_expression()
            return self.parse_dict_expression([DictArgumentNode(self.context, key, value, key.location)])
        return self.parse_set_expression([key])

    def parse_set_expression(self, arguments: MutableSequence[ExpressionNode]):
        # """
        # set_expression:
        #     '{' arguments '}'
        # """
        arguments = self.parse_arguments(arguments)
        tok_array = self.consume(TokenID.RightCurly)
        return SetExpressionNode(self.context, arguments, tok_array.location)

    def parse_dict_expression(self, arguments: MutableSequence[DictArgumentNode]):
        # """
        # dict_expression:
        #     '{' dict_arguments '}'
        # """
        arguments = self.parse_dict_arguments(arguments)
        tok_array = self.consume(TokenID.RightCurly)
        return DictExpressionNode(self.context, arguments, tok_array.location)

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
