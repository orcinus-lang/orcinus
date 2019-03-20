# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import contextlib
import io
import sys

from orcinus.parser import Parser
from orcinus.scanner import Scanner
from orcinus.syntax import *


@contextlib.contextmanager
def parse(content: str, is_newline=False, can_errors=False) -> Parser:
    # Create stream for parsing
    stream = io.StringIO(content)
    if is_newline:
        stream.seek(0, io.SEEK_END)
        stream.write("\n")
        stream.seek(0, io.SEEK_SET)

    # Create parser
    context = SyntaxContext()
    with Scanner('test', stream, diagnostics=context.diagnostics) as scanner:
        parser = Parser('test', scanner, context)
        yield parser

    # Check if parser has errors
    if can_errors and context.diagnostics.has_errors:
        for diagnostic in context.diagnostics:
            sys.stderr.write(str(diagnostic))
        assert not context.diagnostics.has_errors

    # Consume new line or end of file
    while parser.current_token.id == TokenID.NewLine:
        parser.consume()

    # Check if all content is read
    assert parser.match(TokenID.EndOfFile), "Can not fully parsed string"


def parse_string(content: str, can_errors=False) -> SyntaxTree:
    with parse(content, can_errors=can_errors) as parser:
        return parser.parse()


def parse_expression(content: str) -> ExpressionNode:
    with parse(content) as parser:
        return parser.parse_expression()


def parse_statement(content: str) -> StatementNode:
    with parse(content, True) as parser:
        return parser.parse_statement()


# def test_function():
#     tree = parse_string("""
# def main() -> int: ...
#     """)
#
#     assert len(tree.members) == 1
#     func = tree.members[0]
#     assert isinstance(func, FunctionNode)
#     assert func.name == 'main'
#     assert len(func.parameters) == 0
#
#
# def test_incorrect_function():
#     tree = parse_string("""
# def main(a: , b: int) -> :
# """, True)
#
#     assert len(tree.members) == 1
#
#     func = tree.members[0]
#     assert isinstance(func, FunctionNode)
#     assert func.name == 'main'
#     assert len(func.parameters) == 2
#
#     arg1 = func.parameters[0]
#     assert isinstance(arg1, ParameterNode)
#     assert arg1.name == 'a'
#     assert isinstance(arg1.type, NamedTypeNode)
#     assert arg1.type.name == ''
#
#     arg2 = func.parameters[1]
#     assert isinstance(arg2, ParameterNode)
#     assert arg2.name == 'b'


def test_parse_unary_expressions():
    expr = parse_expression("+1")
    assert isinstance(expr, UnaryExpressionNode)
    assert isinstance(expr.argument, IntegerExpressionNode)
    assert expr.opcode == UnaryID.Pos
    assert expr.token_operator.id == TokenID.Plus

    expr = parse_expression("-1")
    assert isinstance(expr, UnaryExpressionNode)
    assert isinstance(expr.argument, IntegerExpressionNode)
    assert expr.opcode == UnaryID.Neg
    assert expr.token_operator.id == TokenID.Minus

    expr = parse_expression("not 1")
    assert isinstance(expr, UnaryExpressionNode)
    assert isinstance(expr.argument, IntegerExpressionNode)
    assert expr.opcode == UnaryID.Not
    assert expr.token_operator.id == TokenID.Not

    expr = parse_expression("~1")
    assert isinstance(expr, UnaryExpressionNode)
    assert isinstance(expr.argument, IntegerExpressionNode)
    assert expr.opcode == UnaryID.Inv
    assert expr.token_operator.id == TokenID.Tilde


def test_parse_binary_expressions():
    expr = parse_expression("1 + 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Add
    assert expr.token_operator.id == TokenID.Plus

    expr = parse_expression("1 - 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Sub
    assert expr.token_operator.id == TokenID.Minus

    expr = parse_expression("1 * 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Mul
    assert expr.token_operator.id == TokenID.Star

    expr = parse_expression("1 @ 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.MatrixMul
    assert expr.token_operator.id == TokenID.At

    expr = parse_expression("1 / 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.TrueDiv
    assert expr.token_operator.id == TokenID.Slash

    expr = parse_expression("1 // 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.FloorDiv
    assert expr.token_operator.id == TokenID.DoubleSlash

    expr = parse_expression("1 % 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Mod
    assert expr.token_operator.id == TokenID.Percent

    expr = parse_expression("1 | 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Or
    assert expr.token_operator.id == TokenID.VerticalLine

    expr = parse_expression("1 & 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.And
    assert expr.token_operator.id == TokenID.Ampersand

    expr = parse_expression("1 ^ 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Xor
    assert expr.token_operator.id == TokenID.Circumflex

    expr = parse_expression("1 << 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.LeftShift
    assert expr.token_operator.id == TokenID.LeftShift

    expr = parse_expression("1 >> 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.RightShift
    assert expr.token_operator.id == TokenID.RightShift

    expr = parse_expression("1 ** 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Pow
    assert expr.token_operator.id == TokenID.DoubleStar


def test_parse_logical_expressions():
    expr = parse_expression("1 and 1")
    assert isinstance(expr, LogicExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == LogicID.And
    assert expr.token_operator.id == TokenID.And

    expr = parse_expression("1 or 1")
    assert isinstance(expr, LogicExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == LogicID.Or
    assert expr.token_operator.id == TokenID.Or


def test_parse_compare_expression():
    expr = parse_expression("1 == 1")
    assert isinstance(expr, CompareExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert len(expr.comparators) == 1
    assert isinstance(expr.comparators[0].right_argument, IntegerExpressionNode)
    assert expr.comparators[0].opcode == CompareID.Equal
    assert expr.comparators[0].token_prefix.id == TokenID.DoubleEqual
    assert expr.comparators[0].token_suffix is None

    expr = parse_expression("1 != 1")
    assert isinstance(expr, CompareExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert len(expr.comparators) == 1
    assert isinstance(expr.comparators[0].right_argument, IntegerExpressionNode)
    assert expr.comparators[0].opcode == CompareID.NotEqual
    assert expr.comparators[0].token_prefix.id == TokenID.NotEqual
    assert expr.comparators[0].token_suffix is None

    expr = parse_expression("1 < 1")
    assert isinstance(expr, CompareExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert len(expr.comparators) == 1
    assert isinstance(expr.comparators[0].right_argument, IntegerExpressionNode)
    assert expr.comparators[0].opcode == CompareID.Less
    assert expr.comparators[0].token_prefix.id == TokenID.Less
    assert expr.comparators[0].token_suffix is None

    expr = parse_expression("1 <= 1")
    assert isinstance(expr, CompareExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert len(expr.comparators) == 1
    assert isinstance(expr.comparators[0].right_argument, IntegerExpressionNode)
    assert expr.comparators[0].opcode == CompareID.LessEqual
    assert expr.comparators[0].token_prefix.id == TokenID.LessEqual
    assert expr.comparators[0].token_suffix is None

    expr = parse_expression("1 > 1")
    assert isinstance(expr, CompareExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert len(expr.comparators) == 1
    assert isinstance(expr.comparators[0].right_argument, IntegerExpressionNode)
    assert expr.comparators[0].opcode == CompareID.Great
    assert expr.comparators[0].token_prefix.id == TokenID.Great
    assert expr.comparators[0].token_suffix is None

    expr = parse_expression("1 >= 1")
    assert isinstance(expr, CompareExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert len(expr.comparators) == 1
    assert isinstance(expr.comparators[0].right_argument, IntegerExpressionNode)
    assert expr.comparators[0].opcode == CompareID.GreatEqual
    assert expr.comparators[0].token_prefix.id == TokenID.GreatEqual
    assert expr.comparators[0].token_suffix is None

    expr = parse_expression("1 is 1")
    assert isinstance(expr, CompareExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert len(expr.comparators) == 1
    assert isinstance(expr.comparators[0].right_argument, IntegerExpressionNode)
    assert expr.comparators[0].opcode == CompareID.Is
    assert expr.comparators[0].token_prefix.id == TokenID.Is
    assert expr.comparators[0].token_suffix is None

    expr = parse_expression("1 is not 1")
    assert isinstance(expr, CompareExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert len(expr.comparators) == 1
    assert isinstance(expr.comparators[0].right_argument, IntegerExpressionNode)
    assert expr.comparators[0].opcode == CompareID.IsNot
    assert expr.comparators[0].token_prefix.id == TokenID.Is
    assert expr.comparators[0].token_suffix.id == TokenID.Not

    expr = parse_expression("1 in 1")
    assert isinstance(expr, CompareExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert len(expr.comparators) == 1
    assert isinstance(expr.comparators[0].right_argument, IntegerExpressionNode)
    assert expr.comparators[0].opcode == CompareID.In
    assert expr.comparators[0].token_prefix.id == TokenID.In
    assert expr.comparators[0].token_suffix is None

    expr = parse_expression("1 not in 1")
    assert isinstance(expr, CompareExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert len(expr.comparators) == 1
    assert isinstance(expr.comparators[0].right_argument, IntegerExpressionNode)
    assert expr.comparators[0].opcode == CompareID.NotIn
    assert expr.comparators[0].token_prefix.id == TokenID.Not
    assert expr.comparators[0].token_suffix.id == TokenID.In


def test_parse_condition_expression():
    expr = parse_expression("1 if 1 else 0")
    assert isinstance(expr, ConditionExpressionNode)
    assert isinstance(expr.condition, IntegerExpressionNode)
    assert isinstance(expr.then_value, IntegerExpressionNode)
    assert isinstance(expr.else_value, IntegerExpressionNode)
    assert expr.token_if.id == TokenID.If
    assert expr.token_else.id == TokenID.Else


def test_parse_augmented_assignment():
    expr = parse_statement("a += 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Add
    assert expr.token_operator.id == TokenID.PlusEqual

    expr = parse_statement("a -= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Sub
    assert expr.token_operator.id == TokenID.MinusEqual

    expr = parse_statement("a *= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Mul
    assert expr.token_operator.id == TokenID.StarEqual

    expr = parse_statement("a @= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.MatrixMul
    assert expr.token_operator.id == TokenID.AtEqual

    expr = parse_statement("a /= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.TrueDiv
    assert expr.token_operator.id == TokenID.SlashEqual

    expr = parse_statement("a //= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.FloorDiv
    assert expr.token_operator.id == TokenID.DoubleSlashEqual

    expr = parse_statement("a %= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Mod
    assert expr.token_operator.id == TokenID.PercentEqual

    expr = parse_statement("a |= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Or
    assert expr.token_operator.id == TokenID.VerticalLineEqual

    expr = parse_statement("a &= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.And
    assert expr.token_operator.id == TokenID.AmpersandEqual

    expr = parse_statement("a ^= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Xor
    assert expr.token_operator.id == TokenID.CircumflexEqual

    expr = parse_statement("a <<= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.LeftShift
    assert expr.token_operator.id == TokenID.LeftShiftEqual

    expr = parse_statement("a >>= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.RightShift
    assert expr.token_operator.id == TokenID.RightShiftEqual

    expr = parse_statement("a **= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Pow
    assert expr.token_operator.id == TokenID.DoubleStarEqual
