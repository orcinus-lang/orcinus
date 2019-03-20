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

    expr = parse_expression("-1")
    assert isinstance(expr, UnaryExpressionNode)
    assert isinstance(expr.argument, IntegerExpressionNode)
    assert expr.opcode == UnaryID.Neg

    expr = parse_expression("not 1")
    assert isinstance(expr, UnaryExpressionNode)
    assert isinstance(expr.argument, IntegerExpressionNode)
    assert expr.opcode == UnaryID.Not

    expr = parse_expression("~1")
    assert isinstance(expr, UnaryExpressionNode)
    assert isinstance(expr.argument, IntegerExpressionNode)
    assert expr.opcode == UnaryID.Inv


def test_parse_binary_expressions():
    expr = parse_expression("1 + 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Add

    expr = parse_expression("1 - 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Sub

    expr = parse_expression("1 * 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Mul

    expr = parse_expression("1 @ 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.MatrixMul

    expr = parse_expression("1 / 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.TrueDiv

    expr = parse_expression("1 // 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.FloorDiv

    expr = parse_expression("1 % 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Mod

    expr = parse_expression("1 | 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Or

    expr = parse_expression("1 & 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.And

    expr = parse_expression("1 ^ 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Xor

    expr = parse_expression("1 << 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.LeftShift

    expr = parse_expression("1 >> 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.RightShift

    expr = parse_expression("1 ** 1")
    assert isinstance(expr, BinaryExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Pow


def test_parse_logical_expressions():
    expr = parse_expression("1 and 1")
    assert isinstance(expr, LogicExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == LogicID.And

    expr = parse_expression("1 or 1")
    assert isinstance(expr, LogicExpressionNode)
    assert isinstance(expr.left_argument, IntegerExpressionNode)
    assert isinstance(expr.right_argument, IntegerExpressionNode)
    assert expr.opcode == LogicID.Or


def test_parse_augmented_assignment():
    expr = parse_statement("a += 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Add

    expr = parse_statement("a -= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Sub

    expr = parse_statement("a *= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Mul

    expr = parse_statement("a @= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.MatrixMul

    expr = parse_statement("a /= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.TrueDiv

    expr = parse_statement("a //= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.FloorDiv

    expr = parse_statement("a %= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Mod

    expr = parse_statement("a |= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Or

    expr = parse_statement("a &= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.And

    expr = parse_statement("a ^= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Xor

    expr = parse_statement("a <<= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.LeftShift

    expr = parse_statement("a >>= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.RightShift

    expr = parse_statement("a **= 1")
    assert isinstance(expr, AugmentedAssignStatementNode)
    assert isinstance(expr.target, NamedExpressionNode)
    assert isinstance(expr.source, IntegerExpressionNode)
    assert expr.opcode == BinaryID.Pow
