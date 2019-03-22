# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from orcinus.syntax import *
from orcinus.tests.utils.parser import parse_statement, parse_expression, parse_member


def test_parse_unary():
    _, node = parse_expression("+1")
    assert isinstance(node, UnaryExpressionNode)
    assert isinstance(node.argument, IntegerExpressionNode)
    assert node.opcode == UnaryID.Pos
    assert node.token_operator.id == TokenID.Plus

    _, node = parse_expression("-1")
    assert isinstance(node, UnaryExpressionNode)
    assert isinstance(node.argument, IntegerExpressionNode)
    assert node.opcode == UnaryID.Neg
    assert node.token_operator.id == TokenID.Minus

    _, node = parse_expression("not 1")
    assert isinstance(node, UnaryExpressionNode)
    assert isinstance(node.argument, IntegerExpressionNode)
    assert node.opcode == UnaryID.Not
    assert node.token_operator.id == TokenID.Not

    _, node = parse_expression("~1")
    assert isinstance(node, UnaryExpressionNode)
    assert isinstance(node.argument, IntegerExpressionNode)
    assert node.opcode == UnaryID.Inv
    assert node.token_operator.id == TokenID.Tilde


def test_parse_binary():
    _, node = parse_expression("1 + 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.Add
    assert node.token_operator.id == TokenID.Plus

    _, node = parse_expression("1 - 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.Sub
    assert node.token_operator.id == TokenID.Minus

    _, node = parse_expression("1 * 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.Mul
    assert node.token_operator.id == TokenID.Star

    _, node = parse_expression("1 @ 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.MatrixMul
    assert node.token_operator.id == TokenID.At

    _, node = parse_expression("1 / 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.TrueDiv
    assert node.token_operator.id == TokenID.Slash

    _, node = parse_expression("1 // 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.FloorDiv
    assert node.token_operator.id == TokenID.DoubleSlash

    _, node = parse_expression("1 % 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.Mod
    assert node.token_operator.id == TokenID.Percent

    _, node = parse_expression("1 | 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.Or
    assert node.token_operator.id == TokenID.VerticalLine

    _, node = parse_expression("1 & 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.And
    assert node.token_operator.id == TokenID.Ampersand

    _, node = parse_expression("1 ^ 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.Xor
    assert node.token_operator.id == TokenID.Circumflex

    _, node = parse_expression("1 << 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.LeftShift
    assert node.token_operator.id == TokenID.LeftShift

    _, node = parse_expression("1 >> 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.RightShift
    assert node.token_operator.id == TokenID.RightShift

    _, node = parse_expression("1 ** 1")
    assert isinstance(node, BinaryExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == BinaryID.Pow
    assert node.token_operator.id == TokenID.DoubleStar


def test_parse_logical():
    _, node = parse_expression("1 and 1")
    assert isinstance(node, LogicExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == LogicID.And
    assert node.token_operator.id == TokenID.And

    _, node = parse_expression("1 or 1")
    assert isinstance(node, LogicExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert isinstance(node.right_argument, IntegerExpressionNode)
    assert node.opcode == LogicID.Or
    assert node.token_operator.id == TokenID.Or


def test_parse_compare():
    _, node = parse_expression("1 == 1")
    assert isinstance(node, CompareExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert len(node.comparators) == 1
    assert isinstance(node.comparators[0].right_argument, IntegerExpressionNode)
    assert node.comparators[0].opcode == CompareID.Equal
    assert node.comparators[0].token_prefix.id == TokenID.DoubleEqual
    assert node.comparators[0].token_suffix is None

    _, node = parse_expression("1 != 1")
    assert isinstance(node, CompareExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert len(node.comparators) == 1
    assert isinstance(node.comparators[0].right_argument, IntegerExpressionNode)
    assert node.comparators[0].opcode == CompareID.NotEqual
    assert node.comparators[0].token_prefix.id == TokenID.NotEqual
    assert node.comparators[0].token_suffix is None

    _, node = parse_expression("1 < 1")
    assert isinstance(node, CompareExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert len(node.comparators) == 1
    assert isinstance(node.comparators[0].right_argument, IntegerExpressionNode)
    assert node.comparators[0].opcode == CompareID.Less
    assert node.comparators[0].token_prefix.id == TokenID.Less
    assert node.comparators[0].token_suffix is None

    _, node = parse_expression("1 <= 1")
    assert isinstance(node, CompareExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert len(node.comparators) == 1
    assert isinstance(node.comparators[0].right_argument, IntegerExpressionNode)
    assert node.comparators[0].opcode == CompareID.LessEqual
    assert node.comparators[0].token_prefix.id == TokenID.LessEqual
    assert node.comparators[0].token_suffix is None

    _, node = parse_expression("1 > 1")
    assert isinstance(node, CompareExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert len(node.comparators) == 1
    assert isinstance(node.comparators[0].right_argument, IntegerExpressionNode)
    assert node.comparators[0].opcode == CompareID.Great
    assert node.comparators[0].token_prefix.id == TokenID.Great
    assert node.comparators[0].token_suffix is None

    _, node = parse_expression("1 >= 1")
    assert isinstance(node, CompareExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert len(node.comparators) == 1
    assert isinstance(node.comparators[0].right_argument, IntegerExpressionNode)
    assert node.comparators[0].opcode == CompareID.GreatEqual
    assert node.comparators[0].token_prefix.id == TokenID.GreatEqual
    assert node.comparators[0].token_suffix is None

    _, node = parse_expression("1 is 1")
    assert isinstance(node, CompareExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert len(node.comparators) == 1
    assert isinstance(node.comparators[0].right_argument, IntegerExpressionNode)
    assert node.comparators[0].opcode == CompareID.Is
    assert node.comparators[0].token_prefix.id == TokenID.Is
    assert node.comparators[0].token_suffix is None

    _, node = parse_expression("1 is not 1")
    assert isinstance(node, CompareExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert len(node.comparators) == 1
    assert isinstance(node.comparators[0].right_argument, IntegerExpressionNode)
    assert node.comparators[0].opcode == CompareID.IsNot
    assert node.comparators[0].token_prefix.id == TokenID.Is
    assert node.comparators[0].token_suffix.id == TokenID.Not

    _, node = parse_expression("1 in 1")
    assert isinstance(node, CompareExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert len(node.comparators) == 1
    assert isinstance(node.comparators[0].right_argument, IntegerExpressionNode)
    assert node.comparators[0].opcode == CompareID.In
    assert node.comparators[0].token_prefix.id == TokenID.In
    assert node.comparators[0].token_suffix is None

    _, node = parse_expression("1 not in 1")
    assert isinstance(node, CompareExpressionNode)
    assert isinstance(node.left_argument, IntegerExpressionNode)
    assert len(node.comparators) == 1
    assert isinstance(node.comparators[0].right_argument, IntegerExpressionNode)
    assert node.comparators[0].opcode == CompareID.NotIn
    assert node.comparators[0].token_prefix.id == TokenID.Not
    assert node.comparators[0].token_suffix.id == TokenID.In


def test_parse_condition():
    _, node = parse_expression("1 if 1 else 0")
    assert isinstance(node, ConditionExpressionNode)
    assert isinstance(node.condition, IntegerExpressionNode)
    assert isinstance(node.then_value, IntegerExpressionNode)
    assert isinstance(node.else_value, IntegerExpressionNode)
    assert node.token_if.id == TokenID.If
    assert node.token_else.id == TokenID.Else


def test_parse_augmented_assignment():
    _, node = parse_statement("a += 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.Add
    assert node.token_operator.id == TokenID.PlusEqual

    _, node = parse_statement("a -= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.Sub
    assert node.token_operator.id == TokenID.MinusEqual

    _, node = parse_statement("a *= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.Mul
    assert node.token_operator.id == TokenID.StarEqual

    _, node = parse_statement("a @= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.MatrixMul
    assert node.token_operator.id == TokenID.AtEqual

    _, node = parse_statement("a /= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.TrueDiv
    assert node.token_operator.id == TokenID.SlashEqual

    _, node = parse_statement("a //= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.FloorDiv
    assert node.token_operator.id == TokenID.DoubleSlashEqual

    _, node = parse_statement("a %= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.Mod
    assert node.token_operator.id == TokenID.PercentEqual

    _, node = parse_statement("a |= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.Or
    assert node.token_operator.id == TokenID.VerticalLineEqual

    _, node = parse_statement("a &= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.And
    assert node.token_operator.id == TokenID.AmpersandEqual

    _, node = parse_statement("a ^= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.Xor
    assert node.token_operator.id == TokenID.CircumflexEqual

    _, node = parse_statement("a <<= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.LeftShift
    assert node.token_operator.id == TokenID.LeftShiftEqual

    _, node = parse_statement("a >>= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.RightShift
    assert node.token_operator.id == TokenID.RightShiftEqual

    _, node = parse_statement("a **= 1")
    assert isinstance(node, AugmentedAssignStatementNode)
    assert isinstance(node.target, NamedExpressionNode)
    assert isinstance(node.source, IntegerExpressionNode)
    assert node.opcode == BinaryID.Pow
    assert node.token_operator.id == TokenID.DoubleStarEqual


def test_parse_generic_functions():
    _, node = parse_member("""
def test[A, B: First, C: Second | Third](): ...
    """)

    assert isinstance(node, FunctionNode)
    assert node.name == 'test'
    assert isinstance(node.return_type, AutoTypeNode)
    assert len(node.generic_parameters) == 3

    gen1 = node.generic_parameters[0]
    gen2 = node.generic_parameters[1]
    gen3 = node.generic_parameters[2]

    assert isinstance(gen1, GenericParameterNode)
    assert gen1.name == 'A'
    assert len(gen1.concepts) == 0

    assert isinstance(gen2, GenericParameterNode)
    assert gen2.name == 'B'
    assert len(gen2.concepts) == 1

    gen2_c1 = gen2.concepts[0]
    assert isinstance(gen2_c1, NamedTypeNode)
    assert gen2_c1.name == 'First'

    assert isinstance(gen3, GenericParameterNode)
    assert gen3.name == 'C'
    assert len(gen3.concepts) == 2

    gen3_c1 = gen3.concepts[0]
    gen3_c2 = gen3.concepts[1]

    assert isinstance(gen3_c1, NamedTypeNode)
    assert gen3_c1.name == 'Second'
    assert isinstance(gen3_c2, NamedTypeNode)
    assert gen3_c2.name == 'Third'
