# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import os
import sys
from typing import Sequence, Iterator

import pytest

from orcinus.parser import Parser
from orcinus.scanner import Scanner
from orcinus.syntax import SyntaxTree, SyntaxContext, SyntaxToken, TokenID, SyntaxNode
from orcinus.tests import find_scripts

TEST_FIXTURES = sorted(s for s in find_scripts('./tests/syntax/'))


def tokenize_file(name: str, filename: str) -> Sequence[SyntaxToken]:
    """ Tokenize file """
    tokens = []
    context = SyntaxContext()
    with Scanner(name, open(filename, 'r', encoding='utf-8'), diagnostics=context.diagnostics) as scanner:
        while True:
            token = scanner.consume_token()
            tokens.append(token)
            if token.id == TokenID.EndOfFile:
                break
    return tokens


def parse_file(name: str, filename: str) -> SyntaxTree:
    """ Parse file """
    context = SyntaxContext()
    with Scanner(name, open(filename, 'r', encoding='utf-8'), diagnostics=context.diagnostics) as scanner:
        parser = Parser(name, scanner, context)
        tree = parser.parse()

    if context.diagnostics.has_errors:
        for diagnostic in context.diagnostics:
            sys.stderr.write(str(diagnostic))

    assert not context.diagnostics.has_errors
    return tree


def source_name(fixture_value):
    root_path = os.path.dirname(__file__)
    fullname = os.path.join(root_path, fixture_value)
    basename = os.path.dirname(root_path)
    return os.path.relpath(fullname, basename)


@pytest.fixture(params=TEST_FIXTURES, ids=source_name)
def source_cases(request):
    fullname = "{}.orx".format(request.param)
    fixture = os.path.relpath(fullname, os.getcwd())

    return fixture, fullname


def compare_symbols(node: SyntaxNode, tokens: Iterator[SyntaxToken]):
    for child in node.children:
        if isinstance(child, SyntaxToken):
            token = next(tokens)
            assert child == token
        elif isinstance(child, SyntaxNode):
            compare_symbols(child, tokens)


def test_tokens_utilization(source_cases):
    name, filename = source_cases

    tokens = tokenize_file(name, filename)
    tree = parse_file(name, filename)
    compare_symbols(tree, iter(tokens))
