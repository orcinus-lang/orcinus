# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import contextlib
import io
import sys
from typing import Tuple

from orcinus.parser import Parser
from orcinus.scanner import Scanner
from orcinus.syntax import *


def check_diagnostics(diagnostics: DiagnosticManager):
    for diagnostic in diagnostics:
        sys.stderr.write(str(diagnostic))
    assert not diagnostics.has_errors


@contextlib.contextmanager
def make_parser(content: str, is_newline=False, can_errors=False) -> Parser:
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
    if not can_errors:
        check_diagnostics(context.diagnostics)

    # Consume new line or end of file
    while parser.current_token.id == TokenID.NewLine:
        parser.consume()

    # Check if all content is read
    assert parser.match(TokenID.EndOfFile), "Can not fully parsed string"


def parse_string(content: str, can_errors=False) -> Tuple[SyntaxContext, SyntaxTree]:
    with make_parser(content, can_errors=can_errors) as parser:
        return parser.context, parser.parse()


def parse_expression(content: str) -> Tuple[SyntaxContext, ExpressionNode]:
    with make_parser(content) as parser:
        return parser.context, parser.parse_expression()


def parse_statement(content: str) -> Tuple[SyntaxContext, StatementNode]:
    with make_parser(content, True) as parser:
        return parser.context, parser.parse_statement()


def parse_member(content: str) -> Tuple[SyntaxContext, MemberNode]:
    with make_parser(content, True) as parser:
        return parser.context, parser.parse_member()
