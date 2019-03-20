# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from collections import deque as Queue
from collections import deque as Stack
from io import StringIO, SEEK_CUR
from typing import TextIO

from orcinus.diagnostics import DiagnosticManager
from orcinus.locations import Location
from orcinus.syntax import TokenID, SyntaxToken


class Scanner:
    stream: TextIO
    queue: Queue
    indentations: Stack
    whitespace: str
    level: int
    is_level: bool
    is_new_line: bool

    def __init__(self, filename: str, stream: TextIO = None, *, diagnostics: DiagnosticManager):
        self.diagnostics = diagnostics
        self.stream = stream or open(filename, 'r+', encoding='utf-8')
        self.is_new_line = True
        self.level = 0
        self.queue = Queue()
        self.indentations = Stack([0])
        self.__current_symbol = self.stream.read(1)
        self.value = StringIO()
        self.unputed = Queue()
        self.__location = Location(filename)
        self.whitespace = '', self.__location

    def __enter__(self) -> Scanner:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def current_symbol(self) -> str:
        return self.__current_symbol

    def close(self):
        self.stream.close()

    def consume_token(self) -> SyntaxToken:
        while not self.queue:
            self.scan_token()
        return self.queue.popleft()

    def scan_token(self):
        self.__location = self.__location.step()
        token_id = self.consume_id()
        value = self.consume_value()
        location = self.consume_location(value)

        # new line: can be trivia or token
        if token_id == TokenID.NewLine:
            # ignore new lines in brackets
            if self.level:
                return

            # return new line token
            elif not self.is_new_line:
                self.push_token(token_id, value, location)

            self.is_new_line = True
            return

        # whitespace: under normal circumstances it is trivia but it can translated to
        elif token_id == TokenID.Whitespace:
            if self.is_new_line:
                self.whitespace = value, location
            return

        elif token_id == TokenID.EndOfFile:
            if not self.is_new_line:
                # Push extra new line
                self.push_token(TokenID.NewLine, "\n", location)

            while self.indentations[-1] > 0:
                self.push_token(TokenID.Undent, "", location)
                self.indentations.pop()

            self.push_token(token_id, value, location)
            return

        # skip trivia tokens
        elif token_id == TokenID.Comment:
            return

        # process indentations
        if self.is_new_line:
            whitespace, white_location = self.whitespace
            if whitespace:
                self.whitespace = '', self.__location
            indent = len(whitespace)

            if self.indentations[-1] < indent:
                self.push_token(TokenID.Indent, whitespace, white_location)
                self.indentations.append(indent)

            while self.indentations[-1] > indent:
                self.push_token(TokenID.Undent, "", white_location)
                self.indentations.pop()

        # disable or enable indentation processing for brackets
        if token_id in {TokenID.LeftParenthesis, TokenID.LeftSquare, TokenID.LeftCurly}:
            self.level += 1
        elif token_id in {TokenID.RightParenthesis, TokenID.RightSquare, TokenID.RightCurly}:
            self.level -= 1

        # push token
        self.push_token(token_id, value, location)
        self.is_new_line = False

    def advance_symbol(self):
        self.value.write(self.current_symbol)
        if self.unputed:
            self.__current_symbol = self.unputed.pop()
        else:
            self.__current_symbol = self.stream.read(1)
        return self.__current_symbol

    def unput_symbol(self):
        self.value.seek(-1, SEEK_CUR)
        symbol = self.value.read(1)
        self.value.seek(-1, SEEK_CUR)
        self.value.truncate()
        self.unputed.append(symbol)

    def push_token(self, token_id: TokenID, value: str, location: Location):
        self.queue.append(SyntaxToken(token_id, value, location))

    def consume_location(self, value):
        for c in value[:-1]:
            if c == '\n':
                self.__location = self.__location.lines(1)
            elif len(value) > 1:
                self.__location = self.__location.columns(1)
        location = self.__location
        if value:
            if value[-1] == '\n':
                self.__location = self.__location.lines(1)
            else:
                self.__location = self.__location.columns(1)
        return location

    def consume_value(self) -> str:
        value = self.value.getvalue()
        self.value.seek(0)
        self.value.truncate()
        return value

    def consume_string(self, end_symbol):
        self.advance_symbol()

        while self.current_symbol not in {'', end_symbol, '\n'}:
            # escape sequence
            if self.current_symbol == '\\':
                self.advance_symbol()
            self.advance_symbol()

        if self.current_symbol == end_symbol:
            self.advance_symbol()
            return TokenID.String
        return TokenID.Error

    def consume_keyword(self) -> TokenID:
        value = self.value.getvalue()

        if value == "and":
            return TokenID.And

        if value == "as":
            return TokenID.As

        if value == "break":
            return TokenID.Break

        if value == "class":
            return TokenID.Class

        if value == "continue":
            return TokenID.Continue

        if value == "def":
            return TokenID.Def

        if value == "elif":
            return TokenID.Elif

        if value == "else":
            return TokenID.Else

        if value == "enum":
            return TokenID.Enum

        if value == "except":
            return TokenID.Except

        if value == "finally":
            return TokenID.Finally

        if value == "for":
            return TokenID.For

        if value == "from":
            return TokenID.From

        if value == "if":
            return TokenID.If

        if value == "import":
            return TokenID.Import

        if value == "in":
            return TokenID.In

        if value == "interface":
            return TokenID.Interface

        if value == "is":
            return TokenID.Is

        if value == "not":
            return TokenID.Not

        if value == "or":
            return TokenID.Or

        if value == "pass":
            return TokenID.Pass

        if value == "raise":
            return TokenID.Raise

        if value == "return":
            return TokenID.Return

        if value == "struct":
            return TokenID.Struct

        if value == "try":
            return TokenID.Try

        if value == "with":
            return TokenID.With

        if value == "while":
            return TokenID.While

        if value == "yield":
            return TokenID.Yield

        return TokenID.Name

    def consume_id(self) -> TokenID:
        # End of file
        if self.current_symbol == '':
            return TokenID.EndOfFile

        # New line: \n+
        if self.current_symbol == '\n':
            while self.current_symbol == '\n':
                self.advance_symbol()
            return TokenID.NewLine

        # Whitespace: \s+
        if self.current_symbol.isspace():
            while self.current_symbol.isspace():
                self.advance_symbol()
            return TokenID.Whitespace

        # Comment: #[^\n]*
        if self.current_symbol == '#':
            self.advance_symbol()
            while self.current_symbol != '\n':
                self.advance_symbol()

            return TokenID.Comment

        # Numbers: \d+(\,\d+)?
        if self.current_symbol.isnumeric():
            while self.current_symbol.isnumeric():
                self.advance_symbol()

            if self.current_symbol == '.':
                self.advance_symbol()
                if not self.current_symbol.isnumeric():
                    return TokenID.Error
                while self.current_symbol.isnumeric():
                    self.advance_symbol()
                return TokenID.Double
            return TokenID.Integer

        # Name and keywords: [\w_][\w\d_]*
        if self.current_symbol.isalpha() or self.current_symbol == '_':
            while self.current_symbol.isalnum() or self.current_symbol == '_':
                self.advance_symbol()
            return self.consume_keyword()

        # Special symbols
        if self.current_symbol == '~':
            self.advance_symbol()
            return TokenID.Tilde

        if self.current_symbol == '^':
            self.advance_symbol()
            if self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.CircumflexEqual
            return TokenID.Circumflex

        if self.current_symbol == '%':
            self.advance_symbol()
            if self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.PercentEqual
            return TokenID.Percent

        if self.current_symbol == '<':
            self.advance_symbol()
            if self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.LessEqual
            if self.current_symbol == '<':
                self.advance_symbol()
                if self.current_symbol == '=':
                    self.advance_symbol()
                    return TokenID.LeftShiftEqual
                return TokenID.LeftShift
            return TokenID.Less

        if self.current_symbol == '>':
            self.advance_symbol()
            if self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.GreatEqual
            if self.current_symbol == '>':
                self.advance_symbol()
                if self.current_symbol == '=':
                    self.advance_symbol()
                    return TokenID.RightShiftEqual
                return TokenID.RightShift
            return TokenID.GreatEqual

        if self.current_symbol == '=':
            self.advance_symbol()
            if self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.EqEqual
            return TokenID.Equal

        if self.current_symbol == '!':
            self.advance_symbol()
            if self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.NotEqual

        if self.current_symbol == '.':
            self.advance_symbol()
            if self.current_symbol == '.':
                self.advance_symbol()
                if self.current_symbol == '.':
                    self.advance_symbol()
                    return TokenID.Ellipsis
                self.unput_symbol()
            return TokenID.Dot

        if self.current_symbol == '@':
            self.advance_symbol()
            if self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.AtEqual
            return TokenID.At

        if self.current_symbol == '(':
            self.advance_symbol()
            return TokenID.LeftParenthesis

        if self.current_symbol == ')':
            self.advance_symbol()
            return TokenID.RightParenthesis

        if self.current_symbol == '[':
            self.advance_symbol()
            return TokenID.LeftSquare

        if self.current_symbol == ']':
            self.advance_symbol()
            return TokenID.RightSquare

        if self.current_symbol == '{':
            self.advance_symbol()
            return TokenID.LeftCurly

        if self.current_symbol == '}':
            self.advance_symbol()
            return TokenID.RightCurly

        if self.current_symbol == ':':
            self.advance_symbol()
            return TokenID.Colon

        if self.current_symbol == ';':
            self.advance_symbol()
            return TokenID.Semicolon

        if self.current_symbol == ',':
            self.advance_symbol()
            return TokenID.Comma

        if self.current_symbol == '+':
            self.advance_symbol()
            if self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.PlusEqual
            return TokenID.Plus

        if self.current_symbol == '-':
            self.advance_symbol()
            if self.current_symbol == '>':
                self.advance_symbol()
                return TokenID.Then
            elif self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.MinusEqual
            return TokenID.Minus

        if self.current_symbol == '|':
            self.advance_symbol()
            if self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.VerticalLineEqual
            return TokenID.VerticalLine

        if self.current_symbol == '&':
            self.advance_symbol()
            if self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.AmpersandEqual
            return TokenID.Ampersand

        if self.current_symbol == '*':
            self.advance_symbol()
            if self.current_symbol == '*':
                self.advance_symbol()
                if self.current_symbol == '=':
                    self.advance_symbol()
                    return TokenID.DoubleStarEqual
                return TokenID.DoubleStar
            elif self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.StarEqual
            return TokenID.Star

        if self.current_symbol == '/':
            self.advance_symbol()
            if self.current_symbol == '/':
                self.advance_symbol()
                if self.current_symbol == '=':
                    self.advance_symbol()
                    return TokenID.DoubleSlashEqual
                return TokenID.DoubleSlash
            elif self.current_symbol == '=':
                self.advance_symbol()
                return TokenID.SlashEqual
            return TokenID.Slash

        if self.current_symbol == '\'':
            return self.consume_string('\'')

        if self.current_symbol == '"':
            return self.consume_string('"')

        # Error:
        self.diagnostics.error(self.__location, "Unknown symbol")
        self.advance_symbol()
        return TokenID.Error
