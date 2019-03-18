# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import dataclasses
import enum
import io
import itertools
import os
from typing import Sequence

from orcinus.locations import Location


# Enumeration contains diagnostic severities
@enum.unique
class DiagnosticSeverity(enum.IntEnum):
    # Reports an error.
    Error = 1

    # Reports a warning.
    Warning = 2

    # Reports an information.
    Information = 3

    # Reports a hint.
    Hint = 4


# The Diagnostic class is represented a diagnostic, such as a compiler error or warning.
@dataclasses.dataclass(repr=False)
class Diagnostic:
    # The location at which the message applies
    location: Location

    # The diagnostic's severity.
    severity: DiagnosticSeverity

    # The diagnostic's message.
    message: str

    # A human-readable string describing the source of this diagnostic, e.g. 'orcinus' or 'doxygen'.
    source: str = "orcinus"

    def __str__(self):
        lines = show_source_lines(self.location)
        if lines:
            return "[{}] {}:\n{}".format(self.location, self.message, lines)
        return "[{}] {}".format(self.location, self.message)

    def __repr__(self):
        return '"' + str(self) + '"'


# The DiagnosticManager class is represented collection of diagnostics, and used for simple appending new diagnostic
class DiagnosticManager(Sequence[Diagnostic]):
    def __init__(self):
        self.__diagnostics = []
        self.has_error = False
        self.has_warnings = False

    def __iter__(self):
        return iter(self.__diagnostics)

    def __getitem__(self, idx: int) -> Diagnostic:
        return self.__diagnostics[idx]

    def __len__(self) -> int:
        return len(self.__diagnostics)

    def add(self, location: Location, severity: DiagnosticSeverity, message: str, source: str = "orcinus"):
        self.has_error |= severity == DiagnosticSeverity.Error
        self.has_warnings |= severity == DiagnosticSeverity.Warning

        self.__diagnostics.append(
            Diagnostic(location, severity, message, source)
        )

    def error(self, location: Location, message: str, source: str = "orcinus"):
        return self.add(location, DiagnosticSeverity.Error, message, source)

    def warning(self, location: Location, message: str, source: str = "orcinus"):
        return self.add(location, DiagnosticSeverity.Warning, message, source)

    def info(self, location: Location, message: str, source: str = "orcinus"):
        return self.add(location, DiagnosticSeverity.Information, message, source)

    def hint(self, location: Location, message: str, source: str = "orcinus"):
        return self.add(location, DiagnosticSeverity.Hint, message, source)

    def clear(self):
        self.__diagnostics.clear()


def load_source_content(location: Location, before: int = 2, after: int = 2):
    # """ Load selected line and it's neighborhood lines """
    try:
        with open(location.filename, 'r', encoding='utf-8') as stream:
            at_before = max(0, location.begin.line - before)
            at_after = location.end.line + after

            idx = 0
            results = []
            for idx, line in itertools.islice(enumerate(stream), at_before, at_after):
                results.append((idx + 1, line.rstrip("\n")))
    except IOError:
        return []
    else:
        results.append([idx + 2, ""])
        return results


def show_source_lines(location: Location, before: int = 2, after: int = 2, columns: int = None):
    # """
    # Convert selected lines to error message, e.g.:
    #
    # ```
    #     1 : from module import system =
    #       : --------------------------^
    # ```
    # """
    ANSI_COLOR_RED = "\033[31m"
    ANSI_COLOR_GREEN = "\x1b[32m"
    ANSI_COLOR_BLUE = "\x1b[34m"
    ANSI_COLOR_CYAN = "\x1b[36m"
    ANSI_COLOR_RESET = "\x1b[0m"

    stream = io.StringIO()
    if not columns:
        try:
            _, columns = os.popen('stty size', 'r').read().split()
        except (ValueError, IOError):
            pass

    strings = load_source_content(location, before, after)
    if not strings:
        return

    width = 5
    for idx, _ in strings:
        width = max(len(str(idx)), width)

    for line, string in strings:
        s_line = str(line).rjust(width)

        stream.write(ANSI_COLOR_CYAN)
        stream.write(s_line)
        stream.write(" : ")
        stream.write(ANSI_COLOR_BLUE)
        for column, char in enumerate(string):
            column += 1
            is_error = False
            if location.begin.line == line:
                is_error = column >= location.begin.column
            if location.end.line == line:
                is_error = is_error and column <= location.end.column

            if is_error:
                stream.write(ANSI_COLOR_RED)
            else:
                stream.write(ANSI_COLOR_GREEN)
            stream.write(char)

        stream.write(ANSI_COLOR_RESET)
        stream.write("\n")

        # write error line
        if location.begin.line <= line <= location.end.line:
            stream.write("·" * width)
            stream.write(" : ")

            for column, char in itertools.chain(enumerate(string), ((len(string), None),)):
                column += 1

                is_error = False
                if location.begin.line == line:
                    is_error = column >= location.begin.column
                if location.end.line == line:
                    is_error = is_error and column <= location.end.column

                if is_error:
                    stream.write(ANSI_COLOR_RED)
                    stream.write("^")
                    stream.write(ANSI_COLOR_RESET)
                elif char is not None:
                    stream.write("·")
            stream.write("\n")

    return stream.getvalue()
