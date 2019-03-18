# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(order=True, repr=False)
class Position:
    # Line position in a document (one-based).
    line: int = 1

    # Character offset on a line in a document (one-based).
    column: int = 1

    # Compute max(min, lhs+rhs) (provided min <= lhs).
    @staticmethod
    def __add(lhs: int, rhs: int, min: int) -> int:
        return rhs + lhs if 0 < rhs or -rhs < lhs else min

    # (line related) Advance to the COUNT next lines.
    def lines(self, count: int = 1) -> Position:
        if count:
            line = self.__add(self.line, count, 1)
            return Position(line, 1)
        return self

    # (column related) Advance to the COUNT next columns.
    def columns(self, count: int = 1) -> Position:
        column = self.__add(self.column, count, 1)
        return Position(self.line, column)

    def __str__(self):
        return "{}:{}".format(self.line, self.column)

    def __repr__(self):
        return '{}'.format(self)


@dataclasses.dataclass(repr=False)
class Location:
    # The location's filename
    filename: str

    # The location's begin position.
    begin: Position = Position()

    # The end's begin position.
    end: Position = Position()

    # Reset initial location to final location.
    def step(self) -> Location:
        return Location(self.filename, self.end, self.end)

    # Extend the current location to the COUNT next columns.
    def columns(self, count: int = 1) -> Location:
        end = self.end.columns(count)
        return Location(self.filename, self.begin, end)

    # Extend the current location to the COUNT next lines.
    def lines(self, count: int = 1) -> Location:
        end = self.end.lines(count)
        return Location(self.filename, self.begin, end)

    def __add__(self, other: Location) -> Location:
        return Location(self.filename, self.begin, other.end)

    def __str__(self):
        if self.begin == self.end:
            return "{}:{}".format(self.filename, self.begin)
        elif self.begin.line == self.end.line:
            return "{}:{}-{}".format(self.filename, self.begin, self.end.column)
        else:
            return "{}:{}-{}".format(self.filename, self.begin, self.end)

    def __repr__(self):
        return '"' + str(self) + '"'
