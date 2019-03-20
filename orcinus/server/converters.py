# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from orcinus.diagnostics import Diagnostic
from orcinus.locations import Location, Position
from orcinus.server.constants import CompletionItemKind
from orcinus.symbols import Symbol, ClassType, InterfaceType, StructType, Module, Function, Type, GenericType


def to_lsp_position(value: Position, *, is_end=False) -> dict:
    return {
        'line': value.line - 1,
        'character': value.column if is_end else value.column - 1
    }


def to_lsp_range(value: Location) -> dict:
    return {
        'start': to_lsp_position(value.begin),
        'end': to_lsp_position(value.end, is_end=True)
    }


def to_lsp_location(value: Location) -> dict:
    return {
        'uri': value.filename,
        'range': to_lsp_range(value)
    }


def to_lsp_diagnostic(value: Diagnostic) -> dict:
    assert isinstance(value.source, str)
    assert isinstance(value.message, str)

    return {
        'range': to_lsp_range(value.location),
        'severity': int(value.severity),
        'source': value.source,
        'message': value.message,
        'relatedInformation': None,
    }


def from_lsp_position(position, *, is_end=False):
    return Position(position['line'] + 1, position['character'] if is_end else position['character'] + 1)


def from_lsp_location(uri, range):
    begin = from_lsp_position(range['start'])
    end = from_lsp_position(range['end'])
    return Location(uri, begin, end)


def to_lsp_completion_kind(symbol: Symbol) -> CompletionItemKind:
    if isinstance(symbol, ClassType):
        return CompletionItemKind.Class
    elif isinstance(symbol, InterfaceType):
        return CompletionItemKind.Interface
    elif isinstance(symbol, StructType):
        return CompletionItemKind.Struct
    elif isinstance(symbol, GenericType):
        return CompletionItemKind.TypeParameter
    elif isinstance(symbol, Type):
        return CompletionItemKind.Struct
    elif isinstance(symbol, Module):
        return CompletionItemKind.Module
    elif isinstance(symbol, Function):
        if isinstance(symbol.owner, Type):
            return CompletionItemKind.Method
        return CompletionItemKind.Function

    return CompletionItemKind.Text
