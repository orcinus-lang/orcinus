# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from typing import Sequence

from orcinus.diagnostics import Diagnostic, DiagnosticSeverity
from orcinus.locations import Location


class OrcinusError(Exception):
    pass


class DiagnosticError(OrcinusError):
    def __init__(self, location: Location, message: str):
        super().__init__(Diagnostic(location, DiagnosticSeverity.Error, message))


class DiagnosticCollectionError(OrcinusError):
    def __init__(self, diagnostics: Sequence[Diagnostic]):
        super().__init__(diagnostics)
