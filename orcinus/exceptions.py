# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from orcinus.diagnostics import Diagnostic, DiagnosticSeverity
from orcinus.locations import Location


class OrcinusError(Exception):
    pass


class DiagnosticError(Diagnostic, OrcinusError):
    def __init__(self, location: Location, message: str):
        super(DiagnosticError, self).__init__(location, DiagnosticSeverity.Error, message)
