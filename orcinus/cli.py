#!/usr/bin/env python
from __future__ import annotations

import logging
import os

from orcinus.diagnostics import Diagnostic, DiagnosticSeverity, DiagnosticManager
from orcinus.workspace import Workspace

logger = logging.getLogger('orcinus')

DIAGNOSTIC_LOGGERS = {
    DiagnosticSeverity.Error: logger.error,
    DiagnosticSeverity.Warning: logger.warning,
    DiagnosticSeverity.Information: logger.info,
    DiagnosticSeverity.Hint: logger.info,
}


def log_diagnostics(diagnostics: DiagnosticManager):
    for diagnostic in diagnostics:  # type: Diagnostic
        DIAGNOSTIC_LOGGERS.get(diagnostic.severity, logger.info)(diagnostic)


def main():
    workspace = Workspace([os.getcwd()])

    builtins = workspace.get_or_create_module('__builtins__')
    tree = builtins.tree
    module = builtins.module
    breakpoint()


if __name__ == '__main__':
    main()
