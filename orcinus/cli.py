#!/usr/bin/env python
from __future__ import annotations

import logging
import os

import sys

from orcinus.diagnostics import Diagnostic, DiagnosticSeverity, DiagnosticManager
from orcinus.exceptions import OrcinusError
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

    try:
        builtins = workspace.get_or_create_document(sys.argv[1])
        tree = builtins.syntax_tree
        module = builtins.module
    except OrcinusError as ex:
        sys.stderr.write(str(ex))
        sys.exit(1)
    else:
        breakpoint()


if __name__ == '__main__':
    main()
