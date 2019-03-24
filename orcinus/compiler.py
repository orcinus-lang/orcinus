#!/usr/bin/env python
# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import dataclasses
import enum
import os
from _ast import Tuple
from typing import Sequence

from orcinus import proctools
from orcinus.codegen import initialize_codegen, ModuleEmitter
from orcinus.exceptions import DiagnosticCollectionError
from orcinus.utils import write_stream
from orcinus.workspace import Workspace


@enum.unique
class TargetID(enum.Enum):
    LLVM = 'llvm'
    Object = 'object'


class Compiler:
    def __init__(self, paths: Sequence[str]):
        self.workspace = Workspace(paths=paths)

    def compile_llvm(self, filename: str) -> Tuple[TranslationUnit, bytes]:
        """ Emit LLVM bytecode """
        document = self.workspace.get_or_create_document(filename)
        module = document.analyze()

        if document.diagnostics.has_errors:
            raise DiagnosticCollectionError(document.diagnostics)

        dependencies = [dependency.location.filename for dependency in module.dependencies]
        unit = TranslationUnit(filename, module.name, dependencies)

        initialize_codegen()
        emitter = ModuleEmitter(module.name)
        emitter.emit(module)
        return unit, bytes(str(emitter), encoding='utf-8')

    def compile_native(self, filename: str) -> Tuple[TranslationUnit, bytes]:
        unit, assembly = self.compile_llvm(filename)
        return unit, proctools.execute(proctools.ASSEMBLER, '-filetype=obj', '-relocation-model=pic', input=assembly)


@dataclasses.dataclass
class TranslationUnit:
    filename: str
    module_name: str
    dependencies: Sequence[str]


def compile_module(filename: str, output: str = "-", target: TargetID = TargetID.Object,
                   paths: Sequence[str] = None) -> TranslationUnit:
    """
    Compile module

    :param paths:       Packages paths
    :param filename: Source path
    :param output: Target path
    :param target:      Target
    """
    compiler = Compiler(paths or [os.getcwd()])
    if target == TargetID.LLVM:
        unit, bytecode = compiler.compile_llvm(filename)
    else:
        unit, bytecode = compiler.compile_native(filename)
    write_stream(output, bytecode)
    return unit
