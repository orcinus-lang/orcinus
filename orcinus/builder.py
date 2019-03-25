#!/usr/bin/env python
# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import concurrent
import fnmatch
import os
import re
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures.process import ProcessPoolExecutor
from typing import Sequence, MutableSequence

from orcinus import proctools
from orcinus.compiler import compile_module, TranslationUnit
from orcinus.workspace import Workspace

OBJECT_EXTENSION = 'obj' if os.name == 'nt' else 'o'
LIBRARIES = [
    'liborcinus-stdlib.a',
    'libcoro.a',
    'libgc.a',
    'libuv_a.a',
]


class Builder:
    def __init__(self, paths: Sequence[str]):
        self.paths = paths
        self.workspace = Workspace(paths)
        self.build_dir = os.path.join(os.getcwd(), "build/")
        self.dist_dir = os.path.join(os.getcwd(), "dist/")
        self.pattern = fnmatch.translate('*.orx')

        self.futures = set()
        self.units = list()
        self.seen = set()
        self.pool = ProcessPoolExecutor()
        self.objects = []
        self.name = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown()

    def build(self, path: str):
        """
        Build package.

        If path is directory then build from this source
        """
        fullpath = os.path.abspath(path)

        if os.path.isdir(fullpath):
            self.name = os.path.basename(fullpath)
            return self.build_package(path)

        self.name, _ = os.path.splitext(os.path.basename(fullpath))
        return self.build_modules([path])

    def build_package(self, path: str):
        """ Build package from directory """
        sources: MutableSequence[str] = []

        for root, dirs, files in os.walk(path):
            files = [os.path.join(root, name) for name in files]
            files = [name for name in files if re.match(self.pattern, name)]

            sources.extend(files)

        return self.build_modules(sources)

    def build_modules(self, filenames: Sequence[str]):
        """ Build modules """
        os.makedirs(os.path.dirname(self.build_dir), exist_ok=True)

        for filename in filenames:
            self.compile_module(filename)
        self.compile_package()

    def compile_module(self, filename: str):
        fullname = os.path.abspath(filename)
        if filename in self.seen:
            return
        self.seen.add(fullname)

        filename = os.path.relpath(filename, os.getcwd())

        # construct path to object file
        package = self.workspace.get_package_for_document(filename)
        module_name = package.get_module_name(filename)
        object_name = os.path.join(self.build_dir, f"{module_name}.{OBJECT_EXTENSION}")

        # start work
        future = self.pool.submit(compile_module, filename, output=object_name, paths=self.paths)
        self.futures.add(future)
        self.objects.append(object_name)

    def compile_package(self):
        # wait for compilation of all required sources
        while self.futures:
            completed, _ = concurrent.futures.wait(self.futures, return_when=FIRST_COMPLETED)
            for future in completed:
                self.futures.remove(future)
                unit: TranslationUnit = future.result()
                print(f'[\33[32m✔\33[39m] Compile {unit.module_name}')

                for filename in unit.dependencies:
                    self.compile_module(filename)

    def link(self, name=None):
        output_path = os.path.join(self.dist_dir, 'bin', name or self.name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        libraries = proctools.select_libraries(LIBRARIES, paths=proctools.LIBRARY_PATHS)
        if os.name != 'nt':
            libraries.append("-lpthread")
        proctools.execute(proctools.LINKER, *self.objects, *libraries, "-fPIC", "-O0", "-g", "-ggdb", "-o", output_path)

        result_path = os.path.relpath(output_path, os.getcwd())
        print(f'[\33[32m✔\33[39m] Link {result_path}')
        return output_path


def build_package(path: str, name: str = "", paths: Sequence[str] = None) -> str:
    with Builder(paths or [os.getcwd()]) as builder:
        builder.build(path)
        return builder.link(name)
