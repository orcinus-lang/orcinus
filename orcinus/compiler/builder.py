# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
import os
from concurrent.futures.process import ProcessPoolExecutor
from typing import List


class PackageBuilder:
    def __init__(self, build_path: str = None, dist_path: str = None):
        self.pool = ProcessPoolExecutor()

        # TODO: using Golang style? e.t.c ORX_PATH or `~/.orcinus/`
        self.build_path = build_path or os.path.join(os.getcwd(), './build')
        self.dist_path = dist_path or os.path.join(os.getcwd(), './dist')

    @staticmethod
    def compile(self, filename: str):
        """
        Compile single file with source code to object file

        :param self:
        :param filename:
        :return:
        """
        pass

    @staticmethod
    def link(self, filenames: List[str]):
        """
        Link

        :param self:
        :param filenames:
        :return:
        """
        pass

    # def compile(filename):
    #     basename, _ = os.path.splitext(filename)
    #     basename = os.path.relpath(basename, os.getcwd())
    #
    #     build_dir = os.path.join(os.getcwd(), "build/..orcinus/")
    #     build_path = os.path.join(build_dir, f"{basename}.{OBJECT_EXTENSION}")
    #     os.makedirs(os.path.dirname(build_path), exist_ok=True)
    #
    #     assembly = execute(COMPILER, 'compile', filename)
    #     execute(ASSEMBLER, '-filetype=obj', '-relocation-model=pic', f'-o={build_path}', input=assembly)
    #     return build_path
