#!/usr/bin/env python
# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import io
import os
import subprocess
import sys

from orcinus.exceptions import OrcinusError


class ExecuteError(OrcinusError):
    pass


def select_executable(name, *, hints=None, paths=None):
    hints = hints or []
    paths = paths or []
    hints.append(name)
    paths.extend(os.getenv('PATH').split(os.pathsep))
    for original in hints:
        if os.name == 'nt':
            original = f"{original}.exe"

        for path in paths:
            filename = os.path.join(path, original)
            filename = os.path.abspath(filename)
            if os.path.exists(filename):
                return filename

    stream = io.StringIO()
    stream.write(f"Not found executable {name}\n")
    stream.write(f"Search paths:\n")
    for path in paths:
        stream.write("  - ")
        stream.write(path)
        stream.write("\n")
    raise ExecuteError(stream.getvalue())


def select_libraries(filenames, *, paths):
    results = []
    for original in filenames:
        is_founded = False
        for path in paths:
            filename = os.path.join(path, original)
            if os.path.exists(filename):
                results.append(filename)
                is_founded = True
                break

        if not is_founded:
            sys.stderr.write(f"Not found library {original}\n")
            sys.stderr.write(f"Search paths:\n")
            for path in paths:
                sys.stderr.write("  - ")
                sys.stderr.write(path)
                sys.stderr.write("\n")
            sys.exit(-1)
    return results


def execute(*args, input=None):
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    stdout, stderr = process.communicate(input)
    if process.returncode:
        stderr = stderr.decode("utf-8") if stderr else ''
        raise ExecuteError(stderr)
    return stdout


EXECUTABLE_PATHS = [
    os.path.join(os.getcwd(), ".venv/bin/"),
]

LIBRARY_PATHS = [
    os.path.join(os.getcwd(), "./runtime/dist/lib/"),

    "/usr/lib/",
    "/usr/local/lib/",
]

COMPILER = select_executable("orcinus", paths=EXECUTABLE_PATHS)
ASSEMBLER = select_executable("llc", paths=EXECUTABLE_PATHS, hints=[
    "llc-7",
])
LINKER = select_executable("g++", paths=EXECUTABLE_PATHS)

LIBRARIES = [
    'liborcinus-stdlib.a',
    # 'libcord.a',
    'libcoro.a',
    'libgc.a',
    'libuv_a.a',
]

# def main():
#     libraries = select_libraries(LIBRARIES, paths=LIBRARY_PATHS)
#     if os.name != 'nt':
#         libraries.append("-lpthread")
#     link(list(map(compile, sys.argv[2:])), libraries, sys.argv[1])
#
#
# if __name__ == '__main__':
#     main()
