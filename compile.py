#!/usr/bin/env python3.6
# Copyright (C) 2017 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
import os
import subprocess
import sys

OBJECT_EXTENSION = 'obj' if os.name == 'nt' else 'o'


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

    sys.stderr.write(f"Not found executable {original}\n")
    sys.stderr.write(f"Search paths:\n")
    for path in paths:
        sys.stderr.write("  - ")
        sys.stderr.write(path)
        sys.stderr.write("\n")
    sys.exit(-1)


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
    command = args[0]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    stdout, stderr = process.communicate(input)
    if process.returncode:
        sys.stderr.write(f"Command {command} exited with error: {process.returncode}\n")
        if stderr:
            stderr = stderr.decode("utf-8")
            sys.stderr.write(f"Error:\n{stderr}")
        sys.exit(process.returncode)
    return stdout


def compile(filename):
    basename, _ = os.path.splitext(filename)
    basename = os.path.relpath(basename, os.getcwd())

    build_dir = os.path.join(os.getcwd(), "build/..orcinus/")
    build_path = os.path.join(build_dir, f"{basename}.{OBJECT_EXTENSION}")
    os.makedirs(os.path.dirname(build_path), exist_ok=True)

    assembly = execute(COMPILER, filename)
    execute(ASSEMBLER, '-filetype=obj', '-relocation-model=pic', f'-o={build_path}', input=assembly)
    return build_path


def link_unix(filenames, libraries, output):
    return execute(LINKER, *filenames, *libraries, "-fPIC", "-O0", "-g", "-ggdb", "-o", output)


def link_windows(filenames, libraries, output):
    print(' '.join(map(str, [LINKER, *filenames, '/OUT', output, '/VERBOSE'])))
    return execute(LINKER, *filenames, '/OUT', output, '/VERBOSE')


def link(filenames, libraries, output):
    if os.name == 'nt':
        return link_windows(filenames, libraries, output)
    return link_unix(filenames, libraries, output)


EXECUTABLE_PATHS = [
    os.path.join(os.getcwd(), "./build/debug-gcc"),
    os.path.join(os.getcwd(), "./build/release-gcc"),
    os.path.join(os.getcwd(), "./build/debug-clang"),
    os.path.join(os.getcwd(), "./build/release-clang"),

    os.path.join(os.getcwd(), "./build/RelWithDebInfo"),
    os.path.join(os.getcwd(), "./build/Release"),
    os.path.join(os.getcwd(), "./build/Debug"),

    os.path.join(os.getcwd(), "./dist/bin/"),

    'C:/Development/LLVM64/bin',
    'C:/LLVM/bin',
]

LIBRARY_PATHS = [
    os.path.join(os.getcwd(), "./build/debug-gcc"),
    os.path.join(os.getcwd(), "./build/release-gcc"),
    os.path.join(os.getcwd(), "./build/debug-clang"),
    os.path.join(os.getcwd(), "./build/release-clang"),

    os.path.join(os.getcwd(), "./build/RelWithDebInfo"),
    os.path.join(os.getcwd(), "./build/Release"),
    os.path.join(os.getcwd(), "./build/Debug"),

    os.path.join(os.getcwd(), "./dist/lib/"),

    "/usr/lib/",
    "/usr/lib/x86_64-linux-gnu/",
    "/usr/local/lib/",
    '/usr/lib/llvm-5.0/lib',

    'C:/Development/LLVM64/lib',
    'C:/Development/LLVM/lib',
    'C:/Program Files/ICU/lib64',
    'C:/Program Files/ICU/lib',
]

COMPILER = select_executable("bootstrap", paths=EXECUTABLE_PATHS)
ASSEMBLER = select_executable("llc", paths=EXECUTABLE_PATHS, hints=[
    "llc-6.0",
    "llc-5.0",
])
if os.name == 'nt':
    VISUAL_STUDIO_PATHS = [
        "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Tools/MSVC/14.12.25827/bin/Hostx64/x64"
    ]
    LINKER = select_executable("link", paths=VISUAL_STUDIO_PATHS)
else:
    LINKER = 'g++'

if os.name == 'nt':
    LIBRARIES = [
        'orcinus-runtime.lib',
        'LLVMCore.lib',
        "icuio.lib",
        "icuuc.lib",
    ]
else:
    LIBRARIES = [
        'liborcinus-runtime.a',
        'libLLVM-5.0.so',
        "libicuio.a",
        "libicuuc.a",
        "libicudata.a",
    ]


def main():
    libraries = select_libraries(LIBRARIES, paths=LIBRARY_PATHS)
    if os.name != 'nt':
        libraries.append("-ldl")
        libraries.append("-lpthread")
    link(list(map(compile, sys.argv[2:])), libraries, sys.argv[1])


if __name__ == '__main__':
    main()
