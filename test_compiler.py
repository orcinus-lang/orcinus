#!/usr/bin/env python
# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
import os
import subprocess
import sys
import warnings as pywarnings

import pytest

OPT_EXECUTABLE = 'opt-6.0'
LLI_EXECUTABLE = 'lli-6.0'
BOOTSTRAP_SCRIPT = 'orcinus'


def find_scripts(path):
    for path, _, filenames in os.walk(path):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext == '.orx':
                yield os.path.join(path, basename)


TEST_FIXTURES = sorted(s for s in find_scripts('./tests'))


def execute(command, *, input=None, is_binary=False, is_error=False):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    stdout, stderr = process.communicate(input)
    if not is_binary:
        stdout, stderr = stdout.decode('utf-8').rstrip(), stderr.decode('utf-8').rstrip()
    if process.returncode:
        error = stderr if isinstance(stderr, str) else stderr.decode('utf-8')
        sys.stderr.write(error)
        if is_error:
            raise RuntimeError(error)
        sys.stderr.write(error)
    return process.returncode, stdout, stderr


def get_build_options():
    # code, stdout, stderr = execute(['icu-config', '--ldflags', '--ldflags-icuio'])
    # if code:
    #     raise RuntimeError("Cannot select flags for ICU")

    library_path = None
    items = [
        # "-load", os.path.join(os.getcwd(), './dist/lib/libbootstrap-runtime.so')
    ]
    # for item in stdout.split(' '):
    #     item = item.strip()
    #     if not item:
    #         continue
    #
    #     if item.startswith('-L'):
    #         library_path = item[2:]
    #     elif item.startswith('-l'):
    #         name = item[2:]
    #         if library_path:
    #             filename = "{}.so".format(os.path.join(library_path, 'lib' + name))
    #         else:
    #             filename = "{}.so".format(os.path.join('lib' + name))
    #         items.append("-load")
    #         items.append(filename)
    return items


def compile_and_execute(filename, *, name, opt_level, arguments, input=None):
    # orcinus - generate LLVM IR
    code, assembly, stderr = execute([BOOTSTRAP_SCRIPT, 'compile', filename], is_binary=True, is_error=True)
    if code:
        return False, -code, assembly, stderr.decode('utf-8').rstrip()

    # lli-6.0 - compile LLVM IR and execute
    flags = [
        f'-fake-argv0={name}'
    ]
    flags.extend(get_build_options())
    flags.extend(['-'])
    flags.extend(arguments)
    return (True,) + execute([LLI_EXECUTABLE, f'-O{opt_level}'] + flags, input=assembly)


def remove_startswith_and_strip(haystack: str, needle: str) -> str:
    if haystack.startswith(needle):
        return haystack[len(needle):]
    return ""


def source_name(fixture_value):
    root_path = os.path.dirname(__file__)
    fullname = os.path.join(root_path, fixture_value)
    basename = os.path.dirname(root_path)
    return os.path.relpath(fullname, basename)


@pytest.fixture(params=TEST_FIXTURES, ids=source_name)
def source_cases(request):
    fixture, _ = os.path.splitext(request.param)
    root_path = os.path.dirname(__file__)

    fullname = os.path.abspath(os.path.join(root_path, "{}.orx".format(fixture)))

    warnings_list = []
    arguments = []
    result_code = 0
    inputs = []
    outputs = []
    errors = []

    with open(fullname, 'r', encoding='utf-8') as stream:
        for line in stream:
            line = remove_startswith_and_strip(line, '#').strip()
            if not line:
                continue

            # warning
            test_msg = remove_startswith_and_strip(line, "WARNING: ")
            if test_msg:
                warnings_list.append(test_msg.strip())

            # exit code
            test_msg = remove_startswith_and_strip(line, "EXIT: ")
            if test_msg:
                result_code = int(test_msg.strip())

            # argument
            test_msg = remove_startswith_and_strip(line, "ARG: ")
            if test_msg:
                arguments.append(test_msg.strip())

            # input
            test_msg = remove_startswith_and_strip(line, "INPUT: ")
            if test_msg:
                inputs.append(test_msg)

            # output
            test_msg = remove_startswith_and_strip(line, "OUTPUT: ")
            if test_msg:
                outputs.append(test_msg)

            # error
            test_msg = remove_startswith_and_strip(line, "ERROR:")
            if test_msg:
                errors.append(test_msg)

    # Defaults
    input_string = "\n".join(inputs) if inputs else None
    output_string = "\n".join(outputs) if outputs else None
    error_string = "\n".join(errors) if errors else None

    return fixture, fullname, warnings_list, arguments, input_string, output_string, error_string, result_code


def test_compile_and_execution(source_cases):
    name, filename, warnings, arguments, input, expected_output, expected_error, expected_code = source_cases

    for warning in warnings:
        pywarnings.warn(UserWarning(warning))

    # for opt_level in [0, 1, 2, 3]:  # Test on all optimization levels
    for opt_level in (0,):
        result = compile_and_execute(filename, name=name, opt_level=opt_level, arguments=arguments, input=input)
        result_full, result_code, result_output, result_error = result

        if expected_code is not None:
            assert result_code == expected_code
        if expected_error is not None:
            assert expected_error in result_error
        if expected_output is not None:
            assert expected_output in result_output
