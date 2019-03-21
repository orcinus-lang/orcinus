# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import warnings as pywarnings
from typing import Optional, MutableSequence

import pytest

from orcinus.cli import compile_module
from orcinus.tests import find_scripts

OPT_EXECUTABLE = 'opt-6.0'
LLI_EXECUTABLE = 'lli-6.0'
BOOTSTRAP_SCRIPT = 'orcinus'

TEST_FIXTURES = sorted(s for s in find_scripts('./tests/codegen'))


@dataclasses.dataclass
class ScriptInput:
    name: str
    filename: str
    arguments: MutableSequence[str] = dataclasses.field(default_factory=list)
    result_code: Optional[int] = None
    warnings: MutableSequence[str] = dataclasses.field(default_factory=list)
    inputs: MutableSequence[str] = dataclasses.field(default_factory=list)
    outputs: MutableSequence[str] = dataclasses.field(default_factory=list)
    errors: MutableSequence[str] = dataclasses.field(default_factory=list)

    def input_string(self) -> Optional[str]:
        return "\n".join(self.inputs) if self.inputs else None

    def output_string(self) -> Optional[str]:
        return "\n".join(self.outputs) if self.outputs else None

    def error_string(self) -> Optional[str]:
        return "\n".join(self.errors) if self.errors else None


@dataclasses.dataclass
class ScriptOutput:
    completed: bool
    result_code: int
    output: str
    error: str


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


def compile_and_execute(capsys, filename, *, name, opt_level, arguments, input=None):
    # orcinus - generate LLVM IR
    error_code = compile_module(filename)

    captured = capsys.readouterr()
    if error_code:
        return ScriptOutput(False, error_code, captured.out, captured.err)

    assembly = captured.out.encode('utf-8')

    # lli-6.0 - compile LLVM IR and execute
    flags = [
        f'-fake-argv0={name}'
    ]
    flags.extend(get_build_options())
    flags.extend(['-'])
    flags.extend(arguments)
    return ScriptOutput(True, *execute([LLI_EXECUTABLE, f'-O{opt_level}'] + flags, input=assembly))


def remove_startswith_and_strip(haystack: str, needle: str) -> str:
    if haystack.startswith(needle):
        return haystack[len(needle):].strip()
    return ""


def source_name(fixture_value):
    root_path = os.path.dirname(__file__)
    fullname = os.path.join(root_path, fixture_value)
    basename = os.path.dirname(root_path)
    return os.path.relpath(fullname, basename)


@pytest.fixture(params=TEST_FIXTURES, ids=source_name)
def source_cases(request):
    fullname = "{}.orx".format(request.param)
    fixture = os.path.relpath(fullname, os.getcwd())

    script = ScriptInput(fixture, fullname)

    with open(fullname, 'r', encoding='utf-8') as stream:
        for line in stream:
            line = remove_startswith_and_strip(line, '#').strip()
            if not line:
                continue

            # warning
            test_msg = remove_startswith_and_strip(line, "WARNING: ")
            if test_msg:
                script.warnings.append(test_msg)

            # exit code
            test_msg = remove_startswith_and_strip(line, "EXIT: ")
            if test_msg:
                script.result_code = int(test_msg)

            # argument
            test_msg = remove_startswith_and_strip(line, "ARG: ")
            if test_msg:
                script.arguments.append(test_msg)

            # input
            test_msg = remove_startswith_and_strip(line, "INPUT: ")
            if test_msg:
                script.inputs.append(test_msg)

            # output
            test_msg = remove_startswith_and_strip(line, "OUTPUT: ")
            if test_msg:
                script.outputs.append(test_msg)

            # error
            test_msg = remove_startswith_and_strip(line, "ERROR:")
            if test_msg:
                script.errors.append(test_msg)

    return script


def test_compile_and_execution(caplog, capsys, source_cases):
    script: ScriptInput = source_cases

    for warning in script.warnings:
        pywarnings.warn(UserWarning(warning))

    # for opt_level in [0, 1, 2, 3]:  # Test on all optimization levels
    for opt_level in (0,):
        result: ScriptOutput = compile_and_execute(
            capsys,
            script.filename,
            name=script.name,
            opt_level=opt_level,
            arguments=script.arguments,
            input=script.inputs
        )

        if script.result_code is not None:
            assert result.result_code == script.result_code

        for error in script.errors:
            assert any(error in rec.message for rec in caplog.records)

        for output in script.outputs:
            assert output in result.output
