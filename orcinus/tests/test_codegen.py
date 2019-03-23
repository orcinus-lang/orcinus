# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import tempfile
import warnings as pywarnings
from typing import Optional, MutableSequence

import pytest

from orcinus.cli import compile_module
from orcinus.tests import find_scripts

OPT_EXECUTABLE = 'opt-6.0'
LLI_EXECUTABLE = 'lli-6.0'
STRACEC_EXECUTABLE = 'strace'

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
    trace: str


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
    return [
        "-load", os.path.join(os.getcwd(), './runtime/dist/lib/liborcinus-stdlib.so')
    ]


def compile_and_execute(capsys, trace_filename, *, name, opt_level, arguments, input=None, is_trace=False):
    # orcinus - generate LLVM IR
    result_code = compile_module(trace_filename)

    captured = capsys.readouterr()
    if result_code:
        return ScriptOutput(False, result_code, captured.out, captured.err, "")

    assembly = captured.out.encode('utf-8')

    args = [
        LLI_EXECUTABLE,
        f'-O{opt_level}',
        f'-fake-argv0={name}'
    ]
    args.extend(get_build_options())
    args.extend(arguments)

    if is_trace:
        trace_filename = f'{tempfile.mktemp()}.strace'
        args = [
                   STRACEC_EXECUTABLE,
                   # trace children threads
                   '-f',
                   # output trace to file
                   f'-o', trace_filename,
               ] + args
    else:
        trace_filename = None

    stdtrace = ''
    result_code, stdout, stderr = execute(args, input=assembly)
    if is_trace:
        with open(trace_filename, 'r', os.O_NONBLOCK) as stream:
            print(stream.read(), file=sys.stderr)

    return ScriptOutput(True, result_code, stdout, stderr, '')


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

        assert result.completed != bool(script.errors), "Compilation is failed with errors"

        if script.result_code is not None:
            assert result.result_code == script.result_code

        for error in script.errors:
            assert any(error in rec.message for rec in caplog.records)

        for output in script.outputs:
            assert output in result.output
