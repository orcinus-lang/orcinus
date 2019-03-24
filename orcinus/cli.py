#!/usr/bin/env python
# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import argparse
import functools
import logging
import sys

from colorlog import ColoredFormatter

from orcinus import __name__ as app_name, __version__ as app_version
from orcinus.builder import build_package
from orcinus.compiler import compile_module, TargetID
from orcinus.diagnostics import log_diagnostics
from orcinus.exceptions import OrcinusError, DiagnosticError, DiagnosticCollectionError

logger = logging.getLogger('orcinus')

# noinspection PyProtectedMember
LEVELS = list(map(str.lower, logging._nameToLevel.keys()))
DEFAULT_LEVEL = "warning"
KEY_ACTION = '__action__'
KEY_LEVEL = '__level__'
KEY_PDB = '__pdb__'


def initialize_logging():
    """ Prepare rules for loggers """
    # Prepare console logger
    console = logging.StreamHandler()

    # Prepare console formatter
    if sys.stderr.isatty():
        formatter = ColoredFormatter(
            '%(reset)s%(message_log_color)s%(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red',
            },
            secondary_log_colors={
                'message': {
                    'ERROR': 'red',
                    'CRITICAL': 'red'
                }
            }
        )
    else:
        formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)

    # Setup logging in console
    logger.addHandler(console)


def process_errors(action):
    @functools.wraps(action)
    def wrapper(*args, **kwargs):
        try:
            return action(*args, **kwargs)
        except DiagnosticError as ex:
            log_diagnostics([ex.args[0]])
            return 1
        except DiagnosticCollectionError as ex:
            log_diagnostics(ex.args[0])
            return 1
        except OrcinusError as ex:
            logger.fatal(ex)
            return 2
        except Exception as ex:
            logger.exception(ex)
            return 1

    return wrapper


def process_pdb(action):
    @functools.wraps(action)
    def wrapper(*args, **kwargs):
        try:
            import ipdb as pdb
        except ImportError:
            import pdb

        try:
            return action(*args, **kwargs)
        except Exception as ex:
            logger.fatal(ex)
            pdb.post_mortem()
            raise ex

    return wrapper


def main():
    # initialize default logging
    initialize_logging()

    # create arguments parser
    parser = argparse.ArgumentParser(app_name)
    parser.add_argument('--pdb', dest=KEY_PDB, action='store_true', help="post-mortem mode")
    parser.add_argument('-l', '--level', dest=KEY_LEVEL, choices=LEVELS, default=DEFAULT_LEVEL)
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {app_version}')

    # create subparser
    subparsers = parser.add_subparsers()

    # compile single module
    targets = [target for target in TargetID]
    target_names = [target.name for target in TargetID]

    compile_cmd = subparsers.add_parser('compile')
    compile_cmd.add_argument('filename', type=str, help="input file")
    compile_cmd.add_argument(
        '--output', '-O', type=str, default='-', help="output result to file, use `-` for stdout")
    compile_cmd.add_argument(
        '--target', type=TargetID, choices=targets, metavar=target_names, default=TargetID.LLVM, help="result target")
    compile_cmd.add_argument(dest=KEY_ACTION, help=argparse.SUPPRESS, action='store_const', const=compile_module)

    # build package or single module
    build_cmd = subparsers.add_parser('build')
    build_cmd.add_argument('path', type=str, default='.', help='package path or main module for anonymous package')
    build_cmd.add_argument('--name', type=str, default='', help="output file name")
    build_cmd.add_argument(dest=KEY_ACTION, help=argparse.SUPPRESS, action='store_const', const=build_package)

    # parse arguments
    kwargs = parser.parse_args().__dict__
    action = kwargs.pop(KEY_ACTION, None)
    is_pdb = kwargs.pop(KEY_PDB, False)

    # change logging level
    logger.setLevel(kwargs.pop(KEY_LEVEL, DEFAULT_LEVEL).upper())

    if action:
        if is_pdb:  # enable pdb if required
            action = process_pdb(action)
        if not sys.gettrace():
            action = process_errors(action)
        return action(**kwargs)
    else:
        parser.print_usage()
        return 2
