# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import os


def find_scripts(path):
    for path, _, filenames in os.walk(path):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext == '.orx':
                yield os.path.abspath(os.path.join(path, basename))
