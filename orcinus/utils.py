# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.


# noinspection PyPep8Naming
import sys


class cached_property(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls=None):
        if instance is not None:
            result = instance.__dict__[self.func.__name__] = self.func(instance)
            return result
        return None  # ABC


def write_stream(filename: str, content: bytes):
    """ Write content to stream. """
    if filename == '-':
        sys.stdout.write(str(content, encoding='utf-8', errors="backslashreplace"))
    else:
        with open(filename, mode='wb') as stream:
            stream.write(content)
