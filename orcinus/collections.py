# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import itertools
from typing import MutableMapping, Callable, Iterator, TypeVar

K = TypeVar('K')
V = TypeVar('V')


class LazyDictionary(MutableMapping[K, V]):
    def __init__(self, seq=None, *, constructor: Callable[[K], V], initializer: Callable[[K], None] = None, **kwargs):
        self.__items = dict(seq or (), **kwargs)
        self.__constructor = constructor
        self.__initializer = initializer or (lambda x: None)

    def __len__(self) -> int:
        return len(self.__items)

    def __iter__(self) -> Iterator[K]:
        return iter(self.__items)

    def __getitem__(self, key: K) -> V:
        try:
            return self.__items[key]
        except KeyError:
            value = self.__constructor(key)
            if value is None:
                return value

            self.__items[key] = value
            self.__initializer(key)
            return value

    def __setitem__(self, key: K, value: V):
        self.__items[key] = value
        self.__initializer(key)

    def __delitem__(self, key: K):
        del self.__items[key]

    def __contains__(self, key: K) -> bool:
        return key in self.__items


class NamedScope:
    def __init__(self):
        self.__names = set()
        self.__counter = itertools.count()
        next(self.__counter)  # begin from one

    def add(self, original: str = None, previous=None) -> str:
        if previous and previous in self.__names:
            self.__names.remove(previous)

        counter = itertools.count()
        next(counter)  # begin from one

        name = original
        while not name or name in self.__names:
            if original:
                name = '{}.{}'.format(original, next(counter))
            else:
                name = str(next(self.__counter))

        self.__names.add(name)
        return name
