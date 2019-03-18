# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.
import weakref


class Signal:
    def __init__(self):
        self.__receivers = []

    def connect(self, receiver, weak=True):
        lookup_key = self.__make_id(receiver)
        if weak:
            receiver = weakref.ref(receiver, self.disconnect)

        if lookup_key in self.__receivers:
            return
        self.__receivers.append((lookup_key, receiver))

    def disconnect(self, receiver=None):
        lookup_key = self.__make_id(receiver)
        for index in range(len(self.__receivers)):
            r_key, _ = self.__receivers[index]
            if r_key == lookup_key:
                del self.__receivers[index]
                break

    @staticmethod
    def __make_id(target):
        if hasattr(target, 'im_func'):
            return id(target.im_self), id(target.im_func)
        return id(target)

    def __lived_receivers(self):
        receivers = []
        for receiver in self.__receivers:
            if isinstance(receiver, weakref.ReferenceType):
                receiver = receiver()
            if receiver:
                receiver.append(receiver)
        return receivers

    def __call__(self, *args, **kwargs):
        for receiver in self.__lived_receivers():
            receiver(*args, **kwargs)
