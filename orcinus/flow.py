# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from typing import Sequence

from orcinus.syntax import ExpressionNode


class FlowGraph:
    def __init__(self):
        self.__blocks = []
        self.__enter_node = FlowNode()
        self.__exit_node = FlowNode()

    @property
    def enter_node(self) -> FlowNode:
        return self.__enter_node

    @property
    def exit_node(self) -> FlowNode:
        return self.__exit_node

    @property
    def blocks(self) -> Sequence[FlowBlock]:
        return self.__blocks

    def append_block(self, name: str, enter_node: FlowNode) -> FlowBlock:
        block = FlowBlock(name, enter_node)
        self.__blocks.append(block)
        return block


class FlowNode:
    pass


class FlowBlock:
    def __init__(self, name: str, enter_node: FlowNode):
        self.__name = name
        self.__enter_nodes = [enter_node]
        self.__exit_node = None
        self.__instructions = []

    @property
    def name(self) -> str:
        return self.__name

    @property
    def enter_nodes(self) -> Sequence[FlowNode]:
        return self.__enter_nodes

    @property
    def exit_node(self) -> FlowNode:
        return self.__exit_node

    @exit_node.setter
    def exit_node(self, value: FlowNode):
        self.__exit_node = value

    @property
    def instructions(self) -> Sequence[ExpressionNode]:
        return self.__instructions

    def append_instruction(self, node: ExpressionNode):
        self.__instructions.append(node)

    def __str__(self):
        return self.__name

    def __repr__(self):
        return f'{type(self).__name__}: {self}'
