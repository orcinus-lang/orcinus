# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import contextlib
import weakref
from typing import Sequence, Optional, Set

from orcinus.collections import NamedScope
from orcinus.syntax import SyntaxNode
from orcinus.utils import cached_property


class FlowGraph:
    def __init__(self):
        self.__blocks = []
        self.__scope = NamedScope()
        self.__enter_block = self.append_block('entry')
        self.__exit_block = FlowBlock(self, ':exit')

    @property
    def scope(self) -> NamedScope:
        return self.__scope

    @property
    def enter_block(self) -> FlowBlock:
        return self.__enter_block

    @property
    def exit_block(self) -> FlowBlock:
        return self.__exit_block

    @property
    def throw_block(self) -> FlowBlock:
        return self.__throw_block

    @property
    def blocks(self) -> Sequence[FlowBlock]:
        return self.__blocks

    def append_block(self, name: str) -> FlowBlock:
        block = FlowBlock(self, self.scope.add(name))
        self.__blocks.append(block)
        return block


class FlowEdge:
    __parent: weakref.ReferenceType
    __successor: weakref.ReferenceType

    def __init__(self, parent: FlowBlock, successor: FlowBlock):
        self.__parent = weakref.ref(parent)
        self.__successor = weakref.ref(successor)

    @property
    def parent(self) -> FlowBlock:
        return self.__parent()

    @property
    def successor(self) -> FlowBlock:
        return self.__successor()

    def __str__(self):
        return f'{self.parent.name} -> {self.successor.name}'

    def __repr__(self):
        return f'{type(self).__name__}: {self}'


class FlowBlock:
    def __init__(self, graph: FlowGraph, name: str):
        self.__graph = weakref.ref(graph)
        self.__name = name
        self.__enter_edges = []
        self.__exit_edges = []
        self.__instructions = []

    @property
    def graph(self) -> FlowGraph:
        return self.__graph()

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str):
        self.__name = self.graph.scope.add(value, self.__name)

    @property
    def enter_edges(self) -> Sequence[FlowEdge]:
        return self.__enter_edges

    @property
    def exit_edges(self) -> Sequence[FlowEdge]:
        return self.__exit_edges

    @property
    def predecessor(self) -> Set[FlowBlock]:
        return {edge.parent for edge in self.enter_edges}

    @property
    def successors(self) -> Set[FlowBlock]:
        return {edge.successor for edge in self.exit_edges}

    @property
    def is_terminated(self) -> bool:
        return bool(self.exit_edges)

    @property
    def is_succeeded(self) -> bool:
        return bool(self.enter_edges)

    @property
    def instructions(self) -> Sequence[SyntaxNode]:
        return self.__instructions

    def append_link(self, block: FlowBlock):
        assert block

        edge = FlowEdge(self, block)
        self.__exit_edges.append(edge)
        block.__enter_edges.append(edge)
        return edge

    def append_instruction(self, node: SyntaxNode):
        self.__instructions.append(node)

    def __str__(self):
        return self.__name

    def __repr__(self):
        return f'{type(self).__name__}: {self}'


class FlowBuilder:
    def __init__(self, graph: FlowGraph):
        self.__graph = graph
        self.__block = graph.enter_block
        self.__continue_block = None
        self.__break_block = None

    @property
    def graph(self) -> FlowGraph:
        return self.__graph

    @property
    def exit_block(self) -> FlowBlock:
        return self.__graph.exit_block

    @property
    def block(self) -> Optional[FlowBlock]:
        return self.__block

    @block.setter
    def block(self, block: Optional[FlowBlock]):
        self.__block = block

    @cached_property
    def unreached_block(self) -> FlowBlock:
        return self.append_block(':unreached')

    @property
    def required_block(self) -> FlowBlock:
        if not self.__block:
            self.__block = self.unreached_block
        return self.block

    @property
    def continue_block(self) -> Optional[FlowBlock]:
        return self.__continue_block

    @property
    def break_block(self) -> Optional[FlowBlock]:
        if not self.__break_block and self.__continue_block:
            self.__break_block = self.append_block('break')
        return self.__break_block

    def append_block(self, name: str):
        return self.graph.append_block(name)

    def append_instruction(self, node: SyntaxNode):
        self.required_block.append_instruction(node)

    def append_link(self, block: FlowBlock):
        return self.required_block.append_link(block)

    def unreachable(self):
        self.__block = None

    @contextlib.contextmanager
    def block_helper(self, name: str) -> BlockHelper:
        helper = BlockHelper(self.append_block(name))
        self.append_link(helper.enter_block)

        self.block = helper.enter_block
        yield helper
        helper.exit_block = self.block

    @contextlib.contextmanager
    def loop_helper(self, continue_block: FlowBlock) -> LoopHelper:
        old_continue = self.__continue_block
        old_break = self.__break_block

        self.__continue_block = continue_block
        self.__break_block = None

        helper = LoopHelper(continue_block)
        yield helper

        helper.break_block = self.__break_block

        self.__continue_block = old_continue
        self.__break_block = old_break


class BlockHelper:
    enter_block: FlowBlock
    exit_block: Optional[FlowBlock]

    def __init__(self, enter_block: FlowBlock):
        self.exit_block = self.enter_block = enter_block


class LoopHelper:
    continue_block: FlowBlock
    break_block: Optional[FlowBlock]

    def __init__(self, continue_block: FlowBlock):
        self.continue_block = continue_block
        self.break_block = None
