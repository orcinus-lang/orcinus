# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from orcinus.semantic import *
from orcinus.tests.utils.parser import parse_member, check_diagnostics


def annotate_control_flow(content: str):
    context, node = parse_member(content)
    assert isinstance(node, FunctionNode)

    annotator = FlowAnnotator(context.diagnostics)
    flow = annotator.annotate(node)
    check_diagnostics(context.diagnostics)
    return flow


def test_condition_simple_flow():
    flow = annotate_control_flow("""
def main():
    if cond:
        pass
    """)

    assert len(flow.blocks) == 3

    block_enter = flow.blocks[0]
    block_then = flow.blocks[1]
    block_else = flow.blocks[2]
    block_exit = flow.exit_block

    assert block_enter.successors, {block_then, block_else}
    assert block_then.successors, {block_else}
    assert block_else.successors, {block_exit}


def test_condition_simple_flow_with_return():
    flow = annotate_control_flow("""
def main():
    if cond:
        return
    """)

    assert len(flow.blocks) == 3

    block_enter = flow.blocks[0]
    block_then = flow.blocks[1]
    block_else = flow.blocks[2]
    block_exit = flow.exit_block

    # enter -> {then, else}
    assert block_enter.successors == {block_then, block_else}

    # then -> {exit}
    assert block_then.successors == {block_exit}

    # else -> {exit}
    assert block_else.successors == {block_exit}


def test_condition_else_flow():
    flow = annotate_control_flow("""
def main():
    if cond:
        pass
    else:
        pass
    """)

    assert len(flow.blocks) == 4

    block_enter = flow.blocks[0]
    block_then = flow.blocks[1]
    block_else = flow.blocks[2]
    block_next = flow.blocks[3]
    block_exit = flow.exit_block

    assert block_enter.successors == {block_then, block_else}
    assert block_then.successors == {block_next}
    assert block_else.successors == {block_next}
    assert block_next.successors == {block_exit}


def test_condition_else_flow_with_return():
    flow = annotate_control_flow("""
def main():
    if cond:
        return
    else:
        return
    """)

    assert len(flow.blocks) == 3

    block_enter = flow.blocks[0]
    block_then = flow.blocks[1]
    block_else = flow.blocks[2]
    block_exit = flow.exit_block

    assert block_enter.successors == {block_then, block_else}
    assert block_then.successors == {block_exit}
    assert block_else.successors == {block_exit}


def test_condition_else_flow_with_partial_return():
    flow = annotate_control_flow("""
def main():
    if cond:
        return
    else:
        pass
    """)

    assert len(flow.blocks) == 4

    block_enter = flow.blocks[0]
    block_then = flow.blocks[1]
    block_else = flow.blocks[2]
    block_next = flow.blocks[3]
    block_exit = flow.exit_block

    assert block_enter.successors == {block_then, block_else}
    assert block_then.successors == {block_exit}
    assert block_else.successors == {block_next}
    assert block_next.successors == {block_exit}


def test_condition_else_flow_with_unreachable():
    flow = annotate_control_flow("""
def main():
    if cond:
        return
    else:
        return
    return
        """)

    assert len(flow.blocks) == 4

    block_enter = flow.blocks[0]
    block_then = flow.blocks[1]
    block_else = flow.blocks[2]
    block_unreached = flow.blocks[3]
    block_exit = flow.exit_block

    assert block_enter.successors == {block_then, block_else}
    assert block_then.successors == {block_exit}
    assert block_else.successors == {block_exit}
    assert block_unreached.predecessor == set()
    assert block_unreached is not flow.enter_block


def test_while_flow():
    flow = annotate_control_flow("""
def main():
    while True:
        pass
    """)

    assert len(flow.blocks) == 5

    block_enter = flow.blocks[0]
    block_loop = flow.blocks[1]
    block_cond = flow.blocks[2]
    block_then = flow.blocks[3]
    block_else = flow.blocks[4]
    block_exit = flow.exit_block

    assert block_enter.successors == {block_loop}
    assert block_loop.successors == {block_cond}
    assert block_cond.successors == {block_then, block_else}
    assert block_then.successors == {block_cond}
    assert block_else.successors == {block_exit}


def test_while_flow_with_return():
    flow = annotate_control_flow("""
def main():
    while True:
        return
    """)

    assert len(flow.blocks) == 5

    block_enter = flow.blocks[0]
    block_loop = flow.blocks[1]
    block_cond = flow.blocks[2]
    block_then = flow.blocks[3]
    block_else = flow.blocks[4]
    block_exit = flow.exit_block

    assert block_enter.successors == {block_loop}
    assert block_loop.successors == {block_cond}
    assert block_cond.successors == {block_then, block_else}
    assert block_then.successors == {block_exit}
    assert block_else.successors == {block_exit}


def test_while_else_flow():
    flow = annotate_control_flow("""
def main():
    while True:
        pass
    else:
        pass
    """)

    assert len(flow.blocks) == 6

    block_enter = flow.blocks[0]
    block_loop = flow.blocks[1]
    block_cond = flow.blocks[2]
    block_then = flow.blocks[3]
    block_else = flow.blocks[4]
    block_next = flow.blocks[5]
    block_exit = flow.exit_block

    assert block_enter.successors == {block_loop}
    assert block_loop.successors == {block_cond}
    assert block_cond.successors == {block_then, block_else}
    assert block_then.successors == {block_cond}
    assert block_else.successors == {block_next}
    assert block_next.successors == {block_exit}


def test_while_else_flow_with_return():
    flow = annotate_control_flow("""
def main():
    while True:
        return
    else:
        return
    """)

    assert len(flow.blocks) == 5

    block_enter = flow.blocks[0]
    block_loop = flow.blocks[1]
    block_cond = flow.blocks[2]
    block_then = flow.blocks[3]
    block_else = flow.blocks[4]
    block_exit = flow.exit_block

    assert block_enter.successors == {block_loop}
    assert block_loop.successors == {block_cond}
    assert block_cond.successors == {block_then, block_else}
    assert block_then.successors == {block_exit}
    assert block_else.successors == {block_exit}


def test_while_else_flow_with_partial_return():
    flow = annotate_control_flow("""
def main():
    while True:
        pass
    else:
        return
    """)

    assert len(flow.blocks) == 5

    block_enter = flow.blocks[0]
    block_loop = flow.blocks[1]
    block_cond = flow.blocks[2]
    block_then = flow.blocks[3]
    block_else = flow.blocks[4]
    block_exit = flow.exit_block

    assert block_enter.successors == {block_loop}
    assert block_loop.successors == {block_cond}
    assert block_cond.successors == {block_then, block_else}
    assert block_then.successors == {block_cond}
    assert block_else.successors == {block_exit}

def test_while_else_flow_with_break():
    flow = annotate_control_flow("""
def main():
    while True:
        break
    """)

    assert len(flow.blocks) == 5

    block_enter = flow.blocks[0]
    block_loop = flow.blocks[1]
    block_cond = flow.blocks[2]
    block_then = flow.blocks[3]
    block_else = flow.blocks[4]
    block_exit = flow.exit_block

    assert block_enter.successors == {block_loop}
    assert block_loop.successors == {block_cond}
    assert block_cond.successors == {block_then, block_else}
    assert block_then.successors == {block_cond}
    assert block_else.successors == {block_exit}
