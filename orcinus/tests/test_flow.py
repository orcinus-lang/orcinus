# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

# def annotate_control_flow(content: str, has_errors=False):
#     context, node = parse_member(content)
#     assert isinstance(node, FunctionNode)
#
#     annotator = FlowAnnotator(context.diagnostics)
#     flow = annotator.annotate(node)
#     if not has_errors:
#         check_diagnostics(context.diagnostics)
#     return context, node, flow
#
#
# def test_condition_simple_flow():
#     _, _, flow = annotate_control_flow("""
# def main():
#     if cond:
#         pass
#     """)
#
#     assert len(flow.blocks) == 3
#
#     block_enter = flow.blocks[0]
#     block_then = flow.blocks[1]
#     block_else = flow.blocks[2]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors, {block_then, block_else}
#     assert block_then.successors, {block_else}
#     assert block_else.successors, {block_exit}
#
#
# def test_condition_simple_flow_with_return():
#     _, _, flow = annotate_control_flow("""
# def main():
#     if cond:
#         return
#     """)
#
#     assert len(flow.blocks) == 3
#
#     block_enter = flow.blocks[0]
#     block_then = flow.blocks[1]
#     block_else = flow.blocks[2]
#     block_exit = flow.exit_block
#
#     # enter -> {then, else}
#     assert block_enter.successors == {block_then, block_else}
#
#     # then -> {exit}
#     assert block_then.successors == {block_exit}
#
#     # else -> {exit}
#     assert block_else.successors == {block_exit}
#
#
# def test_condition_else_flow():
#     _, _, flow = annotate_control_flow("""
# def main():
#     if cond:
#         pass
#     else:
#         pass
#     """)
#
#     assert len(flow.blocks) == 4
#
#     block_enter = flow.blocks[0]
#     block_then = flow.blocks[1]
#     block_else = flow.blocks[2]
#     block_next = flow.blocks[3]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_then, block_else}
#     assert block_then.successors == {block_next}
#     assert block_else.successors == {block_next}
#     assert block_next.successors == {block_exit}
#
#
# def test_condition_else_flow_with_return():
#     _, _, flow = annotate_control_flow("""
# def main():
#     if cond:
#         return
#     else:
#         return
#     """)
#
#     assert len(flow.blocks) == 3
#
#     block_enter = flow.blocks[0]
#     block_then = flow.blocks[1]
#     block_else = flow.blocks[2]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_then, block_else}
#     assert block_then.successors == {block_exit}
#     assert block_else.successors == {block_exit}
#
#
# def test_condition_else_flow_with_partial_return():
#     _, _, flow = annotate_control_flow("""
# def main():
#     if cond:
#         return
#     else:
#         pass
#     """)
#
#     assert len(flow.blocks) == 4
#
#     block_enter = flow.blocks[0]
#     block_then = flow.blocks[1]
#     block_else = flow.blocks[2]
#     block_next = flow.blocks[3]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_then, block_else}
#     assert block_then.successors == {block_exit}
#     assert block_else.successors == {block_next}
#     assert block_next.successors == {block_exit}
#
#
# def test_condition_else_flow_with_unreachable():
#     _, _, flow = annotate_control_flow("""
# def main():
#     if cond:
#         return
#     else:
#         return
#     return
#         """)
#
#     assert len(flow.blocks) == 4
#
#     block_enter = flow.blocks[0]
#     block_then = flow.blocks[1]
#     block_else = flow.blocks[2]
#     block_unreached = flow.blocks[3]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_then, block_else}
#     assert block_then.successors == {block_exit}
#     assert block_else.successors == {block_exit}
#     assert block_unreached.predecessor == set()
#     assert block_unreached is not flow.enter_block
#
#
# def test_while_flow():
#     _, _, flow = annotate_control_flow("""
# def main():
#     while True:
#         pass
#     """)
#
#     assert len(flow.blocks) == 5
#
#     block_enter = flow.blocks[0]
#     block_loop = flow.blocks[1]
#     block_cond = flow.blocks[2]
#     block_then = flow.blocks[3]
#     block_else = flow.blocks[4]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_loop}
#     assert block_loop.successors == {block_cond}
#     assert block_cond.successors == {block_then, block_else}
#     assert block_then.successors == {block_cond}
#     assert block_else.successors == {block_exit}
#
#
# def test_while_flow_with_return():
#     _, _, flow = annotate_control_flow("""
# def main():
#     while True:
#         return
#     """)
#
#     assert len(flow.blocks) == 5
#
#     block_enter = flow.blocks[0]
#     block_loop = flow.blocks[1]
#     block_cond = flow.blocks[2]
#     block_then = flow.blocks[3]
#     block_else = flow.blocks[4]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_loop}
#     assert block_loop.successors == {block_cond}
#     assert block_cond.successors == {block_then, block_else}
#     assert block_then.successors == {block_exit}
#     assert block_else.successors == {block_exit}
#
#
# def test_while_else_flow():
#     _, _, flow = annotate_control_flow("""
# def main():
#     while True:
#         pass
#     else:
#         pass
#     """)
#
#     assert len(flow.blocks) == 6
#
#     block_enter = flow.blocks[0]
#     block_loop = flow.blocks[1]
#     block_cond = flow.blocks[2]
#     block_then = flow.blocks[3]
#     block_else = flow.blocks[4]
#     block_next = flow.blocks[5]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_loop}
#     assert block_loop.successors == {block_cond}
#     assert block_cond.successors == {block_then, block_else}
#     assert block_then.successors == {block_cond}
#     assert block_else.successors == {block_next}
#     assert block_next.successors == {block_exit}
#
#
# def test_while_else_flow_with_return():
#     _, _, flow = annotate_control_flow("""
# def main():
#     while True:
#         return
#     else:
#         return
#     """)
#
#     assert len(flow.blocks) == 5
#
#     block_enter = flow.blocks[0]
#     block_loop = flow.blocks[1]
#     block_cond = flow.blocks[2]
#     block_then = flow.blocks[3]
#     block_else = flow.blocks[4]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_loop}
#     assert block_loop.successors == {block_cond}
#     assert block_cond.successors == {block_then, block_else}
#     assert block_then.successors == {block_exit}
#     assert block_else.successors == {block_exit}
#
#
# def test_while_else_flow_with_partial_return():
#     _, _, flow = annotate_control_flow("""
# def main():
#     while True:
#         pass
#     else:
#         return
#     """)
#
#     assert len(flow.blocks) == 5
#
#     block_enter = flow.blocks[0]
#     block_loop = flow.blocks[1]
#     block_cond = flow.blocks[2]
#     block_then = flow.blocks[3]
#     block_else = flow.blocks[4]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_loop}
#     assert block_loop.successors == {block_cond}
#     assert block_cond.successors == {block_then, block_else}
#     assert block_then.successors == {block_cond}
#     assert block_else.successors == {block_exit}
#
#
# def test_while_else_flow_with_break():
#     _, _, flow = annotate_control_flow("""
# def main():
#     while True:
#         break
#     """)
#
#     assert len(flow.blocks) == 5
#
#     block_enter = flow.blocks[0]
#     block_loop = flow.blocks[1]
#     block_cond = flow.blocks[2]
#     block_then = flow.blocks[3]
#     block_else = flow.blocks[4]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_loop}
#     assert block_loop.successors == {block_cond}
#     assert block_cond.successors == {block_then, block_else}
#     assert block_then.successors == {block_else}
#     assert block_else.successors == {block_exit}
#
#
# def test_while_else_flow_with_continue():
#     _, _, flow = annotate_control_flow("""
# def main():
#     while True:
#         continue
#     """)
#
#     assert len(flow.blocks) == 5
#
#     block_enter = flow.blocks[0]
#     block_loop = flow.blocks[1]
#     block_cond = flow.blocks[2]
#     block_then = flow.blocks[3]
#     block_else = flow.blocks[4]
#     block_exit = flow.exit_block
#
#     assert block_enter.successors == {block_loop}
#     assert block_loop.successors == {block_cond}
#     assert block_cond.successors == {block_then, block_else}
#     assert block_then.successors == {block_cond}
#     assert block_else.successors == {block_exit}
#
#
# def test_continue_without_loop():
#     context, _, flow = annotate_control_flow("""
# def main():
#     continue
#     """, has_errors=True)
#
#     assert len(flow.blocks) == 1
#
#     block_enter = flow.blocks[0]
#     block_exit = flow.exit_block
#
#     # assert block_enter.successors == {block_exit} # TODO: successors is empty set
#
#
# def test_break_without_loop():
#     context, _, flow = annotate_control_flow("""
# def main():
#     break
#     """, has_errors=True)
#
#     assert len(flow.blocks) == 1
#
#     block_enter = flow.blocks[0]
#     block_exit = flow.exit_block
#
#     # assert block_enter.successors == {block_exit} # TODO: successors is empty set
#
#     assert len(context.diagnostics) == 1
#     assert 'break' in context.diagnostics[0].message
