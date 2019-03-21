# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import logging
from typing import Mapping

from llvmlite import ir, binding

from orcinus.collections import LazyDictionary
from orcinus.exceptions import OrcinusError, DiagnosticError
from orcinus.symbols import *

logger = logging.getLogger('orcinus.codegen')


class ModuleEmitter:
    def __init__(self, name: str):
        # create llvm module
        self.__llvm_module = ir.Module(name)
        self.__llvm_module.triple = binding.Target.from_default_triple().triple
        self.__llvm_ref = None
        self.__is_normalize = False
        self.__llvm_types = LazyDictionary[Type, ir.Type](constructor=self.declare_type)
        self.__llvm_functions = LazyDictionary[Function, ir.Function](constructor=self.declare_function)

    @property
    def llvm_context(self) -> ir.Context:
        return self.llvm_module.context

    @property
    def llvm_module(self) -> ir.Module:
        return self.__llvm_module

    @property
    def llvm_types(self) -> Mapping[Type, ir.Type]:
        return self.__llvm_types

    @property
    def llvm_functions(self) -> Mapping[Function, ir.Function]:
        return self.__llvm_functions

    @property
    def is_normalize(self) -> bool:
        return self.__is_normalize

    @is_normalize.setter
    def is_normalize(self, value: bool):
        self.__is_normalize = value

    def verify(self):
        try:
            self.__llvm_ref = binding.parse_assembly(str(self.llvm_module))
            self.__llvm_ref.verify()
        except RuntimeError as ex:
            raise OrcinusError(str(ex))

    def __str__(self):
        if self.is_normalize:
            if not self.__llvm_ref:
                self.verify()
            return str(self.__llvm_ref)
        return str(self.llvm_module)

    def declare_function(self, func: Function) -> ir.Function:
        if func.is_generic:
            raise RuntimeError(u"Can not convert generic function to LLVM")

        llvm_arguments = [self.llvm_types[param.type] for param in func.parameters]
        llvm_returns = self.llvm_types[func.return_type]
        llvm_type = ir.FunctionType(llvm_returns, llvm_arguments)
        return ir.Function(self.llvm_module, llvm_type, func.native_name or func.name)

    def declare_type(self, symbol: Type) -> ir.Type:
        if isinstance(symbol, BooleanType):
            return ir.IntType(1)
        elif isinstance(symbol, IntegerType):
            return ir.IntType(64)
        elif isinstance(symbol, VoidType):
            return ir.LiteralStructType([])

        raise DiagnosticError(symbol.location, f'Conversion to LLVM is not implemented: {type(symbol).__name__}')

    def emit_function(self, func: Function):
        emitter = FunctionEmitter(self, func, self.llvm_functions[func])
        emitter.emit()

    def emit(self, module: Module):
        for func in module.functions:
            if not func.is_generic and not func.is_abstract:
                self.emit_function(func)


class FunctionEmitter:
    llvm_builder: ir.IRBuilder

    def __init__(self, parent: ModuleEmitter, func: Function, llvm_func: ir.Function):
        self.parent = parent
        self.function = func
        self.llvm_function = llvm_func
        self.llvm_blocks = {}
        self.llvm_instructions = {}
        self.llvm_builder = None

    @property
    def llvm_types(self):
        return self.parent.llvm_types

    def emit(self):
        # generate blocks
        for block in self.function.blocks:
            self.llvm_blocks[block] = self.llvm_function.append_basic_block(block.name)

        # generate code
        for block in self.function.blocks:
            self.llvm_builder = ir.IRBuilder(self.llvm_blocks[block])

            for inst in block.instructions:
                self.emit_instruction(inst)

    def get_value(self, value: Value) -> ir.Value:
        if isinstance(value, IntegerConstant):
            return ir.Constant(self.llvm_types[value.type], value.value)
        elif isinstance(value, BooleanConstant):
            return ir.Constant(self.llvm_types[value.type], value.value)
        elif isinstance(value, NoneConstant):
            return ir.Constant.literal_struct([])

        raise DiagnosticError(value.location, f'Conversion to LLVM is not implemented: {type(value).__name__}')

    def emit_instruction(self, inst: Instruction) -> ir.Value:
        if isinstance(inst, ReturnInstruction):
            result = self.emit_return_instruction(inst)
        elif isinstance(inst, BranchInstruction):
                result = self.emit_branch_instruction(inst)
        else:
            raise DiagnosticError(inst.location, f'Conversion to LLVM is not implemented: {type(inst).__name__}')

        self.llvm_instructions[inst] = result
        return result

    def emit_return_instruction(self, inst: ReturnInstruction) -> ir.Value:
        llvm_value = self.get_value(inst.value)
        return self.llvm_builder.ret(llvm_value)

    def emit_branch_instruction(self, inst: BranchInstruction) -> ir.Value:
        if inst.condition:
            llvm_cond = self.get_value(inst.condition)
            llvm_true = self.llvm_blocks[inst.then_block]
            llvm_false = self.llvm_blocks[inst.else_block]
            return self.llvm_builder.cbranch(llvm_cond, llvm_true, llvm_false)
        else:
            llvm_target = self.llvm_blocks[inst.then_block]
            return self.llvm_builder.branch(llvm_target)


def initialize_codegen():
    # init LLVM
    logger.debug('Initialize LLVM bindings')
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmparser()
    binding.initialize_native_asmprinter()
