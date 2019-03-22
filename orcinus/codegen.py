# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import itertools
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
        self.__llvm_module = ir.Module(name, context=ir.Context())
        self.__llvm_module.triple = binding.Target.from_default_triple().triple
        self.__llvm_ref = None
        self.__is_normalize = True
        self.__llvm_types = LazyDictionary[Type, ir.Type](constructor=self.declare_type, initializer=self.emit_type)
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

    @cached_property
    def llvm_malloc(self) -> ir.Function:
        llvm_type = ir.FunctionType(ir.IntType(8).as_pointer(), [ir.IntType(64)])
        return ir.Function(self.llvm_module, llvm_type, 'malloc')

    @cached_property
    def llvm_size(self) -> ir.Type:
        return ir.IntType(64)

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
            raise DiagnosticError(func.location, f'Conversion to LLVM for generic function is not allowed')

        is_main = func.name == 'main'

        llvm_arguments = [self.llvm_types[param.type] for param in func.parameters]
        llvm_returns = self.llvm_types[func.return_type]
        llvm_type = ir.FunctionType(llvm_returns, llvm_arguments)
        llvm_func = ir.Function(self.llvm_module, llvm_type, func.name if is_main else func.mangled_name)
        for param, arg in zip(func.parameters, llvm_func.args):
            arg.name = param.name
        llvm_func.linkage = 'external' if func.is_abstract or is_main else 'internal'

        return llvm_func

    def declare_type(self, symbol: Type) -> ir.Type:
        if symbol.is_generic:
            raise DiagnosticError(symbol.location, f'Conversion to LLVM for generic type is not allowed')

        if isinstance(symbol, BooleanType):
            return ir.IntType(1)
        elif isinstance(symbol, IntegerType):
            return ir.IntType(64)
        elif isinstance(symbol, VoidType):
            return ir.LiteralStructType([])
        elif isinstance(symbol, VoidType):
            return ir.LiteralStructType([])
        elif isinstance(symbol, StringType):
            return ir.IntType(8).as_pointer()
        elif isinstance(symbol, ClassType):
            return self.llvm_context.get_identified_type(symbol.mangled_name).as_pointer()
        elif isinstance(symbol, ArrayType):
            # {size_t, T*}
            llvm_size = self.llvm_size
            llvm_element = self.llvm_types[symbol.element_type]
            llvm_array = llvm_element.as_pointer()
            return ir.LiteralStructType([llvm_size, llvm_array])

        raise DiagnosticError(symbol.location, f'Conversion to LLVM is not implemented: {type(symbol).__name__}')

    def emit_type(self, symbol: Type):
        if isinstance(symbol, ClassType):
            llvm_fields = []
            for field in symbol.fields:
                llvm_fields.append(self.llvm_types[field.type])

            llvm_struct: ir.IdentifiedStructType = cast(ir.PointerType, self.llvm_types[symbol]).pointee
            llvm_struct.set_body(*llvm_fields)

    def emit_function(self, func: Function):
        emitter = FunctionEmitter(self, func, self.llvm_functions[func])
        emitter.emit()

    def emit(self, module: Module):
        for func in module.declared_functions:
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
        self.llvm_parameters = {}
        self.llvm_builder = None

    @property
    def llvm_types(self):
        return self.parent.llvm_types

    @property
    def llvm_functions(self):
        return self.parent.llvm_functions

    def emit(self):
        # generate parameters
        for param, arg in zip(self.function.parameters, self.llvm_function.args):
            self.llvm_parameters[param] = arg

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
        elif isinstance(value, Parameter):
            return self.llvm_parameters[value]
        elif isinstance(value, Instruction):
            llvm_inst = self.llvm_instructions.get(value)
            if not llvm_inst:
                raise DiagnosticError(value.location, f'Instructions order are broken: {type(value).__name__}')
            return llvm_inst

        raise DiagnosticError(value.location, f'Conversion to LLVM is not implemented: {type(value).__name__}')

    def emit_instruction(self, inst: Instruction) -> ir.Value:
        if isinstance(inst, ReturnInstruction):
            result = self.emit_return_instruction(inst)
        elif isinstance(inst, BranchInstruction):
            result = self.emit_branch_instruction(inst)
        elif isinstance(inst, AllocaInstruction):
            result = self.emit_alloca_instruction(inst)
        elif isinstance(inst, StoreInstruction):
            result = self.emit_store_instruction(inst)
        elif isinstance(inst, LoadInstruction):
            result = self.emit_load_instruction(inst)
        elif isinstance(inst, CallInstruction):
            result = self.emit_call_instruction(inst)
        elif isinstance(inst, NewInstruction):
            result = self.emit_new_instruction(inst)
        elif isinstance(inst, InsertValueInstruction):
            result = self.emit_insert_value_instruction(inst)
        elif isinstance(inst, ExtractValueInstruction):
            result = self.emit_extract_value_instruction(inst)
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

    def emit_alloca_instruction(self, inst: AllocaInstruction) -> ir.Value:
        llvm_type = self.llvm_types[inst.type]
        return self.llvm_builder.alloca(llvm_type, name=inst.name)

    def emit_load_instruction(self, inst: LoadInstruction) -> ir.Value:
        llvm_source = self.get_value(inst.source)
        return self.llvm_builder.load(llvm_source, name=inst.name)

    def emit_store_instruction(self, inst: StoreInstruction) -> ir.Value:
        llvm_source = self.get_value(inst.source)
        llvm_target = self.get_value(inst.target)
        return self.llvm_builder.store(llvm_source, llvm_target)

    def emit_call_instruction(self, inst: CallInstruction) -> ir.Value:
        llvm_arguments = [self.get_value(arg) for arg in inst.arguments]

        # TODO: Builtins calls
        if inst.function.name == '__neg__':
            return self.llvm_builder.neg(llvm_arguments[0], name=inst.name)
        if inst.function.name == '__pos__':
            return llvm_arguments[0]
        elif inst.function.name == '__add__':
            return self.llvm_builder.add(llvm_arguments[0], llvm_arguments[1], name=inst.name)
        elif inst.function.name == '__eq__':
            return self.llvm_builder.icmp_signed('==', llvm_arguments[0], llvm_arguments[1], name=inst.name)
        elif inst.function.name == '__ne__':
            return self.llvm_builder.icmp_signed('!=', llvm_arguments[0], llvm_arguments[1], name=inst.name)
        elif inst.function.name == '__lt__':
            return self.llvm_builder.icmp_signed('<', llvm_arguments[0], llvm_arguments[1], name=inst.name)
        elif inst.function.name == '__le__':
            return self.llvm_builder.icmp_signed('<=', llvm_arguments[0], llvm_arguments[1], name=inst.name)
        elif inst.function.name == '__gt__':
            return self.llvm_builder.icmp_signed('>', llvm_arguments[0], llvm_arguments[1], name=inst.name)
        elif inst.function.name == '__ge__':
            return self.llvm_builder.icmp_signed('>=', llvm_arguments[0], llvm_arguments[1], name=inst.name)

        # TODO: Invoke calls

        llvm_function = self.llvm_functions[inst.function]
        return self.llvm_builder.call(llvm_function, llvm_arguments, name=inst.name)

    def emit_new_instruction(self, inst: NewInstruction) -> ir.Value:
        llvm_struct: ir.PointerType = cast(ir.PointerType, self.llvm_types[inst.type])
        llvm_offset = self.llvm_builder.gep(ir.Constant(llvm_struct, None), [ir.Constant(ir.IntType(32), 1)])
        llvm_sizeof = self.llvm_builder.ptrtoint(llvm_offset, ir.IntType(64))
        llvm_pointer = self.llvm_builder.call(self.parent.llvm_malloc, [llvm_sizeof])
        llvm_instance = self.llvm_builder.bitcast(llvm_pointer, llvm_struct)

        llvm_arguments = list(itertools.chain((llvm_instance,), (self.get_value(arg) for arg in inst.arguments)))
        llvm_function = self.llvm_functions[inst.function]
        self.llvm_builder.call(llvm_function, llvm_arguments, name=inst.name)
        return llvm_instance

    def emit_insert_value_instruction(self, inst: InsertValueInstruction) -> ir.Value:
        index = inst.instance.type.fields.index(inst.field)
        llvm_instance = self.get_value(inst.instance)
        llvm_index = self.llvm_builder.gep(llvm_instance, [
            ir.Constant(ir.IntType(32), 0),
            ir.Constant(ir.IntType(32), index),
        ])
        llvm_value = self.get_value(inst.source)
        return self.llvm_builder.store(llvm_value, llvm_index)

    def emit_extract_value_instruction(self, inst: ExtractValueInstruction) -> ir.Value:
        index = inst.instance.type.fields.index(inst.field)
        llvm_instance = self.get_value(inst.instance)
        llvm_index = self.llvm_builder.gep(llvm_instance, [
            ir.Constant(ir.IntType(32), 0),
            ir.Constant(ir.IntType(32), index),
        ])
        return self.llvm_builder.load(llvm_index)


def initialize_codegen():
    # init LLVM
    logger.debug('Initialize LLVM bindings')
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmparser()
    binding.initialize_native_asmprinter()
