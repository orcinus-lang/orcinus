# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import itertools
import logging
from typing import Mapping, Callable

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
    def llvm_exit(self) -> ir.Function:
        # TODO: Import from module `system`?

        llvm_exit = self.get_runtime_function('orx_exit', ir.VoidType(), [ir.IntType(64)])
        llvm_exit.attributes.add('noreturn')
        return llvm_exit

    @cached_property
    def llvm_malloc(self) -> ir.Function:
        return self.get_runtime_function('orx_malloc', ir.IntType(8).as_pointer(), [ir.IntType(64)])

    @cached_property
    def llvm_size(self) -> ir.Type:
        return ir.IntType(64)

    @cached_property
    def llvm_wire_func(self) -> ir.FunctionType:
        # typedef void (*orx_wire_func)(void*);
        llvm_void_ptr = ir.IntType(8).as_pointer()
        return ir.FunctionType(ir.VoidType(), [llvm_void_ptr])

    @cached_property
    def llvm_floor_double(self):
        return self.get_runtime_function('llvm.floor.f64', ir.DoubleType(), [ir.DoubleType()])

    @property
    def is_normalize(self) -> bool:
        return self.__is_normalize

    @is_normalize.setter
    def is_normalize(self, value: bool):
        self.__is_normalize = value

    def get_runtime_function(self, name: str, llvm_returns: ir.Type, llvm_params: Sequence[ir.Type]) -> ir.Function:
        try:
            return self.llvm_module.get_global(name)
        except KeyError:
            llvm_type = ir.FunctionType(llvm_returns, llvm_params)
            return ir.Function(self.llvm_module, llvm_type, name)

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

        # declare function
        llvm_arguments = [self.llvm_types[param.type] for param in func.parameters]
        llvm_returns = self.llvm_types[func.return_type]
        llvm_type = ir.FunctionType(llvm_returns, llvm_arguments)
        llvm_func = ir.Function(self.llvm_module, llvm_type, func.mangled_name)

        # parameters name
        for param, arg in zip(func.parameters, llvm_func.args):
            arg.name = param.name

        # function attributes
        llvm_func.linkage = 'external'
        if func.is_noreturn:
            llvm_func.attributes.add('noreturn')

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
            return ir.LiteralStructType([llvm_size, llvm_array]).as_pointer()

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

    def emit_main(self, func: Function):
        # fix main attributes
        llvm_entry = self.llvm_functions[func]
        llvm_entry.linkage = 'internal'

        # Create full blow function for main
        string_type = func.context.string_type
        arguments_type = ArrayType(func.module, string_type, location=func.location)

        # __orx_main prototype
        llvm_main_type = ir.FunctionType(ir.VoidType(), [self.llvm_types[arguments_type]])
        llvm_main = ir.Function(self.llvm_module, llvm_main_type, '__orx_main')
        llvm_main.linkage = 'internal'

        # __orx_main call
        llvm_builder = ir.IRBuilder(llvm_main.append_basic_block('entry'))
        if len(func.parameters) == 1:
            llvm_result = llvm_builder.call(llvm_entry, [llvm_main.args[0]])
        else:
            llvm_result = llvm_builder.call(llvm_entry, [])

        if isinstance(func.return_type, IntegerType):
            llvm_builder.call(self.llvm_exit, [llvm_result])
        llvm_builder.ret_void()

        # orx_start prototype: `void orx_main(int64_t argc, const char** argv, orx_wire_func main_func)`
        llvm_string_args = ir.IntType(8).as_pointer().as_pointer()
        llvm_init_args = [self.llvm_size, llvm_string_args, self.llvm_wire_func.as_pointer()]
        llvm_init = self.get_runtime_function('orx_start', ir.VoidType(), llvm_init_args)

        # C main function
        llvm_ctype = ir.FunctionType(ir.IntType(32), [ir.IntType(32), llvm_string_args])
        llvm_cmain = ir.Function(self.llvm_module, llvm_ctype, name="main")

        argc, argv = llvm_cmain.args
        argc.name = 'argc'
        argv.name = 'argv'

        llvm_builder = ir.IRBuilder(llvm_cmain.append_basic_block('entry'))
        llvm_builder.call(llvm_init, [
            llvm_builder.sext(argc, self.llvm_size),
            argv,
            llvm_builder.bitcast(llvm_main, self.llvm_wire_func.as_pointer())
        ])
        llvm_builder.ret(ir.Constant(ir.IntType(32), 0))

    def emit(self, module: Module):
        # emit declared functions
        for func in module.declared_functions:
            if not func.is_generic and not func.is_abstract:
                self.emit_function(func)

        # emit main function
        main_func = module.get_member('main')
        if isinstance(main_func, Function):
            self.emit_main(main_func)


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
    def llvm_module(self):
        return self.parent.llvm_module

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
                if inst.is_terminator and not self.llvm_builder.block.is_terminated:
                    self.llvm_builder.unreachable()

    def get_value(self, value: Value) -> ir.Value:
        if isinstance(value, IntegerConstant):
            return ir.Constant(self.llvm_types[value.type], value.value)
        elif isinstance(value, BooleanConstant):
            return ir.Constant(self.llvm_types[value.type], value.value)
        elif isinstance(value, NoneConstant):
            return ir.Constant.literal_struct([])
        elif isinstance(value, StringConstant):
            constant = ir.Constant.literal_array(
                [ir.Constant(ir.IntType(8), c) for c in bytes(value.value, encoding='utf-8')] +
                [ir.Constant(ir.IntType(8), 0)]
            )
            global_v = ir.GlobalVariable(self.llvm_module, constant.type, self.llvm_module.get_unique_name("string"))
            global_v.initializer = constant
            global_v.linkage = 'internal'
            return global_v.bitcast(ir.IntType(8).as_pointer())
        elif isinstance(value, Parameter):
            return self.llvm_parameters[value]
        elif isinstance(value, Instruction):
            llvm_inst = self.llvm_instructions.get(value)
            if not llvm_inst:
                raise DiagnosticError(value.location, f'Instructions order are broken: {type(value).__name__}')
            return llvm_inst

        raise DiagnosticError(value.location, f'Conversion to LLVM is not implemented: {type(value).__name__}')

    @staticmethod
    def get_builtin(func: Function) -> Optional[BuiltinEmitter]:
        if isinstance(func.owner, IntegerType):
            return INTEGER_BUILTINS.get(func.name)

        elif isinstance(func.owner, BooleanType):
            return BOOLEAN_BUILTINS.get(func.name)

        elif isinstance(func.owner, ArrayType):
            return ARRAY_BUILTINS.get(func.name)

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

        builtin_emitter = self.get_builtin(inst.function)
        if builtin_emitter:
            return builtin_emitter(self, llvm_arguments, name=inst.name)

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


BuiltinEmitter = Callable[[FunctionEmitter, Sequence[ir.Value], str], ir.Value]


class IntegerBuiltin:
    @staticmethod
    def emit_neg(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.neg(*arguments, name=name or '')

    @staticmethod
    def emit_pos(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return arguments[0]

    @staticmethod
    def emit_inv(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.not_(*arguments, name=name or '')

    @staticmethod
    def emit_add(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.add(*arguments, name=name or '')

    @staticmethod
    def emit_sub(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.sub(*arguments, name=name or '')

    @staticmethod
    def emit_mul(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.mul(*arguments, name=name or '')

    @staticmethod
    def emit_and(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.and_(*arguments, name=name or '')

    @staticmethod
    def emit_or(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.or_(*arguments, name=name or '')

    @staticmethod
    def emit_xor(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.xor(*arguments, name=name or '')

    @staticmethod
    def emit_eq(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.icmp_signed('==', *arguments, name=name or '')

    @staticmethod
    def emit_ne(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.icmp_signed('!=', *arguments, name=name or '')

    @staticmethod
    def emit_lt(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.icmp_signed('<', *arguments, name=name or '')

    @staticmethod
    def emit_le(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.icmp_signed('<=', *arguments, name=name or '')

    @staticmethod
    def emit_gt(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.icmp_signed('>', *arguments, name=name or '')

    @staticmethod
    def emit_ge(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.icmp_signed('>=', *arguments, name=name or '')

    @staticmethod
    def emit_idiv(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        # Floor division
        llvm_dbl1 = emitter.llvm_builder.sitofp(arguments[0], ir.DoubleType())
        llvm_dbl2 = emitter.llvm_builder.sitofp(arguments[1], ir.DoubleType())
        llvm_result = emitter.llvm_builder.fdiv(llvm_dbl1, llvm_dbl2)
        llvm_result = emitter.llvm_builder.call(emitter.parent.llvm_floor_double, [llvm_result])
        return emitter.llvm_builder.fptosi(llvm_result, arguments[0].type, name=name or '')

    @staticmethod
    def emit_imod(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        # Python style `%`, e.g. a mod b == a - b * a // b
        llvm_div = IntegerBuiltin.emit_idiv(emitter, arguments)
        llvm_mul = emitter.llvm_builder.mul(arguments[1], llvm_div)
        return emitter.llvm_builder.sub(arguments[0], llvm_mul, name=name or '')

    @staticmethod
    def emit_div(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        lhr = emitter.llvm_builder.sitofp(arguments[0], ir.DoubleType())
        rhr = emitter.llvm_builder.sitofp(arguments[1], ir.DoubleType())
        return emitter.llvm_builder.fdiv(lhr, rhr, name=name or '')


class BooleanBuiltins:
    @staticmethod
    def emit_not(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.not_(*arguments, name=name or '')

    @staticmethod
    def emit_and(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.and_(*arguments, name=name or '')

    @staticmethod
    def emit_or(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.or_(*arguments, name=name or '')

    @staticmethod
    def emit_xor(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.xor(*arguments, name=name or '')

    @staticmethod
    def emit_eq(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.icmp_signed('==', *arguments, name=name or '')

    @staticmethod
    def emit_ne(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.icmp_signed('!=', *arguments, name=name or '')


class ArrayBuiltins:
    @staticmethod
    def emit_len(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        return emitter.llvm_builder.extract_value(arguments[0], [0], name=name or '')

    @staticmethod
    def emit_getitem(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        llvm_array = emitter.llvm_builder.extract_value(arguments[0], [1])
        llvm_value = emitter.llvm_builder.gep(llvm_array, [arguments[1]])
        return emitter.llvm_builder.load(llvm_value, name=name or '')

    @staticmethod
    def emit_setitem(emitter: FunctionEmitter, arguments: Sequence[ir.Value], name: str = None) -> ir.Value:
        llvm_array = emitter.llvm_builder.extract_value(arguments[0], [1])
        llvm_value = emitter.llvm_builder.gep(llvm_array, [arguments[1]])
        emitter.llvm_builder.store(arguments[2], llvm_value)
        return ir.Constant.literal_struct([])


INTEGER_BUILTINS = {
    '__neg__': IntegerBuiltin.emit_neg,
    '__pos__': IntegerBuiltin.emit_pos,
    '__inv__': IntegerBuiltin.emit_inv,
    '__add__': IntegerBuiltin.emit_add,
    '__sub__': IntegerBuiltin.emit_sub,
    '__mul__': IntegerBuiltin.emit_mul,
    '__div__': IntegerBuiltin.emit_div,
    '__floordiv__': IntegerBuiltin.emit_idiv,
    '__mod__': IntegerBuiltin.emit_imod,
    '__and__': IntegerBuiltin.emit_and,
    '__or__': IntegerBuiltin.emit_or,
    '__xor__': IntegerBuiltin.emit_xor,
    '__eq__': IntegerBuiltin.emit_eq,
    '__ne__': IntegerBuiltin.emit_ne,
    '__lt__': IntegerBuiltin.emit_lt,
    '__le__': IntegerBuiltin.emit_le,
    '__gt__': IntegerBuiltin.emit_gt,
    '__ge__': IntegerBuiltin.emit_ge,
}

BOOLEAN_BUILTINS = {
    '__not__': BooleanBuiltins.emit_not,
    '__and__': BooleanBuiltins.emit_and,
    '__or__': BooleanBuiltins.emit_or,
    '__xor__': BooleanBuiltins.emit_xor,
    '__eq__': BooleanBuiltins.emit_eq,
    '__ne__': BooleanBuiltins.emit_ne,
}

ARRAY_BUILTINS = {
    '__len__': ArrayBuiltins.emit_len,
    '__getitem__': ArrayBuiltins.emit_getitem,
    '__setitem__': ArrayBuiltins.emit_setitem,
}


def initialize_codegen():
    # init LLVM
    logger.debug('Initialize LLVM bindings')
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmparser()
    binding.initialize_native_asmprinter()
