/*******************************************************************************
 *     Copyright (c) 2019 Vasiliy Sheredeko vasiliy@sheredeko.me
 *         All Rights Reserved
 *
 *     THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE OF Vasiliy Sheredeko
 *     The copyright notice above does not evidence any
 *     actual or intended publication of such source code. The
 *     intellectual and technical concepts contained herein are proprietary to
 *     Vasiliy Sheredeko are protected by trade secret or copyright law.
 *     Dissemination of this information or reproduction of this material is
 *     strictly forbidden unless prior written permission is obtained from
 *     Vasiliy Sheredeko.
 ******************************************************************************/
#include <coro.h>
#include "memory.h"
#include "processor.h"
#include "wire.h"

#if CORO_USE_VALGRIND
#    include <valgrind/valgrind.h>
#endif

typedef struct coro_context orx_context_t;
typedef struct orx_stack_t {
    void*      pointer;
    orx_size_t size;
#if CORO_USE_VALGRIND
    int valgrind_id;
#endif
} orx_stack_t;

typedef struct orx_wire_t {
    orx_processor_t* processor;
    orx_context_t    context;
    orx_stack_t      stack;
    orx_wire_func    function;
    void*            argument;
} orx_wire_t;

static const orx_size_t ORX_WIRE_INITIAL_STACK_SIZE = 256 * 1024;

static void orx_wire_main(void* ptr) {
    orx_wire_t* wire = (orx_wire_t*) ptr;
    wire->function(wire->argument);
    orx_processor_yield(wire->processor); // TODO: Where go completed wire?
}

extern orx_wire_t* orx_wire_initial() {
    orx_wire_t* wire = orx_malloc(sizeof(orx_wire_t));
    coro_create(&wire->context, NULL, NULL, NULL, 0);
    wire->stack    = (orx_stack_t){NULL, 0, 0};
    wire->function = NULL;
    wire->argument = NULL;
    return wire;
}

extern orx_wire_t* orx_wire_create(orx_wire_func func, void* ptr) {
    orx_wire_t* wire    = orx_malloc(sizeof(orx_wire_t));
    wire->function      = func;
    wire->argument      = ptr;
    wire->stack.size    = ORX_WIRE_INITIAL_STACK_SIZE;
    wire->stack.pointer = orx_malloc(ORX_WIRE_INITIAL_STACK_SIZE);
#if CORO_USE_VALGRIND
    wire->stack.valgrind_id =
        VALGRIND_STACK_REGISTER((char*) wire->stack.size, ((char*) wire->stack.size) + (size_t) wire->stack.pointer);
#endif

    coro_create(&wire->context, orx_wire_main, wire, wire->stack.pointer, (size_t) wire->stack.size);
    return wire;
}

/// Register wire in processor
extern void orx_wire_register(orx_wire_t* wire, orx_processor_t* processor) {
    wire->processor = processor;
}

extern void orx_wire_transfer(orx_wire_t* current_wire, orx_wire_t* next_wire) {
    coro_transfer(&current_wire->context, &next_wire->context);
}
