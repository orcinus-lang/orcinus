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
#include <stdio.h>
#include <stdlib.h>
#include <uv.h>
#include "memory.h"
#include "processor.h"
#include "runtime.h"
#include "wire.h"

typedef struct orx_processor_t {
    bool         is_executed;
    orx_int64_t  exit_code;
    uv_loop_t*   uv_loop;
    orx_wire_t*  current_wire;
    orx_wire_t*  work_wire;
    orx_wire_t** wires;
    orx_size_t   wire_count;
    orx_size_t   wire_capacity;
} orx_processor_t;

void orx_processor_main(void* ptr) {
    orx_processor_t* processor = (orx_processor_t*) ptr;
    orx_wire_t*      wire      = NULL;

    while (orx_processor_is_executed(processor)) {
        uv_run(processor->uv_loop, UV_RUN_NOWAIT);

        wire = orx_processor_pop(processor);
        if (wire) {
            orx_processor_transfer(processor, wire);
        }
    }

    // stop UV loop
    uv_stop(processor->uv_loop);

    // :( it worked, but it smells..
    fflush(stdout);
    fflush(stderr);
    quick_exit((int) processor->exit_code);
}

orx_processor_t* orx_processor_create() {
    orx_processor_t* processor = orx_malloc(sizeof(orx_processor_t));
    processor->is_executed     = true;
    processor->exit_code       = 0;
    processor->work_wire       = orx_wire_create(orx_processor_main, processor);
    processor->current_wire    = processor->work_wire;
    processor->wire_count      = 0;
    processor->wire_capacity   = 8;
    processor->wires           = orx_malloc(sizeof(orx_wire_t*) * processor->wire_capacity);

    processor->uv_loop = malloc(sizeof(uv_loop_t));
    uv_loop_init(processor->uv_loop);
    return processor;
}

void orx_processor_exit(orx_processor_t* processor, orx_int64_t code) {
    processor->is_executed = false;
    processor->exit_code   = code;
}

bool orx_processor_is_executed(orx_processor_t* processor) {
    return processor->is_executed && (processor->wire_count > 0 || uv_loop_alive(processor->uv_loop));
}

orx_wire_t* orx_processor_current_wire(orx_processor_t* processor) {
    return processor->current_wire;
}

uv_loop_t* orx_processor_loop(orx_processor_t* processor) {
    return processor->uv_loop;
}

void orx_processor_push(orx_processor_t* processor, orx_wire_t* wire) {
    if (processor->wire_count == processor->wire_capacity) {
        processor->wire_capacity = (orx_size_t)(processor->wire_capacity * 1.4);
        processor->wires         = orx_realloc(processor->wires, sizeof(orx_wire_t*) * processor->wire_capacity);
    }
    processor->wires[processor->wire_count++] = wire;
}

orx_wire_t* orx_processor_pop(orx_processor_t* processor) {
    if (processor->wire_count == 0) {
        return NULL;
    }
    return processor->wires[--processor->wire_count];
}

void orx_processor_transfer(orx_processor_t* processor, orx_wire_t* wire) {
    orx_wire_t* previous_wire = processor->current_wire;
    processor->current_wire   = wire;
    orx_wire_register(processor->current_wire, processor);
    orx_wire_transfer(previous_wire, processor->current_wire);
}

void orx_processor_yield(orx_processor_t* processor) {
    orx_processor_transfer(processor, processor->work_wire);
}

void orx_processor_run(orx_processor_t* processor) {
    orx_wire_t* initial_wire = orx_wire_initial();
    orx_wire_transfer(initial_wire, processor->work_wire);
    __builtin_unreachable();
}
