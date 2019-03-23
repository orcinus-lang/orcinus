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

#include <gc.h>
#include <stdio.h>
#include <stdlib.h>
#include "memory.h"
#include "processor.h"
#include "runtime.h"
#include "wire.h"

typedef struct orx_arguments_t {
    orx_size_t         length;
    const orx_byte_t** items;
} orx_arguments_t;

static orx_processor_t* global_processor;

void orx_initialize() {
//    GC_enable_incremental();
//    GC_INIT();
//    GC_disable();
}

extern void orx_start(int64_t argc, const char** argv, orx_wire_func main_func) {
    // initialize runtime
    orx_initialize();

    // initialize program arguments
    orx_arguments_t* arguments = orx_malloc(sizeof(orx_arguments_t));
    arguments->length          = argc;
    arguments->items           = orx_malloc(sizeof(const orx_byte_t*) * arguments->length);
    for (int i = 0; i < argc; ++i) {
        arguments->items[i] = (const orx_byte_t*) argv[i];
    }

    // initialize wire
    global_processor = orx_processor_create();
    orx_wire_t* wire = orx_wire_create(main_func, arguments);

    //    GC_add_roots(&global_processor, &global_processor + 1);

    orx_processor_push(global_processor, wire);
    orx_processor_run(global_processor);
}

extern void orx_exit(int64_t code) {
    orx_processor_exit(global_processor, code);
    orx_processor_yield(global_processor);
}

orx_processor_t* orx_processor_current() {
    return global_processor;
}
