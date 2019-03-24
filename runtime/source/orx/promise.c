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
#include <orx/memory.h>
#include <orx/processor.h>
#include <orx/promise.h>
#include <orx/runtime.h>
#include <orx/wire.h>

void orx_promise_init(orx_promise_t* promise) {
    promise->wire     = NULL;
    promise->is_error = true;
    promise->result   = NULL;
}

void orx_promise_wait(orx_promise_t* promise) {
    orx_processor_t* processor = orx_processor_current();
    promise->wire              = orx_processor_current_wire(processor);

    // yield control to another wire
    orx_processor_yield(processor);
}

void orx_promise_set_error(orx_promise_t* promise, void* error) {
    promise->is_error = true;
    promise->result   = error;

    // resume work of wire
    orx_processor_t* processor = orx_processor_current();
    orx_processor_push(processor, promise->wire);
}

void orx_promise_set_result(orx_promise_t* promise, void* result) {
    promise->is_error = false;
    promise->result   = result;

    // resume work of wire
    orx_processor_t* processor = orx_processor_current();
    orx_processor_push(processor, promise->wire);
}
