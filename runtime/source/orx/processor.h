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
#pragma once

#include "types.h"

typedef struct uv_loop_s uv_loop_t;

/// Create new processor
orx_processor_t* orx_processor_create();

/// Exit from process
void orx_processor_exit(orx_processor_t* processor, orx_int64_t code);

/// Returns true, if processor has not executed wires or not completed loop events
bool orx_processor_is_executed(orx_processor_t* processor);

/// Returns current wire
orx_wire_t* orx_processor_current_wire(orx_processor_t* processor);

/// Returns uv loop for processor
uv_loop_t* orx_processor_loop(orx_processor_t* processor);

/// Push wire to execution
void orx_processor_push(orx_processor_t* processor, orx_wire_t* wire);

/// Push wire to execution
orx_wire_t* orx_processor_pop(orx_processor_t* processor);

/// Transfer execution to wire
void orx_processor_transfer(orx_processor_t* processor, orx_wire_t* wire);

/// Transfer execution to processor
void orx_processor_run(orx_processor_t* processor) __attribute__((noreturn));

/// Yield execution to processor
void orx_processor_yield(orx_processor_t* processor);
