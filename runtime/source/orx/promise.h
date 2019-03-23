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

#include <orx/types.h>

typedef struct orx_result_t {
    bool  is_error;
    void* result;
} orx_result_t;

/// Create new promise
orx_promise_t* orx_promise_create();

/// Wait for promise result. E.g. current wire is transfer control to next scheduler wire.
orx_result_t orx_promise_wait(orx_promise_t* promise);

/// Set error to promise and resume execution of waited wire
void orx_promise_set_error(orx_promise_t* promise, void* error);

/// Set error to promise and resume execution of waited wire
void orx_promise_set_result(orx_promise_t* promise, void* result);
