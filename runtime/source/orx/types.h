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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef int64_t orx_size_t;
typedef int64_t orx_int64_t;
typedef uint8_t orx_byte_t;

typedef void (*orx_wire_func)(void*);

typedef struct orx_wire_t      orx_wire_t;
typedef struct orx_processor_t orx_processor_t;
