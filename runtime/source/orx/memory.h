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

/**
 * Allocate memory for buffer
 *
 * @param size Size of buffer in bytes
 * @return Pointer to buffer
 */
void* orx_malloc(orx_size_t size);

/**
 * Allocate atomic memory for buffer, e.g. memory that not have references to another objects
 *
 * @param size Size of buffer in bytes
 * @return Pointer to buffer
 */
void* orx_malloc_atomic(orx_size_t size);

/**
 * Reallocate memory for buffer
 *
 * @param ptr   Old pointer to buffer
 * @param size  New size of buffer
 * @return Pointer to reallocated buffer
 */
void* orx_realloc(void* ptr, orx_size_t size);
