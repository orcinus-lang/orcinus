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
#include <stdlib.h>
#include "memory.h"

void* orx_malloc(orx_size_t size) {
    return GC_MALLOC((size_t) size);
}

void* orx_malloc_atomic(orx_size_t size) {
    return GC_MALLOC_ATOMIC((size_t) size);
}

void* orx_realloc(void* ptr, orx_size_t size) {
    return GC_REALLOC(ptr, (size_t) size);
}

void orx_free(void* ptr) {
    GC_FREE(ptr);
}
