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

#include <stdint.h>
#include <stdio.h>
#include <utf8.h>
#include "types.h"

void orx_print(const char* message, bool is_newline) {
    fputs(message, stdout);
    if (is_newline) {
        fputs("\n", stdout);
    }
}

orx_int64_t orx_string_compare(const char* self, const char* other) {
    return utf8cmp(self, other);
}

orx_int64_t orx_string_length(const char* self) {
    return utf8len(self);
}
