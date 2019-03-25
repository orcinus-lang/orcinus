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
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <orx/types.h>

const char* orx_int64_str(orx_int64_t value) {
    char buffer[100];
    int  count    = sprintf(buffer, "%" PRId64, value);
    buffer[count] = '\0';
    return strdup(buffer); // TODO: Memory leak
}
