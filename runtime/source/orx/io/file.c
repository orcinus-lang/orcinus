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
#include <uv.h>
#include <orx/io/file.h>
#include <orx/memory.h>
#include <orx/processor.h>
#include <orx/promise.h>
#include <orx/runtime.h>
#include <orx/types.h>

typedef struct orx_file_t {
    orx_type_t* type;
    ssize_t     fd;
} orx_file_t;

void __orx_on_file_open(uv_fs_t* req) {
    orx_promise_t* promise = (orx_promise_t*) req->data;
    if (req->result >= 0) {
        orx_promise_set_result(promise, NULL);
    } else {
        // Fail errors
        orx_promise_set_error(promise, NULL);
        fprintf(stderr, "error opening file: %s\n", uv_strerror((int) req->result));
        orx_exit(-1);
    }
}

void __orx_on_file_close(uv_fs_t* req) {
    orx_promise_t* promise = (orx_promise_t*) req->data;
    if (req->result >= 0) {
        orx_promise_set_result(promise, NULL);
    } else {
        // Fail errors
        orx_promise_set_error(promise, NULL);
        fprintf(stderr, "error closing file: %s\n", uv_strerror((int) req->result));
        orx_exit(-1);
    }
}

orx_file_t* orx_file_open(const orx_byte_t* path, const orx_byte_t* mode) {
    orx_processor_t* processor = orx_processor_current();
    uv_loop_t*       uv_loop   = orx_processor_loop(processor);
    orx_file_t*      file      = orx_malloc(sizeof(orx_file_t));
    orx_promise_t*   promise   = orx_promise_create();

    // start work
    struct uv_fs_s* request = orx_malloc(sizeof(struct uv_fs_s));
    request->data           = promise;
    file->fd = uv_fs_open(uv_loop, request, (const char*) path, S_IRUSR | S_IWUSR, 0, __orx_on_file_open);

    // Wait when promise fulfilled
    orx_promise_wait(promise);
    orx_free(promise);
    uv_fs_req_cleanup(request);
    orx_free(request);
    return file;
}

void orx_file_close(orx_file_t* file) {
    orx_processor_t* processor = orx_processor_current();
    uv_loop_t*       uv_loop   = orx_processor_loop(processor);
    orx_promise_t*   promise   = orx_promise_create();

    // start work
    struct uv_fs_s* request = orx_malloc(sizeof(struct uv_fs_s));
    request->data           = promise;
    uv_fs_close(uv_loop, request, (uv_file) file->fd, __orx_on_file_close);

    // Wait when promise fulfilled
    orx_promise_wait(promise);
    orx_free(promise);
    uv_fs_req_cleanup(request);
    orx_free(request);
}
