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
#include <string.h>
#include <utf8.h>
#include <uv.h>
#include <orx/modules/io/file.h>
#include <orx/memory.h>
#include <orx/processor.h>
#include <orx/promise.h>
#include <orx/runtime.h>
#include <orx/types.h>

#define ORX_MIN(a, b) (((a) < (b)) ? (a) : (b))

static const size_t BUFFER_SIZE = 2;

typedef struct orx_file_t {
    orx_type_t* type;
    ssize_t     fd;
} orx_file_t;

typedef struct orx_file_request_t {
    orx_promise_t  promise; // promise
    struct uv_fs_s uv_req;  // uv file request
} orx__request_s;

static void orx__file_request_init(orx__request_s* request) {
    request->uv_req.data = &request->promise;
}

static int orx__file_request_wait(orx__request_s* request) {
    orx_promise_wait(&request->promise);
    int* ptr    = request->promise.result; // TODO: Throw error
    int  result = ptr == NULL ? 0 : *ptr;
    return result;
}

static void orx__file_request_cleanup(orx__request_s* request) {
    uv_fs_req_cleanup(&request->uv_req);
}

static void orx__on_file_open(uv_fs_t* req) {
    orx_promise_t* promise = (orx_promise_t*) req->data;
    if (req->result >= 0) {
        int* result = orx_malloc(sizeof(int));
        *result     = (int) req->result;
        orx_promise_set_result(promise, result);
    } else {
        // Fail errors
        orx_promise_set_error(promise, NULL);
        fprintf(stderr, "error opening file: %s\n", uv_strerror((int) req->result));
        orx_exit(-1);
    }
}

static void orx__on_file_close(uv_fs_t* req) {
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

static void orx__on_file_read(uv_fs_t* req) {
    orx_promise_t* promise = (orx_promise_t*) req->data;
    if (req->result >= 0) {
        int* result = orx_malloc(sizeof(int));
        *result     = (int) req->result;
        orx_promise_set_result(promise, result);
    } else {
        // Fail errors
        orx_promise_set_error(promise, NULL);
        fprintf(stderr, "error reading file: %s\n", uv_strerror((int) req->result));
        orx_exit(-1);
    }
}

static void orx__on_file_write(uv_fs_t* req) {
    orx_promise_t* promise = (orx_promise_t*) req->data;
    if (req->result >= 0) {
        orx_promise_set_result(promise, NULL);
    } else {
        // Fail errors
        orx_promise_set_error(promise, NULL);
        fprintf(stderr, "error writing file: %s\n", uv_strerror((int) req->result));
        orx_exit(-1);
    }
}

orx_file_t* orx_file_open(const char* path, const char* mode) {
    orx_processor_t* processor = orx_processor_current();
    uv_loop_t*       uv_loop   = orx_processor_loop(processor);
    orx_file_t*      file      = orx_malloc(sizeof(orx_file_t));

    int  flags     = 0;
    bool created   = false;
    bool appending = false;
    bool readable  = false;
    bool writable  = false;

    /// TODO: Must be as Python 3.7 (https://github.com/python/cpython/blob/master/Modules/_io/fileio.c#L290)
    const char* s = mode;
    while (*s) {
        switch (*s++) {
            case 'x':
                flags |= O_EXCL | O_CREAT;
                created  = true;
                writable = true;
                break;
            case 'r':
                readable = true;
                break;
            case 'w':
                flags |= O_CREAT | O_TRUNC;
                writable = true;
                break;
            case 'a':
                flags |= O_APPEND | O_CREAT;
                writable  = true;
                appending = true;
                break;
            default:
                /// TODO: Error
                break;
        }
    }
    if (readable && writable) {
        flags |= O_RDWR;
    } else if (readable) {
        flags |= O_RDONLY;
    } else {
        flags |= O_WRONLY;
    }

    orx__request_s request;
    orx__file_request_init(&request);
    uv_fs_open(uv_loop, &request.uv_req, utf8dup(path), flags, 0666, orx__on_file_open);
    file->fd = orx__file_request_wait(&request);
    orx__file_request_cleanup(&request);
    return file;
}

void orx_file_close(orx_file_t* file) {
    orx_processor_t* processor = orx_processor_current();
    uv_loop_t*       uv_loop   = orx_processor_loop(processor);

    orx__request_s request;
    orx__file_request_init(&request);
    uv_fs_close(uv_loop, &request.uv_req, (uv_file) file->fd, orx__on_file_close);
    orx__file_request_wait(&request);
    orx__file_request_cleanup(&request);
}

const char* orx_file_read(orx_file_t* file, orx_int64_t size) {
    // size == -1 => read all content of file
    orx_processor_t* processor = orx_processor_current();
    uv_loop_t*       uv_loop   = orx_processor_loop(processor);

    int      count  = 0;
    int      length = 0;
    char     data[BUFFER_SIZE + 1];
    char*    result = NULL;
    uv_buf_t buffer;

    orx__request_s request;
    orx__file_request_init(&request);

    do {
        buffer = uv_buf_init(data, (unsigned int) ORX_MIN(size == -1 ? BUFFER_SIZE : size - length, BUFFER_SIZE));
        uv_fs_read(uv_loop, &request.uv_req, (uv_file) file->fd, &buffer, 1, -1, orx__on_file_read);
        count       = orx__file_request_wait(&request);
        data[count] = '\0';

        if (count > 0) {
            size_t ssize   = (size_t) length + count + 1;
            result         = result ? orx_realloc(result, sizeof(char) * ssize) : orx_malloc(sizeof(char) * ssize);
            result[length] = '\0';
            utf8cat(result, data);

            length += count;
        } else {
            break;
        }
    } while (size == -1 || length < size);

    orx__file_request_cleanup(&request);
    return result ? result : "";
}

void orx_file_write(orx_file_t* file, char* value) {
    // size == -1 => read all content of file
    orx_processor_t* processor = orx_processor_current();
    uv_loop_t*       uv_loop   = orx_processor_loop(processor);

    uv_buf_t buffer;

    orx__request_s request;
    orx__file_request_init(&request);

    buffer = uv_buf_init(value, (unsigned int) strlen(value));
    uv_fs_write(uv_loop, &request.uv_req, (uv_file) file->fd, &buffer, 1, -1, orx__on_file_write);
    orx__file_request_wait(&request);
}
