/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Capture the Mesa X11 WSI DMA-BUF exports for one scoped swapchain creation. */
#define _GNU_SOURCE
#include "odin3_wsi_dma_capture.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <xcb/dri3.h>

typedef struct {
    uint32_t drawable;
    Odin3WsiDmaExport *exports;
    size_t capacity, count;
    bool prefer_linear;
    xcb_connection_t *modifier_connection;
    unsigned modifier_sequence;
} Capture;

static _Thread_local Capture capture;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static __typeof__(&xcb_dri3_pixmap_from_buffers_checked) submit_buffers;
static __typeof__(&xcb_dri3_pixmap_from_buffer_checked) submit_buffer;
static __typeof__(&xcb_dri3_get_supported_modifiers) request_modifiers;
static __typeof__(&xcb_dri3_get_supported_modifiers_reply) receive_modifiers;

int odin3_wsi_dma_begin(uint32_t drawable, Odin3WsiDmaExport *exports, size_t capacity, int prefer_linear) {
    if (capture.drawable || !drawable || !exports || !capacity) return 0;
    capture = (Capture){.drawable = drawable, .exports = exports, .capacity = capacity, .prefer_linear = prefer_linear != 0};
    return 1;
}

size_t odin3_wsi_dma_end(void) {
    size_t count = capture.count;
    capture = (Capture){0};
    return count;
}

static void resolve(void) {
    void *library = dlopen("libxcb-dri3.so.0", RTLD_NOW | RTLD_LOCAL);
    if (!library) abort();
    submit_buffers = dlsym(library, "xcb_dri3_pixmap_from_buffers_checked");
    submit_buffer = dlsym(library, "xcb_dri3_pixmap_from_buffer_checked");
    request_modifiers = dlsym(library, "xcb_dri3_get_supported_modifiers");
    receive_modifiers = dlsym(library, "xcb_dri3_get_supported_modifiers_reply");
    if (!submit_buffers || !submit_buffer || !request_modifiers || !receive_modifiers) abort();
}

xcb_dri3_get_supported_modifiers_cookie_t xcb_dri3_get_supported_modifiers(xcb_connection_t *connection,
        uint32_t window, uint8_t depth, uint8_t bpp) {
    pthread_once(&resolve_once, resolve);
    const xcb_dri3_get_supported_modifiers_cookie_t cookie = request_modifiers(connection, window, depth, bpp);
    if (capture.prefer_linear && capture.drawable == window && bpp == 32) {
        capture.modifier_connection = connection;
        capture.modifier_sequence = cookie.sequence;
    }
    return cookie;
}

static bool has_linear(const uint64_t *modifiers, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i)
        if (modifiers[i] == 0) return true;
    return false;
}

xcb_dri3_get_supported_modifiers_reply_t *xcb_dri3_get_supported_modifiers_reply(xcb_connection_t *connection,
        xcb_dri3_get_supported_modifiers_cookie_t cookie, xcb_generic_error_t **error) {
    pthread_once(&resolve_once, resolve);
    xcb_dri3_get_supported_modifiers_reply_t *reply = receive_modifiers(connection, cookie, error);
    if (!reply || connection != capture.modifier_connection || cookie.sequence != capture.modifier_sequence) return reply;
    capture.modifier_connection = NULL;
    const uint32_t window = reply->num_window_modifiers, screen = reply->num_screen_modifiers;
    if ((uint64_t)window + screen > reply->length / 2) return reply;
    uint64_t *modifiers = (uint64_t *)(reply + 1);
    const uint32_t linear_window = has_linear(modifiers, window), linear_screen = has_linear(modifiers + window, screen);
    fprintf(stderr, "Odin3 WSI: modifiers window_count=%u screen_count=%u linear_window=%u linear_screen=%u\n",
        window, screen, linear_window, linear_screen);
    if (linear_window || linear_screen) {
        reply->num_window_modifiers = linear_window;
        reply->num_screen_modifiers = linear_screen;
        reply->length = 2 * (linear_window + linear_screen);
        for (uint32_t i = 0; i < linear_window + linear_screen; ++i) modifiers[i] = 0;
    }
    return reply;
}

static void record(uint32_t drawable, Odin3WsiDmaExport entry) {
    if (drawable != capture.drawable || !capture.exports) return;
    const size_t index = capture.count++;
    if (index >= capture.capacity) return;
    entry.fd = entry.planes == 1 && entry.fd >= 0 ? fcntl(entry.fd, F_DUPFD_CLOEXEC, 0) : -1;
    capture.exports[index] = entry;
}

xcb_void_cookie_t xcb_dri3_pixmap_from_buffers_checked(xcb_connection_t *connection,
        xcb_pixmap_t pixmap, xcb_window_t window, uint8_t planes, uint16_t width, uint16_t height,
        uint32_t stride0, uint32_t offset0, uint32_t stride1, uint32_t offset1,
        uint32_t stride2, uint32_t offset2, uint32_t stride3, uint32_t offset3,
        uint8_t depth, uint8_t bpp, uint64_t modifier, const int32_t *buffers) {
    pthread_once(&resolve_once, resolve);
    record(window, (Odin3WsiDmaExport){buffers && planes ? buffers[0] : -1, pixmap,
        width, height, planes, bpp, stride0, offset0, modifier});
    return submit_buffers(connection, pixmap, window, planes, width, height, stride0, offset0,
        stride1, offset1, stride2, offset2, stride3, offset3, depth, bpp, modifier, buffers);
}

xcb_void_cookie_t xcb_dri3_pixmap_from_buffer_checked(xcb_connection_t *connection,
        xcb_pixmap_t pixmap, xcb_drawable_t drawable, uint32_t size, uint16_t width,
        uint16_t height, uint16_t stride, uint8_t depth, uint8_t bpp, int32_t fd) {
    pthread_once(&resolve_once, resolve);
    /* The original DRI3 request does not identify the allocation modifier. */
    record(drawable, (Odin3WsiDmaExport){fd, pixmap, width, height, 1, bpp, stride, 0, UINT64_MAX});
    return submit_buffer(connection, pixmap, drawable, size, width, height, stride, depth, bpp, fd);
}
