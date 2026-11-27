/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef ODIN3_WSI_DMA_CAPTURE_H
#define ODIN3_WSI_DMA_CAPTURE_H

#include <stddef.h>
#include <stdint.h>

/* An eager X11 WSI image export. The caller owns each nonnegative fd. */
typedef struct {
    int fd;
    uint32_t pixmap;
    uint16_t width, height;
    uint8_t planes, bpp;
    uint32_t stride, offset;
    uint64_t modifier;
} Odin3WsiDmaExport;

#ifdef __cplusplus
extern "C" {
#endif
int odin3_wsi_dma_begin(uint32_t drawable, Odin3WsiDmaExport *exports, size_t capacity, int prefer_linear);
size_t odin3_wsi_dma_end(void);
#ifdef __cplusplus
}
#endif

#endif
