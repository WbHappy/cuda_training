#ifndef SHARED_IMAGE_PROCESSING_CUH_
#define SHARED_IMAGE_PROCESSING_CUH_

#include "gpu_errchk.cuh"

#include <cuda.h>

#define OFFSET 7

#define BLOCK_X 32
#define BLOCK_Y 32

#define SMEM_X (BLOCK_X + 2*OFFSET)
#define SMEM_Y (BLOCK_Y + 2*OFFSET)
#define SMAP_SIZE (SMAP_X * SMAP_Y)

#define HMAP_X 256
#define HMAP_Y 256
#define HMAP_SIZE (HMAP_X * HMAP_Y)

#define CMAP_X (HMAP_X - 2*OFFSET)
#define CMAP_Y (HMAP_Y - 2*OFFSET)
#define CMAP_SIZE (CMAP_X * CMAP_Y)


class SharedImageProcessing
{
    uint8_t *dev_hmap, *dev_cmap, *dev_tmap;
public:
    uint8_t *hmap, *cmap, *tmap;


public:
    SharedImageProcessing(uint8_t *hmap, uint8_t *cmap, uint8_t *tmap)
    {
        this->hmap = hmap;
        this->cmap = cmap;
        this->tmap = tmap;
    }

    void gpuAllocateMemory();

    void gpuCopyInputToDevice(uint8_t *hmap, uint8_t *cmap, uint8_t *tmap);

    void gpuExecuteKernel();

    void gpuCopyOutputToHost(uint8_t *hmap, uint8_t *cmap, uint8_t *tmap);

    void gpuFree();
};

#endif
