#include "shared_image_processing.cuh"




__global__ void kernel(uint8_t *dev_hmap, uint8_t *dev_cmap, uint8_t *dev_tmap)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = idx + idy * gridDim.x * blockDim.x;
    int tid2;


    int sidx = threadIdx.x;
    int sidy = threadIdx.y;
    int stid = sidy * blockDim.x + sidx;

    if(idx >= CMAP_X || idy >= CMAP_Y)
    {
        return;
    }


    dev_cmap[tid] = 255;

    __shared__ uint8_t smem_hmap[SMEM_Y][SMEM_X];

    //1
    smem_hmap[sidy][sidx] = dev_hmap[tid];

    //2
    if(sidy < 2*OFFSET)
    {
        tid2 = idx + (idy + BLOCK_Y) * gridDim.x * blockDim.x;
        smem_hmap[BLOCK_Y + sidy][sidx] = dev_hmap[tid2];
    }

    //3
    if(sidx < 2*OFFSET)
    {
        tid2 = (idx + BLOCK_X) + idy * gridDim.x * blockDim.x;
        smem_hmap[sidy][BLOCK_X + sidx] = dev_hmap[tid2];
    }

    //4
    if(sidx < 2*OFFSET && sidy < 2*OFFSET)
    {
        tid2 = (idx + BLOCK_X) + (idy + BLOCK_Y) * gridDim.x * blockDim.x;
        smem_hmap[BLOCK_Y + sidy][BLOCK_X + sidx] = dev_hmap[tid2];
    }

    tid2 = (idx + 2*OFFSET) + (idy + 2*OFFSET) * gridDim.x * blockDim.x;
    __syncthreads();

    // if(blockIdx.x == 1 && blockIdx.y == 1)
    // {
        dev_tmap[tid2] = smem_hmap[2*OFFSET + sidy][2*OFFSET + sidx];
    // }
}




void SharedImageProcessing::gpuAllocateMemory()
{
    gpuErrchk( cudaMalloc((void**)&dev_hmap,  HMAP_SIZE * sizeof(uint8_t)) );
    gpuErrchk( cudaMalloc((void**)&dev_cmap,  CMAP_SIZE * sizeof(uint8_t)) );
    gpuErrchk( cudaMalloc((void**)&dev_tmap,  CMAP_SIZE * sizeof(uint8_t)) );
}





void SharedImageProcessing::gpuCopyInputToDevice(uint8_t *hmap, uint8_t *cmap, uint8_t *tmap)
{
    gpuErrchk( cudaMemcpy(dev_hmap, hmap, HMAP_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dev_cmap, cmap, CMAP_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dev_tmap, tmap, CMAP_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice) );
}





void SharedImageProcessing::gpuExecuteKernel()
{
    // Wątków napewno nie mniej niż potrzeba, konieczny warunek wykonania
    int GRID_X = (CMAP_X + BLOCK_X - 1) / BLOCK_X;
    int GRID_Y = (CMAP_Y + BLOCK_Y - 1) / BLOCK_Y;

    dim3 grid(GRID_X, GRID_Y,1);
    dim3 block(BLOCK_X, BLOCK_Y,1);

    kernel <<<grid, block>>> (dev_hmap, dev_cmap, dev_tmap);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}





void SharedImageProcessing::gpuCopyOutputToHost(uint8_t *hmap, uint8_t *cmap, uint8_t *tmap)
{
    gpuErrchk( cudaMemcpy(hmap, dev_hmap, HMAP_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(cmap, dev_cmap, CMAP_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(tmap, dev_tmap, CMAP_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost) );
}





void SharedImageProcessing::gpuFree()
{
    gpuErrchk( cudaFree(dev_hmap) );
    gpuErrchk( cudaFree(dev_cmap) );
    gpuErrchk( cudaFree(dev_tmap) );

}
