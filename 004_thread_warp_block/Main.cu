#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>

__global__ void what_is_my_id(unsigned int * const block,
                              unsigned int * const thread,
                              unsigned int * const warp,
                              unsigned int * const calc_thread)
{
    // Thread_ID is block_index * block_size + thread_index inside this block
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    block[idx] = blockIdx.x;
    thread[idx] = threadIdx.x;
    warp[idx] = threadIdx.x / warpSize; // Use build in variable warpSize=32 to calculate actual warp

    calc_thread[idx] = idx;
}

#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_warp[ARRAY_SIZE];
unsigned int cpu_calc_thread[ARRAY_SIZE];


int main(void){
    // Total threads: 64 * 2 = 128
    const unsigned int num_blocks = 2;
    const unsigned int num_threads = 64;

    // Declare pointers for GPU based params
    unsigned int * gpu_block;
    unsigned int * gpu_thread;
    unsigned int * gpu_warp;
    unsigned int * gpu_calc_thread;

    // Declaration of loop iterator
    unsigned int i;

    // Allocate four arrays on GPU
    cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_warp, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);

    // Execute kernel
    what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread, gpu_warp, gpu_calc_thread);

    cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(gpu_block);
    cudaFree(gpu_thread);
    cudaFree(gpu_warp);
    cudaFree(gpu_calc_thread);

    for(i=0;i<ARRAY_SIZE;i++){
        printf("Calculcated thread: %3u | Block: %3u | Warp: %3u | Thread: %3u\n",
                cpu_calc_thread[i], cpu_block[i], cpu_warp[i], cpu_thread[i]);
    }
}
