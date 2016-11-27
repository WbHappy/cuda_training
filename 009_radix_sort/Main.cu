#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>


#define THREADS 32
#define BLOCKS 1
#define N (THREADS*BLOCKS)
#define BIT 32

__host__ void cpuRadixSort(uint32_t* array, uint32_t length)
{
    static uint32_t cpu_temp0[N];
    static uint32_t cpu_temp1[N];

    for(int bit=0;bit<BIT;bit++){
        uint32_t base_cnt0 = 0;
        uint32_t base_cnt1 = 0;

        for(int i=0;i<N;i++){
            const uint32_t elem = array[i];
            const uint32_t bit_mask = (1 << bit);

            if((elem & bit_mask) > 0){
                cpu_temp1[base_cnt1] = elem;
                base_cnt1++;
            }else{
                cpu_temp0[base_cnt0] = elem;
                base_cnt0++;
            }
        }

        for(int i=0;i<base_cnt0;i++){
            array[i] = cpu_temp0[i];
        }
        for(int i=0;i<base_cnt1;i++){
            array[i+base_cnt0] = cpu_temp1[i];
        }
    }
}

__device__ void gpuRadixSort(uint32_t* const array,
                             uint32_t* const sort_tmp0,
                             uint32_t* const sort_tmp1,
                             const uint32_t num_list,
                             const uint32_t num_elements,
                             const uint32_t tid)
{
    for(int bit=0;bit<N;bit++){
        uint32_t base_cnt0 = 0;
        uint32_t base_cnt1 = 0;

        for(int i=0;i<num_elements;i+=num_list){
            const uint32_t elem = array[i + tid];
            const uint32_t bit_mask = (1 << bit);

            if((elem & bit_mask) > 0){
                sort_tmp1[base_cnt1 + tid] = elem;
                base_cnt1+=num_list;
            }else{
                sort_tmp0[base_cnt0 + tid] = elem;
                base_cnt0+=num_list;
            }
        }

        for(int i=0;i<base_cnt0;i+=num_list){
            array[i+tid] = sort_tmp0[i+tid];
        }
        for(int i=0;i<base_cnt1;i+=num_list){
            array[base_cnt0+i+tid] = sort_tmp1[i+tid];
        }

    }
    __syncthreads();
}

__global__ void kernel_RadixSort(uint32_t* array, uint32_t* sort_tmp0, uint32_t* sort_tmp1, uint32_t num_elements){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    gpuRadixSort(array, sort_tmp0, sort_tmp1, (int)BIT, (int)N, idx);
}

int main(int argc, char** argv){
    srand(time(NULL));

    uint32_t* array = (uint32_t*)malloc(N*sizeof(uint32_t));

    for(int i=0;i<N;i++){
        array[i] = rand()%256;
    }

    for(int i=0;i<N;i++){
        printf("%d\n", array[i]);
    }
//  CPU
    // cpuRadixSort(array, N);

//  GPU
    uint32_t* d_array; cudaMalloc((void**)&d_array, N*sizeof(uint32_t));
    uint32_t* d_sort_tmp0; cudaMalloc((void**)&d_sort_tmp0, N*sizeof(uint32_t));
    uint32_t* d_sort_tmp1; cudaMalloc((void**)&d_sort_tmp1, N*sizeof(uint32_t));

    cudaMemcpy(d_array, array, N*sizeof(uint32_t), cudaMemcpyHostToDevice);

    kernel_RadixSort <<<BLOCKS, THREADS>>> (d_array, d_sort_tmp0, d_sort_tmp0, (uint32_t)N);

    cudaMemcpy(array, d_array, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("\n\n");


    for(int i=0;i<N;i++){
        printf("%d\n", array[i]);
    }

    cudaFree(d_array);
    cudaFree(d_sort_tmp0);
    cudaFree(d_sort_tmp1);
    free(array);
    return 0;
}
