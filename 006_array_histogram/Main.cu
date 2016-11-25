#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <stdfloat.h>
#include <cstdlib>

#define BLOCKS (65536)
#define THREADS 256
#define N (BLOCKS * THREADS)
#define MAX_VAL 256

float32_t aqq;

__global__ void kernel_tid(unsigned int * d_tid){
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

    d_tid[tid] = tid;

}

__global__ void kernel_1(const unsigned char * const d_hist_data, unsigned int * const d_bin_data){
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

    const unsigned char value = d_hist_data[tid];

    atomicAdd(&(d_bin_data[value]), 1);
}

__global__ void kernel_4( uint32_t * d_hist_data, uint32_t * d_bin_data){
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

    uint8_t value1 = (d_hist_data[tid] & 0x000000FF);
    uint8_t value2 = (d_hist_data[tid] & 0x0000FF00) >> 8;
    uint8_t value3 = (d_hist_data[tid] & 0x00FF0000) >> 16;
    uint8_t value4 = (d_hist_data[tid] & 0xFF000000) >> 24;

    atomicAdd(&(d_bin_data[value1]), 1);
    atomicAdd(&(d_bin_data[value2]), 1);
    atomicAdd(&(d_bin_data[value3]), 1);
    atomicAdd(&(d_bin_data[value4]), 1);
}

__shared__ unsigned int d_bin_data_shared[256];

__global__ void kernel_4_shared( uint32_t * d_hist_data, uint32_t * d_bin_data){
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    const uint32_t tid = idx + idy * blockIdx.x * gridDim.x;

    // Shared data osobne dla każdego bloku, więc czyścimy po indeksie wątku! :)
    d_bin_data_shared[threadIdx.x] = 0;

    // One fetch from memory per 4 pixels
    const uint32_t data = d_hist_data[tid];

    uint8_t value1 = (data & 0x000000FF);
    uint8_t value2 = (data & 0x0000FF00) >> 8;
    uint8_t value3 = (data & 0x00FF0000) >> 16;
    uint8_t value4 = (data & 0xFF000000) >> 24;

    // Wait for all threads to update shared memory
    __syncthreads();

    atomicAdd(&(d_bin_data_shared[value1]), 1);
    atomicAdd(&(d_bin_data_shared[value2]), 1);
    atomicAdd(&(d_bin_data_shared[value3]), 1);
    atomicAdd(&(d_bin_data_shared[value4]), 1);

    // Wait for all threads to update shared memory
    __syncthreads();

    // Write accumulated data do global memory in blocks, not scattered
    atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);

}

void cpu_hist(const unsigned char* array, unsigned int* hist, int i){
    hist[array[i]]++;
}

int main(int argc, char** argv){


    // CPU
    unsigned int* tid = (unsigned int*) malloc(N * sizeof(unsigned int));
    unsigned char* array = (unsigned char*) malloc(N * sizeof(unsigned char));
    unsigned int* hist = (unsigned int*) malloc(MAX_VAL * sizeof(unsigned int));

    srand(time(NULL));
    for(int i = 0; i < MAX_VAL; i++){
        hist[i] = 0;
    }

    for(int i = 0; i < N; i++){
        array[i] = rand() % MAX_VAL;
        // array[i] = rand() % 1;
        // printf("%d\t", array[i]);
    }
    // printf("\n\n");


    //CPU

    for(int i = 0; i < N; i++){
        cpu_hist(array, hist, i);
    }

    // GPU 1

    // unsigned char* d_hist_data; cudaMalloc((void**)&d_hist_data, N * sizeof(unsigned char));
    // unsigned int* d_bin_data; cudaMalloc((void**)&d_bin_data, MAX_VAL * sizeof(unsigned int));
    //
    // cudaMemcpy(d_hist_data, array, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_bin_data, hist, MAX_VAL * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // kernel_1<<<dim3(BLOCKS/64, 64),THREADS>>>(d_hist_data, d_bin_data);
    // cudaMemcpy(hist, d_bin_data, MAX_VAL * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //
    // cudaFree(d_hist_data);
    // cudaFree(d_bin_data);


    // GPU 4
    // uint32_t* d_hist_data; cudaMalloc((void**) &d_hist_data, N * sizeof(unsigned char));
    // uint32_t* d_bin_data; cudaMalloc((void**) &d_bin_data, MAX_VAL * sizeof(uint32_t));
    //
    // cudaMemcpy(d_hist_data, array, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_bin_data, hist, MAX_VAL * sizeof(unsigned int), cudaMemcpyHostToDevice);
    //
    // kernel_4<<<dim3(BLOCKS/64/4, 64), THREADS>>>(d_hist_data, d_bin_data);
    //
    // cudaMemcpy(hist, d_bin_data, MAX_VAL * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //
    // cudaFree(d_hist_data);
    // cudaFree(d_bin_data);

    //



    // GPU 4 SHARED
    // uint32_t* d_hist_data; cudaMalloc((void**) &d_hist_data, N * sizeof(unsigned char));
    // uint32_t* d_bin_data; cudaMalloc((void**) &d_bin_data, MAX_VAL * sizeof(uint32_t));
    //
    // cudaMemcpy(d_hist_data, array, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_bin_data, hist, MAX_VAL * sizeof(unsigned int), cudaMemcpyHostToDevice);
    //
    // kernel_4<<<dim3(BLOCKS/64/4, 64), THREADS>>>(d_hist_data, d_bin_data);
    //
    // cudaMemcpy(hist, d_bin_data, MAX_VAL * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //
    // cudaFree(d_hist_data);
    // cudaFree(d_bin_data);



    unsigned int* d_tid; cudaMalloc((void**)&d_tid, N * sizeof(unsigned int));
    kernel_tid<<<dim3(BLOCKS/64, 64),THREADS>>>(d_tid);
    cudaMemcpy(tid, d_tid, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_tid);

    bool ok = true;
    int errors = 0;
    for(int i = 0; i < N; i++){
        if(tid[i] != i){
            // printf("%d\t%d\n", i, tid[i]);
            errors++;
            ok = false;
        }
    }
    if(ok){
        printf("Everything OK!");
    }else{
        printf("ERROR: %d mistakes", errors);
    }

    printf("\n\n");


    //  CPU
    for(int i = 0; i < MAX_VAL; i++){
        printf("%d\t", hist[i]);
    }
    printf("\n\n");

    free(array);
    free(hist);
    return 0;

}
