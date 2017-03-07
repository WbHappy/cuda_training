
#include <stdio.h>

#include <cuda.h>

#include "../Stopwatch.hpp"

// #define CUDA_CALL(x){
//     const cudaError_t a = (x);
//     if(a != cudaSucces){
//         printf("\nCUDA Errors: %s (err_num = %d)\n", cudaGetErrorString(a), a);
//         cudaDeviceReset();
//         assert(0);
//     }
// }

#define KERNEL_LOOP 65636
#define THREADS 128
#define BLOCKS 1024
#define N (THREADS * BLOCKS)

__global__ void const_test_gpu_literal(uint32_t * const data, const uint32_t num_elements){
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < num_elements){
        uint32_t d = tid;
        for(int i = 0; i < KERNEL_LOOP; i++){
            d ^= 0x55555555;
            d |= 0x77777777;
            d &= 0x33333333;
            d |= 0x11111111;
        }

        data[tid] = d;
    }

};

__constant__ static const uint32_t const_data_01 = 0x55555555;
__constant__ static const uint32_t const_data_02 = 0x77777777;
__constant__ static const uint32_t const_data_03 = 0x33333333;
__constant__ static const uint32_t const_data_04 = 0x11111111;

__global__ void const_test_gpu_const( uint32_t * const data, const uint32_t num_elements){
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements){
        uint32_t d = tid;
        for(int i = 0; i < KERNEL_LOOP; i++ ){
            d ^= const_data_01;
            d |= const_data_02;
            d &= const_data_03;
            d |= const_data_04;
        }

        data[tid] = d;
    }

};

__device__ static uint32_t data_01 = 0x55555555;
__device__ static uint32_t data_02 = 0x77777777;
__device__ static uint32_t data_03 = 0x33333333;
__device__ static uint32_t data_04 = 0x11111111;

__global__ void const_test_gpu_gmem( uint32_t * const data, const uint32_t num_elements){
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements){
        uint32_t d = tid;
        for(int i = 0; i < KERNEL_LOOP; i++ ){
            d ^= data_01;
            d |= data_02;
            d &= data_03;
            d |= data_04;
        }

        data[tid] = d;
    }
};


int main(int argc, char** argv){

    Stopwatch stopwatch;

    uint32_t* h_data = (uint32_t*) malloc(N * sizeof(uint32_t));
    uint32_t* h_result1 = (uint32_t*) malloc(N * sizeof(uint32_t));
    uint32_t* h_result2 = (uint32_t*) malloc(N * sizeof(uint32_t));
    uint32_t* h_result3 = (uint32_t*) malloc(N * sizeof(uint32_t));

    uint32_t* d_data; cudaMalloc((void**)&d_data, N * sizeof(uint32_t));


    stopwatch.Start();
    cudaMemcpy(d_data, h_data, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    const_test_gpu_const <<<BLOCKS, THREADS>>> (d_data, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result1, d_data, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopwatch.Check_n_Reset("CONSTANT: ");


    stopwatch.Start();
    cudaMemcpy(d_data, h_data, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    const_test_gpu_literal <<<BLOCKS, THREADS>>> (d_data, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result2, d_data, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopwatch.Check_n_Reset("LITERAL: ");


    stopwatch.Start();
    cudaMemcpy(d_data, h_data, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    const_test_gpu_gmem <<<BLOCKS, THREADS>>> (d_data, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result3, d_data, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopwatch.Check_n_Reset("GLOBAL: ");

    printf("\n%d %d %d\n", h_result1[10], h_result2[10], h_result3[10]);


    cudaFree(d_data);
    free(h_data);
    free(h_result1);
    free(h_result2);
    free(h_result3);

    return 0;
}
