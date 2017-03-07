
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
#define THREADS 8
#define BLOCKS 8
#define N (THREADS * BLOCKS)

__global__ void const_test_gpu_literal(uint32_t * const data, const uint32_t num_elements){
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < num_elements){
        uint32_t d = 0x55555555;
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
    const uint32_t tid = (blockIdx.x * blockDim.x) + blockIdx.x;
    if(tid < num_elements){
        uint32_t d = const_data_01;
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
    const uint32_t tid = (blockIdx.x * blockDim.x) + blockIdx.x;
    if(tid < num_elements){
        uint32_t d = const_data_01;
        for(int i = 0; i < KERNEL_LOOP; i++ ){
            d ^= data_01;
            d |= data_02;
            d &= data_03;
            d |= data_04;
        }

        data[tid] = d;
    }
};

// __host__ void wait_exit(void){
//     char ch;
//
//     printf("\nPress any key to exit");
//     ch = getch();
// };
//
// __host__ void cuda_error_check(const char * prefix, const char * postfix){
//     if(cudaPeekAtLastError() != cudaSucces){
//         printf("\n%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
//         cudaDeviceReset();
//         waitExit();
//         exit(-1);
//     }
// };
//
//
// __host__ void gpu_kernel(void){
//     const uint32_t num_elements = (128*1024);
//     const uint32_t num_threads = 256;
//     const uint32_t num_blocks = (num_elements + num_threads -1)/num_threads;
//     const uint32_t num_bytes = num_elements + sizeof(uint32_t);
//     int max_device_num;
//     const int max_runs = 6;
//
//     CUDA_CALL(cudaGetDeviceCount(&max_device_num));
//
//     for(int device_num = 0; device_num < max_device_num; device++){
//         CUDA_CALL(cudaSetDevice(device_num));
//
//         for(int num_test = 0; num_test < max_runs; num_test++){
//             uint32_t * data_gpu;
//             cudaEvent_t kernel_start1, kernel_stop1;
//             cudaEvent_t kernel_start2, kernel_stop2;
//
//             float delta_time1 = 0.0F, delta_time2 = 0.0F;
//             struct cudaDeviceProp device_prop;
//             char device_prefix[261];
//
//             CUDA_CALL(cudaMalloc(%data_gpu, num_bytes));
//             CUDA_CALL(cudaEventCreate(&kernel_start1));
//             CUDA_CALL(cudaEventCreate(&kernel_start2));
//             CUDA_CALL(cudaEventCreateWithFlags(&kernel_stop1, cudaEventBlockingSync));
//             CUDA_CALL(cudaEventCreateWithFlags(&kernel_stop2, cudaEventBlockingSync));
//
//             CUDA_CALL(cudaGetDeviceProperties(&device_prop, device_num));
//             sprintf(device_prefix, "ID:%d %s:", device_num, device_prop.name);
//
//             const_test_gpu_literal <<<num_blocks, num_threads>>>(data_gpu, num_elements);
//             cuda_error_check("Error ", " returned from literal startup kernel");
//
//             CUDA_CALL(cudaEventRecord(kernel_start1,0));
//             const_test_gpu_literal <<<num_blocks, num_threads>>>(data_gpu, num_elements);
//
//             cuda_error_check("Error ", " returned from literal runtime kernel");
//
//             CUDA_CALL(cudaEventRecord(kernel_stop1,0));
//             CUDA_CALL(cudaEventSynchronize(kernel_stop1));
//             CUDA_CALL(cudaEventElapsedTime(&delta_time1, kernel_start1, kernel_stop1));
//
//
//             const_test_gpu_const <<<num_blocks, num_threads>>>(data_gpu, num_elements);
//             cuda_error_check("Error ", " returned from constant startup kernel");
//             // Do the constant kernel
//             // printf("\nLaunching constant kernel");
//             CUDA_CALL(cudaEventRecord(kernel_start2,0));
//             const_test_gpu_const <<<num_blocks, num_threads>>>(data_gpu, num_elements);
//             cuda_error_check("Error ", " returned from constant runtime kernel");
//             CUDA_CALL(cudaEventRecord(kernel_stop2,0));
//             CUDA_CALL(cudaEventSynchronize(kernel_stop2));
//             CUDA_CALL(cudaEventElapsedTime(&delta_time2, kernel_start2, kernel_stop2));
//             // printf("\nConst Elapsed time: %.3fms", delta_time2);
//
//
//             if (delta_time1 > delta_time2)
//             printf("\n%sConstant version is faster by: %.2fms (Const1⁄4%.2fms vs. Literal1⁄4%.2fms)", device_prefix, delta_time1-delta_time2, delta_time1, delta_time2);
//             else
//             printf("\n%sLiteral version is faster by: %.2fms (Const1⁄4%.2fms vs. Literal1⁄4%.2fms)", device_prefix, delta_time2-delta_time1, delta_time1, delta_time2);
//             CUDA_CALL(cudaEventDestroy(kernel_start1));
//             CUDA_CALL(cudaEventDestroy(kernel_start2));
//             CUDA_CALL(cudaEventDestroy(kernel_stop1));
//             CUDA_CALL(cudaEventDestroy(kernel_stop2));
//             CUDA_CALL(cudaFree(data_gpu));
//             }
//             CUDA_CALL(cudaDeviceReset());
//             printf("\n");
//             }
//             wait_exit();
//             }
//         }
//     }
// };


int main(int argc, char** argv){

    Stopwatch stopwatch;

    uint32_t* h_data = (uint32_t*) malloc(N * sizeof(uint32_t));
    uint32_t* h_result1 = (uint32_t*) malloc(N * sizeof(uint32_t));
    uint32_t* h_result2 = (uint32_t*) malloc(N * sizeof(uint32_t));
    uint32_t* h_result3 = (uint32_t*) malloc(N * sizeof(uint32_t));

    uint32_t* d_data; cudaMalloc((void**)&d_data, N * sizeof(uint32_t));


    cudaMemcpy(d_data, h_data, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopwatch.Start();
    const_test_gpu_const <<<BLOCKS, THREADS>>> (d_data, N);
    cudaDeviceSynchronize();
    stopwatch.Check_n_Reset("CONSTANT: ");
    cudaMemcpy(h_result1, d_data, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    cudaMemcpy(d_data, h_data, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopwatch.Start();
    const_test_gpu_literal <<<BLOCKS, THREADS>>> (d_data, N);
    cudaDeviceSynchronize();
    stopwatch.Check_n_Reset("LITERAL: ");
    cudaDeviceSynchronize();
    cudaMemcpy(h_result2, d_data, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    cudaMemcpy(d_data, h_data, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopwatch.Start();
    const_test_gpu_gmem <<<BLOCKS, THREADS>>> (d_data, N);
    cudaDeviceSynchronize();
    stopwatch.Check_n_Reset("GLOBAL: ");
    cudaMemcpy(h_result3, d_data, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("\n%d %d %d\n", h_result1[10], h_result2[10], h_result3[10]);


    cudaFree(d_data);
    free(h_data);
    free(h_result1);
    free(h_result2);
    free(h_result3);

    return 0;
}
