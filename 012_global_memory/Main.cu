#include "../Stopwatch.hpp"

#include <cuda.h>

// #include "const_common.h"

#define NUM_ELEMENTS 4096

#define CUDA_CALL(x) {
    const cudaError_t a = (x);
    if (a != cudaSuccess){
        printf("\nCUDA Error: %s (err_num=%d) \n", cudaGetErrorString(a), a);
        cudaDeviceReset();
        // assert(0);
    }
}

// INTERLEAVED
typedef struct
{
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
} INTERLEAVED_T;
typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[NUM_ELEMENTS];

// NON INTERLEAVED
typedef uint32_t ARRAY_MEMBER_T[NUM_ELEMENTS];
typedef struct
{
    ARRAY_MEMBER_T a;
    ARRAY_MEMBER_T b;
    ARRAY_MEMBER_T c;
    ARRAY_MEMBER_T d;
} NON_INTERLAVED_T;


__host__ void add_test_non_interleaved_cpu(NON_INTERLAVED_T * const host_dest_ptr,
                                            const NON_INTERLAVED_T * const host_src_ptr,
                                            const uint32_t iter,
                                            const uint32_t num_elements)
{

    for(uint32_t tid = 0; tid < num_elements; tid++){
        for(uint32_t i = 0; i < iter; i++){
            host_dest_ptr->a[tid] += host_src_ptr->a[tid];
            host_dest_ptr->b[tid] += host_src_ptr->b[tid];
            host_dest_ptr->c[tid] += host_src_ptr->c[tid];
            host_dest_ptr->d[tid] += host_src_ptr->d[tid];
        }
    }

}

__host__ void add_test_interleaved_cpu(INTERLEAVED_T * const host_dest_ptr,
                                        const INTERLEAVED_T * const host_src_ptr,
                                        const uint32_t iter,
                                        const uint32_t num_elements)
{

    for(uint32_t tid = 0; tid < num_elements; tid++){
        for(uint32_t i = 0; i < iter; i++){
            host_dest_ptr[tid].a += host_src_ptr[tid].a;
            host_dest_ptr[tid].b += host_src_ptr[tid].b;
            host_dest_ptr[tid].c += host_src_ptr[tid].c;
            host_dest_ptr[tid].d += host_src_ptr[tid].d;
        }
    }
}

__global__ void add_kernel_interleaved(
    INTERLEAVED_T * const dest_ptr,
    const INTERLEAVED_T * const src_ptr,
    const uint32_t iter,
    const uint32_t num_elements
){
    const uint32_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if(tid < num_elements){
        for(int i = 0; i < iter; i++){
            dest_ptr[tid].a += src_ptr[tid].a;
            dest_ptr[tid].b += src_ptr[tid].b;
            dest_ptr[tid].c += src_ptr[tid].c;
            dest_ptr[tid].d += src_ptr[tid].d;
        }
    }
}

__global__ void add_kernel_non_interleaved(
    NON_INTERLAVED_T * const dest_ptr,
    const NON_INTERLAVED_T * const src_ptr,
    const uint32_t iter,
    const uint32_t num_elements
){
    const uint32_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if(tid < num_elements){
        for(int i = 0; i < iter; i++){
            dest_ptr->a[tid] += src_ptr->a[tid];
            dest_ptr->b[tid] += src_ptr->b[tid];
            dest_ptr->c[tid] += src_ptr->c[tid];
            dest_ptr->d[tid] += src_ptr->d[tid];
        }
    }
}

int main(int argc, char** argv){

    Stopwatch stopwatch;

    INTERLEAVED_T array_of_structs1[NUM_ELEMENTS];
    INTERLEAVED_T array_of_structs2[NUM_ELEMENTS];

    NON_INTERLAVED_T struct_of_arrays1;
    NON_INTERLAVED_T struct_of_arrays2;

    for(int i = 0; i < NUM_ELEMENTS; i++){
        array_of_structs1[i].a = 1;
        array_of_structs1[i].b = 2;
        array_of_structs1[i].c = 3;
        array_of_structs1[i].d = 4;

        struct_of_arrays1.a[i] = 1;
        struct_of_arrays1.b[i] = 2;
        struct_of_arrays1.c[i] = 3;
        struct_of_arrays1.d[i] = 4;

        array_of_structs2[i].a = 0;
        array_of_structs2[i].b = 0;
        array_of_structs2[i].c = 0;
        array_of_structs2[i].d = 0;

        struct_of_arrays2.a[i] = 0;
        struct_of_arrays2.b[i] = 0;
        struct_of_arrays2.c[i] = 0;
        struct_of_arrays2.d[i] = 0;
    }

    stopwatch.Start();
    add_test_interleaved_cpu(array_of_structs2, array_of_structs1, 100, NUM_ELEMENTS);
    stopwatch.Check_n_Reset("ARRAY OF STRUCTS");

    add_test_non_interleaved_cpu(&struct_of_arrays2, &struct_of_arrays1, 100, NUM_ELEMENTS);
    stopwatch.Check_n_Reset("STRUCT OF ARRAYS");


    printf("CPU RESULTS INTERLEAVED:     %d %d %d %d\n", array_of_structs2[64].a, array_of_structs2[64].b, array_of_structs2[64].c, array_of_structs2[64].d);
    printf("CPU RESULTS NON INTERLEAVED: %d %d %d %d\n", struct_of_arrays2.a[64], struct_of_arrays2.b[64], struct_of_arrays2.c[64], struct_of_arrays2.d[64]);

    for(int i = 0; i < NUM_ELEMENTS; i++){
        array_of_structs2[i].a = 0;
        array_of_structs2[i].b = 0;
        array_of_structs2[i].c = 0;
        array_of_structs2[i].d = 0;

        struct_of_arrays2.a[i] = 0;
        struct_of_arrays2.a[i] = 0;
        struct_of_arrays2.a[i] = 0;
        struct_of_arrays2.a[i] = 0;
    }


    const uint32_t num_threads = 64;
    const uint32_t num_blocks = 64;
    const uint32_t size_inter = NUM_ELEMENTS * sizeof(INTERLEAVED_T);
    const uint32_t size_noninter = sizeof(NON_INTERLAVED_T);

    // INTERLEAVED
    INTERLEAVED_T * d_inter1; cudaMalloc((void**)&d_inter1, size_inter);
    INTERLEAVED_T * d_inter2; cudaMalloc((void**)&d_inter2, size_inter);

    cudaDeviceSynchronize();

    stopwatch.Start();
    cudaMemcpy(d_inter1, array_of_structs1, size_inter, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inter2, array_of_structs2, size_inter, cudaMemcpyHostToDevice);
    add_kernel_interleaved <<<num_blocks, num_threads>>> (array_of_structs2, array_of_structs1, 100, NUM_ELEMENTS);
    cudaMemcpy(array_of_structs2, d_inter2, size_inter, cudaMemcpyDeviceToHost);
    stopwatch.Check_n_Reset("INTERLEAVED GPU: ");

    cudaFree(d_inter1);
    cudaFree(d_inter2);


    // NON INTERLEAVED

    NON_INTERLAVED_T * d_noninter1; cudaMalloc((void**)&d_noninter1, size_noninter);
    NON_INTERLAVED_T * d_noninter2; cudaMalloc((void**)&d_noninter2, size_noninter);

    cudaDeviceSynchronize();

    stopwatch.Start();
    cudaMemcpy(d_noninter1, &struct_of_arrays1, size_inter, cudaMemcpyHostToDevice);
    cudaMemcpy(d_noninter2, &struct_of_arrays2, size_inter, cudaMemcpyHostToDevice);
    add_kernel_non_interleaved <<<num_blocks, num_threads>>> (&struct_of_arrays2, &struct_of_arrays1, 100, NUM_ELEMENTS);
    cudaMemcpy(&struct_of_arrays2, d_noninter2, size_inter, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopwatch.Check_n_Reset("NON INTERLEAVED GPU: ");

    cudaFree(d_noninter1);
    cudaFree(d_noninter2);

    for(int i = 2; i < 3; i++){
        printf("GPU RESULTS INTERLEAVED:     %d %d %d %d\n", array_of_structs2[i].a, array_of_structs2[i].b, array_of_structs2[i].c, array_of_structs2[i].d);
    }
    for(int i = 2; i < 3; i++){
        printf("GPU RESULTS NON INTERLEAVED: %d %d %d %d\n", struct_of_arrays2.a[i], struct_of_arrays2.b[i], struct_of_arrays2.c[i], struct_of_arrays2.d[i]);
    }

    return 0;
}
