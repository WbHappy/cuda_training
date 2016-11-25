#include "CudaClass.cuh"
#include "Stopwatch.hpp"

#define X 640
#define Y 480
#define N (X*Y)


int main(int argc, char** argv){

    Stopwatch stopwatch;

    uint32_t* h_a = (uint32_t*) malloc(N*sizeof(uint32_t));
    uint32_t* h_b = (uint32_t*) malloc(N*sizeof(uint32_t));
    uint32_t* h_c = (uint32_t*) malloc(N*sizeof(uint32_t));

    for(int i = 0; i < N; i++){
        h_a[i] = i;
        h_b[i] = 4*i;
    }

    CudaClass cuda_object(h_a, h_b, h_c, N);

    uint32_t block_x = 32;
    uint32_t block_y = 32;
    uint32_t grid_x = 20;
    uint32_t grid_y = 15;


    stopwatch.Start();
    cuda_object.DefineGridLayout(block_x, block_y, grid_x, grid_y);
    stopwatch.Check_n_Reset("DefineGridLayout");
    cuda_object.AllocateDeviceMemory();
    stopwatch.Check_n_Reset("AllocateDeviceMemory");
    cuda_object.CopyInputToDevice();
    stopwatch.Check_n_Reset("CopyInputToDevice");
    cuda_object.RunKernelGPU();
    stopwatch.Check_n_Reset("RunKernelGPU\t");
    cuda_object.CopyOutputToHost();
    stopwatch.Check_n_Reset("CopyOutputToHost");
    cuda_object.FreeDeviceMemory();
    stopwatch.Check_n_Reset("FreeDeviceMemory");

    // for(int i = 0; i < N; i++){
    //     printf("%d\n", h_c[i]);
    // }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;

 }
