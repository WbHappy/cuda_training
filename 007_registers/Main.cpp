
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "GlobalVariableLoop.hpp"
#include "../Stopwatch.hpp"

#define N (384*1024)

int main(int argc, char** argv){

    bool* h_inputs = (bool*)malloc(N * 32 * sizeof(bool));

    uint32_t* h_outputs = (uint32_t*)malloc(N*sizeof(uint32_t));

    srand(time(NULL));

    for(int i = 0; i < N; i++){
        h_outputs[i] = 0;
        for(int j = 0; j < 32; j++){
            h_inputs[i * 32 + j] = rand()%2;
        }
    }

    GlobalVariableLoop gvl(h_inputs, h_outputs, N);

    Stopwatch stopwatch;

    stopwatch.Start();
    gvl.DefineGridLayout(384,1,32,32);
    gvl.AllocateDeviceMemory();
    stopwatch.Check_n_Reset("AllocateDeviceMemory");

    // gvl.CopyInputToDevice();
    // stopwatch.Check_n_Reset("CopyInputToDevice");
    // gvl.RunKernelGPUSimple();
    // stopwatch.Check_n_Reset("RunKernelGPUSimple");
    // gvl.CopyOutputToHost();
    // stopwatch.Check_n_Reset("CopyOutputToHost");
    // printf("\nOUTPUT: %d\n\n", h_outputs[259]);

    gvl.CopyInputToDevice();
    stopwatch.Check_n_Reset("CopyInputToDevice");
    gvl.RunKernelGPUDevice();
    stopwatch.Check_n_Reset("RunKernelGPUDevice");
    gvl.CopyOutputToHost();
    stopwatch.Check_n_Reset("CopyOutputToHost");
    printf("\nOUTPUT: %d\n\n", h_outputs[259]);

    gvl.CopyInputToDevice();
    stopwatch.Check_n_Reset("CopyInputToDevice");
    gvl.RunKernelGPURegister();
    stopwatch.Check_n_Reset("RunKernelGPURegister");
    gvl.CopyOutputToHost();
    stopwatch.Check_n_Reset("CopyOutputToHost");
    printf("\nOUTPUT: %d\n\n", h_outputs[259]);

    gvl.FreeDeviceMemory();
    stopwatch.Check_n_Reset("FreeDeviceMemory");
    //

    for(int tid = 0; tid < N; tid++){
        for(int i = 0; i < 10; i++){
            h_outputs[tid] |= (h_inputs[tid * 32 + i] << i);
        }
    }
    stopwatch.Check_n_Reset("CPU_version\t");

    // for(int i = 0; i < N; i++){
    // printf("%d\n", h_outputs[i]);
    // }
    return 0;

}
