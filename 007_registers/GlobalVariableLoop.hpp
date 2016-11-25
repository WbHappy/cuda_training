#ifndef GLOBAL_VARIABLE_LOOP_HPP_
#define GLOBAL_VARIABLE_LOOP_HPP_

#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

class GlobalVariableLoop{

    bool* d_inputs;
    bool* h_inputs;

    uint32_t* h_outputs;
    uint32_t* d_outputs;

    uint32_t block_x;
    uint32_t block_y;
    uint32_t grid_x;
    uint32_t grid_y;

    int N;

public:
    GlobalVariableLoop(bool* h_inputs, uint32_t* h_outputs, int N);

    void DefineGridLayout(uint32_t block_x, uint32_t block_y, uint32_t grid_x, uint32_t grid_y);
    void AllocateDeviceMemory();
    void CopyInputToDevice();
    void RunKernelGPUSimple();
    void RunKernelGPUDevice();
    void RunKernelGPURegister();
    void CopyOutputToHost();
    void FreeDeviceMemory();

};

#endif
