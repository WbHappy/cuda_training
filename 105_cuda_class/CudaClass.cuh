#ifndef MY__CUDA__CLASS__HEADER__
#define MY__CUDA__CLASS__HEADER__


#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

class CudaClass{
    uint32_t *d_a, *d_b, *d_c;
    uint32_t *h_a, *h_b, *h_c;

    uint32_t mem_size;
    uint32_t length;

public:
    CudaClass(uint32_t *h_a, uint32_t *h_b, uint32_t *h_c, uint32_t length);

    void AllocateDeviceMemory();
    void CopyInputToDevice();
    void RunKernelGPU();
    void CopyOutputToHost();
    void FreeDeviceMemory();

};

#endif
