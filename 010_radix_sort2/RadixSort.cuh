#ifndef RADIX_SORT_CUH_
#define RADIX_SORT_CUH_

#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define THREADS 32
#define BLOCKS 1
#define LISTS 32
#define N 1024

typedef uint32_t u32;

class RadixSort{
    u32* d_array;

public:
    u32* h_array;


public:
    RadixSort();

    ~RadixSort();

    void CopyResults();

    void RunKernelGPU();

    void PrintResults(){
        for(int i=0; i<N; i++){
            printf("%d\n", h_array[i]);
        }
    }


};

#endif
