#include "RadixSort.cuh"

int main(int argc, char** argv){

    RadixSort gpu_sort;

    gpu_sort.RunKernelGPU();
    gpu_sort.CopyResults();
    gpu_sort.PrintResults();

    return 0;
}
