#include "RadixSort.cuh"
#include "../Stopwatch.hpp"

int main(int argc, char** argv){

    Stopwatch stopwatch;

    stopwatch.Start();
    RadixSort gpu_sort;         stopwatch.Check_n_Reset("GPU ready to start");

    gpu_sort.RunKernelGPU();    stopwatch.Check_n_Reset("Kernel executed   ");
    gpu_sort.CopyResults();     stopwatch.Check_n_Reset("Results copied    ");
    // gpu_sort.PrintResults();

    return 0;
}
