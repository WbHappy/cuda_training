#include <cuda.h>
#include <stdio.h>

__global__ void add_number(int* a, int* b, int* c){
    *c = *a + *b;
}

int main(int argc, char **argv){
    int a, b, c;    // Host copies of variables
    int *d_a, *d_b, *d_c;   // Device copies of variables
    int size = sizeof(int);

    // Allocation of device's memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Setting up variables on host
    a = 64;
    b = 641;

    // Copy inputs to device memory
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // Launching kernel on GPU
    add_number<<<1,1>>>(d_a, d_b, d_c);

    // Copy results back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("result: %d\n", c);

    return 0;
}
