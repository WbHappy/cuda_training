#include <cuda.h>
#include <stdio.h>

#define N 512

__global__ void add_number(int* a, int* b, int* c){
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main(int argc, char **argv){
    int *a, *b, *c;    // Host copies of variables
    int *d_a, *d_b, *d_c;   // Device copies of variables
    int size = N * sizeof(int);

    // Allocation of device's memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocation of space for variables on host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Setting up input variables
    for(int i = 0; i < N; i++){
        *(a+i) = i;
        *(b+i) = i;
    }

    // Copy inputs to device memory
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launching kernel on GPU
    add_number<<<1,N>>>(d_a, d_b, d_c);

    // Copy results back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for(int i = 1; i < N; i++){
        printf("%d\n", *(c+i));
    }

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
