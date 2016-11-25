#include <cuda.h>
#include <stdio.h>

#define N 100000

__global__ void kernel_add(int* a, int* b, int* c){
    *c = *a + *b;
}

int main(int argc, char** argv){


    int* host_a = (int*) malloc(sizeof(int));
    int* host_b = (int*) malloc(sizeof(int));
    int* host_c = (int*) malloc(sizeof(int));

    int* device_a; cudaMalloc((void**) &device_a, sizeof(int));
    int* device_b; cudaMalloc((void**) &device_b, sizeof(int));
    int* device_c; cudaMalloc((void**) &device_c, sizeof(int));
    for(int i = 0; i < N; i++){

        *host_a = 7;
        *host_b = 2;

        cudaMemcpy(device_a, host_a, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, host_b, sizeof(int), cudaMemcpyHostToDevice);

        kernel_add<<<1,1>>>(device_a, device_b, device_c);

        cudaMemcpy(host_c, device_c, sizeof(int), cudaMemcpyDeviceToHost);

    }
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    printf("%d\n", *host_c);
    return 0;
}
