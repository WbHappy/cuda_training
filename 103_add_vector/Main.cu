#include <cuda.h>
#include <stdio.h>

#define N (1024*1024)

__global__ void kernel(int* a, int* b, int* c){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    *(c + index) = *(a + index) + *(b + index);
}

int main(int argc, char** argv){

    int size = N * sizeof(int);

    int* host_a = (int*) malloc(size);
    int* host_b = (int*) malloc(size);
    int* host_c = (int*) malloc(size);

    int* device_a; cudaMalloc((void**)&device_a, size);
    int* device_b; cudaMalloc((void**)&device_b, size);
    int* device_c; cudaMalloc((void**)&device_c, size);


    for(int i = 0; i < N; i++){
        *(host_a+i) = i;
        *(host_b+i) = 2*i;
        *(host_c+i) = *(host_a+i) + *(host_b+i);
    }
    printf("CPU\n");


    for(int j = 0; j< 20000; j++){

        cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
        kernel<<<1024,1024>>>(device_a, device_b, device_c);
    }

    // for(int j = 0; j< 20000; j++){
    //     for(int i = 0; i < N; i++){
    //         *(host_c+i) = *(host_a+i) + *(host_b+i);
    //     }
    // }


    printf("GPU\n");


    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    // for(int i = 0; i < N; i++){
    //     printf("%d, %d\n", i, *(host_c+i));
    // }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c);

    return 0;

}
