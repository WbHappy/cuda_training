#include <cuda.h>
#include <stdio.h>

#define N 100000

__global__ void kernel_add(int a, int b, int *c){
    *c = a + b;
}

int main(int argc, char **argv){
    int* host_a = (int*) malloc(sizeof(int));
    int* host_b = (int*) malloc(sizeof(int));
    int* host_c = (int*) malloc(sizeof(int));
    int* device_c;

    cudaMalloc((void**) &device_c, sizeof(int));
    for(int i = 0; i < N; i++){
        *host_a = 2;
        *host_b = 7;

        kernel_add<<<1,1>>>(*host_a, *host_b, device_c);

        cudaMemcpy(host_c, device_c, sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaFree(&device_c);


    printf("%d\n", *host_c);

    return 0;

}
