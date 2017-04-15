#include "gpu_errchk.cuh"

#include <cuda.h>

#define imin(a,b) (a<b?a:b)

const int N = 1024 * 1024;
const int THREADS_PER_BLOCK = 256;
const int BLOCKS_PER_GRID = imin(32, (N+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK); // lim -> N / THREADS_PER_BLOCK


// ILOCZYN SKALARNY WEKTORÓW
// obliczamy najpierw iloczyny odpowiadających sobie elementów
// następnie je sumujemy

__global__ void dot_shared(float *a, float *b, float *c)
{

    // Iloczyny odpowiadających sobie elementów są zapisywane do tablicy w pamięci współdzielonej

    __shared__ float cache[THREADS_PER_BLOCK];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0;

    while(tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[threadIdx.x] = temp;

    // Konieczna synchronizacja wątków przed rozpoczęciem czytania z tablicy
    __syncthreads();

    // Sumowanie elemntów tablicy (redukcja)

    int i = blockDim.x / 2;

    while( i != 0)
    {
        if(threadIdx.x < i)
        {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Zapisywanie wyniku z bloku do wektora

    if(threadIdx.x == 0)
    {
        c[blockIdx.x] = cache[0];
    }

}


int main(int argc, char** argv)
{
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    c = (float*)malloc(BLOCKS_PER_GRID * sizeof(float));

    gpuErrchk( cudaMalloc((void**) &dev_a, N * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**) &dev_b, N * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**) &dev_c, BLOCKS_PER_GRID * sizeof(float)) );

    for(int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i*2;
    }

    gpuErrchk( cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice) );

    dot_shared<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_c);

    gpuErrchk( cudaMemcpy(c, dev_c, BLOCKS_PER_GRID * sizeof(float), cudaMemcpyDeviceToHost) );

    float result = 0;

    for(int i = 0; i < BLOCKS_PER_GRID; i++)
    {
        result += c[i];
    }

    printf("%f\n", result);

    gpuErrchk( cudaFree(dev_a) );
    gpuErrchk( cudaFree(dev_b) );
    gpuErrchk( cudaFree(dev_c) );

    free(a);
    free(b);
    free(c);

    return 0;

}
