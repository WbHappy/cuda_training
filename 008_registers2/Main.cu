#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

#include "../Stopwatch.hpp"

#define THREADS 384
#define BLOCKS 1024

#define N (THREADS*BLOCKS)
#define M (16)

__global__ void kernelGlobal(uint32_t* input, uint32_t* output, uint32_t* adds){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t tid = idx + idy * gridDim.x * blockDim.x;

    output[tid] += input[tid];

    for(int i = 0; i < M; i++){
        output[tid] += adds[i];
    }
}

__global__ void kernelRegister(uint32_t* input, uint32_t* output, uint32_t* adds){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t tid = idx + idy * gridDim.x * blockDim.x;

    uint32_t d_temp = input[tid];

    for(int i = 0; i < M; i++){
        d_temp += adds[i];
    }
    output[tid] = d_temp;
}

__global__ void kernelRegisterFast(uint32_t* input, uint32_t* output, uint32_t* adds){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t tid = idx + idy * gridDim.x * blockDim.x;

    uint32_t d_temp = input[tid];

    d_temp += adds[0];
    d_temp += adds[1];
    d_temp += adds[2];
    d_temp += adds[3];
    d_temp += adds[4];
    d_temp += adds[5];
    d_temp += adds[6];
    d_temp += adds[7];
    d_temp += adds[8];
    d_temp += adds[9];
    d_temp += adds[10];
    d_temp += adds[11];
    d_temp += adds[12];
    d_temp += adds[13];
    d_temp += adds[14];
    d_temp += adds[15];

    output[tid] = d_temp;
}

__global__ void kernelRegisterFast2(uint32_t* input, uint32_t* output, uint32_t* adds){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t tid = idx + idy * gridDim.x * blockDim.x;

    uint32_t d_temp1 = input[tid];
    uint32_t d_temp2 = 0;
    uint32_t d_temp3 = 0;
    uint32_t d_temp4 = 0;

    d_temp1 += adds[0];
    d_temp2 += adds[1];
    d_temp3 += adds[2];
    d_temp4 += adds[3];

    d_temp1 += adds[4];
    d_temp2 += adds[5];
    d_temp3 += adds[6];
    d_temp4 += adds[7];

    d_temp1 += adds[8];
    d_temp2 += adds[9];
    d_temp3 += adds[10];
    d_temp4 += adds[11];

    d_temp1 += adds[12];
    d_temp2 += adds[13];
    d_temp3 += adds[14];
    d_temp4 += adds[15];

    output[tid] = d_temp1 + d_temp2 + d_temp3 + d_temp4;
}


int main(int argc, char** argv){

    Stopwatch stopwatch;

    uint32_t* h_inputs = (uint32_t*)malloc(N*sizeof(uint32_t));
    uint32_t* h_outputs = (uint32_t*)malloc(N*sizeof(uint32_t));
    uint32_t* h_adds = (uint32_t*)malloc(M*sizeof(uint32_t));

    for(int i = 0; i < N; i++){
        h_inputs[i] = 10000;
        h_outputs[i] = 1000;
    }

    for(int i = 0; i < M; i++){
        h_adds[i] = 1;
    }

    stopwatch.Start();

    uint32_t* d_inputs; cudaMalloc((void**)&d_inputs, N*sizeof(uint32_t));
    uint32_t* d_outputs; cudaMalloc((void**)&d_outputs, N*sizeof(uint32_t));
    uint32_t* d_adds; cudaMalloc((void**)&d_adds, M*sizeof(uint32_t));
    cudaDeviceSynchronize();

    stopwatch.Check_n_Reset("dev mem allocated");

// DEVICE MEMORY KERNEL

    cudaMemcpy(d_inputs, h_inputs, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputs, h_outputs, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adds, h_adds, M*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    stopwatch.Check_n_Reset("mem copied to dev");

    kernelGlobal<<< BLOCKS, THREADS >>> (d_inputs, d_outputs, d_adds);
    cudaDeviceSynchronize();

    stopwatch.Check_n_Reset("DEV-KERNEL EXECUTED");

// REGISTER MEMORY KERNEL

    cudaMemcpy(d_inputs, h_inputs, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputs, h_outputs, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adds, h_adds, M*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    stopwatch.Check_n_Reset("mem copied to dev");

    kernelRegister<<< BLOCKS, THREADS >>> (d_inputs, d_outputs, d_adds);
    cudaDeviceSynchronize();

    stopwatch.Check_n_Reset("REG-KERNEL EXECUTED");

// NO LOOP REGISTER KERNEL

    cudaMemcpy(d_inputs, h_inputs, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputs, h_outputs, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adds, h_adds, M*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    stopwatch.Check_n_Reset("mem copied to dev");

    kernelRegisterFast<<< BLOCKS, THREADS >>> (d_inputs, d_outputs, d_adds);
    cudaDeviceSynchronize();

    stopwatch.Check_n_Reset("NO-LOOP EXECUTED");

// ONE-LINE NO LOOP KERNEL

    cudaMemcpy(d_inputs, h_inputs, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputs, h_outputs, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adds, h_adds, M*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    stopwatch.Check_n_Reset("mem copied to dev");

    kernelRegisterFast2<<< BLOCKS, THREADS >>> (d_inputs, d_outputs, d_adds);
    cudaDeviceSynchronize();

    stopwatch.Check_n_Reset("ONE-LINE EXECUTED");

    cudaMemcpy(h_outputs, d_outputs, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    stopwatch.Check_n_Reset("mem copied to host");

    printf("%d\n", h_outputs[999]);

    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_adds);

    free(h_inputs);
    free(h_outputs);
    free(h_adds);

    return 0;
}
