#include "GlobalVariableLoop.hpp"

__global__ void KernelSimple(bool* d_inputs, uint32_t* d_outputs){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t tid = idx + idy * gridDim.x * blockDim.x;

    for(int i = 0; i < 10; i++){
        d_outputs[tid] |= (d_inputs[tid * 32 + i] << i);
    }
}

__device__ static uint32_t temp = 0;
__global__ void KernelDevice(bool* d_inputs, uint32_t* d_outputs){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t tid = idx + idy * gridDim.x * blockDim.x;

    for(int i = 0; i < 10; i++){
        temp |= (d_inputs[tid * 32 + i] << i);
    }
    d_outputs[tid] = temp;
}

__global__ void KernelRegister(bool* d_inputs, uint32_t* d_outputs){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t tid = idx + idy * gridDim.x * blockDim.x;

    uint32_t reg = 0;  // Value store in REGISTER -> The fastest memory available!!!!

    for(int i = 0; i < 10; i++){
        reg |= (d_inputs[tid * 32 + i] << i);
    }
    d_outputs[tid] = reg;
}

GlobalVariableLoop::GlobalVariableLoop(bool* h_inputs, uint32_t* h_outputs, int N){
    this->h_inputs = h_inputs;
    this->h_outputs = h_outputs;
    this->N = N;
}

void GlobalVariableLoop::DefineGridLayout(uint32_t block_x, uint32_t block_y, uint32_t grid_x, uint32_t grid_y){
    this->block_x = block_x;
    this->block_y = block_y;
    this->grid_x = grid_x;
    this->grid_y = grid_y;
}

void GlobalVariableLoop::AllocateDeviceMemory(){
    cudaMalloc((void**) &d_inputs, N*32*sizeof(bool));
    cudaMalloc((void**) &d_outputs, N*sizeof(uint32_t));
    cudaDeviceSynchronize();
}

void GlobalVariableLoop::CopyInputToDevice(){
    cudaMemcpy(d_inputs, h_inputs, N*32*sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputs, h_outputs, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void GlobalVariableLoop::RunKernelGPUSimple(){
    KernelSimple<<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>>(d_inputs, d_outputs);
    cudaDeviceSynchronize();
}

void GlobalVariableLoop::RunKernelGPUDevice(){
    KernelDevice<<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>>(d_inputs, d_outputs);
    cudaDeviceSynchronize();
}

void GlobalVariableLoop::RunKernelGPURegister(){
    KernelRegister<<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>>(d_inputs, d_outputs);
    cudaDeviceSynchronize();
}

void GlobalVariableLoop::CopyOutputToHost(){
    cudaMemcpy(h_inputs, d_inputs, N*32*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputs, d_outputs, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void GlobalVariableLoop::FreeDeviceMemory(){
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaDeviceSynchronize();

}
