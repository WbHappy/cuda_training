#include "CudaClass.cuh"

__global__ void Kernel (uint32_t *d_a, uint32_t *d_b, uint32_t *d_c){
        uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
        uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
        uint32_t tid = idx + idy * blockDim.x * gridDim.x;

        d_c[tid] = d_a[tid] + d_b[tid];
};

CudaClass::CudaClass(uint32_t *h_a, uint32_t *h_b, uint32_t *h_c, uint32_t length){

    this->length = length;
    this->mem_size = length * sizeof(uint32_t);

    this->h_a = h_a;
    this->h_b = h_b;
    this->h_c = h_c;

}

void CudaClass::DefineGridLayout(uint32_t block_x, uint32_t block_y, uint32_t grid_x, uint32_t grid_y){
    this->block_x = block_x;
    this->block_y = block_y;
    this->grid_x = grid_x;
    this->grid_y = grid_y;

}

void CudaClass::AllocateDeviceMemory(){
    cudaMalloc((void**) &d_a, mem_size);
    cudaMalloc((void**) &d_b, mem_size);
    cudaMalloc((void**) &d_c, mem_size);
    cudaDeviceSynchronize();
}

void CudaClass::CopyInputToDevice(){
    cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void CudaClass::RunKernelGPU(){
    Kernel<<< dim3(grid_x, grid_y), dim3(block_x, block_y) >>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
}

void CudaClass::CopyOutputToHost(){
    cudaMemcpy(h_c, d_c, mem_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void CudaClass::FreeDeviceMemory(){
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceSynchronize();
}
