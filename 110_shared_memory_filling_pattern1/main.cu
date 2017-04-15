#include "gpu_errchk.cuh"

#include <cuda.h>
#include <opencv2/opencv.hpp>

#define X 32
#define Y 32
#define SIZE (X*Y)

#define OFFSET_X 4
#define OFFSET_Y 4

#define IMG_X 256
#define IMG_Y 256
#define IMG_SIZE (IMG_X*IMG_Y)

#define THREADS SIZE
#define BLOCKS (IMG_SIZE/SIZE)

__global__ void shared_filling_pattern(uint8_t *hmap, uint8_t *cmap, uint8_t* tmap)
{
    __shared__ uint8_t hmap_shared[Y][X];

    int sidx = threadIdx.x * OFFSET_X;
    int sidy = threadIdx.y * OFFSET_Y;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * OFFSET_X;
    int idy = (blockIdx.y * blockDim.y + threadIdx.y) * OFFSET_Y;
    int tid = idy * gridDim.x * blockDim.x * OFFSET_X + idx;
    tmap[tid] = 255;

    int pid;

    for(int oy = 0; oy < OFFSET_Y; oy++)
    {
        for(int ox = 0; ox < OFFSET_X; ox++)
        {
            pid = (idy+oy) * gridDim.x * blockDim.x * OFFSET_X + (idx+ox);
            hmap_shared[sidy+oy][sidx+ox] = hmap[pid];
        }
    }


    if(blockIdx.x%2 == blockIdx.y%2)
    {
        for(int oy = 0; oy < OFFSET_Y; oy++)
        {
            for(int ox = 0; ox < OFFSET_X; ox++)
            {
                hmap_shared[sidy+oy][sidx+ox] = 255 - hmap_shared[sidy+oy][sidx+ox];
            }
        }
    }

    for(int oy = 0; oy < OFFSET_Y; oy++)
    {
        for(int ox = 0; ox < OFFSET_X; ox++)
        {
            pid = (idy+oy) * gridDim.x * blockDim.x * OFFSET_X + (idx+ox);
            cmap[pid] = hmap_shared[sidy+oy][sidx+ox];
        }
    }



}

int main(int argc, char const *argv[])
{

    cv::Mat hmap = cv::imread("hmap.png", cv::IMREAD_GRAYSCALE);
    cv::Mat cmap(hmap.rows, hmap.cols, hmap.type());
    cv::Mat tmap(hmap.rows, hmap.cols, hmap.type());
    for(int i = 0; i < hmap.rows * hmap.cols; i++){
        tmap.data[i] = 0;
    }
    uint8_t *dev_hmap, *dev_cmap, *dev_tmap;



    dim3 grid(IMG_X/X, IMG_Y/Y,1);
    dim3 block(X/OFFSET_X,Y/OFFSET_Y,1);

    gpuErrchk( cudaMalloc((void**)&dev_hmap, hmap.rows * hmap.cols * sizeof(uint8_t)) );
    gpuErrchk( cudaMalloc((void**)&dev_cmap, hmap.rows * hmap.cols * sizeof(uint8_t)) );
    gpuErrchk( cudaMalloc((void**)&dev_tmap, hmap.rows * hmap.cols * sizeof(uint8_t)) );
    gpuErrchk( cudaMemcpy(dev_hmap, hmap.data, hmap.rows * hmap.cols * sizeof(uint8_t), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dev_tmap, tmap.data, hmap.rows * hmap.cols * sizeof(uint8_t), cudaMemcpyHostToDevice) );

    shared_filling_pattern<<< grid, block >>>(dev_hmap, dev_cmap, dev_tmap);

    gpuErrchk( cudaMemcpy(cmap.data, dev_cmap, hmap.rows * hmap.cols * sizeof(uint8_t), cudaMemcpyDeviceToHost) ) ;
    gpuErrchk( cudaMemcpy(tmap.data, dev_tmap, hmap.rows * hmap.cols * sizeof(uint8_t), cudaMemcpyDeviceToHost) ) ;

    gpuErrchk( cudaFree(dev_hmap) );
    gpuErrchk( cudaFree(dev_cmap) );



    cv::namedWindow("cmap", 0);
    cv::namedWindow("tmap", 0);
    cv::imshow("cmap", cmap);
    cv::imshow("tmap", tmap);
    cv::waitKey(0);

    return 0;
}
