#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>

#define THRESH 127

__global__ void kernel_bgr2gray(uint8_t* input, uint8_t* output){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    output[idx] = (uint8_t)((input[3*idx] + input[3*idx+1] + input[3*idx+2])/3);
}

__global__ void kernel_binary(uint8_t* input, uint8_t* output){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(input[idx] > THRESH){
        output[idx] = 255;
    }else{
        output[idx] = 0;
    }
}

int main(int argc, char** argv){

    cv::VideoCapture cap;
    cv::Mat color(480,640, CV_8UC3);
    cv::Mat gray(480,640, CV_8UC1);
    cv::Mat binary(480,640, CV_8UC1);

    cv::namedWindow("window1", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("window2", CV_WINDOW_AUTOSIZE);

    uint8_t* device_color; cudaMalloc((void**)&device_color, 640*480*3*sizeof(uint8_t));
    uint8_t* device_gray; cudaMalloc((void**)&device_gray, 640*480*sizeof(uint8_t));
    uint8_t* device_binary; cudaMalloc((void**)&device_binary, 640*480*sizeof(uint8_t));

    cap.open(0);
    while(1){
        cap >> color;

        cudaMemcpy(device_color, color.data, 640*480*3*sizeof(uint8_t), cudaMemcpyHostToDevice);

        kernel_bgr2gray <<<640,480>>> (device_color, device_gray);
        kernel_binary <<<640,480>>> (device_gray, device_binary);

        cudaMemcpy(gray.data, device_gray, 640*480*sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(binary.data, device_binary, 640*480*sizeof(uint8_t), cudaMemcpyDeviceToHost);

        cv::imshow("window1", gray);
        cv::waitKey(1);

        cv::imshow("window2", binary);
        cv::waitKey(1);
        // printf("%d %d\n", frame.cols, frame.rows);
    }

    cudaFree(device_gray);
    cudaFree(device_binary);

    return 0;
}
