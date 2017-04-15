#include "shared_image_processing.cuh"

#include "opencv2/opencv.hpp"

int main(int argc, char const *argv[])
{

    cv::Mat hmap = cv::imread("hmap.png", cv::IMREAD_GRAYSCALE);
    cv::Mat cmap(hmap.rows, hmap.cols, hmap.type());
    cv::Mat tmap(hmap.rows, hmap.cols, hmap.type());
    for(int i = 0; i < hmap.rows * hmap.cols; i++){
        tmap.data[i] = 0;
    }

    SharedImageProcessing shared_image_processing(hmap.data, cmap.data, tmap.data);

    shared_image_processing.gpuAllocateMemory();
    shared_image_processing.gpuCopyInputToDevice(hmap.data, cmap.data, tmap.data);
    shared_image_processing.gpuExecuteKernel();
    shared_image_processing.gpuCopyOutputToHost(hmap.data, cmap.data, tmap.data);
    shared_image_processing.gpuFree();

    cv::namedWindow("cmap", 0);
    cv::namedWindow("tmap", 0);
    cv::imshow("cmap", cmap);
    cv::imshow("tmap", tmap);
    cv::waitKey(0);

    return 0;
}
