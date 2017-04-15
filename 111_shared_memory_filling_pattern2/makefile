CC=nvcc
SRC=main.cpp shared_image_processing.cu gpu_errchk.cu
INCL=shared_image_processing.cuh gpu_errchk.cuh

arch=sm_30

Main : $(SRC) $(INCL)
	$(CC) -arch $(arch) -o $@ $(SRC) `pkg-config --libs opencv-3.2.0` -I/opt/ros/kinetic/include/opencv-3.2.0
