All : Main1 Main2

Main1 : Main1.cu
	nvcc -arch $(arch) -o $@ $^ -std=c++11

Main2 : Main2.cu
	nvcc -arch $(arch) -o $@ $^ -std=c++11
