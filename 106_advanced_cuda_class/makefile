CC=nvcc
SRC=Main.cpp CudaClass.cu
INCL=CudaClass.cuh Stopwatch.hpp

Main : $(SRC) $(INCL)
	$(CC) -o $@ $(SRC) -v
