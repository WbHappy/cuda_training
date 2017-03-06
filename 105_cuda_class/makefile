CC=nvcc
SRC=Main.cpp CudaClass.cu
INCL=CudaClass.cuh

Main : $(SRC) $(INCL)
	$(CC)-arch $(arch) -o $@ $(SRC) -v
