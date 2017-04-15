CC=nvcc
SRC=main.cu gpu_errchk.cu
INCL=gpu_errchk.cuh

arch=sm_30

Main : $(SRC) $(INCL)
	$(CC) -arch $(arch) -o $@ $(SRC) -v
