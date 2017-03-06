CC=nvcc
SRC=Main.cu
INCL=

Main : $(SRC) $(INCL)
	$(CC)-arch $(arch) -o $@ $(SRC) --resource-usage
