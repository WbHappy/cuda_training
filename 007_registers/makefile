CC=nvcc
SRC=Main.cpp GlobalVariableLoop.cu
INCL=GlobalVariableLoop.hpp

Main : $(SRC) $(INCL)
	$(CC)-arch $(arch) -o $@ $(SRC) --resource-usage
