All : Main_cc2.0 Main_cc3.0

Main_cc2.0 : Main.cpp RadixSort.cu RadixSort.cuh
	nvcc -o $@ Main.cpp RadixSort.cu --resource-usage -arch sm_20 -Wno-deprecated-gpu-targets

Main_cc3.0 : Main.cpp RadixSort.cu RadixSort.cuh
	nvcc -o $@ Main.cpp RadixSort.cu --resource-usage -arch sm_30

.PHONY : clean

clean :
	rm Main_cc*
