NVCC=nvcc
ALLFILES=edge.cu lodepng.cpp benchfun.cpp cudafun.cu
CCFLAGS=-std=c++11 -Xcompiler -fopenmp -Wno-deprecated-gpu-targets -O2 # Compiles with optimization

default: edge

edge:	
		$(NVCC) -o $@ $(ALLFILES) $< $(CCFLAGS) 
		
clean:
	rm -f edge
	rm -f *_cpu.png
	rm -f *_omp.png
	rm -f *_gpu.png
