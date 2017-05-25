CC-FLAGS=-Wno-deprecated-gpu-targets
CC=nvcc

all:  rot-13.o encrypt_cuda.o encrypt_cuda

encrypt_cuda : rot-13.o encrypt_cuda.o
	$(CC) rot-13.o encrypt_cuda.o $(CC-FLAGS) -o encrypt_cuda

encrypt_cuda.o : encrypt_cuda.c rot-13.cuh
	$(CC) -c encrypt_cuda.c $(CC-FLAGS)

rot-13.o : rot-13.cu rot-13.cuh
	$(CC) -c rot-13.cu $(CC-FLAGS)
clean :
	rm encrypt_cuda *.o