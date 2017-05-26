CC-FLAGS=-Wno-deprecated-gpu-targets
CC=nvcc

all:  rot-13.o encode_cuda.o encode_cuda

encode_cuda : rot-13.o encode_cuda.o
	$(CC) rot-13.o encode_cuda.o $(CC-FLAGS) -o encode_cuda

encode_cuda.o : encode_cuda.c rot-13.cuh
	$(CC) -c encode_cuda.c $(CC-FLAGS)

rot-13.o : rot-13.cu rot-13.cuh
	$(CC) -c rot-13.cu $(CC-FLAGS)
clean :
	rm encode_cuda *.o