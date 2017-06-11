CC-FLAGS=-Wno-deprecated-gpu-targets
CC=nvcc
GCC=gcc
DEP=util.o base64.o rot-13.o arcfour.o encode_cuda.o seq-rot-13.o seq-base64.o seq-arcfour.o
 
all: $(DEP) encode_cuda

encode_cuda : $(DEP)
	$(CC) $(DEP) $(CC-FLAGS) -o encode_cuda

encode_cuda.o : encode_cuda.c rot-13.cuh
	$(CC) -c encode_cuda.c $(CC-FLAGS)

rot-13.o : rot-13.cu rot-13.cuh
	$(CC) -c rot-13.cu $(CC-FLAGS)

base64.o : base64.cu base64.cuh
	$(CC) -c base64.cu $(CC-FLAGS)

arcfour.o : arcfour.cu arcfour.cuh seq-arcfour.o
	$(CC) -c seq-arcfour.o arcfour.cu $(CC-FLAGS)

util.o : util.c util.h
	$(CC) -c util.c $(CC-FLAGS) 

seq-rot-13.o : seq-rot-13.c seq-rot-13.h
	$(GCC) -c seq-rot-13.c

seq-base64.o : seq-base64.c seq-base64.h
	$(GCC) -c seq-base64.c

seq-arcfour.o : seq-arcfour.c arcfour.h
	$(GCC) -c seq-arcfour.c

clean :
	rm encode_cuda *.o