CC-FLAGS=-Wno-deprecated-gpu-targets
CC=nvcc
DEP=util.o base64.o rot-13.o encode_cuda.o
 
all: $(DEP) encode_cuda

encode_cuda : $(DEP)
	$(CC) $(DEP) $(CC-FLAGS) -o encode_cuda

encode_cuda.o : encode_cuda.c rot-13.cuh
	$(CC) -c encode_cuda.c $(CC-FLAGS)

rot-13.o : rot-13.cu rot-13.cuh
	$(CC) -c rot-13.cu $(CC-FLAGS)

base64.o : base64.cu base64.cuh
	$(CC) -c base64.cu $(CC-FLAGS)

util.o : util.c util.h
	$(CC) -c util.c $(CC-FLAGS) 

clean :
	rm encode_cuda *.o