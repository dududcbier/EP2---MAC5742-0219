#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
extern "C" {
    #include "arcfour.cuh"
    #include "util.h"
    #include "arcfour.h"
}

__global__
void arcfour_cuda(int n, BYTE *data, BYTE *key) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    data[i] = data[i] ^ key[i];
}

extern "C"
void arcfour_encode(BYTE *data, size_t length, BYTE *key, unsigned long key_length) {
    BYTE state[256], *d_data, *d_key;
    BYTE *key_stream = (BYTE *) malloc(length * sizeof(BYTE));

    arcfour_key_setup(state, key, key_length);
    arcfour_generate_stream(state, key_stream, length);

    errorCheck(cudaMalloc(&d_data, length * sizeof(BYTE)), (char *) "cudaMalloc d_data");
    errorCheck(cudaMemcpy(d_data, data, length * sizeof(BYTE), cudaMemcpyHostToDevice), (char *) "memcpy host to device");
    errorCheck(cudaMalloc(&d_key, length * sizeof(BYTE)), (char *) "cudaMalloc 3");
    errorCheck(cudaMemcpy(d_key, key_stream, length * sizeof(BYTE), cudaMemcpyHostToDevice), (char *) "memcpy host to device");

    arcfour_cuda<<<(length + 127)/128, 128>>>(length, d_data, d_key);

    errorCheck(cudaMemcpy(data, d_data, length * sizeof(BYTE), cudaMemcpyDeviceToHost), (char *) "memcpy device to host");
    errorCheck(cudaFree(d_data), (char *) "cudaFree");
    errorCheck(cudaFree(d_key), (char *) "cudaFree");
    free(key_stream);
  
}
extern "C"
void arcfour_decode(BYTE *data, size_t length, BYTE *key, unsigned long key_length) {
    arcfour_encode(data, length, key, key_length);
}


