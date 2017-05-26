#include <stdio.h>
#include <string.h>
#include <iostream>
extern "C" {
    #include "rot-13.cuh"
    #include "util.h"
}

__global__
void rot13_cuda(int n, char *str) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int case_type;
    if (i < n && (str[i] >= 'A' && str[i] <= 'Z') || (str[i] >= 'a' && str[i] <= 'z')) {
        if (str[i] >= 'a') 
            case_type = 'a';
        else 
            case_type = 'A';
        str[i] = (str[i] + 13) % (case_type + 26);
        if (str[i] < 26)
            str[i] += case_type;
    }
}
extern "C"
void rot13_encode(unsigned char *data, unsigned long length) {
    char *enc;

    cudaMalloc(&enc, length * sizeof(char));
    errorCheck(cudaMemcpy(enc, data, length * sizeof(char), cudaMemcpyHostToDevice), (char *) "memcpy host to device");

    rot13_cuda<<<(length+255)/256, 256>>>(length, enc);

    errorCheck(cudaMemcpy(data, enc, length * sizeof(char), cudaMemcpyDeviceToHost), (char *) "memcpy device to host");
    errorCheck(cudaFree(enc), (char *) "cudaFree");
}
extern "C"
void rot13_decode(unsigned char *data, unsigned long length) {
    rot13_encode(data, length);
}



