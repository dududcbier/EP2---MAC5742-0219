#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cuda_runtime.h>
extern "C" {
    #include "rot-13.cuh"
}
#define OK 0
#define ERROR 1

int errorCheck(cudaError_t code, char origin[]);

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
void rot13_encrypt(char *data) {
    char *enc;
    int n = strlen(data);

    cudaMalloc(&enc, n * sizeof(char));
    errorCheck(cudaMemcpy(enc, data, n * sizeof(char), cudaMemcpyHostToDevice), (char *) "memcpy host to device");

    rot13_cuda<<<(n+255)/256, 256>>>(n, enc);

    errorCheck(cudaMemcpy(data, enc, n * sizeof(char), cudaMemcpyDeviceToHost), (char *) "memcpy device to host");
    errorCheck(cudaFree(enc), (char *) "cudaFree");
}
extern "C"
void rot13_decrypt(char *data) {
    rot13_encrypt(data);
}

// Returns OK if no errors occur and ERROR otherwise
int errorCheck(cudaError_t code, char origin[]) {
    if (code != cudaSuccess) {
        printf("%s: %s\n", origin, cudaGetErrorString(code));
        return ERROR;
    }
    return OK;
}


