#include <stdio.h>
#include <string.h>
#include <iostream>
extern "C" {
    #include "base64.cuh"
    #include "util.h"
}

#define NEWLINE_INVL 76

static const BYTE charset[]={"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"};

int errorCheck(cudaError_t code, char origin[]);

__device__
BYTE reverse_char(char ch) {
    if (ch >= 'A' && ch <= 'Z')
        ch -= 'A';
    else if (ch >= 'a' && ch <='z')
        ch = ch - 'a' + 26;
    else if (ch >= '0' && ch <='9')
        ch = ch - '0' + 52;
    else if (ch == '+')
        ch = 62;
    else if (ch == '/')
        ch = 63;

    return(ch);
}

__global__
void base64_cuda_encode(size_t n, BYTE *in, BYTE *out, BYTE *d_charset) {
    size_t i = (blockDim.x * blockIdx.x + threadIdx.x) * 3;
    size_t j, left_over = n % 3;

    if (i < n) {
        if (i >= n - left_over)
            left_over = n - i;
        else 
            left_over = 0;
        
        j = i / 3 * 4;
        j += j / 76; // Account for the number of newlines already put

        out[j] = d_charset[in[i] >> 2];
        if (left_over == 1) {
            out[j + 1] = d_charset[((in[i] & 0x03) << 4)];
            out[j + 2] = '=';
            out[j + 3] = '=';
        }
        else {
            out[j + 1] = d_charset[((in[i] & 0x03) << 4) | (in[i + 1] >> 4)];
            if (left_over == 2) {
                out[j + 2] = d_charset[((in[i + 1] & 0x0f) << 2)];
                out[j + 3] = '=';
            }
            else {
                out[j + 2] = d_charset[((in[i + 1] & 0x0f) << 2) | (in[i + 2] >> 6)];
                out[j + 3] = d_charset[in[i + 2] & 0x3F];
            }
        }
    
        if ((j - j / 76 + 4) % NEWLINE_INVL == 0)
            out[j + 4] = '\n';
    }

} 

__global__
void base64_cuda_decode(size_t n, BYTE *in, BYTE *out, BYTE *d_charset) {
    size_t i, j, left_over;

    if (in[n - 1] == '=')
        n--;
    if (in[n - 1] == '=')
        n--;

    i = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    j = i / 4 * 3;
    i += i / 76; // Account for the number of newlines already put
    
    left_over = n % 4;
    if (i < n - left_over)
        left_over = 0;
    if (i < n) {
        if (in[i] == '\n') i++;
        out[j]     = (reverse_char(in[i]) << 2) | ((reverse_char(in[i + 1]) & 0x30) >> 4);
        if (left_over != 2){
            out[j + 1] = (reverse_char(in[i + 1]) << 4) | (reverse_char(in[i + 2]) >> 2);
            if (left_over != 3)
                out[j + 2] = (reverse_char(in[i + 2]) << 6) | reverse_char(in[i + 3]);
        }
    }
} 

extern "C"
void base64_encode(BYTE *data, size_t length, BYTE *output, size_t out_length) {
    BYTE *in, *out, *d_charset;

    cudaMalloc(&in, length * sizeof(BYTE));
    cudaMalloc(&out, out_length * sizeof(BYTE));
    cudaMalloc(&d_charset, 64 * sizeof(BYTE));
    errorCheck(cudaMemcpy(in, data, length * sizeof(BYTE), cudaMemcpyHostToDevice), (char *) "memcpy host to device");
    errorCheck(cudaMemcpy(d_charset, charset, 64 * sizeof(BYTE), cudaMemcpyHostToDevice), (char *) "memcpy host to device");

    base64_cuda_encode<<<(length+255)/256, 256>>>(length, in, out, d_charset);

    errorCheck(cudaMemcpy(output, out, out_length * sizeof(BYTE), cudaMemcpyDeviceToHost), (char *) "memcpy device to host");
    errorCheck(cudaFree(out), (char *) "cudaFree");
    errorCheck(cudaFree(in), (char *) "cudaFree");
}

extern "C"
void base64_decode(BYTE *data, size_t length, BYTE *output, size_t out_length) {
    BYTE *in, *out, *d_charset;

    cudaMalloc(&in, length * sizeof(BYTE));
    cudaMalloc(&out, out_length * sizeof(BYTE));
    cudaMalloc(&d_charset, 64 * sizeof(BYTE));
    errorCheck(cudaMemcpy(in, data, length * sizeof(BYTE), cudaMemcpyHostToDevice), (char *) "memcpy host to device");
    errorCheck(cudaMemcpy(d_charset, charset, 64 * sizeof(BYTE), cudaMemcpyHostToDevice), (char *) "memcpy host to device");

    base64_cuda_decode<<<(length+255)/256, 256>>>(length, in, out, d_charset);

    errorCheck(cudaMemcpy(output, out, out_length * sizeof(BYTE), cudaMemcpyDeviceToHost), (char *) "memcpy device to host");
    errorCheck(cudaFree(out), (char *) "cudaFree");
    errorCheck(cudaFree(in), (char *) "cudaFree");
}
