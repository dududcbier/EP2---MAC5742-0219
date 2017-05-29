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
    size_t j, left_over = n % 3, newlines;

    if (i < n) {
        if (i >= n - left_over)
            left_over = n - i;
        else 
            left_over = 0;
        
        j = i / 3 * 4;
        newlines = j / 76;
        j += newlines; // Account for the number of newlines already put

		out[j]     = d_charset[in[i] >> 2];
        switch(left_over) {
            case 0:
                out[j + 1] = d_charset[((in[i] & 0x03) << 4) | (in[i + 1] >> 4)];
                out[j + 2] = d_charset[((in[i + 1] & 0x0f) << 2) | (in[i + 2] >> 6)];
                out[j + 3] = d_charset[in[i + 2] & 0x3F];
                break;
            case 1:
                out[j + 1] = d_charset[(in[i] & 0x03) << 4];
                out[j + 2] = '=';
                out[j + 3] = '=';
                break;
            case 2:
                out[j + 1] = d_charset[((in[i] & 0x03) << 4) | (in[i + 1] >> 4)];
                out[j + 2] = d_charset[(in[i + 1] & 0x0F) << 2];
                out[j + 3] = '=';
                break;
        }

        if ((j - newlines + 4) % NEWLINE_INVL == 0)
            out[j + 4] = '\n';
    
    }

} 

__global__
void base64_cuda_decode(size_t n, BYTE *in, BYTE *out) {
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
        switch(left_over) {
            case 0:
                if (in[i] == '\n')
                    i++;
                out[j + 2] = (reverse_char(in[i + 2]) << 6) | reverse_char(in[i + 3]);

            case 3:
                out[j + 1] = (reverse_char(in[i + 1]) << 4) | (reverse_char(in[i + 2]) >> 2);

            case 2:
                out[j]     = (reverse_char(in[i]) << 2) | ((reverse_char(in[i + 1]) & 0x30) >> 4);
                break;
        }
    }
} 

extern "C"
void base64_encode(BYTE *data, size_t length, BYTE *output, size_t out_length) {
    BYTE *in, *out, *d_charset;

    errorCheck(cudaMalloc(&in, length * sizeof(BYTE)), (char *) "cudaMalloc 1");
    errorCheck(cudaMalloc(&out, out_length * sizeof(BYTE)), (char *) "cudaMalloc 2");
    errorCheck(cudaMalloc(&d_charset, 64 * sizeof(BYTE)), (char *) "cudaMalloc 3");
    errorCheck(cudaMemcpy(in, data, length * sizeof(BYTE), cudaMemcpyHostToDevice), (char *) "memcpy host to device");
    errorCheck(cudaMemcpy(d_charset, charset, 64 * sizeof(BYTE), cudaMemcpyHostToDevice), (char *) "memcpy host to device");

    base64_cuda_encode<<<(length+1023)/1024, 1024>>>(length, in, out, d_charset);

    errorCheck(cudaMemcpy(output, out, out_length * sizeof(BYTE), cudaMemcpyDeviceToHost), (char *) "memcpy device to host");
    errorCheck(cudaFree(out), (char *) "cudaFree");
    errorCheck(cudaFree(in), (char *) "cudaFree");
}

extern "C"
void base64_decode(BYTE *data, size_t length, BYTE *output, size_t out_length) {
    BYTE *in, *out;

    errorCheck(cudaMalloc(&in, length * sizeof(BYTE)), (char *) "cudaMalloc 4");
    errorCheck(cudaMalloc(&out, out_length * sizeof(BYTE)), (char *) "cudaMalloc 5");
    errorCheck(cudaMemcpy(in, data, length * sizeof(BYTE), cudaMemcpyHostToDevice), (char *) "memcpy host to device");

    base64_cuda_decode<<<(length+1023)/256, 1024>>>(length, in, out);

    errorCheck(cudaMemcpy(output, out, out_length * sizeof(BYTE), cudaMemcpyDeviceToHost), (char *) "memcpy device to host");
    errorCheck(cudaFree(out), (char *) "cudaFree");
    errorCheck(cudaFree(in), (char *) "cudaFree");
}
