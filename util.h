#ifndef _UTIL
#define _UTIL

#include <cuda_runtime.h>

#define OK 0
#define ERROR 1
#define BYTE unsigned char

int errorCheck(cudaError_t code, char origin[]);

#endif