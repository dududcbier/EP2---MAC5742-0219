#include "util.h"
#include <stdio.h>

// Returns OK if no errors occur and ERROR otherwise
int errorCheck(cudaError_t code, char origin[]) {
    if (code != cudaSuccess) {
        printf("%s: %s\n", origin, cudaGetErrorString(code));
        return ERROR;
    }
    return OK;
}