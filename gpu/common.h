#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#include <Windows.h>

#define CheckCUDAError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    static char buffer[512];
    if (code != cudaSuccess)
    {
        sprintf_s(buffer, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        MessageBox(NULL, buffer, "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
        if (abort) exit(code);
    }
}
