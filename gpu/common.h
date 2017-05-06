#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#ifdef USE_OPENGL

#include <Windows.h>

void ShowErrorAndExit(const char *s)
{
    MessageBox(NULL, s, "SHUTDOWN ERROR", MB_OK | MB_ICONINFORMATION);
}

#include <fstream>
static std::ofstream Logger("log.txt");

#else

void ShowErrorAndExit(const char *s)
{
    fprintf(stderr, "%s\n", s);
}

#include <iostream>
static std::ostream &Logger = std::cout;

#endif

void LogError(const char *s)
{
    Logger << "Error: " << s << std::endl;
    Logger.flush();
}
void LogInfo(const char *s)
{
    Logger << "Info: " << s << std::endl;
}

#define CheckCUDAError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    static char buffer[512];
    if (code != cudaSuccess)
    {
#ifdef USE_OPENGL
        sprintf_s(buffer, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        ShowErrorAndExit(buffer);
        // PrintError(buffer);
#else
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
#endif
        if (abort) exit(code);
    }
}
