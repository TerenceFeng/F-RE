#pragma once

#include "common.h"

#include <cassert>

enum PoolType
{
    IN_HOST = 1, IN_DEVICE = 2
};

template <typename T>
class Pool
{
    size_t size, flag;
    T *host_p, *device_p;
public:
    Pool(size_t _size, size_t _flag = 0u)
        : size(_size), flag(_flag), host_p(nullptr), device_p(nullptr)
    {}
    ~Pool()
    {
        //if (device_p != nullptr) CheckCUDAError(cudaFree(device_p));
        if (host_p != nullptr) delete[] host_p;
        device_p = host_p = nullptr;
        size = flag = 0;
    }
    void copyToDevice()
    {
        assert(size != 0);
        if ((flag & IN_HOST) && (flag & IN_DEVICE))
            CheckCUDAError(cudaMemcpy(getDevice(), getHost(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copyFromDevice()
    {
        assert(size != 0);
        if ((flag & IN_HOST) && (flag & IN_DEVICE))
            CheckCUDAError(cudaMemcpy(getHost(), getDevice(), size * sizeof(T), cudaMemcpyDeviceToHost));
    }
    T * getHost()
    {
        assert(size != 0);
        if ((flag & IN_HOST) && host_p == nullptr)
        {
            host_p = new T[size];
        }
        while (host_p == nullptr);
        return host_p;
    }
    T * getDevice()
    {
        assert(size != 0);
        if ((flag & IN_DEVICE) && device_p == nullptr)
        {
            CheckCUDAError(cudaMalloc((void**)&device_p, size * sizeof(T)));
        }
        return device_p;
    }
    size_t getSize() const
    {
        return size;
    }
    void swap(Pool<T> &o)
    {
        Pool<T> tmp(*this);
        *this = o;
        o = tmp;
    }
};

// store data list-like on CPU, array-like on GPU
