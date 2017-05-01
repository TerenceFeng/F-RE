#pragma once

#include "common.h"

#include <cassert>
#include <vector>

enum PoolType
{
    IN_HOST = 1, IN_DEVICE = 2
};

// TODO: apply this to all pointers to device memory
template <typename T>
struct device_addr_t
{
    T *value;
};

// TODO: maybe reference counted ? (instead of unique ownership)
template <typename T>
class Pool
{
    size_t size, flag;
    T *host_p, *device_p;

    inline void release()
    {
        device_p = host_p = nullptr;
        size = flag = 0;
    }
public:
    Pool(size_t _size, size_t _flag = 0u)
        : size(_size), flag(_flag), host_p(nullptr), device_p(nullptr)
    {}
    ~Pool()
    {
        clear();
    }
    void clear()
    {
        // TODO: uncomment this when Scene is finished
        //if (device_p != nullptr) CheckCUDAError(cudaFree(device_p));
        if (host_p != nullptr) delete[] host_p;
        device_p = host_p = nullptr;
        size = flag = 0;
    }
    void copyToDevice()
    {
        assert(size != 0);
        if ((flag & IN_HOST) && (flag & IN_DEVICE))
        {
            CheckCUDAError(cudaMemcpy(getDevice(), getHost(), size * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
    void copyFromDevice()
    {
        assert(size != 0);
        if ((flag & IN_HOST) && (flag & IN_DEVICE))
        {
            CheckCUDAError(cudaMemcpy(getHost(), getDevice(), size * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }
    T * getHost()
    {
        assert(size != 0);
        if ((flag & IN_HOST) && host_p == nullptr)
        {
            host_p = new T[size];
        }
        if (host_p == nullptr)
        {
            PrintError("Pool<T>: called getHost(), but IN_HOST not set");
            exit(1);
        }
        return host_p;
    }
    T * getDevice()
    {
        assert(size != 0);
        if ((flag & IN_DEVICE) && device_p == nullptr)
        {
            CheckCUDAError(cudaMalloc((void**)&device_p, size * sizeof(T)));
        }
        if (device_p == nullptr)
        {
            PrintError("Pool<T>: called getDevice(), but IN_DEVICE not set");
            exit(1);
        }
        return device_p;
    }
    size_t getSize() const
    {
        return size;
    }

    // unique ownership
    Pool(Pool<T> &o)
    {
        size = o.size;
        flag = o.flag;
        host_p = o.host_p;
        device_p = o.device_p;
        o.release();
    }
    Pool(Pool<T> &&o)
    {
        size = o.size;
        flag = o.flag;
        host_p = o.host_p;
        device_p = o.device_p;
        o.release();
    }
    Pool & operator = (Pool<T> &o)
    {
        size = o.size;
        flag = o.flag;
        host_p = o.host_p;
        device_p = o.device_p;
        o.release();
        return *this;
    }
    Pool & operator = (Pool<T> &&o)
    {
        size = o.size;
        flag = o.flag;
        host_p = o.host_p;
        device_p = o.device_p;
        o.release();
        return *this;
    }
    void swap(Pool<T> &o)
    {
        Pool<T> tmp = *this;
        *this = o;
        o = tmp;
    }
};

// store data list-like on CPU, array-like on GPU
template <typename T>
class VectorPool
{
    std::vector<T> data;
    T *device_p;
public:
    VectorPool() : device_p(nullptr)
    {}
    ~VectorPool()
    {
        // if (device_p != nullptr) CheckCUDAError(cudaFree(device_p));
        device_p = nullptr;
    }
    void add(const T &i)
    {
        assert(device_p == nullptr);
        data.emplace_back(i);
    }
    void add(T &&i)
    {
        assert(device_p == nullptr);
        data.emplace_back(i);
    }
    T * getHost()
    {
        return data.data();
    }
    T * getDevice()
    {
        if (device_p == nullptr) syncToDevice();
        assert(device_p != nullptr);
        return device_p;
    }
    size_t getSize() const
    {
        return data.size();
    }
    void syncToDevice()
    {
        assert(!data.empty());
        if (device_p == nullptr)
        {
            CheckCUDAError(cudaMalloc((void**)&device_p, data.size() * sizeof(T)));
            CheckCUDAError(cudaMemcpy(device_p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
};