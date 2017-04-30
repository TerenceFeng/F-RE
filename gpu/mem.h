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
	Pool(Pool<T> &o)
	{
		size = o.size;
		flag = o.flag;
		host_p = o.host_p;
		device_p = o.device_p;
		o.size = o.flag = 0;
		o.host_p = o.device_p = nullptr;
	}
	Pool(Pool<T> &&o)
	{
		size = o.size;
		flag = o.flag;
		host_p = o.host_p;
		device_p = o.device_p;
		o.size = o.flag = 0;
		o.host_p = o.device_p = nullptr;
	}
	Pool & operator=(Pool<T> &o)
	{
		size = o.size;
		flag = o.flag;
		host_p = o.host_p;
		device_p = o.device_p;
		o.size = o.flag = 0;
		o.host_p = o.device_p = nullptr;
		return (*this);
	}
	Pool & operator=(Pool<T> &&o)
	{
		size = o.size;
		flag = o.flag;
		host_p = o.host_p;
		device_p = o.device_p;
		o.size = o.flag = 0;
		o.host_p = o.device_p = nullptr;
		return (*this);
	}
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
		if (host_p == nullptr)
		{
			fprintf(stderr, "Pool<T> getHost() without IN_HOST flag\n");
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
			fprintf(stderr, "Pool<T> getDevice() without IN_DEVICE flag\n");
			exit(1);
		}
        return device_p;
    }
    size_t getSize() const
    {
        return size;
    }
    void swap(Pool<T> &o)
    {
		*this = o;
    }
};

// store data list-like on CPU, array-like on GPU
