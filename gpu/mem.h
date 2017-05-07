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

#include <map>
class DeviceMemoryManager
{
    static std::map<void *, size_t> msize;
    static std::map<void *, bool> mfree;
public:
    static void *Alloc(size_t size)
    {
        Logger << "DeviceMemory: Alloc(" << size << ")\n";
        void *p = nullptr;
        CheckCUDAError(cudaMalloc((void**)&p, size));
        if (p == nullptr)
        {
            ShowErrorAndExit("DeviceMemory: Alloc() failed!");
        }
        msize[p] = size;
        mfree[p] = false;
        return p;
    }
    static void Free(void *p)
    {
        Logger << "DeviceMemory: Free(" << p << ")\n";
        if (p != nullptr)
        {
            ShowErrorAndExit("DeviceMemory: Free() nullptr!");
        }
        else if (mfree.find(p) != mfree.end() && mfree[p] == true)
        {
            ShowErrorAndExit("DeviceMemory: Free() twice!");
        }
        else
        {
            CheckCUDAError(cudaFree(p));
            mfree[p] = true;
        }
    }
    static void Summary()
    {
        Logger << "----------- Memory Summary -------------\n";
        for (auto &k : msize)
        {
            Logger << "addr: " << k.first
                << " size: " << k.second
                << " free: " << mfree[k.first]
                << "\n";
        }
    }
};
std::map<void *, size_t> DeviceMemoryManager::msize;
std::map<void *, bool> DeviceMemoryManager::mfree;

//class ReferenceCounted
//{
//    int *ref;
//public:
//    ReferenceCounted()
//    {
//        ref = new int;
//        *ref = 1;
//    }
//    ReferenceCounted(const ReferenceCounted &r)
//    {
//        ref = r.ref;
//        *ref += 1;
//    }
//    ReferenceCounted(ReferenceCounted &&r)
//    {
//        ref = r.ref;
//        *ref += 1;
//    }
//    ~ReferenceCounted()
//    {
//        if (*ref == 1) free(ref);
//        else *ref -= 1;
//    }
//    int get() const
//    {
//        return *ref;
//    }
//    ReferenceCounted & operator = (const ReferenceCounted &r)
//    {
//        ref = r.ref;
//        *ref += 1;
//        return *this;
//    }
//    ReferenceCounted & operator = (ReferenceCounted &&r)
//    {
//        ref = r.ref;
//        *ref += 1;
//        return *this;
//    }
//};

// TODO: maybe reference counted ? (instead of unique ownership)
template <typename T>
class Pool //: private ReferenceCounted
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
        /*if (get() == 1)*/ clear();
    }
    void clear()
    {
        //if (device_p != nullptr) DeviceMemoryManager::Free(device_p);
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
            ShowErrorAndExit("Pool<T>: called getHost(), but IN_HOST not set");
        }
        return host_p;
    }
    T * getDevice()
    {
        assert(size != 0);
        if ((flag & IN_DEVICE) && device_p == nullptr)
        {
            device_p = (T *)DeviceMemoryManager::Alloc(size * sizeof(T));
        }
        if (device_p == nullptr)
        {
            ShowErrorAndExit("Pool<T>: called getDevice(), but IN_DEVICE not set");
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
class VectorPool //: private ReferenceCounted
{
    std::vector<T> data;
    T *device_p;
public:
    VectorPool() : device_p(nullptr)
    {}
    ~VectorPool()
    {
        //if (get() == 1)
        //{
        //    if (device_p != nullptr) DeviceMemoryManager::Free(device_p);
        //}
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
            device_p = (T *)DeviceMemoryManager::Alloc(data.size() * sizeof(T));
            CheckCUDAError(cudaMemcpy(device_p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
};

template <typename T, size_t N>
class ConstVectorPool //: private ReferenceCounted
{
    std::vector<T> data;
    T *device_p;
public:
    void add(const T &i)
    {
        if (data.size() == N)
        {
            LogInfo("ConstVectorPool: can't add more elements.");
        }
        else
        {
            data.emplace_back(i);
        }
    }
    void add(T &&i)
    {
        if (data.size() == N)
        {
            LogInfo("ConstVectorPool: can't add more elements.");
        }
        else
        {
            data.emplace_back(i);
        }
    }
    T * getHost()
    {
        return data.data();
    }
    void setDevice(T *p)
    {
        device_p = p;
        Logger << "ConstMemory: addr: " << device_p << " size: " << (N * sizeof(T)) << "\n";
    }
    T * getDevice()
    {
        return device_p;
    }
    size_t getSize() const
    {
        return data.size();
    }
    //void syncToDevice()
    //{
    //    //CheckCUDAError(cudaMemcpyToSymbol(device_p, data.data(), data.size() * sizeof(T)));
    //}
};