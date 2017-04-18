#pragma once

#include <cstddef>

#include "math.h"

enum EDevice
{
    CPU,
    GPU
};

/* Computation Resource Management */
class ComputeEngine
{
   public:
    template <typename F, size_t N, template <size_t> class MD, typename R,
              typename P1>
    static void Dispatch(EDevice device, F f, R *out, MD<N> &s, P1 *in1)
    {
        if (device == CPU)
        {
            for (auto &p : s)
            {
                f(out, s, p, in1);
                fprintf(stderr, "\r%lu/%lu", p[N - 1], s[N - 1]);
            }
        }
    }
    template <typename F, size_t N, template <size_t> class MD, typename R,
              typename P1, typename P2>
    static void Dispatch(EDevice device, F f, R *out, MD<N> &s, P1 *in1,
                         P2 *in2)
    {
        if (device == CPU)
        {
            for (auto &p : s)
            {
                f(out, s, p, in1, in2);
                fprintf(stderr, "\r%lu/%lu", p[N - 1], s[N - 1]);
            }
        }
    }
    template <typename F, size_t N, template <size_t> class MD, typename R1,
              typename R2, typename P1, typename P2, typename P3>
    static void Dispatch(EDevice device, F f, R1 *out1, R2 *out2, MD<N> &s,
                         P1 *in1, P2 *in2, P3 *in3)
    {
        if (device == CPU)
        {
            for (auto &p : s)
            {
                f(out1, out2, s, p, in1, in2, in3);
                fprintf(stderr, "\r%lu/%lu", p[N - 1], s[N - 1]);
            }
        }
    }
};
