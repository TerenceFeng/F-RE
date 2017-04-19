#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>

/* Multi-dimensional View */
template <size_t N = 3>
class MDSpace;
template <size_t N = 3>
class MDPoint;

template <size_t N>
class MDPoint
{
    size_t d[N];

   public:
    typedef size_t *iterator;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;
    typedef size_t value_type;
    typedef size_t *pointer;
    typedef size_t &reference;
    typedef const size_t *const_iterator;

   public:
    MDPoint<N>()
    {
        std::fill(d, d + N, 0);
    }
    MDPoint<N>(const MDPoint<N> &s)
    {
        std::copy(s.d, s.d + N, d);
    }
    MDPoint<N>(std::initializer_list<size_t> l)
    {
        std::copy(l.begin(), l.end(), d);
    }
    iterator begin()
    {
        return d;
    }
    iterator end()
    {
        return d + N;
    }
    const_iterator begin() const
    {
        return d;
    }
    const_iterator end() const
    {
        return d + N;
    }
    inline size_t &operator[](size_t i)
    {
        // assert( i < N);
        return d[i];
    }
    inline size_t operator[](size_t i) const
    {
        // assert( i < N);
        return d[i];
    }
};
template <size_t N>
class MDSpace : public MDPoint<N>
{
    struct _forward_iterator
    {
        const MDSpace<N> &space;
        MDPoint<N> point;
        // ++, *, ->, ==, !=
        _forward_iterator(const MDSpace<N> &s, const MDPoint<N> &p)
            : space(s), point(p)
        {
        }

        _forward_iterator &operator++()
        {
            // assert( N > 0 );
            if (point[N - 1] == space[N - 1])
                return *this;

            for (int i = 0; i < N; ++i)
            {
                ++point[i];
                if (i < N - 1 && point[i] == space[i])
                    point[i] = 0;
                else
                    break;
            }
            return *this;
        }
        _forward_iterator &operator++(int)
        {
            return _forward_iterator(*this ++);
        }
        const MDPoint<N> &operator*()
        {
            return point;
        }
        bool operator==(const _forward_iterator &o) const
        {
            if (point[N - 1] == space[N - 1])
                return true;

            size_t i = N;
            while (i > 0)
            {
                --i;
                if (point[i] != o.point[i])
                    return false;
            }
            return true;
        }
        bool operator!=(const _forward_iterator &o) const
        {
            return !(*this == o);
        }
    };

   public:
    typedef _forward_iterator iterator;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;
    typedef size_t value_type;
    typedef size_t *pointer;
    typedef size_t &reference;

   public:
    MDSpace<N>(std::initializer_list<size_t> l) : MDPoint<N>(l)
    {
    }
    iterator begin()
    {
        return iterator(*this, MDPoint<N>());
    }
    iterator end()
    {
        return iterator(*this, MDPoint<N>(*this));
    }
};

// Avoid N = 0
template <>
class MDPoint<0>
{
};
template <>
class MDSpace<0>
{
};

/* Linear Algebra */
template <typename T>
struct Vec3
{
    T x, y, z;

    bool isZero() const
    {
        return x == T(0) && y == T(0) && z == T(0);
    }

    Vec3<T> &zero()
    {
        x = y = z = T(0);
        return *this;
    }
    Vec3<T> &norm()
    {
        // T mod = rsqrt(x * x + y * y + z * z);
        T mod = T(1) / sqrt(x * x + y * y + z * z);
        x *= mod;
        y *= mod;
        z *= mod;
        return *this;
    }
    Vec3<T> &assign(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
        return *this;
    }
    Vec3<T> &assign(const Vec3<T> &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
    Vec3<T> &add(const Vec3<T> &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vec3<T> &sub(const Vec3<T> &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    Vec3<T> &mul(const Vec3<T> &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    Vec3<T> &scale(T factor)
    {
        x *= factor;
        y *= factor;
        z *= factor;
        return *this;
    }
    T dot(const Vec3<T> &v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    static Vec3<T> Zero()
    {
        Vec3<T> result = {T(0), T(0), T(0)};
        return result;
    }
    static Vec3<T> Add(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        Vec3<T> result = {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
        return result;
    }
    static Vec3<T> Sub(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        Vec3<T> result = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
        return result;
    }
    static Vec3<T> Mul(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        Vec3<T> result = {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
        return result;
    }
    static Vec3<T> Scale(const Vec3<T> &v, T factor)
    {
        Vec3<T> result = {v.x * factor, v.y * factor, v.z * factor};
        return result;
    }
    static Vec3<T> Norm(const Vec3<T> &v)
    {
        Vec3<T> result = v;
        result.norm();
        return result;
    }
    static T Dot(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }
    static Vec3<T> Cross(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        Vec3<T> result = {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
                          v1.x * v2.y - v1.y * v2.x};
        return result;
    }

    Vec3<T> operator+(const Vec3<T> &v) const
    {
        return Add(*this, v);
    }
    Vec3<T> operator-(const Vec3<T> &v) const
    {
        return Sub(*this, v);
    }
    Vec3<T> &operator+=(const Vec3<T> &v)
    {
        return add(v);
    }
    Vec3<T> operator-=(const Vec3<T> &v)
    {
        return sub(v);
    }

    // Debug
    Vec3<T>(T _x = 0, T _y = 0, T _z = 0) : x(_x), y(_y), z(_z)
    {
    }
};

template <typename T>
struct Mat
{
    T *data;
    // size_t width, height;

    // T & at(size_t w, size_t h);
    // const T & at(size_t w, size_t h) const;
};

/* Geometry */
// typedef Vec3f Vertex;
// Surface
template <typename T>
T DistanceSquared(const Vec3<T> &l, const Vec3<T> &r)
{
    return (l.x - r.x) * (l.x - r.x) + (l.y - r.y) * (l.y - r.y) +
           (l.z - r.z) * (l.z - r.z);
}

// Point, Normal, Ray
// Intersection-Test
