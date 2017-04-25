#include "compute.h"

#include <functional>
#include <iostream>
using namespace std;

void add_vec(int * out, const MDSpace<1> & s, const MDPoint<1> & p,
             const int * a, const int * b)
{
    size_t i = p[0];
    out[i] = a[i] + b[i];
}

int main()
{
    int a[5] = {1, 1, 1, 1, 1};
    int b[5] = {2, 3, 4, 5, 6};
    int c[5] = {0, 0, 0, 0, 0};

    MDSpace<1> s = {5};
    ComputeEngine::Dispatch(CPU, add_vec, c, s, a, b);
    for (int i = 0; i < 5; ++i)
    {
        if (c[i] != a[i] + b[i])
        {
            cout << "Test failed!" << endl;
            return 1;
        }
    }
    cout << "Test pass." << endl;

    return 0;
}
