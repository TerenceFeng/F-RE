#include <iostream>
using namespace std;

#include "math.h"

int main()
{
    MDSpace<3> space = {3, 3, 3};
    for (const auto & p : space)
    {
        for (size_t pi : p)
            cout << pi << ',';
        cout << endl;
    }
    return 0;
}
