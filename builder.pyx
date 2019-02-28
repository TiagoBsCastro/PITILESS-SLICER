def bisection(float a, float b):

    if (func(a) * func(b) >= 0):
        return -1.0

    cdef c = a;
    while ((b-a) >= 1e-2):

        # Find middle point
        c = (a+b)/2;

        # Check if middle point is root
        if (func(c) == 0.0)
            break
        # Decide the side to repeat the steps
        else if (func(c)*func(a) < 0)
            b = c
        else
            a = c

    return c
