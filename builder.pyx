from libc.math cimport pow

cdef float func(float q, float v1, float v2, float v31, float v32, float d
             , float d2, float d31, float d32, a, j):

    return pow(q + (v1*d + v2*d2 + v31*d31 + v32*d32)*pow(a,j), 2)

cpdef void getCrossingScaleParameter (float [:,:] qPos, float [:,:] V1, float [:,:] V2, float [:,:] V31, float [:,:] V32, float [:] aplc,
        int npart, float [:] DPLC, float [:] D, float [:] D2, float [:] D31, float [:] D32, int norder, float amin, float amax):

    cdef Py_ssize_t i,j
    cdef float fa, fb, dplca, dplcb, dplcc, a, b, c

    for i in range(npart):

        a = amin
        b = amax
        c = (a + b)/2
        fa = 0.0
        fb = 0.0
        dplca = 0.0
        dplcb = 0.0

        for j in range(norder+1):

            for k in range(3):

                fa += func(qPos[i,k], V1[i,k], V2[i,k], V31[i,k], V32[i,k], D[j], D2[j], D31[j], D32[j], a, j)
                fb += func(qPos[i,k], V1[i,k], V2[i,k], V31[i,k], V32[i,k], D[j], D2[j], D31[j], D32[j], b, j)

            dplca += DPLC[j]*pow(a,j)
            dplcb += DPLC[j]*pow(b,j)

        fa -= pow(dplca, 2)
        fb -= pow(dplcb, 2)

        if (fa*fb > 0):

            aplc[i] = -1.0

        else:

            while( 2.0*(b - a)/(b + a) > 1e-2):

                c = (a+b)/2
                fc = 0.0
                dplcc = 0.0
                for j in range(norder+1):

                    for k in range(3):

                        fc += func(qPos[i,k], V1[i,k], V2[i,k], V31[i,k], V32[i,k], D[j], D2[j], D31[j], D32[j], c, j)

                    dplcc += DPLC[j]*pow(c,j)

                fc -= pow(dplcc, 2)

                # Check if middle point is root
                if (fc == 0.0):

                    aplc[i] = c
                    continue

                # Decide the side to repeat the steps
                elif (fc*fa < 0):

                    b = c

                else:

                    a = c

            aplc[i] = c
