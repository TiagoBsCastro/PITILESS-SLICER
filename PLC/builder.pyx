from libc.math cimport sin, cos, asin, acos, atan2, pi, sqrt, fabs
cimport cython
cdef int _WRAP_POSITIONS = False

cdef void func(float *q, float *v1, float *v2, float *v31, float *v32, float *d
                 , float *d2, float *d31, float *d32, float *dplc, float a, int norder, float *f, float *df) nogil:

    cdef float aj = 1.0;
    cdef float fpi;
    cdef float dfpi;
    cdef float fp = 0.0;
    cdef float dfp = 0.0;
    cdef float fplc = 0.0;
    cdef float dfplc = 0.0;
    cdef Py_ssize_t i,j

    for j in range(norder+1):

        fplc += dplc[j]*aj
        if j<norder:
          dfplc += (j+1)*dplc[j+1]*aj

        aj *= a;

    for i in range(3):

        fpi = q[i];
        dfpi = 0.0;
        aj = 1.0;
        for j in range(norder+1):

            fpi += (v1[i]*d[j] + v2[i]*d2[j] + v31[i]*d31[j] + v32[i]*d32[j])*aj
            if j < norder:
              dfpi += (j+1)*(v1[i]*d[j+1] + v2[i]*d2[j+1] + v31[i]*d31[j+1] + v32[i]*d32[j+1])*aj

            aj *= a

        fp += fpi*fpi
        dfp += 2*fpi*dfpi

    f[0]  = fp - fplc*fplc;
    df[0] = dfp - 2*( fplc*dfplc )

cpdef void getCartesianCoordinates (float [:,:] qPos, float[:] replication, float [:,:] V1, float [:,:] V2, float [:,:] V31, float [:,:] V32, float [:] a, float[:,:] coord,
                        int npart, float [:] D, float [:] D2, float [:] D31, float [:] D32, int norder) nogil:

    cdef Py_ssize_t i,j,k;
    cdef float fpij;
    cdef float x[3];
    cdef float ak;

    with cython.boundscheck(False):

        for i in range(npart):

            if a[i] > 0:

                for j in range(3):

                    fpij = qPos[i,j];
                    ak = 1.0;
                    for k in range(norder+1):

                        fpij += (V1[i,j]*D[k] + V2[i,j]*D2[k] + V31[i,j]*D31[k] + V32[i,j]*D32[k])*ak
                        ak   *= a[i]

                    if _WRAP_POSITIONS:

                        if fpij > 0.5:
                            x[j] = fpij - 1.0 + replication[j]
                        elif fpij < -0.5:
                            x[j] = 1.0 - fpij + replication[j]
                        else:
                            x[j] = fpij + replication[j]

                    else:

                        x[j] = fpij + replication[j]

                coord[i,0] = x[0]
                coord[i,1] = x[1]
                coord[i,2] = x[2]

            else:

               coord[i,0] = -1.0;
               coord[i,1] = -1.0;
               coord[i,2] = -1.0;

cpdef void getSkyCoordinates (float [:,:] qPos, float[:] replication, float [:,:] V1, float [:,:] V2, float [:,:] V31, float [:,:] V32, float [:] a, float[:,:] coord,
                                       int npart, float [:] D, float [:] D2, float [:] D31, float [:] D32, int norder) nogil:

    cdef Py_ssize_t i,j,k;
    cdef float fpij;
    cdef float x[3];
    cdef float ak;

    with cython.boundscheck(False):

        for i in range(npart):

            if a[i] > 0:

                for j in range(3):

                    fpij = qPos[i,j];
                    ak = 1.0;
                    for k in range(norder+1):

                        fpij += (V1[i,j]*D[k] + V2[i,j]*D2[k] + V31[i,j]*D31[k] + V32[i,j]*D32[k])*ak
                        ak   *= a[i]

                    if _WRAP_POSITIONS:

                        if fpij > 0.5:
                            x[j] = fpij - 1.0 + replication[j]
                        elif fpij < -0.5:
                            x[j] = 1.0 - fpij + replication[j]
                        else:
                            x[j] = fpij + replication[j]

                    else:

                        x[j] = fpij + replication[j]

                coord[i,0] = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
                coord[i,1] = -acos(x[2]/coord[i,0]) + pi/2.0;
                coord[i,2] = atan2(x[1],x[0]);

                if coord[i,2] < 0:

                  coord[i,2] += 2*pi

            else:

                coord[i,0] = -1.0;
                coord[i,1] = -1.0;
                coord[i,2] = -1.0;

cpdef void getCrossingScaleParameterBisection (float [:,:] qPos, float [:,:] V1, float [:,:] V2, float [:,:] V31, float [:,:] V32, float [:] aplc,
        int npart, float [:] DPLC, float [:] D, float [:] D2, float [:] D31, float [:] D32, int norder, float amin, float amax):

    cdef Py_ssize_t i
    cdef float fa, fb, fc, a, b, c, f, df

    for i in range(npart):

        a = amin
        b = amax
        c = (a + b)/2

        func(&qPos[i,0], &V1[i,0], &V2[i,0], &V31[i,0], &V32[i,0], &D[0], &D2[0], &D31[0], &D32[0], &DPLC[0], a, norder, &fa, &df)
        func(&qPos[i,0], &V1[i,0], &V2[i,0], &V31[i,0], &V32[i,0], &D[0], &D2[0], &D31[0], &D32[0], &DPLC[0], b, norder, &fb, &df)

        if (fa*fb > 0):

            aplc[i] = -1.0

        else:

            while( 2.0*(b - a)/(b + a) > 1e-2):

                c = (a+b)/2
                func(&qPos[i,0], &V1[i,0], &V2[i,0], &V31[i,0], &V32[i,0], &D[0], &D2[0], &D31[0], &D32[0], &DPLC[0], c, norder, &fc, &df)

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

cpdef void getCrossingScaleParameterNewtonRaphson (float [:,:] qPos, float [:,:] V1, float [:,:] V2, float [:,:] V31, float [:,:] V32, float [:] aplc,
                    int npart, float [:] DPLC, float [:] D, float [:] D2, float [:] D31, float [:] D32, int norder, float amin, float amax) nogil:

    cdef Py_ssize_t i
    cdef float fa, fb, a, b, x, f, df
    cdef float h;
    with cython.boundscheck(False):

        for i in range(npart):

            a = amin
            b = amax

            func(&qPos[i,0], &V1[i,0], &V2[i,0], &V31[i,0], &V32[i,0], &D[0], &D2[0], &D31[0], &D32[0], &DPLC[0], a, norder, &fa, &df)
            func(&qPos[i,0], &V1[i,0], &V2[i,0], &V31[i,0], &V32[i,0], &D[0], &D2[0], &D31[0], &D32[0], &DPLC[0], b, norder, &fb, &df)

            if (fa*fb)>0:
                aplc[i] = -1.0

            else:
                x = (a+b)/2;
                h = 9999.9
                while( fabs(h) > 1e-3):

                    func(&qPos[i,0], &V1[i,0], &V2[i,0], &V31[i,0], &V32[i,0], &D[0], &D2[0], &D31[0], &D32[0], &DPLC[0], x, norder, &f, &df)
                    h = f/df
                    x -= h

                if x >= amin and x < amax:

                    aplc[i] = x

                else:

                    aplc[i] = -1.0
