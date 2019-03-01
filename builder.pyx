cdef void getCrossingScaleParameter (float [:] qPos, float [:] V1, float [:] V2, float [:] V31, float [:] V32, float [:] aplc, 
        int npart, float [:] DPLC, float [:] D, float [:] D2, float [:] D31, float [:] D32, int norder, float amin, float amax):

    cdef Py_ssize_t i,j
    cdef float fa, fb, a, b, c

    for i in range(npart):

        a = amin
        b = amax 
        c = (a + b)/2
        fa = 0.0
        fb = 0.0

        for j in range(norder):

            fa += qPos[i] + (V1[i]*D[j] + V2[i]*D2[j] + V31[i]*D31[j] + V32[i]*D32[j] - DPLC[j])*a**j
            fb += qPos[i] + (V1[i]*D[j] + V2[i]*D2[j] + V31[i]*D31[j] + V32[i]*D32[j] - DPLC[j])*b**j

        if (fa*fb > 0):

            aplc[i] = -1.0

        else:

            while(fb - fa > 1e-2):

                c = (a+b)/2    
                fc = 0.0
                for j in range(norder):
    
                    fc += qPos[i] + (V1[i]*D[j] + V2[i]*D2[j] + V31[i]*D31[j] + V32[i]*D32[j])*c**j

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


    

