import numpy as np

def mass_correction (mf, mthreshold):
    '''
    This function returns the piecewise linear coeficients to correct masses
    enforcing the Analytical Mass function at low-masses.

    mf: Pinocchio mass function instance
        The mf instance read by ReadPinocchio

    mthreshold: float
        Only halos with mass lower than mthreshold are corrected.
    '''

    # Getting the first object more massive than mthreshold
    i0 = np.where(mf.m-mthreshold > 0.0 )[0][0]
    m0 = mf.m[i0]
    # Getting the number of bins
    nbins = mf.m.size

    alpha = mf.dndlnm/mf.dndlnm_teo
    alpha[mf.m>m0] = 1.0

    beta = np.log( mf.m )

    for i in range(i0):

        beta[i] = np.sum( [alpha[j] * np.log(mf.m[j]/mf.m[j+1]) for j in range(i, i0-1) ] ) + np.log(m0)

    return alpha, beta

def get_mass_correction (mx, mf, mthreshold):
    '''
    This function returns the piecewise linear correction of masses
    enforcing the Analytical Mass function at low-masses.

    mf: Pinocchio mass function instance
        The mf instance read by ReadPinocchio

    mthreshold: float
        Only halos with mass lower than mthreshold are corrected.

    mx: float
        Mass to evaluate the piecewise mass correction
    '''

    # Get index
    idx = np.digitize(mx, mf.m)
    if idx == mf.m.size:

        idx -=1
        
    # Get coefficients
    alpha, beta = mass_correction(mf, mthreshold)

    return np.exp(alpha[idx] * np.log(mx/mf.m[idx]) + beta[idx])
