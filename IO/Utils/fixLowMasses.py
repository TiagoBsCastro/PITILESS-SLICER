import numpy as np
from scipy.integrate import cumtrapz, quad
from scipy.optimize import root_scalar

def mass_correction (mf, mthreshold):
    '''
    This function returns the piecewise linear coeficients to correct masses
    enforcing the Analytical Mass function at low-masses.

    mf: Pinocchio mass function instance
        The mf instance read by ReadPinocchio.

    mthreshold: float
        Only halos with mass lower than mthreshold are corrected.
    '''
    # Getting the first object more massive than mthreshold
    i0    = np.where(mf.m-mthreshold > 0.0 )[0][0]
    x0    = np.log(mf.m[i0])
    x     = np.log(mf.m[:i0+1])
    y     = mf.dndlnm[:i0+1]
    y_teo = mf.dndlnm_teo[:i0+1]
    # Getting the number of bins
    nbins = x.size

    # Getting the CMF for both teo and obs
    pdf     = lambda lnm: np.interp(lnm, x, y)
    pdf_teo = lambda lnm: np.interp(lnm, x, y_teo)
    cmf     = lambda lnm: quad(pdf, lnm, x0)[0]
    cmf_teo = lambda lnm: quad(pdf_teo, lnm, x0)[0]

    # Imposing the initial values alpha0 = 1; beta0 = ln(m0)
    alpha = np.ones_like(mf.dndlnm)
    beta = np.log( mf.m )

    sol0 = x0
    soli = [x0]
    for i in range(i0-1, -1, -1):

        cmfi    = cmf(x[i])
        deltaxi = x[i]-x[i+1]
        f = lambda lnm: cmf_teo(lnm) - cmfi
        sol = root_scalar(f, bracket=[x.min(), x.max()])
        alpha[i] = (sol.root-sol0)/deltaxi
        beta[i]  = alpha[i+1] * (x[i]-x[i+1]) + beta[i+1]
        sol0 = sol.root
        soli += [sol.root]

    return soli, alpha, beta

def mass_correction_linear (mf, mthreshold):
    '''
    This function returns the piecewise linear coeficients to correct masses
    enforcing the Analytical Mass function at low-masses.

    mf: Pinocchio mass function instance
        The mf instance read by ReadPinocchio.

    mthreshold: float
        Only halos with mass lower than mthreshold are corrected.
    '''
    # Getting the first object more massive than mthreshold
    i0    = np.where(mf.m-mthreshold > 0.0 )[0][0]
    x0    = mf.m[i0]
    x     = mf.m[:i0+1]
    y     = mf.dndm[:i0+1]
    y_teo = mf.dndm_teo[:i0+1]
    # Getting the number of bins
    nbins = x.size

    # Getting the CMF for both teo and obs
    pdf     = lambda m: np.interp(m, x, y)
    pdf_teo = lambda m: np.interp(m, x, y_teo)
    cmf     = lambda m: quad(pdf, m, x0)[0]
    cmf_teo = lambda m: quad(pdf_teo, m, x0)[0]

    # Imposing the initial values alpha0 = 1; beta0 = ln(m0)
    alpha = np.ones_like(mf.dndlnm)
    beta  = mf.m

    sol0 = x0
    soli = [x0]
    for i in range(i0-1, -1, -1):

        cmfi    = cmf(x[i])
        deltaxi = x[i]-x[i+1]
        f = lambda m: cmf_teo(m) - cmfi
        sol = root_scalar(f, bracket=[x.min(), x.max()])
        alpha[i] = (sol.root-sol0)/deltaxi
        beta[i]  = alpha[i+1] * (x[i]-x[i+1]) + beta[i+1]
        sol0 = sol.root
        soli += [sol.root]

    return soli, alpha, beta

def get_mass_correction (mx, mf, mthreshold, linear=True):
    '''
    This function returns the piecewise linear correction of masses
    enforcing the Analytical Mass function at low-masses.

    mx: float.
        Mass to evaluate the piecewise mass correction

    nx: int
        Number of particles.

    mf: Pinocchio mass function instance
        The mf instance read by ReadPinocchio.

    mthreshold: float
        Only halos with mass lower than mthreshold are corrected.
    '''
    # Get index
    idx = np.digitize(mx, mf.m, right=True)
    idx[ idx == mf.m.size ] = mf.m.size-1

    if linear:

        # Get coefficients
        m, alpha, beta = mass_correction_linear(mf, mthreshold)

        return np.array(m), alpha[idx] * (mx-mf.m[idx]) + beta[idx]

    else:
        # Get coefficients
        lnm, alpha, beta = mass_correction(mf, mthreshold)

        return np.exp(lnm), np.exp(alpha[idx] * np.log(mx/mf.m[idx]) + beta[idx])
