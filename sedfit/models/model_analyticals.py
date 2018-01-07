import numpy as np
import Radiation_Model_Toolkit as rmt

__all__ = ["BlackBody", "Modified_BlackBody", "Power_Law", "Synchrotron",
           "Linear", "Line_Gaussian_L", "Poly3"]

ls_mic = 2.99792458e14 #unit: micron/s
Mpc = 3.08567758e24 #unit: cm
mJy = 1e-26 #1 mJy in erg/s/cm^2/Hz

def BlackBody(logOmega, T, wave):
    """
    Calculate the flux density of a blackbody emitter.

    Parameters
    ----------
    logOmega : float
        The log10 of the solid angle subtended by the emitter.
    T : float
        The temperature of the emitter, unit: Kelvin.
    wave : float array
        The wavelength to be calculated.

    Returns
    -------
    flux : float array
        The flux density of corresponding to the input wavelength, unit: mJy.

    Notes
    -----
    None.
    """
    nu   = ls_mic / wave
    flux = rmt.BlackBody(nu, logOmega=logOmega, T=T) / mJy #unit: mJy
    return flux

def Modified_BlackBody(logM, T, beta, wave, DL, z, kappa0=16.2, lambda0=140, frame="rest"):
    """
    This function is a wrapper to calculate the modified blackbody model.

    Parameters
    ----------
    logM : float
        The dust mass in the unit solar mass.
    T : float
        The temperature of the dust.
    beta : float
        The dust emissivity which should be around 2.
    wave : float array
        The wavelengths of the calculated fluxes.
    DL : float
        The luminosity distance in the unit Mpc.
    z : float
        The redshift of the source.
    frame : string
        "rest" for the rest frame SED and "obs" for the observed frame.
    kappa0 : float, default: 16.2
        The normalisation opacity.
    lambda0 : float, default: 140
        The normalisation wavelength.

    Returns
    -------
    flux : float array
        The flux at the given wavelengths to calculate.

    Notes
    -----
    None.

    """
    nu = ls_mic / wave
    flux = rmt.Dust_Modified_BlackBody(nu, logM, DL, beta, T, z, frame, kappa0, lambda0)
    return flux

def Power_Law(PL_alpha, PL_logsf, wave):
    """
    This function is a wrapper to calculate the power law model.

    Parameters
    ----------
    PL_alpha : float
        The power-law index.
    PL_logsf : float
        The log of the scaling factor.
    wave : float array
        The wavelength.

    Returns
    -------
    flux : float array
        The flux at the given wavelengths to calculate.

    Notes
    -----
    None.
    """
    nu = ls_mic / wave
    flux = rmt.Power_Law(nu, PL_alpha, 10**PL_logsf)
    return flux

def Synchrotron(Sn_alpha, Sn_logsf, wave, lognuc=13, lognum=14):
    """
    Calculate the model of synchrotron emission,
        fnu ~ nu^-alpha (nu < nu0)
            ~ nu^-(alpha+0.5) (nu > nu0),
    which comes from Figure 1 of Pe'er, Space Sci Rev (2014) 183:371.

    Parameters
    ----------
    Sn_alpha : float
        The power-law index.
    Sn_logsf : float
        The log of the scaling factor.
    wave : float array
        The wavelength.
    lognuc : float
        The log of the cooling frequency (unit: Hz).
    lognum : float
        The maximum frequency above which there is no synchrotron emission
        (unit: Hz).

    Returns
    -------
    flux : float array
        The flux at the given wavelengths to calculate.

    Notes
    -----
    None.
    """
    num = 10**(lognum - lognuc)
    nu = np.atleast_1d(ls_mic / wave) / 10**lognuc
    fltr_m = nu < num
    fltr_h = (nu > 1) & fltr_m
    fltr_l = (nu <= 1) & fltr_m
    flux = np.zeros_like(nu)
    sf = 10**Sn_logsf
    flux[fltr_h] = sf * nu[fltr_h]**(-Sn_alpha-0.5)
    flux[fltr_l] = sf * nu[fltr_l]**(-Sn_alpha)
    return flux

def Linear(a, b, x):
    return a * np.atleast_1d(x) + b

def Line_Gaussian_L(wavelength, logLum, lambda0, FWHM, DL):
    """
    The wrapper of the function Line_Profile_Gaussian() to use wavelength and
    luminosity as the parameters.
    Calculate the flux density of the emission line with a Gaussian profile.

    Parameters
    ----------
    wavelength : float array
        The wavelength of the spectrum.
    logLum : float
        The log of luminosity of the line, unit: erg/s.
    lambda0 : float
        The central wavelength of the emission line.
    FWHM : float
        The full width half maximum (FWHM) of the emission line.
    DL : float
        The luminosity distance, unit: Mpc.

    Returns
    -------
    fnu : float array
        The flux density of the spectrum, units: mJy.

    Notes
    -----
    None.
    """
    flux = 10**logLum / (4 * np.pi * (DL * Mpc)**2.0)
    nu  = ls_mic / wavelength
    nu0 = ls_mic / lambda0
    fnu  = rmt.Line_Profile_Gaussian(nu, flux, nu0, FWHM, norm="integrate")
    return fnu

from numpy.polynomial.polynomial import polyval
def Poly3(x, logc0, c1, c2, c3):
    """
    This is a 3rd order polynomial function.  It calls polyval from numpy.

    Parameters
    ----------
    x : array like
        The input active variable.
    logc0 : float
        The log10 of c0 the coefficient of the 0th order.
    c1 -- c3 : floats
        The coefficiants for 1th to 3rd order.

    Returns
    -------
    y : array like
        The function result.
    """
    x = np.atleast_1d(x)
    c0 = 10**logc0
    y = polyval(x, [c0, c1, c2, c3])
    return y

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    wave = np.linspace(-5, 5, 20)
    flux = Poly3(wave, 5, -8., 2, 1)
    #flux = Synchrotron(0.8, 5, wave, lognuc=13, lognum=14)
    plt.plot(wave, flux)
    #plt.axvline(x=ls_mic/1e13, color="r", linestyle=":")
    #plt.axvline(x=ls_mic/1e14, color="r", linestyle=":")
    #plt.xscale('log')
    plt.yscale('log')
    plt.show()
