
# coding: utf-8

# # This page is to release the functions of radiation models
# * The prototype of this page is [SEDToolKit](http://localhost:8888/notebooks/SEDFitting/SEDToolKit.ipynb) in /Users/jinyi/Work/PG_QSO/SEDFitting/

# In[2]:

import numpy as np


# In[1]:

#Func_bgn:
#----------------------------------#
#      by SGJY, Dec. 13, 2015      #
#----------------------------------#
def Single_Planck(nu, T=1e4):
    '''
    This is a single Planck function.
    The input parameter are frequency (nu) with unit Hz and temperature (T) with unit K.
    The system of units is Gaussian units, thus the brightness has unit erg/s/cm^2/Hz/ster.
    '''
    h = 6.62606957e-27 #erg s
    c = 29979245800.0 #cm/s
    k = 1.3806488e-16 #erg/K
    Bnu = 2.0*h*nu**3.0/c**2.0 / (np.e**(h*nu/k/T) - 1.0)
    return Bnu
#Func_end

#Func_bgn:
#----------------------------------#
#      by SGJY, Dec. 13, 2015      #
#----------------------------------#
def Power_Law(nu, alpha, sf):
    '''
    This is a power-law function.
    nu is frequency.
    alpha is the power index.
    sf is the scaling factor.
    The results are in the units erg/s/cm^2/Hz.
    '''
    return sf * nu**(-alpha)
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Apr. 3, 2016     #
#-------------------------------------#
def BlackBody(nu, logOmega, T=1e4):
    '''
    This function calculate the blackbody emission given temperature T. The max
    flux is normalised to 1 if logSF=0.

    Parameter:
    ----------
    nu : float array
        Frequency with unit Hz.
    logOmega : float
        The log10 of the solid angle subtended by the emitter.
    T : float
        The temperature.

    Returns
    -------
    flux : float array
        The flux density (F_nu) calculated from the model, unit: erg/s/cm^2/Hz.

    Notes
    -----
    None.
    '''
    flux_sp = Single_Planck(nu, T)
    flux = 10**logOmega * flux_sp
    return flux
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Feb. 20, 2016    #
#-------------------------------------#
def Dust_Modified_BlackBody(nu, logM, DL, beta, T, z=0.0, frame="rest", kappa0=16.2, lambda0=140):
    '''
    This function calculate the dust emission assuming it is modified blackbody.
    The calculation is in the cgs system.

    Parameters
    ----------
    nu : array
        The frequency corresponding to the flux.
    logM : float
        The dust mass, in the unit solar mass in log10.
    DL : float
        The luminosity distance, in the unit Mpc.
    beta : float
        The dust emissivity.
    T : float
        The blackbody temperature.
    z : float
        The redshift, default: 0.0
    frame : string
        "rest" for the rest frame SED and "obs" for the observed frame.
    kappa0 : float, default: 16.2
        The opacity for reference, by default we choose the value at 140
        micron (Li & Draine 2001).
    lambda0 : float, default: 140
        The wavelength at which we use the opacity as reference, by default
        we choose 140 micron.

    Returns
    -------
    flux : array
        The flux at the corresponding frequences, in the unit mJy.

    Notes
    -----
    None.
    '''
    ls_mic = 2.99792458e14 #micron/s
    Msun = 1.9891e33 #solar mass in gram
    Mpc = 3.08567758e24 #1 megaparsec in centimeter
    mJy = 1e-26 #1 mJy in erg/s/cm^2/Hz

    nu0 = ls_mic/lambda0
    kappa = kappa0 * (nu/nu0)**beta
    mbb = Single_Planck(nu, T)
    Md = Msun * 10**logM
    if frame == "rest":
        idx = 2.0
    elif frame == "obs":
        idx = 1.0
    else:
        raise ValueError("The frame '{0}' is not recognised!".format(frame))
    flux = (1 + z)**idx * Md * kappa * mbb / (DL*Mpc)**2 / mJy
    return flux
#Func_end


#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Oct. 30, 2016    #
#-------------------------------------#
def Line_Profile_Gaussian(nu, flux, nu0, FWHM, units="v", norm="peak"):
    """
    Calculate the flux density of the emission line with a Gaussian profile.

    Parameters
    ----------
    nu : float array
        The frequency of the spectrum.
    flux : float
        The integrated flux of the emission line if the norm parameter is "integrate".
        The peak flux density of the emission line if the norm parameter is "peak".
        Unit: erg/s/cm^2
    nu0 : float
        The central frequency of the emission line.
    FWHM : float
        The full width half maximum (FWHM) of the emission line.
    units : string, default: "v"
        The unit of the FWHM.
        "v": velocity (km/s);
        "lambda": wavelength (micron);
        "nu": frequency (Hz).
    norm : string, default: "peak"
        "peak": use the flux parameter as the peak flux density.
        "integrate": use the flux parameter as the integrated flux of the line.

    Returns
    -------
    fnu : float array
        The flux density of the spectrum.

    Notes
    -----
    None.
    """
    ls_km  = 2.99792458e5 #km/s
    ls_mic = 2.99792458e14 #micron/s
    #Convert the FWHM from velocity into the units of frequency.
    if units == "v":
        FWHM = FWHM / ls_km * nu0
    elif units == "lambda":
        FWHM = ls_mic / FWHM
    elif units == "nu":
        FWHM = FWHM
    else:
        raise ValueError("The units {0} is not recognised!".format(units))
    sigma = FWHM / 2.355
    #Convert the FWHM into the sigma of the Gaussian function.
    if norm == "peak":
        amplitude = flux
    elif norm == "integrate":
        amplitude = flux / (sigma * (2.0 * np.pi)**0.5) * 10**26 #unit: mJy
    fnu = amplitude * np.exp(-0.5 * ((nu - nu0) / sigma)**2.0)
    return fnu
#Func_end
