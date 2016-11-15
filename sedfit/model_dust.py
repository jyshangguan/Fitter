import numpy as np
import cPickle as pickle
import rel_Radiation_Model_Toolkit as rmt
from sedfit.fitter.template import Template

ls_mic = 2.99792458e14 #unit: micron/s
Mpc = 3.08567758e24 #unit: cm
Msun = 1.9891e33 #unit: gram

def Dust_Emission(T, Md, kappa, wave, DL):
    """
    Calculate the dust emission using the dust temperature, mass, opacity and
    the luminosity distance of the source.

    Parameters
    ----------
    T : float
        Temperature, unit: Kelvin.
    Md : float
        The dust mass, unit: Msun.
    kappa : float array
        The opacity array, unit: cm^2/g.
    wave : float array
        The wavelength array to caculate, unit: micron.
    DL : float
        The luminosity distance, unit: Mpc.

    Returns
    -------
    de : float array
        The dust emission SED, unit: mJy (check!!).

    Notes
    -----
    None.
    """
    nu = ls_mic / wave
    bb = rmt.Single_Planck(nu, T)
    de = (Md * Msun) * bb * kappa / (DL * Mpc)**2 * 1e26 #Unit: mJy
    return de

fp = open("/Users/jinyi/Work/mcmc/Fitter/template/dust_grain_kdt.tmplt", "r")
grainModel = pickle.load(fp)
fp.close()




import matplotlib.pyplot as plt
