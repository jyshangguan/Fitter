import h5py
import numpy as np
import cPickle as pickle
import ndiminterpolation as ndip
from ..fitter.template import Template

pi = np.pi
Mpc = 3.08567758e24 #unit: cm
template_dir = "/Users/jinyi/Work/PG_QSO/templates/"

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, May. 3, 2016     #
#-------------------------------------#
#From: dev_CLUMPY_intp.ipynb
### CLUMPY template
try:
    #clumpyFile = template_dir+"clumpy_models_201410_tvavg.hdf5"
    clumpyFile = template_dir+"clumpy_fnu_norm.hdf5"
    h = h5py.File(clumpyFile,"r")
    theta = [np.unique(h[par][:]) for par in ("i","tv","q","N0","sig","Y","wave")]
    data = h["flux_tor"].value
    wave_tmpl = h["wave"].value
    ip = ndip.NdimInterpolation(data,theta)
    h.close()
except:
    print("[model_functions]: Fail to import the CLUMPY template from: {0}".format(clumpyFile))
    ip = None
waveLim = [1e-2, 1e3]
def CLUMPY_intp(logL, i, tv, q, N0, sigma, Y, wave, DL, z, frame="rest", t=ip, waveLim=waveLim):
    """
    This function provide the dust torus MIR flux with CLUMPY model.

    Parameters
    ----------
    logL : float
        The log of the torus luminosity, unit erg/s.
    i : float
        The inclination angle of the torus to the observer.
    tv : float
        The visual optical depth of individual clumps.
    q : float
        The radial distribution power law exponent of the dust clumps.
    N0 : float
        The total number of clumps along the radial equatorial ray.
    sigma : float
        The angular distribution with of the torus.
    Y : float
        The radial torus relative thickness, Y=Ro/Rd.
    wave : float array
        The wavelength at which we want to calculate the flux.
    DL : float
        The luminosity distance
    z : float
        The redshift.
    frame : string
        "rest" for the rest frame SED and "obs" for the observed frame.
    t : NdimInterpolation class
        The NdimInterpolation class obtained from Nikutta"s interpolation code.
    waveLim : list
        The min and max of the wavelength covered by the template.

    Returns
    -------
    flux : array of float
        The flux density (F_nu) from the model.

    Notes
    -----
    None.
    """
    vector = np.array([i, tv, q, N0, sigma, Y])
    if frame == "rest":
        idx = 2.0
    elif frame == "obs":
        idx = 1.0
    else:
        raise ValueError("The frame '{0}' is not recognised!".format(frame))
    f0 = (1 + z)**idx * 10**(logL+26) / (4 * pi * (DL * Mpc)**2.) #Convert to mJy unit
    fltr = (wave > waveLim[0]) & (wave < waveLim[1])
    flux = np.zeros_like(wave)
    flux[fltr] = f0 * t(vector, wave[fltr])
    return flux
#Func_end

'''
fp = open("/Users/jinyi/Work/mcmc/Fitter/template/clumpy_kdt.tmplt")
tp_clumpy = pickle.load(fp)
fp.close()
tclumpy = Template(**tp_clumpy)
def Clumpy(logL, i, tv, q, N0, sigma, Y, wave, DL, t=tclumpy):
    """
    The CLUMPY model generating the emission from the clumpy torus.

    Parameters
    ----------
    logL : float
        The log of the torus luminosity, unit erg/s.
    i : float
        The inclination angle of the torus to the observer.
    tv : float
        The visual optical depth of individual clumps.
    q : float
        The radial distribution power law exponent of the dust clumps.
    N0 : float
        The total number of clumps along the radial equatorial ray.
    sig : float
        The angular distribution with of the torus.
    Y : float
        The radial torus relative thickness, Y=Ro/Rd.
    wave : float array
        The wavelength at which we want to calculate the flux.
    DL : float
        The luminosity distance
    t : NdimInterpolation class
        The NdimInterpolation class obtained from Nikutta"s interpolation code.

    Returns
    -------
    flux : array of float
        The flux density (F_nu) from the model.

    Notes
    -----
    None.
    """
    par = [i, tv, q, N0, sigma, Y]
    f0 = 10**(logL+26) / (4 * np.pi * (DL * Mpc)**2.) #Convert to mJy unit
    flux = f0 * t(wave, par)
    return flux
'''
