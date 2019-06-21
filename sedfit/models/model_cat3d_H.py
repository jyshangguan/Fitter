import numpy as np
import cPickle as pickle
from ..fitter.template import Template
from scipy.interpolate import splev
from ..dir_list import template_path

Msun = 1.9891e33 #unit: gram
Mpc = 3.08567758e24 #unit: cm
m_H = 1.6726219e-24 #unit: gram
r0 = 1.1  # pc

fp = open(template_path+"Cat3d_H.tmplt")
tp_cat3d_H = pickle.load(fp)
fp.close()
tcat3d_H = Template(**tp_cat3d_H)
waveLim = [1.0, 1e4]


def Cat3d_H(a, h, N0, i, logL, DL, z, wave, frame="rest", t=tcat3d_H, waveLim=waveLim):
    """
    This function generates the modified CLUMPY torus radiation from Garcia-Gonzalez et al. 2017.

    Parameters
    ----------
    a : float
        The index of the radial dust cloud distribution power law.
    N0 : float
        The number of clouds along an equatorial line-of-sight.
    h : float
        Vertical Gaussian distribution dimensionless scale height.
    i : degree
        Inclination angle.
    logL : float
        UV luminosity erg/s in log.
    DL : float
        The luminosity distance.
    z : float
        The redshift.
    wave : float array
        The wavelengths of the output flux.
    frame : string
        "rest" for the rest frame SED and "obs" for the observed frame.
    t : Template object
        The template of DL07 model provided by user.
    waveLim : list
        The min and max of the wavelength covered by the template.

    Returns
    -------
    flux : float array
        The flux density of the model.

    Notes
    -----
    None.
    """
    fltr = (wave > waveLim[0]) & (wave < waveLim[1])
    if np.sum(fltr) == 0:
        return np.zeros_like(wave)
    para = [a, h, N0, i]
    if frame == "rest":
        idx = 2.0
    elif frame == "obs":
        idx = 1.0
    else:
        raise ValueError("The frame '{0}' is not recognised!".format(frame))
    f0 = (1 + z)**idx * 10**(logL - 46) * (r0 / DL * 1e-6)**2
    flux = np.zeros_like(wave)
    flux[fltr] = f0 * t(wave[fltr], para) * 1e29  # unit: mJy
    return flux


def Cat3d_H_PosPar(a, h, N0, i, logL, t=tcat3d_H):
    """

    To position the parameters in the parameter grid. If true, the function
    will return the paramteter grid found nearest from the input parameters
    using the KDTree template.

    Parameters
    ----------
    a : float
        The index of the radial dust cloud distribution power law.
    N0 : float
        The number of clouds along an equatorial line-of-sight.
    h : float
        Vertical Gaussian distribution dimensionless scale height.
    i : degree
        Inclination angle.
    logL : float
        Torus luminosity erg/s in log.
    t : Template object
        The template of torus model provided by user.
    logtau_torus : float
        Scale of 9.7 micron extinction strength

    Returns
    -------
    param : list
        The parameters of the template used for the input parameters.

    Notes
    -----
    None.
    """

    par = [a, h, N0, i]
    Par = t.get_nearestParameters(par)
    Pardict = {
        'a': Par[0],
        'h': Par[1],
        'N0': Par[2],
        'i': Par[3],
        'logL': logL
    }
    return Pardict
