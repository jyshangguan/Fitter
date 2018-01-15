import numpy as np
import cPickle as pickle
from ..fitter.template import Template
from scipy.interpolate import splev
from ..dir_list import template_path

Msun = 1.9891e33 #unit: gram
Mpc = 3.08567758e24 #unit: cm
m_H = 1.6726219e-24 #unit: gram

fp = open(template_path+"dl07_kdt_mw.tmplt")
tp_dl07 = pickle.load(fp)
fp.close()
tdl07 = Template(**tp_dl07)
modelInfo = tdl07.get_modelInfo()
qpahList = modelInfo["qpah"]
mdust2mh = modelInfo["mdmh"]
waveLim = [1.0, 1e4]

def DL07(logumin, logumax, qpah, loggamma, logMd, DL, z, wave, frame="rest", t=tdl07, waveLim=waveLim):
    """
    This function generates the dust emission template from Draine & Li (2007).

    Parameters
    ----------
    logumin : float
        The minimum radiation field intensity in log10.
    logumax : float
        The maximum radiation field intensity in log10.
    qpah : float
        The PAH fraction of the dust.
    loggamma : float
        The fraction of the dust in the PDR in log10.
    logMd : float
        The log10 of the dust mass in the unit of solar mass.
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
    umin = 10**logumin
    umax = 10**logumax
    gamma = 10**loggamma
    pmin = t.get_nearestParameters([umin, umin, qpah])
    qpah_min = pmin[2]
    ppl = [umin, umax, qpah_min] # In order to avoid inconsistency, the qpah of
                                 # the pl component is matched to that of the
                                 # min component.
    fltr = (wave > waveLim[0]) & (wave < waveLim[1])
    if np.sum(fltr) == 0:
        return np.zeros_like(wave)
    jnu_min = t(wave[fltr], pmin)
    jnu_pl  = t(wave[fltr], ppl)
    mdmh = mdust2mh[qpahList.index(qpah_min)]
    jnu = (1 - gamma) * jnu_min + gamma * jnu_pl
    if frame == "rest":
        idx = 2.0
    elif frame == "obs":
        idx = 1.0
    else:
        raise ValueError("The frame '{0}' is not recognised!".format(frame))
    flux = np.zeros_like(wave)
    flux[fltr] = (1 + z)**idx * 10**logMd * Msun/m_H * jnu/(DL * Mpc)**2 / mdmh * 1e3 #unit: mJy
    return flux

def DL07_PosPar(logumin, logumax, qpah, loggamma, logMd, t=tdl07):
    """
    To position the parameters in the parameter grid. If true, the function
    will return the paramteter grid found nearest from the input parameters
    using the KDTree template.

    Parameters
    ----------
    logumin : float
        The minimum radiation field intensity in log10.
    logumax : float
        The maximum radiation field intensity in log10.
    qpah : float
        The PAH fraction of the dust.
    loggamma : float
        The fraction of the dust in the PDR, in log10.
    logMd : float
        The log10 of the dust mass in the unit of solar mass.
    tmpl_dl07 : numpy.ndarray
        The template of DL07 model provided by user.

    Returns
    -------
    param : list
        The parameters of the template used for the input parameters.

    Notes
    -----
    None.
    """
    umin = 10**logumin
    umax = 10**logumax
    pmin = [umin, umin, qpah]
    ppl  = [umin, umax, qpah]
    parMin = t.get_nearestParameters(pmin)
    parPl  = t.get_nearestParameters(ppl)
    if parMin[2] != parPl[2]:
        raise RuntimeError("The DL07 model is inconsistent!")
    parDict = {
        "logumin": np.log10(parPl[0]),
        "logumax": np.log10(parPl[1]),
        "qpah": parPl[2],
        "loggamma": loggamma,
        "logMd": logMd
    }
    return parDict
