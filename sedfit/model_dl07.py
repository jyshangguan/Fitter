import numpy as np
import cPickle as pickle
from fitter.template import Template
from scipy.interpolate import splev


Msun = 1.9891e33 #unit: gram
Mpc = 3.08567758e24 #unit: cm
m_H = 1.6726219e-24 #unit: gram
template_dir = "/Users/jinyi/Work/PG_QSO/templates/"

fp = open("/Users/jinyi/Work/mcmc/Fitter/template/dl07_kdt.tmplt")
tp_dl07 = pickle.load(fp)
fp.close()
tdl07 = Template(**tp_dl07)
modelInfo = tdl07.get_modelInfo()
qpahList = modelInfo["qpah"]
mdust2mh = modelInfo["mdmh"]

def DL07(logumin, logumax, qpah, gamma, logMd, DL, wave, t=tdl07):
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
    gamma : float
        The fraction of the dust in the PDR.
    logMd : float
        The log10 of the dust mass in the unit of solar mass.
    DL : float
        The luminosity distance.
    wave : float array
        The wavelengths of the output flux.
    tmpl_dl07 : numpy.ndarray
        The template of DL07 model provided by user.

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
    pmin = [umin, umin, qpah]
    ppl  = [umin, umax, qpah]
    jnu_min = t(wave, pmin)
    jnu_pl  = t(wave, ppl)
    qpah_min = t.get_nearestParameters(pmin)[2]
    qpah_pl = t.get_nearestParameters(ppl)[2]
    if qpah_min != qpah_pl:
        raise RuntimeError("The DL07 model is inconsistent!")
    mdmh = mdust2mh[qpahList.index(qpah_min)]
    jnu = (1 - gamma) * jnu_min + gamma * jnu_pl
    flux = 10**logMd * Msun/m_H * jnu/(DL * Mpc)**2 / mdmh * 1e3 #unit: mJy
    return flux

'''
#DL07 model#
#----------#
try:
    dl07File = template_dir+"DL07spec/dl07.tmplt"
    fp = open(dl07File, "r")
    tmpl_dl07 = pickle.load(fp)
    fp.close()
    waveModel = tmpl_dl07[0]["wavesim"]
except:
    print("[model_functions]: Fail to import the DL07 template from: {0}".format(dl07File))
    tmpl_dl07 = None

uminList = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 1.00, 1.20,
        1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 8.00, 10.0, 12.0,
        15.0, 20.0, 25.0]
umaxList = [1e3, 1e4, 1e5, 1e6]
qpahList = [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58, 0.75, 1.49, 2.37, 0.10]
mdust2mh = [0.01, 0.01, 0.0101, 0.0102, 0.0102, 0.0103, 0.0104, 0.00343,
            0.00344, 0.00359, 0.00206]
qpahList = np.array(qpahList)
mdust2mh = np.array(mdust2mh)
srtIndex = np.argsort(qpahList)
qpahList = list(qpahList[srtIndex])
mdust2mh = list(mdust2mh[srtIndex])

try:
    dl07SplFile = template_dir+"DL07spec/dl07_spl.tmplt"
    fp = open(dl07SplFile, "r")
    tmpl_dl07_spl = pickle.load(fp)
    fp.close()
except:
    print("[model_functions]: Fail to import the DL07_spl template from: {0}".format(dl07SplFile))
    tmpl_dl07_spl = None

def DL07_Model_spl(umin, umax, qpah, gamma, logMd, DL, wave, tmpl_dl07=tmpl_dl07_spl):
    """
    This function generates the dust emission from the Draine & Li (2007) templates.

    Parameters
    ----------
    umin : float
        The minimum radiation field intensity.
    umax : float
        The maximum radiation field intensity.
    qpah : float
        The PAH fraction of the dust.
    gamma : float
        The fraction of the dust in the PDR.
    logMd : float
        The log10 of the dust mass in the unit of solar mass.
    DL : float
        The luminosity distance.
    wave : float array
        The wavelengths of the output flux.
    tmpl_dl07 : numpy.ndarray
        The template of DL07 model provided by user.

    Returns
    -------
    flux : float array
        The flux density of the model.

    Notes
    -----
    The function is tested with the DL07_Model function.
    """
    assert umin in uminList
    assert umax in umaxList
    assert qpah in qpahList
    index    = tmpl_dl07["index"]
    tckList  = tmpl_dl07["tck_list"]
    fltr_qpah = index["qpah"] == qpah
    fltr_umin = index["umin"] == umin
    fltr_min = fltr_qpah & fltr_umin & (index["umax"] == umin)
    fltr_pl  = fltr_qpah & fltr_umin & (index["umax"] == umax)
    mdmh = index[fltr_min][0]["mdmh"]
    index_min = index[fltr_min][0]["index"]
    index_pl  = index[fltr_pl][0]["index"]
    tck_min = tckList[index_min]
    tck_pl  = tckList[index_pl]
    jnu_min = splev(wave, tck_min)
    jnu_pl  = splev(wave, tck_pl)
    jnu = (1 - gamma) * jnu_min + gamma * jnu_pl
    flux = 10**logMd * Msun/m_H * jnu/(DL * Mpc)**2 / mdmh * 1e3 #unit: mJy
    return flux

def DL07_Model(umin, umax, qpah, gamma, logMd, DL, wave, tmpl_dl07=tmpl_dl07):
    """
    This function generates the dust emission template from Draine & Li (2007).

    Parameters
    ----------
    umin : float
        The minimum radiation field intensity.
    umax : float
        The maximum radiation field intensity.
    qpah : float
        The PAH fraction of the dust.
    gamma : float
        The fraction of the dust in the PDR.
    logMd : float
        The log10 of the dust mass in the unit of solar mass.
    DL : float
        The luminosity distance.
    wave : float array
        The wavelengths of the output flux.
    tmpl_dl07 : numpy.ndarray
        The template of DL07 model provided by user.

    Returns
    -------
    flux : float array
        The flux density of the model.

    Notes
    -----
    None.
    """
    chk_umin = umin in uminList
    chk_umax = umax in umaxList
    chk_qpah = qpah in qpahList
    if not chk_umin:
        raise ValueError("The umin={0} is invalid!".format(umin))
    if not chk_umax:
        raise ValueError("The umax={0} is invalid!".format(umax))
    if not chk_qpah:
        raise ValueError("The qpah={0} is invalid!".format(qpah))
    fltr_qpah = tmpl_dl07["qpah"] == qpah
    fltr_umin = tmpl_dl07["umin"] == umin
    fltr_min = fltr_qpah & fltr_umin & (tmpl_dl07["umax"] == umin)
    fltr_pl  = fltr_qpah & fltr_umin & (tmpl_dl07["umax"] == umax)
    jnu_min  = tmpl_dl07[fltr_min][0]["fluxsim"]
    jnu_pl  = tmpl_dl07[fltr_pl][0]["fluxsim"]
    jnu = (1 - gamma) * jnu_min + gamma * jnu_pl
    flux = 10**logMd * Msun/m_H * jnu/(DL * Mpc)**2 / mdust2mh[qpahList.index(qpah)] * 1e3 #unit: mJy
    if np.max( abs(wave - waveModel) ) != 0:
        raise ValueError("The input wavelength is incorrect!")
    return flux

fp = open("/Users/jinyi/Work/mcmc/Fitter/template/dl07_kdt.tmplt")
tp_dl07 = pickle.load(fp)
fp.close()
tdl07 = Template(**tp_dl07)
modelInfo = tdl07.get_modelInfo()
qpahList = modelInfo["qpah"]
mdust2mh = modelInfo["mdmh"]
def DL07_bak(umin, umax, qpah, gamma, logMd, DL, wave, t=tdl07):
    pmin = [umin, umin, qpah]
    ppl  = [umin, umax, qpah]
    jnu_min = t(wave, pmin)
    jnu_pl  = t(wave, ppl)
    qpah_min = t.get_nearestParameters(pmin)[2]
    qpah_pl = t.get_nearestParameters(ppl)[2]
    if qpah_min != qpah_pl:
        raise RuntimeError("The DL07 model is inconsistent!")
    mdmh = mdust2mh[qpahList.index(qpah_min)]
    jnu = (1 - gamma) * jnu_min + gamma * jnu_pl
    flux = 10**logMd * Msun/m_H * jnu/(DL * Mpc)**2 / mdmh * 1e3 #unit: mJy
    return flux
'''
