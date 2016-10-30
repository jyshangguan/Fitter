import h5py
import copy
import numpy as np
import cPickle as pickle
import rel_Radiation_Model_Toolkit as rmt
import ndiminterpolation as ndip
from scipy.interpolate import interp1d, splrep, splev
from collections import OrderedDict
from fitter.template import Template

template_dir = "/Users/jinyi/Work/PG_QSO/templates/"

ls_mic = 2.99792458e14 #unit: micron/s
m_H = 1.6726219e-24 #unit: gram
Msun = 1.9891e33 #unit: gram
Mpc = 3.08567758e24 #unit: cm
mJy = 1e26 #unit: erg/s/cm^2/Hz

fp = open("/Users/jinyi/Work/mcmc/Fitter/template/bc03_kdt.tmplt")
tp_bc03 = pickle.load(fp)
fp.close()
bc03 = Template(**tp_bc03)
def BC03(logMs, age, DL, wave, t=bc03):
    """
    This function call the interpolated BC03 template to generate the stellar
    emission SED with the given parameters.

    Parameters
    ----------
    logMs : float
        The log10 of stellar mass with the unit solar mass.
    age : float
        The age of the stellar population with the unit Gyr.
    DL : float
        The luminosity distance with the unit Mpc.
    wave : float array
        The wavelength of the SED.
    t : Template class
        The interpolated BC03 template.

    Returns
    -------
    fnu : float array
        The flux density of the calculated SED with the unit erg/s/cm^2/Hz.

    Notes
    -----
    None.
    """
    flux = t(wave, [age])
    fnu = flux * 10**logMs / (4 * np.pi * (DL * Mpc)**2) * mJy
    return fnu

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, May. 3, 2016     #
#-------------------------------------#
#From: dev_CLUMPY_intp.ipynb
def Stellar_SED(logMs, age, zs, wave, band="h", zf_guess=1.0, spsmodel="bc03_ssp_z_0.02_chab.model"):
    """
    This function obtain the galaxy stellar SED given the stellar mass, age and redshift. The
    default model is Bruzual & Charlot (2003) with solar metallicity and Chabrier IMF. The stellar
    synthesis models are organised by the module EzGal (http://www.baryons.org/ezgal/).

    Parameters
    ----------
    logMs : float
        The stellar mass in log10 of solar unit.
    age : float
        The age of the galaxy, in the unit of Gyr.
    zs : float
        The redshift of the source.
    wave : array
        The sed wavelength corresponding to the sedflux. In units micron.
    band : str, default: "h"
        The reference band used to calculate the mass-to-light ratio.
    zf_guess : float. zf_guess=1.0 by default.
        The initial guess to solve the zf that allowing the age between
        zs and zf is as required.
    spsmodel : string. spsmodel="bc03_ssp_z_0.02_chab.model" by default.
        The stellar population synthesis model that is used.

    Returns
    -------
    flux : array
        The sed flux of the bulge. In units mJy.

    Notes
    -----
    None.
    """
    import ezgal #Import the package for stellar synthesis.
    from scipy.optimize import fsolve
    ls_mic = 2.99792458e14 #micron/s
    model = ezgal.model(spsmodel) #Choose a stellar population synthesis model.
    model.set_cosmology(Om=0.308, Ol=0.692, h=0.678)

    func_age = lambda zf, zs, age: age - model.get_age(zf, zs) #To solve the formation redshift given the
                                                               #redshift of the source and the stellar age.
    func_MF = lambda Msun, Mstar, m2l: Msun - 2.5*np.log10(Mstar/m2l) #Calculate the absolute magnitude of
                                                                      #the galaxy. Msun is the absolute mag
                                                                      #of the sun. Mstar is the mass of the
                                                                      #star. m2l is the mass to light ratio.
    func_flux = lambda f0, MF, mu: f0 * 10**(-0.4*(MF + mu)) #Calculate the flux density of the galaxy. f0
                                                             #is the zero point. MF is the absolute magnitude
                                                             #of the galaxy at certain band. mu is the distance
                                                             #module.
    Ms = 10**logMs #Calculate the stellar mass.
    age_up = model.get_age(1500., zs)
    if age > age_up:
        raise ValueError("The age is too large!")
    zf = fsolve(func_age, zf_guess, args=(zs, age)) #Given the source redshift and the age, calculate the redshift
                                                    #for the star formation.
    Msun_H = model.get_solar_rest_mags(nzs=1, filters=band, ab=True) #The absolute magnitude of the Sun in given band.
    m2l = model.get_rest_ml_ratios(zf, band, zs) #Calculate the mass-to-light ratio.
    M_H = func_MF(Msun_H, Ms, m2l) #The absolute magnitude of the galaxy in given band.
    #Calculate the flux at given band for comparison.
    f0 = 3.631e6 #Zero point of AB magnitude, in unit of mJy.
    mu = model.get_distance_moduli(zs) #The distance module
    flux_H = func_flux(f0, M_H, mu)
    wave_H = 1.6448 #Pivot wavelength of given band, in unit of micron.
    #Obtain the SED
    wave_rst = model.ls / 1e4 #In unit micron.
    flux_rst = model.get_sed(age, age_units="gyrs", units="Fv") * 1e26 #In unit mJy.
    wave_ext = np.linspace(200, 1000, 30)
    flux_ext = np.zeros(30)
    wave_extd = np.concatenate([wave_rst, wave_ext])
    flux_extd = np.concatenate([flux_rst, flux_ext])
    #Normalize the SED at the given band.
    #The normalization provided by EzGal is not well understood, so I do not use it.
    f_int = interp1d(wave_extd, flux_extd)
    f_H = f_int(wave_H)
    flux = flux_extd * flux_H/f_H
    sedflux = f_int(wave) * flux_H/f_H
    #return sedflux, wave_extd, flux_extd, wave_H, flux_H #For debug
    return sedflux
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, May. 3, 2016     #
#-------------------------------------#
#From: dev_CLUMPY_intp.ipynb
def Stellar_SED_scale(logMs, flux_star_1Msun, wave):
    """
    This function scales the stellar SED to obtain the best-fit stellar mass.
    The input SED flux should be normalised to 1 solar mass.

    Parameters
    ----------
    logMs : float
        The log stellar mass in unit solar mass
    flux_star_1Msun : float array
        The flux of stellar SED model. It is normalized to one solar mass.

    Returns
    -------
    flux : float array

    Notes
    ----
    None.
    """
    Ms = 10**logMs
    flux = Ms*flux_star_1Msun
    if len(wave) != len(flux):
        raise ValueError("The input wavelength is incorrect!")
    return flux
#Func_end

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
except:
    print("[model_functions]: Fail to import the CLUMPY template from: {0}".format(clumpyFile))
    ip = None
def CLUMPY_intp(logL, i, tv, q, N0, sigma, Y, wave, DL, t=ip):
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
    vector = np.array([i, tv, q, N0, sigma, Y])
    f0 = 10**(logL+26) / (4 * np.pi * (DL * Mpc)**2.) #Convert to mJy unit
    flux = f0 * t(vector, wave)
    return flux
#Func_end

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

def Modified_BlackBody(logM, T, beta, wave, DL, kappa0=16.2, lambda0=140):
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
    ls_mic = 2.99792458e14 #micron/s
    nu = ls_mic / wave
    flux = rmt.Dust_Modified_BlackBody(nu, logM, DL, beta, T, kappa0, lambda0)
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
    ls_mic = 2.99792458e14 #micron/s
    nu = ls_mic / wave
    flux = rmt.Power_Law(nu, PL_alpha, 10**PL_logsf)
    return flux

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

def DL07(logumin, logumax, qpah, gamma, logMd, DL, wave, t=tdl07):
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

def Linear(a, b, x):
    return a * x + b

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

#Dict of the supporting functions
funcLib = {
    "Linear":{
        "function": Linear,
        "x_name": "x",
        "param_fit": ["a", "b"],
        "param_add": []
    },
    "BC03":{
        "function": BC03,
        "x_name": "wave",
        "param_fit": ["logMs", "age"],
        "param_add": ["DL", "t"],
    },
    "Stellar_SED":{
        "function": Stellar_SED,
        "x_name": "wave",
        "param_fit": ["logMs", "age"],
        "param_add": ["zs"]
    },
    "Stellar_SED_scale": {
        "function": Stellar_SED_scale,
        "x_name": "wave",
        "param_fit": ["logMs"],
        "param_add": ["flux_star_1Msun"]
    },
    "CLUMPY_intp": {
        "function": CLUMPY_intp,
        "x_name": "wave",
        "param_fit": ["logL", "i", "tv", "q", "N0", "sigma", "Y"],
        "param_add": ["DL", "t"]
    },
    "Clumpy": {
        "function": Clumpy,
        "x_name": "wave",
        "param_fit": ["logL", "i", "tv", "q", "N0", "sigma", "Y"],
        "param_add": ["DL", "t"]
    },
    "DL07_Model": {
        "function": DL07_Model,
        "x_name": "wave",
        "param_fit": ["umin", "umax", "qpah", "gamma", "logMd"],
        "param_add": ["tmpl_dl07", "DL"]
    },
    "DL07": {
        "function": DL07,
        "x_name": "wave",
        "param_fit": ["logumin", "logumax", "qpah", "gamma", "logMd"],
        "param_add": ["t", "DL"]
    },
    "Modified_BlackBody": {
        "function": Modified_BlackBody,
        "x_name": "wave",
        "param_fit": ["logM", "beta", "T"],
        "param_add": ["DL"]
    },
    "Power_Law": {
        "function": Power_Law,
        "x_name": "wave",
        "param_fit": ["PL_alpha", "PL_logsf"],
        "param_add": []
    },
    "Line_Gaussian_L": {
        "function": Line_Gaussian_L,
        "x_name": "wavelength",
        "param_fit": ["logLum", "lambda0", "FWHM"],
        "param_add": ["DL"]
    }
}
