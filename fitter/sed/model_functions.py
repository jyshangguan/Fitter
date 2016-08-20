import h5py
import copy
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import cPickle as pickle
import rel_SED_Toolkit as sedt
import rel_Radiation_Model_Toolkit as rmt
import ndiminterpolation as ndip
from scipy.interpolate import interp1d
from lmfit import minimize, Parameters, fit_report
from collections import OrderedDict

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, May. 3, 2016     #
#-------------------------------------#
#From: dev_CLUMPY_intp.ipynb
def Stellar_SED(logMs, age, zs, wave, band='h', zf_guess=1.0, spsmodel='bc03_ssp_z_0.02_chab.model'):
    '''
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
    band : str, default: 'h'
        The reference band used to calculate the mass-to-light ratio.
    zf_guess : float. zf_guess=1.0 by default.
        The initial guess to solve the zf that allowing the age between
        zs and zf is as required.
    spsmodel : string. spsmodel='bc03_ssp_z_0.02_chab.model' by default.
        The stellar population synthesis model that is used.

    Returns
    -------
    flux : array
        The sed flux of the bulge. In units mJy.

    Notes
    -----
    None.
    '''
    import ezgal #Import the package for stellar synthesis.
    from scipy.optimize import fsolve
    from scipy.interpolate import interp1d
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
        raise ValueError('The age is too large!')
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
    flux_rst = model.get_sed(age, age_units='gyrs', units='Fv') * 1e26 #In unit mJy.
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
    '''
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
    '''
    Ms = 10**logMs
    flux = Ms*flux_star_1Msun
    if len(wave) != len(flux):
        raise ValueError('The input wavelength is incorrect!')
    return flux
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, May. 3, 2016     #
#-------------------------------------#
#From: dev_CLUMPY_intp.ipynb
def CLUMPY_Torus_Model(TORUS_logsf,
                       TORUS_i,
                       TORUS_tv,
                       TORUS_q,
                       TORUS_N0,
                       TORUS_sig,
                       TORUS_Y,
                       wave,
                       TORUS_tmpl_ip):
    '''
    This function provide the dust torus MIR flux with CLUMPY model.

    Parameters
    ----------
    TORUS_logsf : float
        The scaling factor of the model to fit the data.
    TORUS_i : float
        The inclination angle of the torus to the observer.
    TORUS_tv : float
        The visual optical depth of individual clumps.
    TORUS_q : float
        The radial distribution power law exponent of the dust clumps.
    TORUS_N0 : float
        The total number of clumps along the radial equatorial ray.
    TORUS_sig : float
        The angular distribution with of the torus.
    TORUS_Y : float
        The radial torus relative thickness, Y=Ro/Rd.
    wave : float array
        The wavelength at which we want to calculate the flux.
    TORUS_tmpl_ip : NdimInterpolation class
        The NdimInterpolation class obtained from Nikutta's interpolation code.

    Returns
    -------
    flux : array of float
        The flux density (F_nu) from the model.

    Notes
    -----
    None.
    '''
    ls_mic = 2.99792458e14 #micron/s
    nu = ls_mic / wave
    vector = np.array([TORUS_i, TORUS_tv, TORUS_q, TORUS_N0, TORUS_sig, TORUS_Y])
    sed = TORUS_tmpl_ip(vector,wave)/nu * 1e10
    flux = (10**TORUS_logsf) * sed
    return flux
#Func_end

def Modified_BlackBody(logM, T, beta, wave, DL, kappa0=16.2, lambda0=140):
    '''
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

    '''
    ls_mic = 2.99792458e14 #micron/s
    nu = ls_mic / wave
    flux = rmt.Dust_Modified_BlackBody(nu, logM, DL, beta, T, kappa0, lambda0)
    return flux

def Power_Law(PL_alpha, PL_logsf, wave):
    '''
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
    '''
    ls_mic = 2.99792458e14 #micron/s
    nu = ls_mic / wave
    flux = rmt.Power_Law(nu, PL_alpha, 10**PL_logsf)
    return flux

#DL07 model
m_H = 1.6726219e-24 #unit: gram
Msun = 1.9891e33 #unit: gram
Mpc = 3.08567758e24 #unit: cm
uminList = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 1.00, 1.20,
        1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 8.00, 10.0, 12.0,
        15.0, 20.0, 25.0]
umaxList = [1e3, 1e4, 1e5, 1e6]
qpahList = [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58, 0.75, 1.49, 2.37, 0.10]
mdust2mh = [0.01, 0.01, 0.0101, 0.0102, 0.0102, 0.0103, 0.0104, 0.00343,
            0.00344, 0.00359, 0.00206]
def DL07_Model_Intp(umin, umax, qpah, gamma, logMd, tmpl_dl07, wave, DL):
    '''
    This function generates the dust emission template from Draine & Li (2007).
    The fluxes are interpolated at the given wavelength.

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
    tmpl_dl07 : numpy.ndarray
        The template of DL07 model.
    wave : float array
        The wavelengths of the output flux.
    DL : float
        The luminosity distance.

    Returns
    -------
    flux_model : float array
        The flux density of the model at the given wavelength.

    Notes
    -----
    None.
    '''
    chk_umin = umin in uminList
    chk_umax = umax in umaxList
    chk_qpah = qpah in qpahList
    if not chk_umin:
        raise ValueError('The umin={0} is invalid!'.format(umin))
    if not chk_umax:
        raise ValueError('The umax={0} is invalid!'.format(umax))
    if not chk_qpah:
        raise ValueError('The qpah={0} is invalid!'.format(qpah))
    fltr_qpah = tmpl_dl07['qpah'] == qpah
    fltr_umin = tmpl_dl07['umin'] == umin
    fltr_min = fltr_qpah & fltr_umin & (tmpl_dl07['umax'] == umin)
    fltr_pl  = fltr_qpah & fltr_umin & (tmpl_dl07['umax'] == umax)
    wave_min = tmpl_dl07[fltr_min]['wavesim'][0]
    jnu_min  = tmpl_dl07[fltr_min]['fluxsim'][0]
    wave_pl = tmpl_dl07[fltr_pl]['wavesim'][0]
    jnu_pl  = tmpl_dl07[fltr_pl]['fluxsim'][0]
    jnu = (1 - gamma) * jnu_min + gamma * jnu_pl
    flux = 10**logMd * Msun/m_H * jnu/(DL * Mpc)**2 / mdust2mh[qpahList.index(qpah)] * 1e3 #unit: mJy
    flux_model = interp1d(wave_pl, flux)(wave)
    return flux_model

def DL07_Model(umin, umax, qpah, gamma, logMd, tmpl_dl07, DL, wave):
    '''
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
    tmpl_dl07 : numpy.ndarray
        The template of DL07 model provided by user.
    DL : float
        The luminosity distance.

    Returns
    -------
    flux : float array
        The flux density of the model.

    Notes
    -----
    None.
    '''
    chk_umin = umin in uminList
    chk_umax = umax in umaxList
    chk_qpah = qpah in qpahList
    if not chk_umin:
        raise ValueError('The umin={0} is invalid!'.format(umin))
    if not chk_umax:
        raise ValueError('The umax={0} is invalid!'.format(umax))
    if not chk_qpah:
        raise ValueError('The qpah={0} is invalid!'.format(qpah))
    fltr_qpah = tmpl_dl07['qpah'] == qpah
    fltr_umin = tmpl_dl07['umin'] == umin
    fltr_min = fltr_qpah & fltr_umin & (tmpl_dl07['umax'] == umin)
    fltr_pl  = fltr_qpah & fltr_umin & (tmpl_dl07['umax'] == umax)
    jnu_min  = tmpl_dl07[fltr_min]['fluxsim'][0]
    jnu_pl  = tmpl_dl07[fltr_pl]['fluxsim'][0]
    jnu = (1 - gamma) * jnu_min + gamma * jnu_pl
    flux = 10**logMd * Msun/m_H * jnu/(DL * Mpc)**2 / mdust2mh[qpahList.index(qpah)] * 1e3 #unit: mJy
    if len(wave) != len(flux):
        raise ValueError('The input wavelength is incorrect!')
    return flux

#Dict of the supporting functions
funcLib = {
    'Stellar_SED':{
        'function': Stellar_SED,
        'x_name': 'wave',
        'param_fit': ['logMs', 'age'],
        'param_add': ['zs']
    },
    'Stellar_SED_scale': {
        'function': Stellar_SED_scale,
        'x_name': 'wave',
        'param_fit': ['logMs'],
        'param_add': ['flux_star_1Msun']
    },
    'CLUMPY_Torus_Model': {
        'function': CLUMPY_Torus_Model,
        'x_name': 'wave',
        'param_fit': ['TORUS_logsf', 'TORUS_i', 'TORUS_tv', 'TORUS_q', 'TORUS_N0', 'TORUS_sig', 'TORUS_Y'],
        'param_add': ['TORUS_tmpl_ip']
    },
    'DL07_Model': {
        'function': DL07_Model,
        'x_name': 'wave',
        'param_fit': ['umin', 'umax', 'qpah', 'gamma', 'logMd'],
        'param_add': ['tmpl_dl07', 'DL']
    },
    'DL07_Model_Intp': {
        'function': DL07_Model_Intp,
        'x_name': 'wave',
        'param_fit': ['umin', 'umax', 'qpah', 'gamma', 'logMd'],
        'param_add': ['tmpl_dl07', 'DL']
    },
    'Modified_BlackBody': {
        'function': Modified_BlackBody,
        'x_name': 'wave',
        'param_fit': ['logM', 'beta', 'T'],
        'param_add': ['DL']
    },
    'Power_Law': {
        'function': Power_Law,
        'x_name': 'wave',
        'param_fit': ['PL_alpha', 'PL_logsf'],
        'param_add': []
    }
}

#Input model dict
inputModelDict = OrderedDict(
    (
        ('Hot_Dust', {
                'function': 'Modified_BlackBody',
                'logM': {
                    'value': 1.7,
                    'range': [-5.0, 5.0],
                    'type': 'c',
                    'vary': True,
                },
                'beta': {
                    'value': 1.7,
                    'range': [1.5, 2.5],
                    'type': 'c',
                    'vary': True,
                },
                'T': {
                    'value': 641.8,
                    'range': [400.0, 1200.0],
                    'type': 'c',
                    'vary': True,
                }
            }
        ),
        ('Warm_Dust', {
                'function': 'Modified_BlackBody',
                'logM': {
                    'value': 4.5,
                    'range': [0.0, 6.0],
                    'type': 'c',
                    'vary': True,
                },
                'beta': {
                    'value': 2.3,
                    'range': [1.5, 2.5],
                    'type': 'c',
                    'vary': True,
                },
                'T': {
                    'value': 147.4,
                    'range': [60.0, 400.0],
                    'type': 'c',
                    'vary': True,
                }
            }
        ),
        ('Cold_Dust', {
                'function': 'Modified_BlackBody',
                'logM': {
                    'value': 8.8,
                    'range': [5.0, 12.0],
                    'type': 'c',
                    'vary': True,
                },
                'beta': {
                    'value': 2.0,
                    'range': [1.5, 2.5],
                    'type': 'c',
                    'vary': True,
                },
                'T': {
                    'value': 26.1,
                    'range': [5.0, 60.0],
                    'type': 'c',
                    'vary': True,
                }
            }
        ),
    )
)
"""
inputModelDict = OrderedDict(
    (
        ('Hot_Dust', {
                'function': 'Modified_BlackBody',
                'normalisation': ('w1', 'logM'),
                'logM': {
                    'value': -4.,
                    'range': [-10., 3.0],
                    'type': 'c',
                    'vary': True,
                },
                'beta': {
                    'value': 2.0,
                    'range': [1.5, 2.5],
                    'type': 'c',
                    'vary': False,
                },
                'T': {
                    'value': 600.,
                    'range': [500, 1300],
                    'type': 'c',
                    'vary': True,
                }
            }
        ),
        ('CLUMPY', {
                'function': 'CLUMPY_Torus_Model',
                'normalisation': ('w4', 'TORUS_logsf'),
                'TORUS_logsf': {
                    'value': 4.0,
                    'range': [-5.0, 15.0],
                    'type': 'c',
                    'vary': True,
                },
                'TORUS_i': {
                    'value': 13.847,
                    'range': [0.0, 90.0],
                    'type': 'c',
                    'vary': True,
                },
                'TORUS_tv': {
                    'value': 51.390,
                    'range': [10.0, 300.0],
                    'type': 'c',
                    'vary': True,
                },
                'TORUS_q': {
                    'value': 0.356,
                    'range': [0.0, 3.0],
                    'type': 'c',
                    'vary': True,
                },
                'TORUS_N0': {
                    'value': 4.000,
                    'range': [1.0, 15.0],
                    'type': 'c',
                    'vary': True,
                },
                'TORUS_sig': {
                    'value': 69.949,
                    'range': [15.0, 70.0],
                    'type': 'c',
                    'vary': True,
                },
                'TORUS_Y': {
                    'value': 21.889,
                    'range': [5.0, 100.0],
                    'type': 'c',
                    'vary': True,
                }
            }
        ),
        ('DL07', {
                'function': 'DL07_Model',
                'normalisation': ('PACS_100', 'logMd'),
                'umin': {
                    'value': 1.,
                    'range': uminList,
                    'type': 'd',
                    'vary': False,
                },
                'umax': {
                    'value': 1e6,
                    'range': umaxList,
                    'type': 'd',
                    'vary': False,
                },
                'qpah': {
                    'value': 4.58,
                    'range': qpahList,
                    'type': 'd',
                    'vary': False,
                },
                'gamma': {
                    'value': 0.01,
                    'range': [0.01, 0.99],
                    'type': 'c',
                    'vary': True,
                },
                'logMd': {
                    'value': 6.5,
                    'range': [0.0, 15.0],
                    'type': 'd',
                    'vary': True,
                }
            }
        ),
    )
)
"""
