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
#print grainModel["readMe"]
silDict = grainModel["Silicate"]
graDict = grainModel["Graphite"]
tSil = Template(**silDict)
tGra = Template(**graDict)

def Torus_Emission(typeSil, sizeSil, T1Sil, T2Sil, logM1Sil, logM2Sil,
                   typeGra, sizeGra, T1Gra, T2Gra, logM1Gra, logM2Gra,
                   wave, DL, TemplateSil=tSil, TemplateGra=tGra):
    """
    Calculate the emission of the dust torus using the dust opacity and assuming
    it is optical thin situation. In detail, the torus emission model assumes that
    the dust torus consists silicate and graphite dust. Moreover, each type of
    dusts have two average temperature.
    """
    #Calculate the opacity curve
    parSil   = [typeSil, sizeSil]
    parGra   = [typeGra, sizeGra]
    kappaSil = TemplateSil(wave, parSil)
    kappaGra = TemplateGra(wave, parGra)
    #Calculate the dust emission SEDs
    M1Sil = 10**logM1Sil
    M2Sil = 10**logM2Sil
    M1Gra = 10**logM1Gra
    M2Gra = 10**logM2Gra
    de1Sil   = Dust_Emission(T1Sil, M1Sil, kappaSil, wave, DL)
    de2Sil   = Dust_Emission(T2Sil, M2Sil, kappaSil, wave, DL)
    de1Gra   = Dust_Emission(T1Gra, M1Gra, kappaGra, wave, DL)
    de2Gra   = Dust_Emission(T2Gra, M2Gra, kappaGra, wave, DL)
    deTorus  = de1Sil + de2Sil + de1Gra + de2Gra
    return deTorus

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    typeSil = 0
    typeGra = 0
    sizeSil = 0.5
    sizeGra = 0.5
    T1Sil   = 1000.0
    T2Sil   = 300.0
    T1Gra   = 500.0
    T2Gra   = 200.0
    M1Sil   = 1e3
    M2Sil   = 1e5
    M1Gra   = 1e5
    M2Gra   = 1e7
    wave = 10**np.linspace(0, 3, 1000)
    DL = 500.0
    deTorus = Torus_Emission(typeSil, sizeSil, T1Sil, T2Sil, M1Sil, M2Sil,
                             typeGra, sizeGra, T1Gra, T2Gra, M1Gra, M2Gra,
                             wave, DL)
    parSil   = [typeSil, sizeSil]
    parGra   = [typeGra, sizeGra]
    kappaSil = tSil(wave, parSil)
    kappaGra = tGra(wave, parGra)
    de1Sil = Dust_Emission(T1Sil, M1Sil, kappaSil, wave, DL)
    de2Sil = Dust_Emission(T2Sil, M2Sil, kappaSil, wave, DL)
    de1Gra = Dust_Emission(T1Gra, M1Gra, kappaGra, wave, DL)
    de2Gra = Dust_Emission(T2Gra, M2Gra, kappaGra, wave, DL)
    plt.plot(wave, deTorus, color="k", linewidth=1.5)
    plt.plot(wave, de1Sil, color="r", linestyle="--")
    plt.plot(wave, de2Sil, color="r", linestyle=":")
    plt.plot(wave, de1Gra, color="b", linestyle="--")
    plt.plot(wave, de2Gra, color="b", linestyle=":")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([1e4, 1e6])
    plt.show()
