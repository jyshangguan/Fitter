import ezgal
import numpy as np
from sedfit.fitter.template import Template
from sklearn.neighbors import KDTree
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
import cPickle as pickle
ls_mic = 2.99792458e14 #micron/s
ls_aa  = 2.99792458e18 #aa/s
Mpc    = 3.08567758e24 #cm
Lsun   = 3.828e33 #erg/s
mJy    = 1e26 #mJy

def Stellar_SED(logMs, age, wave, zs=0.01, band="h", zf_guess=1.0, spsmodel="bc03_ssp_z_0.02_chab.model"):
    """
    This function obtain the galaxy stellar SED given the stellar mass and age.
    The default model is Bruzual & Charlot (2003) with solar metallicity and
    Chabrier IMF. The stellar synthesis models are organised by the module EzGal
    (http://www.baryons.org/ezgal/). The function is consistent with BC03 templates
    (tested by SGJY in Oct. 6, 2016).

    Parameters
    ----------
    logMs : float
        The stellar mass in log10 of solar unit.
    age : float
        The age of the galaxy, in the unit of Gyr.
    wave : array
        The sed wavelength corresponding to the sedflux. In units micron.
    zs : float, default: 0.01
        The redshift of the source. It does not matter which value zs is, since
        the template is in the rest frame and the unit is erg/s/Hz.
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
        The sed flux of the stellar emission. In units erg/s/Hz.

    Notes
    -----
    None.
    """
    import ezgal #Import the package for stellar synthesis.
    from scipy.optimize import fsolve
    from scipy.interpolate import interp1d
    ls_mic = 2.99792458e14 #micron/s
    mJy    = 1e26 #mJy
    Mpc    = 3.08567758e24 #cm
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
    flux_H = func_flux(f0, M_H, mu) #In unit mJy.
    wave_H = 1.6448 #Pivot wavelength of given band, in unit of micron.
    #Obtain the SED
    wave_rst = model.ls / 1e4 #In unit micron.
    flux_rst = model.get_sed(age, age_units="gyrs", units="Fv") #The unit is not important
                                                                #since it will be normalized.
    wave_ext = np.linspace(200, 1000, 30)
    flux_ext = np.zeros(30)
    wave_extd = np.concatenate([wave_rst, wave_ext])
    flux_extd = np.concatenate([flux_rst, flux_ext])
    #Normalize the SED at the given band.
    #The normalization provided by EzGal is not well understood, so I do not use it.
    f_int = interp1d(wave_extd, flux_extd)
    f_H = f_int(wave_H)
    flux = flux_extd * flux_H/f_H #In unit mJy.
    dm = model.get_distance_moduli(zs, nfilters=1)
    DL = 10**(1. + dm / 5.) / 1e6 * Mpc #In unit cm
    sedflux = f_int(wave) * flux_H/f_H * (4 * np.pi * DL**2) / mJy #In unit: erg/s/Hz
    #return sedflux, wave_extd, flux_extd, wave_H, flux_H #For debug
    return sedflux

#Generate the templates of stellar emission templates
logMs = 0.0
#ageList = 10**np.linspace(8.5, 10., 40)
ageList = np.array([0.5, 1.0, 3.0, 5.0, 9.0])
nAge = len(ageList)
wave = 10**np.linspace(-2, 3, 3000)
fluxList = []
for loop in range(nAge):
    fluxList.append(Stellar_SED(logMs, ageList[loop], wave))
fluxList = np.array(fluxList)
print(fluxList.shape)

#Interpolate with KD tree and spline interpolation
XList = []
tckList = []
for loop in range(nAge):
    age  = ageList[loop]
    flux = fluxList[loop, :]
    tck = splrep(wave, flux)
    tckList.append(tck)
    XList.append([age])
kdt = KDTree(XList)
print("Interpolation finishes!")
modelInfo = {
    "age": ageList,
}
parFormat = ["age"]
readMe = '''
The stellar emission templates are generated with EzGal.
'''
templateDict = {
    "tckList": tckList,
    "kdTree": kdt,
    "parList": XList,
    "modelInfo": modelInfo,
    "parFormat": parFormat,
    "readMe": readMe
}
t = Template(tckList=tckList, kdTree=kdt, parList=XList, modelInfo=modelInfo,
             parFormat=parFormat, readMe=readMe)
t = Template(**templateDict)
fp = open("bc03_kdt.tmplt", "w")
pickle.dump(templateDict, fp)
fp.close()

#Test
cl = ['r', 'g', 'b', 'y', 'c']
for loop in range(nAge):
    pars = [ageList[loop]]
    f = t(wave, pars)
    flux = fluxList[loop]
    plt.plot(wave, f, color=cl[loop], linestyle="--")
    plt.plot(wave, flux, color=cl[loop], linestyle=":")
plt.ylim([1e14, 1e20])
plt.xscale("log")
plt.yscale("log")
plt.show()
