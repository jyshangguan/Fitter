import numpy as np

Msun = 1.9891e33 #unit: gram
Mpc = 3.08567758e24 #unit: cm
mJy = 1e26 #unit: erg/s/cm^2/Hz

waveLim = [1e-2, 1e3]
class BC03_Template(object):
    """
    This function call the interpolated BC03 template to generate the stellar
    emission SED with the given parameters.
    """

    def __init__(self, t):
        """
        Set the template for the function to use.

        Parameters
        ----------
        t : Template class
            The interpolated BC03 template.

        Returns
        -------
        None.
        """
        self.__template = t

    def __call__(self, logMs, age, DL, wave, z, frame="rest", waveLim=waveLim):
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
        z : float
            The redshift.
        frame : string
            "rest" for the rest frame SED and "obs" for the observed frame.
        waveLim : list
            The min and max of the wavelength covered by the template.

        Returns
        -------
        fnu : float array
            The flux density of the calculated SED with the unit erg/s/cm^2/Hz.

        Notes
        -----
        None.
        """
        flux = np.zeros_like(wave)
        fltr = (wave > waveLim[0]) & (wave < waveLim[1])
        if np.sum(fltr) == 0:
            return np.zeros_like(wave)
        flux[fltr] = self.__template(wave[fltr], [age])
        if frame == "rest":
            idx = 2.0
        elif frame == "obs":
            idx = 1.0
        else:
            raise ValueError("The frame '{0}' is not recognised!".format(frame))
        fnu = (1.0 + z)**idx * flux * 10**logMs / (4 * np.pi * (DL * Mpc)**2) * mJy
        return fnu

    def discretize_parameters(self, logMs, age):
        """
        Find the discretized parameters with the template according to input values.

        Parameters
        ----------
        logMs : float
            The log10 of stellar mass with the unit solar mass.
        age : float
            The age of the stellar population with the unit Gyr.

        Returns
        -------
        parDict : dict
            The dict of the parameters of the template w.r.t the input values.
        """
        age_d = self.__template.get_nearestParameters([age])[0]
        parDict = {
            "logMs": logMs,
            "age": age_d
        }
        return parDict

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

if __name__ == "__main__":
    import cPickle as pickle
    import matplotlib.pyplot as plt
    from sedfit.fitter.template import Template
    from sedfit.dir_list import template_path

    fp = open(template_path+"bc03_kdt.tmplt")
    tp_bc03 = pickle.load(fp)
    fp.close()
    bc03 = Template(**tp_bc03)
    wave = 10**np.linspace(-1, 3, 1000)
    BC03 = BC03_Template(bc03)
    print bc03.get_modelInfo()
    flux = BC03(9, 2.456, 50, wave, 0.001)
    print BC03.discretize_parameters(9, 2.456)
    plt.plot(wave, flux)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([5e-4, 1e1])
    plt.show()
