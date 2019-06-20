from __future__ import print_function
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
import cPickle as pickle
ls_mic = 2.99792458e14 #micron/s

def BandAverage(datawave, dataflux, bandwave, bandrsr):
    """
    This code calculate the band average of the given spectrum with the given
    bandpass information.

    Parameters
    ----------
    datawave : float array
        The spectral wavelength.
    dataflux : float array
        The spectral flux density.
    bandwave : float array
        The wavelength of the response curve.
    bandrsr : float array
        The relative spectral response curve.

    Returns
    -------
    ff : float
        The average flux density of the filter.

    Notes
    -----
    None.
    """
    if datawave[-1] <= datawave[0]:
        raise ValueError("The data wavelength is incorrect in sequence!")
    if bandwave[-1] <= bandwave[0]:
        raise ValueError("The filter wavelength is incorrect in sequence!")
    filtering = interp1d(bandwave, bandrsr)
    fltr = (datawave > bandwave[0]) & (datawave < bandwave[-1])
    dw = datawave[fltr] #Make the data wavelength not exceeding the provided band wavelength
    df = dataflux[fltr]
    br = filtering(dw) #Calculate the bandrsr at the same wavelength of the data
    signal = np.trapz(br/dw*df, x=dw)
    norm   = np.trapz(br/dw, x=dw)
    ff = signal/norm
    return ff

def BandFunc_intp(kwargs):
    """
    This function calculate interpolate the input spectrum
    to obtain the flux processed by the bandpass.

    Parameters
    ----------
    kwargs : dict
        The dict of all the necessary parameters.

    Returns
    -------
    fluxFltr : float
        The flux density obtained by the band.

    Notes
    -----
    None.
    """
    wavelength = kwargs['wavelength']
    flux = kwargs['flux']
    bandCenter = kwargs['band_center']
    fluxFltr = interp1d(wavelength, flux)(bandCenter)
    return fluxFltr

def BandFunc_Herschel(kwargs):
    """
    This function calculate the flux density one of the Herschel band obtains.
    Reference: Section 5.2.4, SPIRE Handbook.

    Parameters
    ----------
    kwargs : dict
        The dict of all the necessary parameters.

    Returns
    -------
    fluxFltr : float
        The flux density obtained by the band.

    Notes
    -----
    None.
    """
    K4pDict = { #The K4p parameter I calculated myself.
        'PACS_70'  : 0.994981,
        'PACS_100' : 0.999526,
        'PACS_160' : 1.004355,
        'SPIRE_250': 1.010159,
        'SPIRE_350': 1.009473,
        'SPIRE_500': 1.005581,
    }
    wavelength = kwargs['wavelength']
    rsrList = kwargs['rsr_list']
    flux = kwargs['flux']
    bandName = kwargs['band_name']
    Sbar = np.trapz(rsrList*flux, wavelength) / np.trapz(rsrList, wavelength)
    fluxFltr = K4pDict[bandName] * Sbar
    return fluxFltr

def K_MonP(wave0, waveList, rsrList, alpha=-1):
    '''
    This is the correction factor from band average flux density to
    the monochromatic flux density with assumed power-law function.

    Parameters
    ----------
    wave0 : float
        The effective wavelength of the monochromatic flux density.
    waveList: float
        The wavelength of the relative spectral response curve.
    rsrList : float array
        The relative spectral response.
    alpha : float
        The power-law index of the assumed power-law spectrum; default: -1.

    Returns
    -------
    kmonp : float
        The monochromatic correction factor.

    Notes
    -----
    None.
    '''
    freq = ls_mic / waveList
    nu0 = ls_mic / wave0
    spc = freq**alpha
    k1 = nu0**alpha * np.trapz(rsrList, freq)
    k2 = np.trapz(rsrList*spc, freq)
    kmonp = k1 / k2
    return kmonp

class BandPass(object):
    """
    A class to represent one filter bandpass of a instrument.

    It contains the bandpass information and can convert the spectra
    into the band flux density. The user should understand how the
    instrument works to obtain the measured flux density, in order to
    correctly use the relative spectral response (rsr) curve and the
    band wavelength.

    Parameters
    ----------
    waveList : float array
        The wavelength of the relative system response curve. The provided
        wavelength should be always in the observed frame.
    rsrList : float array
        The relative system response curve.
    bandCenter : float
        The quoted wavelength for the band in the observed frame.
    bandType : string
        The type of band functions.
    bandName : string
        The name of the band.
    redshift : float
        The redshift of the source. Since the SED model is in the rest frame, we
        need to move the wavelength of the filter rsr curve into the rest frame.
    silent : bool
        Stop printing information if True, by default.
    """
    def __init__(self, waveList=None, rsrList=None, bandCenter=None, bandType="mean", bandName='None', redshift=0, silent=True):
        self.__bandType = bandType
        self.redshift = redshift
        if not bandType == "none":
            assert not waveList is None
            assert not rsrList is None
            #->Move the wavelength of the bandpass to match the rest-frame SED.
            waveList = np.array(waveList) / (1 + redshift)
            waveMin = waveList[0]
            waveMax = waveList[-1]
            if waveMin >= waveMax: #-> The waveList should be from small to large.
                raise ValueError("The waveList sequence is incorrect!")
            if len(waveList) == len(rsrList):
                self.__waveList = waveList
                self.__rsrList = rsrList
                self._bandName = bandName
            else:
                raise ValueError("The inputs are not matched!")
            self.__filtertck = splrep(waveList, rsrList)
            if bandCenter is None: #The bandCenter is not specified, the effective wavelength will
                                   #be used (Eq. A21, Bessell&Murphy 2012), assuming fnu~const.
                bandCenter = (1 + redshift) * np.trapz(rsrList, waveList)/np.trapz(rsrList/waveList, waveList)
                if not silent:
                    print("Band {0} center wavelength ({1}) is calculated!".format(bandName, bandCenter))
            self.__bandCenter = bandCenter
            self.__bandCenter_rest = bandCenter / (1 + redshift)
            if bandType == "mono":
                self.k4p = K_MonP(self.__bandCenter_rest, waveList, rsrList, alpha=-1)
                if not silent:
                    print("Band {0} calculates the monochromatic flux density!".format(bandName))
            elif bandType == "mean":
                if not silent:
                    print("Band {0} calculates the averaged flux density!".format(bandName))
            else:
                raise ValueError("The input bandType ({0}) is incorrect!".format(bandType))
        else:
            assert waveList is None
            assert rsrList is None
            assert not bandCenter is None
            self.__waveList = None
            self.__rsrList = None
            self.__bandCenter = bandCenter
            self.__bandCenter_rest = bandCenter / (1 + redshift)
            self.__bandName = bandName
            if not silent:
                print("Band {0} ({1}) does not have bandpass.".format(bandName, bandCenter))

    def get_bandCenter(self):
        return self.__bandCenter

    def get_bandCenter_rest(self):
        return self.__bandCenter_rest

    def get_bandpass(self):
        bandInfo = {
            "wave_list": self.__waveList,
            "rsr_list": self.__rsrList,
            "band_name": self._bandName,
            "band_type": self.__bandType,
            "wave_center": self.__bandCenter,
            "wave_center_rest": self.__bandCenter_rest,
            "redshift": self.redshift
        }
        return bandInfo

    def BandFunc_mono(self, wavelength, flux):
        """
        Calculate the monochromatic flux density with the given data. The function
        applies for the bolometers used by IR satellites.
        To use this function, the relative spectral response should be for the
        bolometer (energy/photon) instead of the CCD (electron/photon).
        Reference: Section 5.2.4, SPIRE Handbook.

        Parameters
        ----------
        wavelength : float array
            The wavelength of the spectrum and the relative spectral response.
        flux : float array
            The flux of the spectrum.

        Returns
        -------
        fluxFltr : float
            The monochromatic flux density calculated from the filter rsr.

        Notes
        -----
        To convert the relative spectral response from electron/photon to
        electron/energy is simply:
            S(electron/energy) = S(electron/photon) / nu
        where nu is the corresponding frequency (Bessell & Murphy 2012).
        """
        waveMin = self.__waveList[0]
        waveMax = self.__waveList[-1]
        fltr = (wavelength > waveMin) & (wavelength < waveMax)
        if np.sum(fltr) == 0:
            raise ValueError("The wavelength is not overlapped with the filter!")
        wavelength = wavelength[fltr]
        freq = ls_mic / wavelength
        flux = flux[fltr]
        rsrList = splev(wavelength, self.__filtertck)
        Sbar = np.trapz(rsrList*flux, freq) / np.trapz(rsrList, freq)
        fluxFltr = self.k4p * Sbar
        return fluxFltr

    def BandFunc_mean(self, wavelength, flux):
        """
        Calculate the band averaged flux density with the given data.
        By default, the rsr is photon response and the band flux is defined as
        eq. A12 (Bessell&Murphy 2012). The rsr is for the CCD detector instead
        of bolometers.

        Parameters
        ----------
        wavelength : float array
            The wavelength of the spectrum and the relative spectral response.
        flux : float array
            The flux of the spectrum.

        Returns
        -------
        fluxFltr : float
            The monochromatic flux density calculated from the filter rsr.

        Notes
        -----
        None.
        """
        waveMin = self.__waveList[0]
        waveMax = self.__waveList[-1]
        fltr = (wavelength > waveMin) & (wavelength < waveMax)
        if np.sum(fltr) == 0:
            raise ValueError("The wavelength is not overlapped with the filter!")
        wavelength = wavelength[fltr]
        flux = flux[fltr]
        rsrList = splev(wavelength, self.__filtertck)
        signal = np.trapz(rsrList/wavelength*flux, x=wavelength)
        norm   = np.trapz(rsrList/wavelength, x=wavelength)
        fluxFltr = signal/norm
        return fluxFltr

    def BandFunc_none(self, wavelength, flux):
        """
        This band function provides the flux without the bandpass. The flux density
        of the model SED at the wavelength closest to the band center is returned.

        Parameters
        ----------
        wavelength : float array
            The wavelength of the spectrum and the relative spectral response.
        flux : float array
            The flux of the spectrum.

        Returns
        -------
        fluxFltr : float
            The flux density at the wavelength closest to the band center.

        Notes
        -----
        None.
        """
        bandCenter = self.__bandCenter_rest
        wave_fdev = np.abs((wavelength - bandCenter) / bandCenter)
        idx = np.argmin(wave_fdev)
        if wave_fdev[idx] > 0.05:
            print("[BandPass warning]: The wavelength deviation at {0} ({1}) is large!".format(self._bandName, bandCenter))
        fluxFltr = flux[idx]
        return fluxFltr

    def filtering(self, wavelength, flux):
        """
        Calculate the flux density of the input spectrum filtered by the bandpass.

        Parameters
        ----------
        wavelength : float array
            The wavelength of the input spectrum.
        flux : float array
            The flux of the input spectrum.

        Returns
        -------
        fluxFltr : float
            The flux density of the spectrum after filtered by the bandpass.

        Notes
        -----
        None.
        """
        bandCenter = self.__bandCenter_rest
        wvMin = wavelength[0]
        wvMax = wavelength[-1]
        if( (bandCenter <= wvMin) or (bandCenter >= wvMax) ):
            raise ValueError("The band center '{0}' is out of the wavelength range '[{1}, {2}]!".format(bandCenter, wvMin, wvMax))
        bandType = self.__bandType
        if bandType == "mean": #By default, the rsr is photon response and the band flux
                             #is defined as eq. A12 (Bessell&Murphy 2012).
            fluxFltr = self.BandFunc_mean(wavelength, flux)
        elif bandType == "mono": #Use the user specified function to get the filtered flux.
            fluxFltr = self.BandFunc_mono(wavelength, flux)
        else: #If there is no filter, direct calculate the flux density.
            fluxFltr = self.BandFunc_none(wavelength, flux)
        return fluxFltr

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

#Default filters
filterDict = {
    "SDSS_u": 0.36069,
    "SDSS_g": 0.46729,
    "SDSS_r": 0.61415,
    "SDSS_i": 0.74585,
    "SDSS_z": 0.89247,
    "UKIDSS_Y": 1.03050,
    "UKIDSS_J": 1.24830,
    "UKIDSS_H": 1.63130,
    "UKIDSS_K": 2.20100,
    "2MASS_J": 1.235,
    "2MASS_H": 1.662,
    "2MASS_Ks": 2.159,
    "WISE_w1": 3.353,
    "WISE_w2": 4.603,
    "WISE_w3": 11.561,
    "WISE_w4": 22.088,
    "Herschel_PACS_70": 70.,
    "Herschel_PACS_100": 100.,
    "Herschel_PACS_160": 160.,
    "Herschel_SPIRE_250": 250.,
    "Herschel_SPIRE_350": 350.,
    "Herschel_SPIRE_500": 500.,
    "Spitzer_IRAC1": 3.550,
    "Spitzer_IRAC2": 4.493,
    "Spitzer_IRAC3": 5.731,
    "Spitzer_IRAC4": 7.872,
    "IRAS_12": 12.,
    "IRAS_25": 25.,
    "IRAS_60": 60.,
    "IRAS_100": 100.,
    "Spitzer_MIPS_24": 24.,
    "Spitzer_MIPS_70": 70.,
    "Spitzer_MIPS_160": 160.,
    "JCMT_SCUBA1_450": 450.,
    "JCMT_SCUBA1_850": 850.,
}
monoFilters = ["Herschel_PACS_70", "Herschel_PACS_100", "Herschel_PACS_160",
               "Herschel_SPIRE_250", "Herschel_SPIRE_350", "Herschel_SPIRE_500",
               "Herschel_SPIRE_250_e", "Herschel_SPIRE_350_e", "Herschel_SPIRE_500_e",
               "Spitzer_IRAC1", "Spitzer_IRAC2", "Spitzer_IRAC3", "Spitzer_IRAC4",
               "Spitzer_MIPS_24", "Spitzer_MIPS_70", "Spitzer_MIPS_160",
               "IRAS_12", "IRAS_25", "IRAS_60", "IRAS_100"]
meanFilters = ["SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z",
               "2MASS_J", "2MASS_H", "2MASS_Ks",
               "UKIDSS_Y", "UKIDSS_J", "UKIDSS_H", "UKIDSS_K",
               "WISE_w1", "WISE_w2", "WISE_w3", "WISE_w4"]

if __name__ == "__main__":
    from dir_list import filter_path as bandPath
    import matplotlib.pyplot as plt
    z = 1.5
    bn = "Herschel_SPIRE_500"
    bandFile = "{0}.dat".format(bn)
    bandPck = np.genfromtxt(bandPath+bandFile)
    bandWave = bandPck[:, 0]
    bandRsr = bandPck[:, 1]
    bandCenter = filterDict[bn]
    bandType = "mono"
    bp1 = BandPass(bandWave, bandRsr, bandCenter, bandType, bandName=bn, redshift=0, silent=False)
    bp2 = BandPass(bandWave, bandRsr, bandCenter, bandType, bandName=bn, redshift=z, silent=False)
    #bp2 = BandPass(bandCenter=bandCenter, bandType="none", bandName=bn, redshift=z, silent=False)

    alpha = -1
    f0 = 10.0
    w0 = bandCenter
    nu0  = ls_mic / w0
    wave_0 = 10**np.linspace(0, 3, 1000)
    freq = ls_mic / wave_0
    flux_0 = f0*(freq / nu0)**alpha
    wave_1 = wave_0 / (1 + z)
    flux_1 = flux_0

    #mF_0 = interp1d(wave_0, flux_0)
    #mF_1 = interp1d(wave_1, flux_1)
    #fb_0 = bp1.filtering(mF_0)
    #fb_1 = bp2.filtering(mF_1)
    fb_0 = bp1.filtering(wave_0, flux_0)
    fb_1 = bp2.filtering(wave_1, flux_1)
    print("w0={0}".format(w0))
    print("f0={0}, fb1={1}".format(f0, fb_0))
    print("f0={0}, fb2={1}".format(f0, fb_1))
    plt.plot(wave_0, flux_0, ":r", label="z=0")
    plt.plot(wave_1, flux_1, ":b", label="z={0}".format(z))
    plt.plot(bp1.get_bandCenter_rest(), fb_0, linestyle="none", marker=".", color="r")
    plt.plot(bp2.get_bandCenter_rest(), fb_1, linestyle="none", marker=".", color="b")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
