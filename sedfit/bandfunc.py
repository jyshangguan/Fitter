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
        The quoted wavelength for the band.
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
            if bandCenter is None: #The bandCenter is not specified, the effective wavelength will
                                   #be used (Eq. A21, Bessell&Murphy 2012), assuming fnu~const.
                bandCenter = np.trapz(rsrList, waveList)/np.trapz(rsrList/waveList, waveList)
                if not silent:
                    print("Band {0} center wavelength ({1}) is calculated!".format(bandName, bandCenter))
            self.__bandCenter = bandCenter
            self.__bandCenter_rest = bandCenter / (1 + redshift)
            if bandType == "mono":
                self.k4p = K_MonP(self.__bandCenter, waveList, rsrList, alpha=-1)
                if not silent:
                    print("Band {0} calculates the monochromatic flux density!".format(bandType))
            elif bandType == "mean":
                if not silent:
                    print("Band {0} calculates the averaged flux density!".format(bandType))
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

    def BandFunc_mono(self, modelFunc):
        """
        Calculate the monochromatic flux density with the given data. The function
        applies for the bolometers used by IR satellites.
        To use this function, the relative spectral response should be for the
        bolometer (energy/photon) instead of the CCD (electron/photon).
        Reference: Section 5.2.4, SPIRE Handbook.

        Parameters
        ----------
        modelFunc : function
            The model function to takes wavelength and returns flux density.

        Returns
        -------
        fluxFltr : float
            The monochromatic flux density calculated from the filter rsr.

        Notes
        -----
        To convert the relative spectral response from electron/photon to
        energy/photon is simply:
            S(energy/photon) = S(electron/photon) / nu
        where nu is the corresponding frequency (Bessell & Murphy 2012).
        """
        waveList = self.__waveList
        rsrList = self.__rsrList
        freq = ls_mic / waveList
        flux = modelFunc(waveList)
        Sbar = np.trapz(rsrList*flux, freq) / np.trapz(rsrList, freq)
        fluxFltr = self.k4p * Sbar
        return fluxFltr

    def BandFunc_mean(self, modelFunc):
        """
        Calculate the band averaged flux density with the given data.
        By default, the rsr is photon response and the band flux is defined as
        eq. A12 (Bessell&Murphy 2012). The rsr is for the CCD detector instead
        of bolometers.

        Parameters
        ----------
        modelFunc : function
            The model function to takes wavelength and returns flux density.

        Returns
        -------
        fluxFltr : float
            The monochromatic flux density calculated from the filter rsr.

        Notes
        -----
        None.
        """
        waveList = self.__waveList
        rsrList = self.__rsrList
        freq = ls_mic / waveList
        flux = modelFunc(waveList)
        signal = np.trapz(rsrList / waveList * flux, x=waveList)
        norm   = np.trapz(rsrList / waveList, x=waveList)
        fluxFltr = signal/norm
        return fluxFltr

    def BandFunc_none(self, modelFunc):
        """
        This band function provides the flux without the bandpass. The flux density
        of the model SED at the wavelength closest to the band center is returned.

        Parameters
        ----------
        modelFunc : function
            The model function to takes wavelength and returns flux density.

        Returns
        -------
        fluxFltr : float
            The flux density at the wavelength closest to the band center.

        Notes
        -----
        None.
        """
        fluxFltr = modelFunc(np.array([self.__bandCenter]))[0]
        return fluxFltr

    def filtering(self, modelFunc):
        """
        Calculate the flux density of the input spectrum filtered by the bandpass.

        Parameters
        ----------
        modelFunc : function
            The model function to takes wavelength and returns flux density.

        Returns
        -------
        fluxFltr : float
            The flux density of the spectrum after filtered by the bandpass.

        Notes
        -----
        None.
        """
        bandType = self.__bandType
        if bandType == "mean": #By default, the rsr is photon response and the band flux
                             #is defined as eq. A12 (Bessell&Murphy 2012).
            fluxFltr = self.BandFunc_mean(modelFunc)
        elif bandType == "mono": #Use the user specified function to get the filtered flux.
            fluxFltr = self.BandFunc_mono(modelFunc)
        else: #If there is no filter, direct calculate the flux density.
            fluxFltr = self.BandFunc_none(modelFunc)
        return fluxFltr

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

#Default filters
filterDict = {
    "j": 1.235,
    "h": 1.662,
    "ks": 2.159,
    "w1": 3.353,
    "w2": 4.603,
    "w3": 11.561,
    "w4": 22.088,
    "PACS_70": 70.,
    "PACS_100": 100.,
    "PACS_160": 160.,
    "SPIRE_250": 250.,
    "SPIRE_350": 350.,
    "SPIRE_500": 500.,
    "IRAC1": 3.550,
    "IRAC2": 4.493,
    "IRAC3": 5.731,
    "IRAC4": 7.872,
    "iras12": 12.,
    "iras25": 25.,
    "iras60": 60.,
    "iras100": 100.,
    "mips24": 24.,
    "mips70": 70.,
    "mips160": 160.,
    "scuba450_new": 450.,
    "scuba850_new": 850.,
}
monoFilters = ["PACS_70", "PACS_100", "PACS_160",
               "SPIRE_250", "SPIRE_350", "SPIRE_500",
               "IRAC1", "IRAC2", "IRAC3", "IRAC4"]
meanFilters = ["j", "h", "ks", "w1", "w2", "w3", "w4"]

if __name__ == "__main__":
    from dir_list import filter_path as bandPath
    import matplotlib.pyplot as plt
    z = 1.5
    bn = "w4"
    bandFile = "/{0}.dat".format(bn)
    bandPck = np.genfromtxt(bandPath+bandFile)
    bandWave = bandPck[:, 0]
    bandRsr = bandPck[:, 1]
    bandCenter = filterDict[bn]
    bp1 = BandPass(bandWave, bandRsr, bandCenter, bandType="mean", bandName=bn, redshift=0, silent=False)
    bp2 = BandPass(bandWave, bandRsr, bandCenter, bandType="mean", bandName=bn, redshift=z, silent=False)
    #bp2 = BandPass(bandCenter=bandCenter, bandType="none", bandName=bn, redshift=z, silent=False)

    alpha = 1
    f0 = 10.0
    w0 = bandCenter
    nu0  = ls_mic / w0
    wave_0 = 10**np.linspace(0, 3, 1000)
    freq = ls_mic / wave_0
    flux_0 = f0*(freq / nu0)**alpha
    wave_1 = wave_0 / (1 + z)
    flux_1 = flux_0

    mF_0 = interp1d(wave_0, flux_0)
    mF_1 = interp1d(wave_1, flux_1)
    fb_0 = bp1.filtering(mF_0)
    fb_1 = bp2.filtering(mF_1)
    print("w0={0}".format(w0))
    print("f0={0}, fb1={1}".format(f0, fb_0))
    print("f0={0}, fb2={1}".format(f0, fb_1))
    plt.plot(wave_0, flux_0, ":r", label="0")
    plt.plot(wave_1, flux_1, ":b", label="1")
    plt.plot(bp1.get_bandCenter_rest(), fb_0, linestyle="none", marker=".", color="r")
    plt.plot(bp2.get_bandCenter_rest(), fb_1, linestyle="none", marker=".", color="b")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
