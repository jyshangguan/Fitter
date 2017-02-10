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
    into the band flux density.

    Parameters
    ----------
    wavelength : float array
        The wavelength of the relative system response curve.
    rsr : float array
        The relative system response curve.
    """
    def __init__(self, waveList=None, rsrList=None, bandCenter=None, bandType="mean", bandName='None', silent=True):
        self.__bandType = bandType
        if not bandType == "none":
            waveMin = waveList[0]
            waveMax = waveList[-1]
            if waveMin >= waveMax:
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
                self._bandCenter = np.trapz(rsrList, waveList)/np.trapz(rsrList/waveList, waveList)
            else:
                self._bandCenter = bandCenter
            if bandType == "mono":
                self.k4p = K_MonP(self._bandCenter, waveList, rsrList, alpha=-1)
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
            self._bandCenter = bandCenter
            self._bandName = bandName
            if not silent:
                print("Band {0} ({1}) does not have bandpass.".format(bandName, bandCenter))

    def get_bandpass(self):
        bandInfo = {
            'wave_list': self.__waveList,
            'rsr_list': self.__rsrList,
            'band_name': self._bandName,
            'wave_center': self._bandCenter,
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
        energy/photon is simply:
            S(energy/photon) = S(electron/photon) / nu
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
        bandCenter = self._bandCenter
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
        bandCenter = self._bandCenter
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
        else:
            fluxFltr = self.BandFunc_none(wavelength, flux)
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
"""
pathList = os.path.abspath(__file__).split("/")
pathList[-1] = "filters"
bandPath = "/".join(pathList)
fp = open(bandPath+"/herschel_bandpass.dat", "r")
herschelBands = pickle.load(fp)
fp.close()
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    bn = "w4"
    bandFile = "/{0}.dat".format(bn)
    bandPck = np.genfromtxt(bandPath+bandFile)
    bandWave = bandPck[:, 0]
    bandRsr = bandPck[:, 1]
    bandCenter = filterDict[bn]
    bp1 = BandPass(bandWave, bandRsr, bandCenter, bandType="mean", bandName=bn, silent=False)
    bp2 = BandPass(bandCenter=bandCenter, bandType="none", bandName=bn, silent=False)
    f0 = 10.0
    w0 = bandCenter
    alpha = 5
    wave = 10**np.linspace(0, 3, 1000)
    freq = ls_mic / wave
    nu0  = ls_mic / w0
    flux = f0*(freq / nu0)**alpha
    fb1 = bp1.filtering(wave, flux)
    fb2 = bp2.filtering(wave, flux)
    print("w0={0}".format(w0))
    print("f0={0}, fb1={1}".format(f0, fb1))
    print("f0={0}, fb2={1}".format(f0, fb2))
    plt.plot(wave, flux, color="k")
    plt.plot(bandCenter, fb1, linestyle="none", marker=".", color="r")
    plt.plot(bandCenter, fb2, linestyle="none", marker=".", color="b")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
