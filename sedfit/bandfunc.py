import os
import numpy as np
from scipy.interpolate import interp1d
import cPickle as pickle


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

#Default filters
filterDict = {
    "j": 1.235,
    "h": 1.662,
    "ks": 2.159,
    "w1": 3.353,
    "w2": 4.603,
    "w3": 11.561,
    "w4": 22.088,
    'PACS_70': 70.,
    'PACS_100': 100.,
    'PACS_160': 160.,
    'SPIRE_250': 250.,
    'SPIRE_350': 350.,
    'SPIRE_500': 500.
}
herschelFilters = ['PACS_70', 'PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350', 'SPIRE_500']
pathList = os.path.abspath(__file__).split("/")
pathList[-1] = "filters"
bandPath = "/".join(pathList)
fp = open(bandPath+"/herschel_bandpass.dat", "r")
herschelBands = pickle.load(fp)
fp.close()
