import numpy as np
from scipy.interpolate import interp1d


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
