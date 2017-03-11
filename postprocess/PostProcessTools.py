#This script provide some functions to do the postprocess of the fitting sampling.
#
import os
import numpy as np
from scipy.interpolate import interp1d
ls_mic = 2.99792458e14 #unit: micron/s
Mpc = 3.08567758e24 #unit: cm
mJy = 1e26 #unit: erg/s/cm^2/Hz

__all__ = ["AddDict", "MatchDict", "parStatistics", "Luminosity_Integrate", "Luminosity_Specific",
           "L_Total", "randomSampler", "CorrectParameters"]

def AddDict(targetDict, quantName, quant, nFillPar=None):
    """
    To add a quantity into the target dict. If there is a parameter omitted
    before, we need to fill the list with nan before the add item.
    """
    #->If the parameter is added before.
    if quantName in targetDict.keys():
        #print "No.{0}: {1} is there!".format(nFillPar, quantName)
        targetDict[quantName].append(quant)
    else:#->If the parameter is not add before.
        #print "No.{0}: {1} is not there!".format(nFillPar, quantName)
        if (nFillPar is None) or (nFillPar == 0):#->If no parameter possibly omitted.
            targetDict[quantName] = [quant]
        else:#->If there is parameter omitted.
            #print "Add nan to {0} for No.{1} source!".format(quantName, nFillPar)
            ngap = nFillPar #->Find how many items are omitted.
            #->Add the items that a omitted.
            parList = [np.nan for loop in range(ngap)]
            #->Add the current item.
            parList.append(quant)
            targetDict[quantName] = parList
    return None

def MatchDict(targetDict):
    """
    To match the lengths of the lists in the dict by filling them with nan.
    """
    lmin = np.inf
    lmax = -1
    for quantName in targetDict.keys():
        lq = len(targetDict[quantName])
        if lq > lmax:
            lmax = lq
        if lq < lmin:
            lmin = lq
    ldiff = lmax-lmin
    if ldiff > 1:
        raise ValueError("The length difference ({0}) is larger than 1!".format(ldiff))
    for quantName in targetDict.keys():
        parList = targetDict[quantName]
        if len(parList) < lmax:
            parList.append(np.nan)
    return None

def parStatistics(ppfunc, nSamples, ps, fargs=[], fkwargs={}):
    """
    Get the statistics of the model calculated parameters.

    Parameters
    ----------
    ppfunc : function
        The function to post process the model posterior sampling.
    nSamples : int
        The number of sampling when it calculate the parameter statistics.
    ps : array
        The posterior sampling.

    Returns
    -------
    pList : array
        The parameter list.

    Notes
    -----
    None.
    """
    pList = []
    for pars in ps[np.random.randint(len(ps), size=nSamples)]:
        pList.append(ppfunc(pars, *fargs, **fkwargs))
    pList = np.array(pList)
    return pList

def Luminosity_Integrate(flux, wave, DL, z, waveRange=[8.0, 1e3], frame="rest"):
    """
    Calculate the integrated luminosity of input SED with given wavelength range.

    Parameters
    ----------
    flux : float array
        The flux density used to integrate to the luminosity.
    wave : float array
        The wavelength of the SED.
    DL : float
        The luminosity distance of the source.
    z : float
        The redshift of the source.
    waveRange : float array
        The short and long end of the wavelength to integrate.
    frame : string
        The flux and wave should be consistently in the rest frame ("rest") or
        observing frame ("obs").

    Returns
    -------
    L : float
        The luminosity of the SED within the given wavelength range, unit: erg/s.

    Notes
    -----
    None.
    """
    nu = ls_mic / wave
    fltr = (wave > waveRange[0]) & (wave < waveRange[1])
    F = -1.0 * np.trapz(flux[fltr], nu[fltr]) / mJy #unit: erg/s/cm^2
    if frame == "rest":
        L = F * 4.0*np.pi * (DL * Mpc)**2.0 / (1 + z)**2
    elif frame == "obs":
        L = F * 4.0*np.pi * (DL * Mpc)**2.0
    else:
        raise ValueError("Cannot recognise the frame: {0}!".format(frame))
    return L

def Luminosity_Specific(flux, wave, wave0, DL, z, frame="rest"):
    """
    Calculate the specific luminosity at wave0 for the SED (wave, flux).

    Parameters
    ----------
    flux : float array
        The flux density of the SED.
    wave : float array
        The wavelength of the SED.
    wave0 : float (array)
        The wavelength(s) to be caculated.
    DL : float
        The luminosity distance.
    z : float
        The redshift.

    Returns
    -------
    Lnu : float (array)
        The specific luminosity (array) at wave0 that is (are) required, unit: erg/s/Hz.

    Notes
    -----
    None.
    """
    S0 = interp1d(wave, flux)(wave0)
    if frame == "rest":
        Lnu = S0 * 4.0*np.pi * (DL * Mpc)**2.0 / (1 + z)**2 / mJy #unit: erg/s/Hz
    elif frame == "obs":
        Lnu = S0 * 4.0*np.pi * (DL * Mpc)**2.0 / (1 + z) / mJy #unit: erg/s/Hz
    return Lnu

def L_Total(pars, sedModel, DL, z):
    """
    Calculate the total luminosity.
    """
    sedModel.updateParList(pars)
    wave = sedModel.get_xList()
    flux = sedModel.combineResult()
    L = Luminosity_Integrate(flux, wave, DL, z, waveRange=[8.0, 1e3], frame="rest")
    return L

def randomSampler(parRange, parType="D"):
    """
    This function randomly sample the given parameter space.
    """
    if parType == "D": #For discrete parameters
        p = np.random.choice(parRange)
    elif parType == "C": #For continuous parameters
        r1, r2 = parRange
        p = (r2 - r1) * np.random.rand() + r1
    else:
        raise ValueError("The parameter type '{0}' is not recognised!".format(parType))
    return p

def CorrectParameters(modelDict, discFuncDict, silent=True):
    """
    Correct the parameters if there are some discrete parameters interpolated
    with nearest neighbor method.

    Parameters
    ----------
    modelDict : OrderedDict
        The ordered model parameter dict generated by ModelCombiner object.
    discFuncDict : dict
        The dict to store all the functions who has the discrete parameters.
    silent : bool; default: True
        Print information and return the parameter names for check if silent is
        False.

    Returns
    -------
    parList : list
        The list of parameters, the sequence of which is important.
    parNameList : list (optional)
        The list of parameter names to check the sequence of the parameters.

    Notes
    -----
    None.
    """
    parList = []
    parNameList = []
    for model in modelDict.keys():
        fitDict = modelDict[model]
        pnList  = fitDict.keys()
        if model in discFuncDict.keys():
            if not silent:
                print("{0} is discrete".format(model))
            inDict  = {}
            for pn in pnList:
                inDict[pn] = fitDict[pn]["value"]
            Func_PosPar = discFuncDict[model]
            outDict = Func_PosPar(**inDict)
            for pn in pnList:
                if fitDict[pn]["vary"]:
                    parList.append(outDict[pn])
                    parNameList.append(pn)
        else:
            if not silent:
                print("{0} is not discrete".format(model))
            for pn in pnList:
                if fitDict[pn]["vary"]:
                    parList.append(fitDict[pn]["value"])
                    parNameList.append(pn)
    if not silent:
        return (parList, parNameList)
    else:
        return parList
