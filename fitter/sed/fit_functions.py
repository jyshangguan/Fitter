import copy
import numpy as np
from scipy.interpolate import interp1d
from .. import basicclass as bc
import sedclass as sc
import model_functions as sedmf
from lmfit import minimize, Parameters, fit_report

inputModelDict = sedmf.inputModelDict

def Model2Data(sedData, sedModel):
    """
    Convert the continual model to the data-like model to directly
    compare with the data.

    Parameters
    ----------
    sedData : SEDClass object
        The data set of SED.
    sedModel : ModelCombiner object
        The combined model.

    Returns
    -------
    fluxModelPht : list
        The model flux list of photometric data.

    Notes
    -----
    None.
    """
    if not isinstance(sedData, sc.SedClass):
        raise TypeError("The sedData type is incorrect!")
    elif not isinstance(sedModel, bc.ModelCombiner):
        raise TypeError("The sedModel type is incorrect!")
    waveModel = sedModel.get_xList()
    fluxModel = sedModel.combineResult()
    fluxModelPht = sedData.model_pht(waveModel, fluxModel)
    return fluxModelPht

def ChiSq(data, model, unct=None):
    '''
    This function calculate the Chi square of the observed data and
    the model. The upper limits are properly deal with using the method
    mentioned by Sawicki (2012).

    Parameters
    ----------
    data : float array
        The observed data.
    model : float array
        The model.
    unct : float array
        The uncertainties.

    Returns
    -------
    chsq : float
        The Chi square

    Notes
    -----
    None.
    '''
    from scipy.special import erf
    if unct is None:
        unct = np.ones(len(data))
    fltr_dtc = unct>0
    fltr_non = unct<0
    if sum(fltr_dtc)>0:
        wrsd_dtc = (data[fltr_dtc] - model[fltr_dtc])/unct[fltr_dtc] #The weighted residual
        chsq_dtc = sum(wrsd_dtc**2)
    else:
        chsq_dtc = 0.
    if sum(fltr_non)>0:
        unct_non = data[fltr_non]/3.0 #The nondetections are 3 sigma upper limits.
        wrsd_non = (data[fltr_non] - model[fltr_non])/unct_non
        chsq_non = sum(-2.* np.log(1 + erf(wrsd_non/2**0.5)))
    else:
        chsq_non = 0.
    chsq = chsq_dtc + chsq_non
    return chsq

#The log_likelihood function: for SED fitting
def logLFunc_SED(params, data, model):
    """
    parDict = model.get_modelParDict()
    pIndex = 0
    for modelName in model._modelList:
        parFitDict = parDict[modelName]
        for parName in parFitDict.keys():
            if parFitDict[parName]["vary"]:
                parFitDict[parName]["value"] = params[pIndex]
                pIndex += 1
            else:
                pass
    """
    model.updateParList(params)
    y = np.array(data.get_List('y'))
    e = np.array(data.get_List('e'))
    #ym = np.array(model.combineResult(x))
    ym = np.array(Model2Data(data, model))
    if len(params) == pIndex:
        s = e
    elif len(params) == (pIndex+1):
        f = np.exp(params[pIndex])
        s = (e**2 + (ym * f)**2)**0.5
    else:
        raise ValueError("The length of params is incorrect!")
    #Calculate the log_likelihood
    logL = -0.5 * (ChiSq(y, ym, s) + np.sum( np.log(2 * np.pi * s**2) ))
    return logL
