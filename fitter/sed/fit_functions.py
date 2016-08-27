import copy
import numpy as np
from scipy.interpolate import interp1d
from .. import basicclass as bc
import sedclass as sc
from lmfit import minimize, Parameters, fit_report

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
    nParVary = len(model.get_parVaryList())
    y = np.array(data.get_List('y'))
    e = np.array(data.get_List('e'))
    #ym = np.array(model.combineResult(x))
    ym = np.array(model.model2Data(data))
    if len(params) == nParVary:
        s = e
    elif len(params) == (nParVary+1):
        f = np.exp(params[nParVary]) #The last par is lnf.
        s = (e**2 + (ym * f)**2)**0.5
    else:
        raise ValueError("The length of params is incorrect!")
    #Calculate the log_likelihood
    logL = -0.5 * (ChiSq(y, ym, s) + np.sum( np.log(2 * np.pi * s**2) ))
    return logL
