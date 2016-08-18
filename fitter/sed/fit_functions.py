import copy
import numpy as np
from scipy.interpolate import interp1d
from .. import basicclass as bc
import sedclass as sc
import model_functions as sedmf
from lmfit import minimize, Parameters, fit_report

inputModelDict = sedmf.inputModelDict

def Model2Data(sedData, sedModel, waveModel):
    """
    Convert the continual model to the data-like model to directly
    compare with the data.

    Parameters
    ----------
    sedData : SEDClass object
        The data set of SED.
    sedModel : ModelCombiner object
        The combined model.
    waveModel : float array
        The model wavelength array.

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
    fluxModel = sedModel.combineResult(waveModel)
    fluxModelPht = sedData.model_pht(waveModel, fluxModel)
    return fluxModelPht

def Parameters_Init(model_dict_init, func_lib):
    '''
    This function create the Parameters() for model fitting using LMFIT.

    Parameters
    ----------
    model_dict_init : dict
        The dict of model components.
    func_lib : dict
        The dict of all the supporting functions.

    Returns
    -------
    params : class Parameters
        The Parameters object used in the LMFIT.

    Notes
    -----
    None.
    '''
    params = Parameters()
    modelList = model_dict_init.keys()
    for modelName in modelList:
        funcName = model_dict_init[modelName]['function']
        func = func_lib[funcName]
        paramFitList = func['param_fit']
        paramsInfo = model_dict_init[modelName]
        #Load the parameters
        for parName in paramFitList:
            parInfo = paramsInfo.get(parName, None)
            if parInfo is None:
                raise KeyError('The fitting parameter: {0} is not found!'.format(parName))
            else:
                name = '{0}_{1}'.format(modelName, parName)
                value = parInfo['value']
                vary  = parInfo['vary']
                vmin, vmax = parInfo['range']
                params.add(name, value=value, vary=vary, min=vmin, max=vmax)
    return params

def Parameters_Dump(params, sed_model):
    '''
    This function load the model parameters into the Parameters()
    used by the LMFIT.

    Parameters
    ----------
    params : class Parameters
        The Parameters object used in the LMFIT.
    sed_model : class ModelCombiner
        The model.

    Returns
    -------
    None

    Notes
    -----
    Since it is not easy to duplicate a new model_dict, we just change
    the original one.
    '''
    if not isinstance(sed_model, bc.ModelCombiner):
        raise ValueError("The sed_model should be ModelCombiner()!")
    model_dict = sed_model.get_modelDict()
    modelList = model_dict.keys()
    for modelName in model_dict.keys():
        parFitDict = model_dict[modelName].parFitDict
        for parName in parFitDict.keys():
            parFitDict[parName] = params['{0}_{1}'.format(modelName, parName)].value
    return None

def Parameters_Load(params, sed_model):
    '''
    This function load the model parameters into the Parameters()
    used by the LMFIT.

    Parameters
    ----------
    params : class Parameters
        The Parameters object used in the LMFIT.
    sed_model : class ModelCombiner
        The model.

    Returns
    -------
    params : class Parameters
        The Parameters object used in the LMFIT.

    Notes
    -----
    None.
    '''
    if not isinstance(sed_model, bc.ModelCombiner):
        raise ValueError("The sed_model should be ModelCombiner()!")
    model_dict = sed_model.get_modelDict()
    modelList = model_dict.keys()
    for modelName in model_dict.keys():
        #print '[Parameters_Load]: {0}'.format(modelName)
        parFitDict = model_dict[modelName].parFitDict
        for parName in parFitDict.keys():
            #print '[Parameters_Load]: {0}: {1}'.format(parName, parFitDict[parName])
            params['{0}_{1}'.format(modelName, parName)].value = parFitDict[parName]
    return params

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
    parDict = model.get_modelParDict()
    pIndex = 0
    for modelName in model._modelList:
        parFitDict = parDict[modelName]
        for parName in parFitDict.keys():
            parFitDict[parName]["value"] = params[pIndex]
            pIndex += 1
    x = np.array(data.get_List('x'))
    y = np.array(data.get_List('y'))
    e = np.array(data.get_List('e'))
    ym = np.array(model.combineResult(x))
    if len(params) == pIndex:
        s = e
    elif len(params) == (pIndex+1):
        s = (e**2 + params[pIndex+1]**2)**0.5
    else:
        raise ValueError("The length of params is incorrect!")
    #Calculate the log_likelihood
    logL = -0.5 * (sedff.ChiSq(y, ym, s) + np.sum( np.log(2 * np.pi * s**2) ))
    return logL
