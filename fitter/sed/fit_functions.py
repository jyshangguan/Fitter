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

def ChiSquare_LMFIT(params, sed_data, sed_model, model_func, wave_model):
    '''
    This function uses the given model to calculate the Chi square.
    This function is designed for the LMFIT package.

    Parameters
    ----------
    params : Parameter class from LMFIT package
    data : float array
        The observed flux.
    unct : float array
        The uncertainties.
    model_func : callable
        Objective function.
    func_kws : dict
        The parameters used by the model_func().

    Returns
    -------
    chsq : float
        The Chi square.

    Notes
    -----
    None.
    '''
    Parameters_Dump(params, sed_model)
    model = np.array( model_func(sed_data, sed_model, wave_model) )
    data = np.array( sed_data.get_dsList('y') )
    unct = np.array( sed_data.get_dsList('e') )
    chsq = ChiSq(data, model, unct)
    return chsq

def SED_Model_Fit(params, sed_data, sed_model, param_dl07, wave_model):
    """
    This function do the SED fitting.

    Parameters
    ----------
    params : class Parameters()
    """
    sed_model = copy.deepcopy(sed_model)
    modelDict = sed_model.get_modelDict()
    umin, umax, qpah = param_dl07
    sed_model.updateParFit('DL07', 'umin', umin, QuietMode=True)
    sed_model.updateParFit('DL07', 'umax', umax, QuietMode=True)
    sed_model.updateParFit('DL07', 'qpah', qpah, QuietMode=True)

    #----------------------
    #Initial guess
    #----------------------
    for modelName in sed_model._modelList:
        #print modelName
        model = modelDict[modelName]
        normPck = inputModelDict[modelName]['normalisation']
        if normPck is None:
            continue
        band_norm, scalePar = normPck
        #Get the flux from the photometric data
        wave_norm = sed_data.get_dsDict()['WISE&Herschel'][band_norm][0]
        flux_norm = sed_data.get_dsDict()['WISE&Herschel'][band_norm][1]
        fluxModel = model.result(wave_model)
        fluxModelPoint = interp1d(wave_model, fluxModel)(wave_norm)
        model.parFitDict[scalePar] += np.log10(flux_norm/fluxModelPoint)
    params = Parameters_Load(params, sed_model)

    chisqKws = {
        'sed_data': sed_data,
        'sed_model': sed_model,
        'model_func': Model2Data,
        'wave_model': wave_model,
    }
    out = minimize(ChiSquare_LMFIT, params, kws=chisqKws, method='nelder')
    parsFits = out.params
    Parameters_Dump(parsFits, sed_model)
    chisq = ChiSquare_LMFIT(parsFits, sed_data, sed_model, Model2Data, wave_model)
    results = {
        'ChiSQ': chisq,
        'sed_model': sed_model
    }
    return results
