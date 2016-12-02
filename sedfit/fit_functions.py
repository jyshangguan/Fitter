import numpy as np
import george
from george import kernels

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
        wrsd_non = (data[fltr_non] - model[fltr_non])/(unct_non * 2**0.5)
        chsq_non = sum( -2.* np.log( 0.5 * (1 + erf(wrsd_non)) ) )
    else:
        chsq_non = 0.
    chsq = chsq_dtc + chsq_non
    return chsq

#Model to data function#
def Model2Data(sedModel, sedData):
    """
    Convert the continual model to the data-like model to directly
    compare with the data.

    Parameters
    ----------
    sedModel : ModelCombiner object
        The combined model.
    sedData : SEDClass object
        The data set of SED.

    Returns
    -------
    fluxModel : list
        The model flux list of the data.

    Notes
    -----
    None.
    """
    waveModel = sedModel.get_xList()
    fluxModel = sedModel.combineResult()
    fluxModelPht = sedData.model_pht(waveModel, fluxModel)
    fluxModelSpc = sedData.model_spc(sedModel.combineResult)
    fluxModel = fluxModelPht + fluxModelSpc
    return fluxModel

#Model to data function using Gaussian process regression
def Model2Data_gp(sedModel, sedData):
    """
    Convert the continual model to the data-like model to directly
    compare with the data.

    Parameters
    ----------
    sedModel : ModelCombiner object
        The combined model.
    sedData : SEDClass object
        The data set of SED.

    Returns
    -------
    fluxModel : list
        The model flux list of the data.

    Notes
    -----
    None.
    """
    waveModel = sedModel.get_xList()
    fluxModel = sedModel.combineResult()
    fluxModelPht = sedData.model_pht(waveModel, fluxModel)
    fluxModelSpc = sedData.model_spc(sedModel.combineResult)
    fluxDict = {
        "pht": fluxModelPht,
        "spc": fluxModelSpc
        }
    return fluxDict

#The log_likelihood function: for SED fitting
def logLFunc(params, data, model):
    """
    Calculate the likelihood of data according to the model and its parameters.

    Parameters
    ----------
    params : list
        The variable parameter list of the model.
    data : DataSet
        The data need to fit.
    model : ModelCombiner
        The model to fit the data.

    Returns
    -------
    logL : float
        The log likelihood.

    Notes
    -----
    None.
    """
    model.updateParList(params)
    nParVary = len(model.get_parVaryList())
    y = np.array(data.get_List('y'))
    e = np.array(data.get_List('e'))
    ym = np.array(Model2Data(model, data))
    #Calculate the log_likelihood
    logL = -0.5 * (ChiSq(y, ym, e) + np.sum( np.log(2 * np.pi * e**2) ))
    #print logL
    return logL

#The log_likelihood function: for SED fitting using Gaussian process regression
def logLFunc_gp(params, data, model):
    """
    Calculate the likelihood of data according to the model and its parameters.

    Parameters
    ----------
    params : list
        The variable parameter list of the model.
    data : DataSet
        The data need to fit.
    model : ModelCombiner
        The model to fit the data.

    Returns
    -------
    lnL : float
        The ln likelihood.

    Notes
    -----
    None.
    """
    #Get the data and error
    xSpc = np.array(data.get_csList("x"))
    yPht = np.array(data.get_dsList("y"))
    ySpc = np.array(data.get_csList("y"))
    ePht = np.array(data.get_dsList("e"))
    eSpc = np.array(data.get_csList("e"))
    #Calculate the model
    model.updateParList(params)
    yDict = Model2Data_gp(model, data)
    yPhtModel = np.array(yDict["pht"])
    ySpcModel = np.array(yDict["spc"])
    nParVary = len(model.get_parVaryList())
    #lnlikelihood for photometric data
    if len(yPhtModel):
        f = np.exp(params[nParVary]) #The last par is lnf.
        fltr_non = ePht < 0 #Find those non-detections
        sPht = (ePht**2 + (yPhtModel * f)**2)**0.5
        sPht[fltr_non] = -1
        lnlPht = -0.5 * ChiSq(yPht, yPhtModel, sPht)
    else:
        lnlPht = 0
    #lnlikelihood for spectral data using Gaussian process regression
    if len(ySpcModel):
        a, tau = np.exp(params[nParVary+1:])
        #a, tau = np.exp(params[nParVary:])
        gp = george.GP(a * kernels.Matern32Kernel(tau))
        sSpc = (eSpc**2 + (ySpcModel * f)**2)**0.5
        gp.compute(xSpc, sSpc)
        #gp.compute(xSpc, eSpc)
        lnlSpc = gp.lnlikelihood(ySpc - ySpcModel)
    else:
        lnlSpc = 0
    lnL = lnlPht + lnlSpc
    print lnL
    return lnL
