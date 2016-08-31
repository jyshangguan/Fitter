import numpy as np

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
    #ym = np.array(model.combineResult(x))
    ym = np.array(Model2Data(model, data))
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
