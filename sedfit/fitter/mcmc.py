import emcee
import numpy as np
from .. import fit_functions as sedff

logLFunc = sedff.logLFunc #The log_likelihood function

def lnprior(params, data, model, ModelUnct):
    """
    Calculate the ln prior probability.
    """
    lnprior = 0.0
    parIndex = 0
    parDict = model.get_modelParDict()
    for modelName in model._modelList:
        parFitDict = parDict[modelName]
        for parName in parFitDict.keys():
            if parFitDict[parName]["vary"]:
                parValue = params[parIndex]
                parIndex += 1
                pr1, pr2 = parFitDict[parName]["range"]
                if (parValue < pr1) or (parValue > pr2):
                    lnprior -= np.inf
            else:
                pass
    #If the model uncertainty is concerned.
    if ModelUnct:
        lnf =  params[parIndex]
        if (lnf < -20) or (lnf > 1.0):
            lnprior -= np.inf
    return lnprior

def log_likelihood(params, data, model):
    """
    Gaussian sampling distrubution.
    """
    logL  = logLFunc(params, data, model)
    return logL

def lnprob(params, data, model, ModelUnct):
    lp = lnprior(params, data, model, ModelUnct)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, data, model)

class EmceeModel(object):
    """
    The MCMC model for emcee.
    """
    def __init__(self, data, model, ModelUnct=False):
        self.__data = data
        self.__model = model
        self.__modelunct = ModelUnct
        if ModelUnct:
            print "[EmceeModel]: ModelUnct is on!"
        else:
            print "[EmceeModel]: ModelUnct is off!"

    def from_prior(self):
        """
        The prior of all the parameters are uniform.
        """
        parList = []
        parDict = self.__model.get_modelParDict()
        for modelName in self.__model._modelList:
            parFitDict = parDict[modelName]
            for parName in parFitDict.keys():
                if parFitDict[parName]["vary"]:
                    parRange = parFitDict[parName]["range"]
                    parType  = parFitDict[parName]["type"]
                    if parType == "c":
                        #print "[DN4M]: continual"
                        r1, r2 = parRange
                        p = (r2 - r1) * np.random.rand() + r1 #Uniform distribution
                    elif parType == "d":
                        #print "[DN4M]: discrete"
                        p = np.random.choice(parRange, 1)[0]
                    else:
                        raise TypeError("The parameter type '{0}' is not recognised!".format(parType))
                    parList.append(p)
                else:
                    pass
        #If the model uncertainty is concerned.
        if self.__modelunct:
            lnf =  20.0 * np.random.rand() - 10.0
            parList.append(lnf)
        parList = np.array(parList)
        return parList

    def EnsembleSampler(self, nwalkers, **kwargs):
        ndim = len(self.__model.get_parVaryList())
        return emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                    args=[self.__data, self.__model, self.__modelunct], **kwargs)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict
