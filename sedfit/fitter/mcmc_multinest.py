import numpy as np
import pymultinest

from .. import fit_functions as sedff

#The log_likelihood function
lnlike = sedff.logLFunc
#The log_likelihood function using Gaussian process regression
lnlike_gp = sedff.logLFunc_gp

#->The object to run PyMultiNest
class MultiNestModel(object):
    """
    The MCMC model for MultiNest.
    """
    def __init__(self, data, model, ModelUnct=False, unctDict=None):
        self.__data = data
        self.__model = model
        self.__modelunct = ModelUnct
        self.__unctDict = unctDict
        if ModelUnct:
            self.__dim = len(model.get_parVaryList()) + 3
            #self.__dim = len(model.get_parVaryList()) + 2
            print("[MultiNestModel]: ModelUnct is on!")
        else:
            self.__dim = len(model.get_parVaryList())
            print "[MultiNestModel]: ModelUnct is off!"

    def prior(self, cube, ndim, nparams):
        """
        The prior of all the parameters are uniform.
        """
        parDict = self.__model.get_modelParDict()
        unctDict = self.__unctDict
        counter = 0
        for modelName in self.__model._modelList:
            parFitDict = parDict[modelName]
            for parName in parFitDict.keys():
                if parFitDict[parName]["vary"]:
                    r1, r2 = parFitDict[parName]["range"]
                    cube[counter] = (r2 - r1) * cube[counter] + r1 #Uniform distribution
                    counter += 1
                else:
                    pass
        #If the model uncertainty is concerned.
        if self.__modelunct:
            if unctDict is None:
                raise Error("No uncertainty model parameter range is provided!")
            parList = ["lnf", "lna", "lntau"]
            for pn in parList:
                r1, r2 = unctDict[pn]
                cube[counter] = (r1 - r2) * cube[counter] + r2
                counter += 1

    def loglike(self, cube, ndim, nparams):
        params = []
        for i in range(ndim):
            params.append(cube[i])
        #print("The cube is: {0}".format(params))
        if self.__modelunct:
            return lnlike_gp(params, self.__data, self.__model)
        else:
            return lnlike(params, self.__data, self.__model)

    def run(self, **kwargs):
        pymultinest.run(self.loglike, self.prior, self.__dim, self.__dim, **kwargs)
