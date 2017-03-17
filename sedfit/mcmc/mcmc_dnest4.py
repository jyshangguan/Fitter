#This script is not ready...
#

import os
from sys import platform
import numpy as np
import pymultinest
import threading, subprocess

from .. import fit_functions as sedff

#The log_likelihood function
lnlike = sedff.logLFunc
#The log_likelihood function using Gaussian process regression
lnlike_gp = sedff.logLFunc_gp

#DNest4 model#
#------------#
#The class follow the format of DNest4

#The combination of a number of models
def Model2Data_Naive(model, data):
    """
    The function gets the model values that can directly compare with the data.
    """
    if not isinstance(data, DataSet):
        raise ValueError("The data is incorrect!")
    if not isinstance(model, ModelCombiner):
        raise ValueError("The model is incorrect!")
    x = np.array(data.get_List("x"))
    y = model.combineResult(x)
    return y

#The log_likelihood function: naive one
def logLFunc_naive(params, data, model):
    """
    This is the simplest log likelihood function.
    """
    model.updateParList(params)
    nParVary = len(model.get_parVaryList())
    y = np.array(data.get_List("y"))
    e = np.array(data.get_List("e"))
    ym = np.array(Model2Data_Naive(model, data))
    if len(params) == nParVary:
        s = e
    elif len(params) == (nParVary+1):
        f = np.exp(params[nParVary]) #The last par is lnf.
        s = (e**2 + (ym * f)**2)**0.5
    else:
        raise ValueError("The length of params is incorrect!")
    #Calculate the log_likelihood
    logL = -0.5 * np.sum( (y - ym)**2 / s**2 + np.log(2 * np.pi * s**2) )
    return logL

#The DNest4 model class
class DNest4Model(object):
    """
    Specify the model
    """
    def __init__(self, data, model, logl=logLFunc_naive, ModelUnct=False):
        if isinstance(data, DataSet):
            self.__data = data
        else:
            raise TypeError("The data type should be DataSet!")
        if isinstance(model, ModelCombiner):
            self.__model = model
        else:
            raise TypeError("The model type should be ModelCombiner!")
        if isinstance(logl, types.FunctionType):
            self._logl = logl
        else:
            raise TypeError("The model type should be a function!")
        if isinstance(ModelUnct, types.BooleanType):
            self.__modelunct = ModelUnct
            if ModelUnct:
                print "[DNest4Model]: ModelUnct is on!"
            else:
                print "[DNest4Model]: ModelUnct is off!"
        else:
            raise TypeError("The ModelUnct type should be Boolean!")

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
                        p = (r2 - r1) * rng.rand() + r1 #Uniform distribution
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
            lnf =  20.0 * rng.rand() - 10.0
            parList.append(lnf)
        parList = np.array(parList)
        return parList

    def perturb(self, params):
        """
        Each step we perturb all the parameters which is more effective from
        computation point of view.
        """
        logH = 0.0
        parDict = self.__model.get_modelParDict()
        pIndex = 0
        for modelName in self.__model._modelList:
            parFitDict = parDict[modelName]
            for parName in parFitDict.keys():
                if parFitDict[parName]["vary"]:
                    parRange = parFitDict[parName]["range"]
                    parType  = parFitDict[parName]["type"]
                    if parType == "c":
                        #print "[DN4M]: continual"
                        r1, r2 = parRange
                        p0 = params[pIndex]
                        #p0 += (r2 - r1) * dnest4.randh() #Uniform distribution
                        p0 += (r2 - r1) * dnest4.randh() / 2.0 #Uniform distribution
                        params[pIndex] = dnest4.wrap(p0, r1, r2)
                        if (params[pIndex] < r1) or (params[pIndex] > r2):
                            logH -= np.inf
                    elif parType == "d":
                        #print "[DN4M]: discrete"
                        rangeLen = len(parRange)
                        iBng = parRange.index(params[pIndex])
                        #iPro = int( iBng + rangeLen * dnest4.randh() ) #Uniform distribution
                        iPro = int( iBng + rangeLen * dnest4.randh() / 2.0 ) #Uniform distribution
                        iPar = dnest4.wrap(iPro, 0, rangeLen)
                        params[pIndex] = parRange[iPar]
                        if not params[pIndex] in parRange:
                            logH -= np.inf
                    else:
                        raise TypeError("The parameter type '{0}' is not recognised!".format(parType))
                    parFitDict[parName]["value"] = params[pIndex]
                    pIndex += 1
                else:
                    pass
        if len(params) == (pIndex+1):
            p0 = params[pIndex]
            p0 += 20.0 * dnest4.randh()
            params[pIndex] = dnest4.wrap(p0, -10.0, 10.0)
            if (params[pIndex] < -10.0) or (params[pIndex] > 10.0):
                logH -= np.inf
        return logH

    def log_likelihood(self, params):
        """
        Gaussian sampling distrubution.
        """
        logL  = self._logl(params, self.__data, self.__model)
        return logL
