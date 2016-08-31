#The code comes from Composite_Model_Fit/dl07/dev_DataClass.ipynb

import types
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import dnest4
from dnest4.utils import rng
from .. import model_functions as sedmf


#Data class#
#----------#

#The basic class of data unit
class DataUnit(object):
    def __init__(self, x, y, e, f=1):
        self.__x = x #Data x
        self.__y = y #Data value
        self.__e = e #Error
        if (f==1) | (f==0):
            self.__f = int(f) #Flag to use the data or not
        else:
            raise ValueError('The flag should be 0 or 1!')

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def get_e(self):
        return self.__e

    def get_f(self):
        return self.__f

    def gdu(self):
        '''
        Get the data unit.
        '''
        return (self.__x, self.__y, self.__e, self.__f)

    def set_flag(self, f):
        '''
        Change the flag of the data.
        '''
        if (f==1) | (f==0):
            self.__f = int(f)
        else:
            raise ValueError('The flag should be 0 or 1!')

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

#The discrete data set unit
class DiscreteSet(object):
    def __init__(self, nameList, xList, yList, eList, fList, dataType=None):
        self.__nameList = nameList
        self.__defaultType = ['name', 'x', 'y', 'e', 'f']
        if dataType is None:
            self.__userType = []
        else:
            if len(dataType) != 5:
                raise ValueError('The dataType should contain 5 strings!')
            self.__userType = dataType
            self.__dataMap = {}
            for loop in range(5):
                #Build a map from the new dataType to the defaultType
                self.__dataMap[dataType[loop]] = self.__defaultType[loop]
        args = [xList, yList, eList, fList]
        self.__unitNumber = len(nameList)
        match = True
        for arg in args:
            match = match&(self.__unitNumber==len(arg))
        if not match:
            raise ValueError('The inputs do not match in length!')

        #Generate the dict of discrete unit
        self.__dataUnitDict = {}
        for loop in range(self.__unitNumber):
            name = nameList[loop]
            x = xList[loop]
            y = yList[loop]
            e = eList[loop]
            f = fList[loop]
            self.__dataUnitDict[name] = DataUnit(x, y, e, f)
        self.__xList = []
        self.__yList = []
        self.__eList = []
        self.__fList = []
        for name in nameList:
            x, y, e, f = self.__dataUnitDict[name].gdu()
            self.__xList.append(x)
            self.__yList.append(y)
            self.__eList.append(e)
            self.__fList.append(f)

    def __getitem__(self, i):
        '''
        Get the data of one unit or one array of data for all the units.
        '''
        nameList = self.__nameList
        if i in nameList:
            item = self.__dataUnitDict[i].gdu()
        elif i in self.__userType:
            ncol = self.__userType.index(i) - 1
            item = []
            for name in nameList:
                if ncol == -1:
                    item.append(name)
                else:
                    item.append( self.__dataUnitDict[name].gdu()[ncol] )
        elif i in self.__defaultType:
            ncol = self.__defaultType.index(i) - 1
            item = []
            for name in nameList:
                if ncol == -1:
                    item.append(name)
                else:
                    item.append( self.__dataUnitDict[name].gdu()[ncol] )
        else:
            raise ValueError('The item is not recognised!')
        return item

    def get_dataUnitDict(self):
        return self.__dataUnitDict

    def get_dataDict(self):
        '''
        Generate the data dict.
        '''
        nameList = self.__nameList
        dataDict = {}
        for name in nameList:
            dataDict[name] = self.__dataUnitDict[name].gdu()
        return dataDict

    def get_nameList(self):
        return self.__nameList

    def get_xList(self):
        return self.__xList

    def get_yList(self):
        return self.__yList

    def get_eList(self):
        return self.__eList

    def get_fList(self):
        return self.__fList

    def get_List(self, typeName):
        if typeName in self.__userType:
            exec "requiredList = self.get_{0}List()".format(self.__dataMap[typeName])
        elif typeName in self.__defaultType:
            exec "requiredList = self.get_{0}List()".format(typeName)
        else:
            raise KeyError("The key '{0}' is not found!".format(typeName))
        return requiredList

    def set_fList(self, fList):
        nameList = self.__nameList
        if len(fList) != self.__unitNumber:
            raise ValueError('The fList size is incorrect!')
            return 0
        for loop in range(self.__unitNumber):
            name = nameList[loop]
            self.__dataUnitDict[name].set_flag(fList[loop])
            self.__fList[loop] = int(fList[loop])
        return 1

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

#The continual data set unit
class ContinueSet(object):
    def __init__(self, xList, yList, eList, fList, dataType=None):
        self.__defaultType = ['x', 'y', 'e', 'f']
        if dataType is None:
            self.__userType = []
        else:
            if len(dataType) != 4:
                raise ValueError('The dataType should contain 4 strings!')
            self.__userType = dataType
            self.__dataMap = {}
            for loop in range(4):
                #Build a map from the new dataType to the defaultType
                self.__dataMap[dataType[loop]] = self.__defaultType[loop]
        args = [yList, eList, fList]
        self.__unitNumber = len(xList)
        match = True
        for arg in args:
            match = match&(self.__unitNumber==len(arg))
        if not match:
            raise ValueError('The inputs do not match in length!')

        #Generate the dict of discrete unit
        self.__dataUnitList = []
        for loop in range(self.__unitNumber):
            x = xList[loop]
            y = yList[loop]
            e = eList[loop]
            f = fList[loop]
            self.__dataUnitList.append( DataUnit(x, y, e, f) )
        self.__xList = []
        self.__yList = []
        self.__eList = []
        self.__fList = []
        for dunit in self.__dataUnitList:
            x, y, e, f = dunit.gdu()
            self.__xList.append(x)
            self.__yList.append(y)
            self.__eList.append(e)
            self.__fList.append(f)

    def __getitem__(self, i):
        '''
        Get the data of one unit or one array of data for all the units.
        '''
        if i in self.__userType:
            ncol = self.__userType.index(i)
            item = []
            for loop in range(self.__unitNumber):
                item.append( self.__dataUnitList[loop].gdu()[ncol] )
        if i in self.__defaultType:
            ncol = self.__defaultType.index(i)
            item = []
            for loop in range(self.__unitNumber):
                item.append( self.__dataUnitList[loop].gdu()[ncol] )
        elif (type(i)==types.IntType) & (i > 0) & (i < self.__unitNumber):
            item = self.__dataUnitList[i].gdu()
        else:
            raise ValueError('The item is not recognised!')
        return item

    def get_dataUnitList(self):
        return self.__dataUnitList

    def get_dataDict(self):
        dataDict = {}
        if len(self.__userType) > 0:
            dtList = self.__userType
        else:
            dtList = self.__defaultType
        for loop_dt in range(len(dtList)):
            dt = dtList[loop_dt]
            dataDict[dt] = []
            for loop_un in range(self.__unitNumber):
                dataDict[dt].append(self.__dataUnitList[loop_un].gdu()[loop_dt])
        return dataDict

    def get_xList(self):
        return self.__xList

    def get_yList(self):
        return self.__yList

    def get_eList(self):
        return self.__eList

    def get_fList(self):
        return self.__fList

    def get_List(self, typeName):
        if typeName in self.__userType:
            exec "requiredList = self.get_{0}List()".format(self.__dataMap[typeName])
        elif typeName in self.__defaultType:
            exec "requiredList = self.get_{0}List()".format(typeName)
        else:
            raise KeyError("The key '{0}' is not found!".format(typeName))
        return requiredList

    def set_fList(self, fList):
        if len(fList) != self.__unitNumber:
            raise ValueError('The fList size is incorrect!')
            return 0
        for loop in range(self.__unitNumber):
            self.__dataUnitList[loop].set_flag(fList[loop])
            self.__fList[loop] = int(fList[loop])
        return 1

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

#The total data set that contains a number of "DiscreteSet"s and "ContinueSet"s
class DataSet(object):
    """
    The data set that contains discrete and continual data.

    Parameters
    ----------
    dSetDict : dict
        The dict of DiscreteSet.
    cSetDict : dict
        The dict of ContinueSet.
    """
    def __init__(self, dSetDict={}, cSetDict={}):
        #Check the data format
        self.__dataType = ['x', 'y', 'e', 'f']
        self.__discreteSetDict = {}
        for dSetName in dSetDict.keys():
            dSet = dSetDict[dSetName]
            if isinstance(dSet, DiscreteSet):
                self.__discreteSetDict[dSetName] = dSet
            else:
                raise ValueError('The {0} discrete set is incorrect!'.format(dSetName))
        self.__continueSetDict = {}
        for cSetName in cSetDict.keys():
            cSet = cSetDict[cSetName]
            if isinstance(cSet, ContinueSet):
                self.__continueSetDict[cSetName] = cSet

    def add_DiscreteSet(self, dSetDict):
        for dSetName in dSetDict.keys():
            dSet = dSetDict[dSetName]
            if isinstance(dSet, DiscreteSet):
                self.__discreteSetDict[dSetName] = dSet
            else:
                raise ValueError('The {0} discrete set is incorrect!'.format(dSetName))

    def add_ContinueSet(self, cSetDict):
        for cSetName in cSetDict.keys():
            cSet = cSetDict[cSetName]
            if isinstance(cSet, ContinueSet):
                self.__continueSetDict[cSetName] = cSet

    def get_DiscreteSetDict(self):
        return self.__discreteSetDict

    def get_ContinueSetDict(self):
        return self.__continueSetDict

    def get_dsDict(self):
        dsDict = {}
        for dSetName in self.__discreteSetDict.keys():
            dSet = self.__discreteSetDict[dSetName]
            dsDict[dSetName] = dSet.get_dataDict()
        return dsDict

    def get_csDict(self):
        csDict = {}
        for cSetName in self.__continueSetDict.keys():
            cSet = self.__continueSetDict[cSetName]
            csDict[cSetName] = cSet.get_dataDict()
        return csDict

    def get_dsArrays(self):
        dsaDict = {}
        for dSetName in self.__discreteSetDict.keys():
            dSet = self.__discreteSetDict[dSetName]
            dsArray = []
            for d in self.__dataType:
                exec "dsArray.append(dSet.get_{0}List())".format(d)
            dsaDict[dSetName] = dsArray
        return dsaDict

    def get_csArrays(self):
        csaDict = {}
        for cSetName in self.__continueSetDict.keys():
            cSet = self.__continueSetDict[cSetName]
            csArray = []
            for d in self.__dataType:
                exec "csArray.append(cSet.get_{0}List())".format(d)
            csaDict[cSetName] = csArray
        return csaDict

    def get_unitNameList(self):
        unlList = []
        for dSetName in self.__discreteSetDict.keys():
            dSet = self.__discreteSetDict[dSetName]
            unlList.append(dSet.get_nameList())
        unList = []
        [unList.extend(unl) for unl in unlList]
        return unList

    def get_dsList(self, typeName):
        if not typeName in self.__dataType:
            raise KeyError("The key '{0}' is not a data type!".format(typeName))
        dslList = []
        for dSetName in self.__discreteSetDict.keys():
            dSet = self.__discreteSetDict[dSetName]
            dslList.append(dSet.get_List(typeName))
        dsList = []
        [dsList.extend(dsl) for dsl in dslList]
        return dsList

    def get_csList(self, typeName):
        if not typeName in self.__dataType:
            raise KeyError("The key '{0}' is not a data type!".format(typeName))
        cslList = []
        for cSetName in self.__continueSetDict.keys():
            cSet = self.__continueSetDict[cSetName]
            cslList.append(cSet.get_List(typeName))
        csList = []
        [csList.extend(csl) for csl in cslList]
        return csList

    def get_List(self, typeName):
        if not typeName in self.__dataType:
            raise KeyError("The key '{0}' is not a data type!".format(typeName))
        dsList = self.get_dsList(typeName)
        csList = self.get_csList(typeName)
        return dsList + csList

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict


#Model class#
#-----------#

#The basic class of model
class ModelFunction(object):
    def __init__(self, function, xName, parFitDict={}, parAddDict={}):
        self.__function = function
        self.xName = xName
        self.parFitDict = parFitDict
        self.parAddDict = parAddDict

    def __call__(self, x):
        kwargs = {}
        #Add in the parameters for fit
        kwargs[self.xName] = x
        for parName in self.parFitDict.keys():
            kwargs[parName] = self.parFitDict[parName]["value"]
        for parName in self.parAddDict.keys():
            kwargs[parName] = self.parAddDict[parName]
        exec "y = sedmf.{0}(**kwargs)".format(self.__function)
        return y

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

"""
class ModelFunction(object):
    def __init__(self, function, xName, parFitDict={}, parAddDict={}):
        self.__function = function
        self.__funcName = function.func_code.co_name
        self.xName = xName
        #Check whether the parameters match the function.
        funcParList = function.func_code.co_varnames
        ##Get a list of all the parameters.
        parList = parFitDict.keys() + parAddDict.keys()
        ##Check whether there is unexpected parameters.
        for loop in range( len(parList) ):
            parName = parList[loop]
            if not parName in funcParList:
                raise TypeError("{0}() got an unexpected keyword argument '{1}'.".format(self.__funcName, parName))
        ##Set the parameters after the checks.
        self.parFitDict = parFitDict
        self.parAddDict = parAddDict

    def result(self, x):
        kwargs = {}
        #Add in the parameters for fit
        kwargs[self.xName] = x
        for parName in self.parFitDict.keys():
            kwargs[parName] = self.parFitDict[parName]["value"]
        for parName in self.parAddDict.keys():
            kwargs[parName] = self.parAddDict[parName]
        y = self.__function(**kwargs)
        return y
"""

class ModelCombiner(object):
    def __init__(self, modelDict, xList):
        self.__modelDict = modelDict
        self._modelList = modelDict.keys()
        self.__x = xList

    def get_xList(self):
        return self.__x

    def set_xList(self, xList):
        self.__x = xList

    def combineResult(self, x=None):
        """
        Return the model result combining all the components.
        """
        if x is None:
            x = self.__x
        result = np.zeros_like(x)
        for modelName in self.__modelDict.keys():
            mf = self.__modelDict[modelName]
            result += mf(x)
        return result

    def componentResult(self, x=None):
        """
        Return the model results of all the components separately.
        """
        result = OrderedDict()
        if x is None:
            x = self.__x
        for modelName in self.__modelDict.keys():
            mf = self.__modelDict[modelName]
            result[modelName] = mf(x)
        return result

    def get_modelDict(self):
        return self.__modelDict

    def get_modelParDict(self):
        modelParDict = OrderedDict()
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict[modelName] = model.parFitDict
        return modelParDict

    def get_parList(self):
        """
        Return the total number of the fit parameters.
        """
        parList = []
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict = model.parFitDict
            for parName in modelParDict.keys():
                parList.append(modelParDict[parName]["value"])
        return parList

    def get_parVaryList(self):
        """
        Return the total number of the fit parameters that can vary.
        """
        parList = []
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict = model.parFitDict
            for parName in modelParDict.keys():
                if modelParDict[parName]["vary"]:
                    parList.append(modelParDict[parName]["value"])
                else:
                    pass
        return parList

    def updateParFit(self, modelName, parName, parValue, QuietMode=True):
        model = self.__modelDict[modelName]
        if not QuietMode:
            orgValue = model.parFitDict[parName]
            print '[{0}][{1}] {2}->{3}'.format(modelName, parName, orgValue, parValue)
        if model.parFitDict[parName]["vary"]:
            model.parFitDict[parName]["value"] = parValue
        else:
            raise RuntimeError("[ModelCombiner]: {0}-{1} is fixed!".format(modelName, parName))

    def updateParList(self, parList):
        """
        Updata the fit parameters from a list.
        """
        counter = 0
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict = model.parFitDict
            for parName in modelParDict.keys():
                if modelParDict[parName]["vary"]:
                    modelParDict[parName]["value"] = parList[counter]
                    counter += 1
                else:
                    pass

    def updateParAdd(self, modelName, parName, parValue, QuietMode=True):
        model = self.__modelDict[modelName]
        if not QuietMode:
            orgValue = model.parAddDict[parName]
            print '[{0}][{1}] {2}->{3}'.format(modelName, parName, orgValue, parValue)
        model.parAddDict[parName] = parValue

    def plot(self, x=None, FigAx=None, DisplayPars=True):
        if x is None:
            x = self.__x
        if FigAx is None:
            fig = plt.figure()
            ax = plt.gca()
            FigAx = (fig, ax)
        else:
            fig, ax = FigAx
        modelDict = self.__modelDict
        modelList = modelDict.keys()
        TextIterm = lambda text, v1, v2: text.format(v1, v2)
        textList = []
        yTotal = np.zeros_like(x)
        colorList = ['r', 'g', 'b', 'c', 'm']
        nColor = len(colorList)
        counter = 0
        for modelName in modelList:
            textList.append( '<{0}>\n'.format(modelName) )
            mf = modelDict[modelName]
            parFitDict = mf.parFitDict
            for parName in parFitDict.keys():
                textList.append( TextIterm('{0}: {1:.2f}\n', parName,
                parFitDict[parName]["value"]) )
            y = mf(x)
            yTotal += y
            ax.plot(x, y, color=colorList[counter%nColor], label=modelName)
            ax.set_xscale('log')
            ax.set_yscale('log')
            counter += 1
        ax.plot(x, yTotal, color='k', linewidth=1.5, label='Total')
        fig.set_size_inches(8, 6)
        text = "".join(textList)
        if DisplayPars:
            ax.text(1.02, 1.0, text, #bbox=dict(facecolor='white', alpha=0.75),
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes, fontsize=14)
        return FigAx

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

#The function generate the ModelCombiner from input model dict
def Model_Generator(input_model_dict, func_lib, x_list, par_add_dict_all={}, **kwargs):
    """
    Generate the ModelCombiner object from the input model dict.
    """
    modelDict = OrderedDict()
    modelNameList = input_model_dict.keys()
    for modelName in modelNameList:
        funcName = input_model_dict[modelName]['function']
        funcInfo = func_lib[funcName]
        xName = funcInfo['x_name']
        parFitList = funcInfo['param_fit']
        parAddList = funcInfo['param_add']
        parFitDict = OrderedDict()
        parAddDict = {}
        for parName in parFitList:
            parFitDict[parName] = input_model_dict[modelName][parName]
        for parName in parAddList:
            par_add_iterm = par_add_dict_all.get(parName, "No this parameter")
            if par_add_iterm == "No this parameter":
                pass
            else:
                parAddDict[parName] = par_add_iterm
        #modelDict[modelName] = ModelFunction(funcInfo['function'], xName, parFitDict, parAddDict)
        modelDict[modelName] = ModelFunction(funcName, xName, parFitDict, parAddDict)
    sed_model = ModelCombiner(modelDict, x_list, **kwargs)
    return sed_model

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
    x = np.array(data.get_List('x'))
    y = model.combineResult(x)
    return y

#The log_likelihood function: naive one
def logLFunc_naive(params, data, model):
    """
    This is the simplest log likelihood function.
    """
    model.updateParList(params)
    nParVary = len(model.get_parVaryList())
    y = np.array(data.get_List('y'))
    e = np.array(data.get_List('e'))
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
