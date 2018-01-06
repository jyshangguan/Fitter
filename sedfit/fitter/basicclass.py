#The code comes from Composite_Model_Fit/dl07/dev_DataClass.ipynb

import types
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
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
            raise ValueError("The flag should be 0 or 1!")

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def get_e(self):
        return self.__e

    def get_f(self):
        return self.__f

    def gdu(self):
        """
        Get the data unit.
        """
        return (self.__x, self.__y, self.__e, self.__f)

    def set_flag(self, f):
        """
        Change the flag of the data.
        """
        if (f==1) | (f==0):
            self.__f = int(f)
        else:
            raise ValueError("The flag should be 0 or 1!")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

#The discrete data set unit
class DiscreteSet(object):
    def __init__(self, nameList, xList, yList, eList, fList, dataType=None):
        self.__nameList = nameList
        self.__defaultType = ["name", "x", "y", "e", "f"]
        if dataType is None:
            self.__userType = []
        else:
            if len(dataType) != 5:
                raise ValueError("The dataType should contain 5 strings!")
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
            raise ValueError("The inputs do not match in length!")

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
        """
        Get the data of one unit or one array of data for all the units.
        """
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
            raise ValueError("The item is not recognised!")
        return item

    def get_dataUnitDict(self):
        return self.__dataUnitDict

    def get_dataDict(self):
        """
        Generate the data dict.
        """
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
            raise ValueError("The fList size is incorrect!")
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
        self.__defaultType = ["x", "y", "e", "f"]
        if dataType is None:
            self.__userType = []
        else:
            if len(dataType) != 4:
                raise ValueError("The dataType should contain 4 strings!")
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
            raise ValueError("The inputs do not match in length!")

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
        """
        Get the data of one unit or one array of data for all the units.
        """
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
            raise ValueError("The item is not recognised!")
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
            raise ValueError("The fList size is incorrect!")
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
        self.__dataType = ["x", "y", "e", "f"]
        self.__discreteSetDict = {}
        for dSetName in dSetDict.keys():
            dSet = dSetDict[dSetName]
            if isinstance(dSet, DiscreteSet):
                self.__discreteSetDict[dSetName] = dSet
            else:
                raise ValueError("The {0} discrete set is incorrect!".format(dSetName))
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
                raise ValueError("The {0} discrete set is incorrect!".format(dSetName))

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

    def check_dsData(self):
        """
        Return the number of discrete data sets.
        """
        return len(self.__discreteSetDict.keys())

    def check_csData(self):
        """
        Return the number of continual data sets.
        """
        return len(self.__continueSetDict.keys())

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict


#Model class#
#-----------#

#The basic class of model
class ModelFunction(object):
    """
    The functions of each model component.  The model is multiplicative if
    multiList is used.

    Parameters
    ----------
    function : string
        The variable name of the function.
    xName : string
        The name of the active variable.
    parFitDict : dict
        The name of the fitting variables.
    parAddDict : dict
        The name of the additional variables for this model.
    multiList (optional): list
        If provided, the model is multiplicative. The model will be
        multiplied to the models in the list.  Otherwise, the model will be
        added with other models that are not multiplicative.

    Notes
    ----
    Revised by SGJY at Jan. 5, 2018 in KIAA-PKU.
    """
    def __init__(self, function, xName, parFitDict={}, parAddDict={}, multiList=None):
        self.__function = function
        self.xName = xName
        self.parFitDict = parFitDict
        self.parAddDict = parAddDict
        self.multiList  = multiList

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

    def if_Add(self):
        """
        Check whether the function is to add or multiply.
        """
        if self.multiList is None:
            return True
        else:
            return False

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

class ModelCombiner(object):
    """
    The object to combine all the model components.

    Parameters
    ----------
    modelDict : dict (better to be ordered dict)
        The dict containing all the model functions.
    xList : array like
        The array of the default active variable.
    QuietMode : bool
        Use verbose mode if False.

    Notes
    -----
    None.
    """
    def __init__(self, modelDict, xList, QuietMode=True):
        self.__modelDict = modelDict
        self._modelList = modelDict.keys()
        self.__x = xList
        self._mltList = [] # The list of models to multiply to other models
        self._addList = [] # The list of models to add together
        for modelName in self._modelList:
            mf = modelDict[modelName]
            if mf.if_Add():
                self._addList.append(modelName)
            else:
                self._mltList.append(modelName)
        if not QuietMode:
            print "Add: {0}".format(self._addList)
            print "Multiply: {0}".format(self._mltList)

    def get_xList(self):
        """
        Get the array of the default active variable.

        Parameters
        ----------
        None.

        Returns
        -------
        The array of the default active variable.
        """
        return self.__x

    def set_xList(self, xList):
        """
        Reset the default active variable array.

        Parameters
        ----------
        xList : array like
            The new array of the active variable.

        Returns
        -------
        None.
        """
        self.__x = xList

    def combineResult(self, x=None):
        """
        Return the model result combining all the components.

        Parameters
        ----------
        x (optional) : array like
            The active variable of the models.

        Returns
        -------
        result : array like
            The result of the models all combined.

        Notes
        -----
        None.
        """
        if x is None:
            x = self.__x
        #-> Calculate the add model components
        addCmpDict = {}
        for modelName in self._addList:
            mf = self.__modelDict[modelName]
            addCmpDict[modelName] = mf(x)
        #-> Manipulate the model components
        for modelName in self._mltList:
            mf = self.__modelDict[modelName]
            my = mf(x) # multiplied y component
            #--> Multiply the current component to the target models
            for tmn in mf.multiList:
                addCmpDict[tmn] *= my
        #-> Add up all the add models
        result = np.zeros_like(x)
        for modelName in self._addList:
            result += addCmpDict[modelName]
        return result

    def componentResult(self, x=None):
        """
        Return the results of all the add components multiplied by the
        multiplicative models correspondingly.

        Parameters
        ----------
        x (optional) : array like
            The active variable of the models.

        Returns
        -------
        result : ordered dict
            The result of the model components.

        Notes
        -----
        None.
        """
        if x is None:
            x = self.__x
        #-> Calculate the add model components
        result = OrderedDict()
        for modelName in self._addList:
            mf = self.__modelDict[modelName]
            result[modelName] = mf(x)
        #-> Manipulate the model components
        for modelName in self._mltList:
            mf = self.__modelDict[modelName]
            my = mf(x) # multiplied y component
            #--> Multiply the current component to the target models
            for tmn in mf.multiList:
                result[tmn] *= my
        return result

    def componentAddResult(self, x=None):
        """
        Return the original results of add models without multiplied other models.

        Parameters
        ----------
        x (optional) : array like
            The active variable of the models.

        Returns
        -------
        result : ordered dict
            The result of the model components.

        Notes
        -----
        None.
        """
        if x is None:
            x = self.__x
        result = OrderedDict()
        for modelName in self._addList:
            result[modelName] = self.__modelDict[modelName](x)
        return result

    def componentMltResult(self, x=None):
        """
        Return the original results of multiplicative models.


        Parameters
        ----------
        x (optional) : array like
            The active variable of the models.

        Returns
        -------
        result : ordered dict
            The result of the model components.

        Notes
        -----
        None.
        """
        if x is None:
            x = self.__x
        result = OrderedDict()
        for modelName in self._mltList:
            result[modelName] = self.__modelDict[modelName](x)
        return result

    def get_modelDict(self):
        """
        Get the dict of all the models.
        """
        return self.__modelDict

    def get_modelAddList(self):
        """
        Get the name list of the add models.
        """
        return self._addList

    def get_modelMltList(self):
        """
        Get the name list of the multiply models.
        """
        return self._mltList

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

    def get_parVaryRanges(self):
        """
        Return a list of ranges for all the variable parameters.
        """
        parRList = []
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict = model.parFitDict
            for parName in modelParDict.keys():
                if modelParDict[parName]["vary"]:
                    parRList.append(modelParDict[parName]["range"])
                else:
                    pass
        return parRList

    def get_parVaryNames(self, latex=True):
        """
        Return a list of names for all the variable parameters. The latex format
        is preferred. If the latex format is not found, the variable name is used.
        """
        parNList = []
        for modelName in self._modelList:
            model = self.__modelDict[modelName]
            modelParDict = model.parFitDict
            for parName in modelParDict.keys():
                if modelParDict[parName]["vary"]:
                    if latex:
                        name = modelParDict[parName].get("latex", parName)
                    else:
                        name = parName
                    parNList.append(name)
                else:
                    pass
        return parNList

    def updateParFit(self, modelName, parName, parValue, QuietMode=True):
        model = self.__modelDict[modelName]
        if not QuietMode:
            orgValue = model.parFitDict[parName]
            print "[{0}][{1}] {2}->{3}".format(modelName, parName, orgValue, parValue)
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
            print "[{0}][{1}] {2}->{3}".format(modelName, parName, orgValue, parValue)
        model.parAddDict[parName] = parValue

    def plot(self, x=None, colorList=None, FigAx=None, DisplayPars=False,
             tKwargs=None, cKwargs={}, useLabel=True):
        if x is None:
            x = self.__x
        if FigAx is None:
            fig = plt.figure()
            ax = plt.gca()
            FigAx = (fig, ax)
        else:
            fig, ax = FigAx
        modelDict = self.__modelDict
        modelList = self.get_modelAddList() #modelDict.keys()
        TextIterm = lambda text, v1, v2: text.format(v1, v2)
        textList = []
        #yTotal = np.zeros_like(x)
        yTotal = self.combineResult(x=x)
        yCmpnt = self.componentResult(x=x) #The best-fit components
        if colorList is None:
            colorList = ["orange", "green", "blue", "magenta", "yellow", "cyan"]
        nColor = len(colorList)
        counter = 0
        for modelName in modelList:
            textList.append( "<{0}>\n".format(modelName) )
            mf = modelDict[modelName]
            parFitDict = mf.parFitDict
            for parName in parFitDict.keys():
                textList.append( TextIterm("{0}: {1:.2f}\n", parName,
                parFitDict[parName]["value"]) )
            y = yCmpnt[modelName]
            if useLabel:
                cLabel = modelName
            else:
                cLabel = None
            ax.plot(x, y, color=colorList[counter%nColor], label=cLabel, **cKwargs)
            counter += 1
        if useLabel:
            tLabel = "Total"
        else:
            tLabel = None
        if tKwargs is None:
            ax.plot(x, yTotal, color="k", label=tLabel)
        else:
            ax.plot(x, yTotal, label=tLabel, **tKwargs)
        #fig.set_size_inches(8, 6)
        text = "".join(textList)
        if DisplayPars:
            ax.text(1.02, 1.0, text, #bbox=dict(facecolor="white", alpha=0.75),
                    verticalalignment="top", horizontalalignment="left",
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

    Parameters
    ----------
    input_model_dict : dict (better to be ordered dict)
        The dict of input model informations.
        An example of the format of the dict elements:
            "Slope": {                    # The name of the model is arbitrary.
                "function": "Linear"      # Necessary to be exactly the same as
                                          # the name of the variable.
                "a": {                    # The name of the first parameter.
                    "value": 3.,          # The value of the parameter.
                    "range": [-10., 10.], # The prior range of the parameter.
                    "type": "c",          # The type (continuous/discrete) of
                                          # the parameter. Currently, it does
                                          # not matter...
                    "vary": True,         # The toggle whether the parameter is
                                          # fixed (if False).
                    "latex": "$a$",       # The format for plotting.
                }
                "b": {...}                # The same format as "a".
            }
    func_lib : dict
        The dict of the information of the functions.
        An example of the format of the dict elements:
            "Linear":{                   # The function name should be exactly
                                         # the same as the name of the function
                                         # variable it refers to.
                "function": Linear,      # The function variable, not necessary.
                "x_name": "x",           # The active variable of the function.
                "param_fit": ["a", "b"], # The name of the parameters that are
                                         # involved in fitting.
                "param_add": [],         # The name of the additional parameters
                                         # necessary for the function.
                "operation": ["+"]       # The operation expected for this
                                         # function, for consistency check.
                                         # "+": to add with other "+" components.
                                         # "*": to multiply to other "+"
                                         # components. One model can be both "+"
                                         # and "*".
    x_list : array like
        The default active variable for the model.
    par_add_dict_all : dict
        The additional parameters for all the models in input_model_dict.
    **kwargs : dict
        Additional keywords for the ModelCombiner.

    Returns
    -------
    sed_model : ModelCombiner object
        The combined SED model.

    Notes
    -----
    None.
    """
    modelDict = OrderedDict()
    modelNameList = input_model_dict.keys()
    for modelName in modelNameList:
        funcName = input_model_dict[modelName]["function"]
        funcInfo = func_lib[funcName]
        xName = funcInfo["x_name"]
        #-> Build up the parameter dictionaries
        parFitList = funcInfo["param_fit"]
        parAddList = funcInfo["param_add"]
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
        #-> Check the consistency if the component is multiply
        multiList = input_model_dict[modelName].get("multiply", None)
        if not multiList is None:
            #--> The "*" should be included in the operation list.
            assert "*" in funcInfo["operation"]
            print "{0} is multiplied to {1}!".format(modelName, multiList)
            #--> Check further the target models are not multiplicative.
            for tmn in multiList:
                f_mlt = input_model_dict[tmn].get("multiply", None)
                if not f_mlt is None:
                    raise ValueError("The multiList includes a multiplicative model ({0})!".format(tmn))
        #modelDict[modelName] = ModelFunction(funcInfo["function"], xName, parFitDict, parAddDict)
        modelDict[modelName] = ModelFunction(funcName, xName, parFitDict,
                                             parAddDict, multiList)
    sed_model = ModelCombiner(modelDict, x_list, **kwargs)
    return sed_model
