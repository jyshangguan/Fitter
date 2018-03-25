## The class of the models for the SED fitting.

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from fitter import basicclass as bc
from SED_Toolkit import WaveFromMicron, WaveToMicron
__all__ = ["SedModel"]

ls_mic = 2.99792458e14 # micron/s


class SedModel(bc.ModelCombiner):
    def __init__(self, input_model_dict, func_lib, x_list, par_add_dict_all={},
                 QuietMode=False, **kwargs):
        """
        Generate the ModelClass object from the input model dict.

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
        This is mainly from bc.Model_Generator.
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
                if not QuietMode:
                    print "[Model_Generator]: {0} is multiplied to {1}!".format(modelName, multiList)
                #--> Check further the target models are not multiplicative.
                for tmn in multiList:
                    f_mlt = input_model_dict[tmn].get("multiply", None)
                    if not f_mlt is None:
                        raise ValueError("The multiList includes a multiplicative model ({0})!".format(tmn))
            modelDict[modelName] = bc.ModelFunction(funcName, xName, parFitDict,
                                                    parAddDict, multiList)
        bc.ModelCombiner.__init__(self, modelDict, x_list, **kwargs)

    def plot(self, wave=None, colorList=None, FigAx=None, DisplayPars=False,
             tKwargs=None, cKwargs={}, useLabel=True, xUnits="micron", yUnits="fnu"):
        """
        Plot the SED model.  The working wavelength units is micron and the
        model output units is assumed mJy.

        Parameters
        ----------
        wave (optional): array_like
            The array of the wavelength, default units: micron.
        colorList (optional): list
            The list of the colors for each model components.
        FigAx (optional): tuple
            The tuple of (fig and ax).
        DisplayPars (optional): bool, default: False
            Display the parameter values in the figure.
        tKwargs (optional): dict or None by default
            The keywords to display the total model results.
        cKwargs (optional): dict
            The keywords to display the component model results.
        useLabel (optional): bool
            If True, add the label of the model into the legend. Defualt: True.
        xUnits (optional): string
            The units of the x-axis, default: micron.  Currently supported units
            are:
                "cm", "mm", "micron", "angstrom", "Hz", "MHz", "GHz"
        yUnits (optional): string
            The form of the y-axis, default: fnu.
                fnu -- mJy
                nufnu -- erg s^-1 cm^-2

        Returns
        -------
        FigAx : tuple
            The tuple of (fig and ax).

        Notes
        -----
        May further adopt more units.
        """
        if wave is None:
            wave = self.get_xList()
        else:
            wave = WaveToMicron(wave, xUnits) # Convert the units to micron
        if FigAx is None:
            fig = plt.figure()
            ax = plt.gca()
            FigAx = (fig, ax)
        else:
            fig, ax = FigAx
        modelDict = self.get_modelDict()
        modelList = self.get_modelAddList() #modelDict.keys()
        TextIterm = lambda text, v1, v2: text.format(v1, v2)
        textList = []
        yTotal = self.combineResult(x=wave)
        yCmpnt = self.componentResult(x=wave) #The best-fit components
        if yUnits == "fnu": # Default settings, units: mJy
            pass
        elif yUnits == "nufnu": # Convert from mJy to erg s^-1 cm^-2
            y_conv = ls_mic / wave * 1.e-26
            yTotal *= y_conv
            for modelName in modelList:
                yCmpnt[modelName] *= y_conv
        else:
            raise ValueError("The yUnits ({0}) is not recognised!".format(yUnits))
        if colorList is None:
            colorList = ["orange", "green", "blue", "magenta", "yellow", "cyan"]
        nColor = len(colorList)
        x = WaveFromMicron(wave, xUnits) # Switch the wavelength units back to
                                         #what assigned
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
        text = "".join(textList)
        if DisplayPars:
            ax.text(1.02, 1.0, text, #bbox=dict(facecolor="white", alpha=0.75),
                    verticalalignment="top", horizontalalignment="left",
                    transform=ax.transAxes, fontsize=14)
        return FigAx


if __name__ == "__main__":
    import sys
    import importlib
    from sedfit.dir_list import root_path
    import sedfit.model_functions as sedmf

    def configImporter(configfile):
        """
        This function import the provided configure file.

        Parameters
        ----------
        configfile : string
            The name of the configure file (with the path).

        Returns
        -------
        config : module object
            The imported module.

        Notes
        -----
        None.
        """
        pathList = configfile.split("/")
        configPath = "/".join(pathList[0:-1])
        sys.path.append(configPath)
        configName = pathList[-1].split(".")[0]
        config = importlib.import_module(configName)
        return config

    config = configImporter("/Users/shangguan/Work/Fitter/configs/config_goals_hon.py")
    funcLib   = sedmf.funcLib
    waveModel = config.waveModel
    modelDict = config.modelDict

    try:
        parAddDict_all = config.parAddDict_all
    except:
        parAddDict_all = {}
    parAddDict_all["DL"]    = config.distance
    parAddDict_all["z"]     = config.redshift
    parAddDict_all["frame"] = "rest"
    sedModel = SedModel(modelDict, funcLib, waveModel, parAddDict_all)
