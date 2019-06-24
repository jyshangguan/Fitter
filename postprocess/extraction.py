#!/Users/jinyi/anaconda/bin/python

from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib_version = eval(matplotlib.__version__.split(".")[0])
if matplotlib_version > 1:
    plt.style.use("classic")
plt.rc('font',family='Times New Roman')
import sys
import types
import numpy as np
import cPickle as pickle
import sedfit.SED_Toolkit as sedt
from sedfit.mcmc import mcmc_emcee as mcmc
from PostProcessTools import *

ls_mic = 2.99792458e14 #micron/s

#Parse the commands#
#-------------------#
fitrsFile = sys.argv[1]
fp = open(fitrsFile, "r")
fitrs = pickle.load(fp)
fp.close()

if len(sys.argv) == 3:
    try:
        nSamples = int(sys.argv[2])
    except:
        raise ValueError("The second argument ({0}) is not an integer!".format(sys.argv[2]))
else:
    nSamples = 100
#--> Dump the model dict
dumpModelDict(fitrs)

#The code starts#
#################
print("#################################")
print("# Galaxy SED Fitter Extraction  #")
print("#################################")

readme = """
The structure of the dict is as follows:
{
 'dataPck': {...},  # Information of the target and the data.
 'waveModel': [...], # The wavelength of the models.
 'Best-Fit': {
    'Total': [...],  # The best-fit total model.
    'Components': {...},  # The best-fit model of each components.
    'Photometry': [...]   # The synthetic photometric result of best-fit model.
    },
 'Variation': {0: {
                    'Total': [...],  # The total model of one set of randomly sampled parameters.
                    'Components': {...},  # The model components of one set of randomly sampled parameters.
                    'Photometry': [...]   # The synthetic photometric result of one set of randomly sampled parameters.
                    },
               1: {'Total': [...], 'Components': {...}},
               2: {'Total': [...], 'Components': {...}}
               ...  # In total %d randomly sampled models.
               },
 'readme': '...'  # This note.
}
"""%(nSamples)

silent = True
dataPck = fitrs["dataPck"]
targname = dataPck["targname"]
redshift = dataPck["redshift"]
distance = dataPck["distance"]
dataDict = dataPck["dataDict"]
modelPck = fitrs["modelPck"]
print("The target info:")
print("Name: {0}".format(targname))
print("Redshift: {0}".format(redshift))
print("Distance: {0}".format(distance))
print("Extracting {0} models as uncertainty.".format(nSamples))
print(readme)

#-> Load the data
sedData = dataLoader(fitrs, silent)

#-> Load the model
sedModel = modelLoader(fitrs, silent)
cleanTempFile()
parTruth = modelPck["parTruth"]   #Whether to provide the truth of the model
modelUnct = False #modelPck["modelUnct"] #Whether to consider the model uncertainty in the fitting

#-> Build the emcee object
em = mcmc.EmceeModel(sedData, sedModel, modelUnct)

#-> Extract the models
waveModel = modelPck["waveModel"]
extractDict = {# The dict of extracted data.
    "dataPck": fitrs["dataPck"],
    "waveModel": waveModel,
    "readme": readme
}
#--> Best-fit model
extractDict["Best-Fit"] = {}
fraction = 0
burnIn = 0
ps = fitrs["posterior_sample"]
pcnt = em.p_median(ps, burnin=burnIn, fraction=fraction)
extractDict["Best-Fit"]["Total"] = sedModel.combineResult(x=waveModel)
extractDict["Best-Fit"]["Components"] = sedModel.componentResult(x=waveModel)
extractDict["Best-Fit"]["Photometry"] = sedData.model_pht(waveModel, extractDict["Best-Fit"]["Total"])
#--> Model variation
extractDict["Variation"] = {}
counter = 0
for pars in ps[np.random.randint(len(ps), size=nSamples)]:
    sedModel.updateParList(pars)
    ytotal = sedModel.combineResult(x=waveModel)
    extractDict["Variation"][counter] = {
        "Total": ytotal,
        "Components": sedModel.componentResult(x=waveModel),
        "Photometry": sedData.model_pht(waveModel, ytotal)
    }
    counter += 1

#-> Save the extracted data
fp = open("{0}.xtr".format(targname), "w")
fitrs = pickle.dump(extractDict, fp)
fp.close()
