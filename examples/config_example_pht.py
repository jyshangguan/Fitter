################################################################################
## This is config is an example of photometric SED fitting.
## The data used is IRSA13120-5453 a luminous infrared galaxy.
## The adopted models are:
##   BC03    -- Stellar emisison
##   Smith07 -- MIR extinction
##   Cat3d_H -- Dust torus extinction
##   DL07    -- Cold dust emission
##
## The example is created by SGJY at Feb-17-2018 in KIAA-PKU.
################################################################################

import numpy as np
from collections import OrderedDict

################################################################################
#                                    Data                                      #
################################################################################
targname = "IRSA13120-5453"
redshift = 0.03076
distance = 143.6 #Luminosity distance
sedFile  = "examples/{0}_obs.csed".format(targname)
dataDict = {
    "phtName": "Phot",
    "spcName": None, #"IRS",
    "bandList_use": [],
    "bandList_ignore":[],
    "frame": "obs",
}

################################################################################
#                                   Model                                      #
################################################################################
waveModel = 10**np.linspace(-0.5, 3.0, 1000)
#parAddDict_all = {}
modelDict = OrderedDict(
    (
        ("BC03", {
                "function": "BC03",
                "logMs":{
                    "value": 11.45,
                    "range": [6., 14.],
                    "type": "c",
                    "vary": True,
                    "latex": r"$\mathrm{log}\,M_\mathrm{s}$",
                },
                "age":{
                    "value": 5,
                    "range": [0.3, 10.],
                    "type": "c",
                    "vary": False, #True, #
                    "latex": r"$Age$",
                },
            }
        ),
        ("Extinction", {
                "function": "Smith07",
                "logtau": {
                    "value": -3.0,
                    "range": [-4.0, 1.5],  # [-4.0, 1.5],
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$\mathrm{log}\,\tau_\mathrm{ext}$",
                },
                "multiply": ["Torus", "DL07"]
            }
        ),
        ("Torus", {
                "function": "Cat3d_H",
                "logL": {
                    "value": 44.,
                    "range": [38., 48.],
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$\mathrm{log}\,L$",
                },
                "i": {
                    "value": 30,
                    "range": [0, 90],  # [47.0, 48.0], #
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$i$",
                },
                "N0": {
                    "value": 7,
                    "range": [4., 11],  # [6.42, 6.44], #
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$N_0$",
                },
                "h": {
                    "value": 0,
                    "range": [0, 1.75],
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$h$",
                },
                "a": {
                    "value": -0.50,
                    "range": [-2.75, 0.0],
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$a$",
                }
            }
        ),
        ("DL07", {
                "function": "DL07",
                "logumin": {
                    "value": 0.0,
                    "range": [-1.0, 1.4], #log[0.1, 25]
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\mathrm{log}\,U_\mathrm{min}$",
                },
                "logumax": {
                    "value": 6,
                    "range": [3, 6], #log[1e3, 1e6]
                    "type": "c",
                    "vary": False, #True, #
                    "latex": r"$\mathrm{log}\,U_\mathrm{max}$",
                },
                "qpah": {
                    "value": 0.47, #10**0.504,
                    "range": [0.3, 4.8],#10**[-1.0, 0.661],
                    "type": "c",
                    "vary": False, #True, #
                    "latex": r"$q_\mathrm{PAH}$",
                },
                "loggamma": {
                    "value": -1.5,
                    "range": [-5.0, 0.0], #[0.01, 0.03],
                    "type": "c",
                    "vary": False, #True, #
                    "latex": r"$\mathrm{log}\,\gamma$",
                },
                "logMd": {
                    "value": 8.70,
                    "range": [5.0, 11.0], #[9.0, 10.0],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\mathrm{log}\,M_\mathrm{d}$",
                }
            }
        ),
    )
)
parTruth = None  #Whether to provide the truth of the model
modelUnct = True #Whether to consider the model uncertainty in the fitting
unctDict = OrderedDict(
    (
        ("lnf"  , [-10, 2]),
        ("lna"  , [-10, 5]),
        ("lntau", [-5, 2.5]),
    )
)

################################################################################
#                                   emcee                                      #
################################################################################

#emcee options#
#-------------#
burnin = OrderedDict(
    (
        ("sampler"  , "EnsembleSampler"),
        ("nwalkers" , 128), #The number of walkers.
        ("iteration", [600, 600]), #[5000, 5000, 1000]), #The iteration of burn-in run.
        ("thin"     , 1), #To thin the recorded sample.
        ("ball-r"   , 0.1), #The radius of the ball as the fraction of the full range.
    )
)
final = OrderedDict(
    (
        ("sampler"  , "EnsembleSampler"),
        ("ntemps"   , 4),
        ("nwalkers" , 128),
        ("iteration", [1000]), #[1000, 600]),
        ("thin"     , 1),
        ("ball-r"   , 0.01),
    )
)
setup = OrderedDict(
    (
        ("threads"  , 4), #Not used if MPI is using.
        ("printfrac", 0.1),
        ("pslow"    , 16),
        ("pscenter" , 50),
        ("pshigh"   , 84),
    )
)
emceeDict = OrderedDict(
    (
        ("BurnIn", burnin),
        ("Final", final),
        ("Setup", setup),
    )
)

#Postprocess#
#-----------#
ppDict = {
    "burn-in" : 500, #This should not be larger than half of the final iteration.
    "low"     : 16,
    "center"  : 50,
    "high"    : 84,
    "nuisance": True, #False, #
    "fraction": 0, #The fraction of walkers to be dropped.
    "savepath": "examples/results/"
}
