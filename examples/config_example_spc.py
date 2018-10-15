################################################################################
## This is config is an example of full SED fitting.
## The data used is IRSA13120-5453 a luminous infrared galaxy.
## The adopted models are:
##   BC03         -- Stellar emisison
##   Smith07      -- MIR extinction (two components applied to the latter two)
##   Cat3d_H_wind -- Dust torus extinction
##   DL07         -- Cold dust emission
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
    "spcName": "IRS",
    "bandList_use": [],
    "bandList_ignore":["WISE_w3", "WISE_w4"],
    "frame": "obs",
}

################################################################################
#                                   Model                                      #
################################################################################
waveModel = 10**np.linspace(-0.1, 7.0, 1000)
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
        ("Ext_torus", {
                "function": "Smith07",
                "logtau": {
                    "value": -3.0,
                    "range": [-4.0, 1.5],  # [-4.0, 1.5],
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$\mathrm{log}\,\tau_\mathrm{torus}$",
                },
                "multiply": ["Torus"]
            }
        ),
        ("Ext_dl07", {
                "function": "Smith07",
                "logtau": {
                    "value": -3.0,
                    "range": [-4.0, 1.5],  # [-4.0, 1.5],
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$\mathrm{log}\,\tau_\mathrm{DL07}$",
                },
                "multiply": ["DL07"]
            }
        ),
        ("Torus", {
                "function": "Cat3d_H_wind",
                "a": {
                    "value": -2.00,
                    "range": [-3.25, 0.0],
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"a",
                },
                "h": {
                    "value": 0.3,
                    "range": [0., 0.6],
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$h$",
                },
                "N0": {
                    "value": 7,
                    "range": [4., 11],  # [6.42, 6.44], #
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$N_0$",
                },
                "i": {
                    "value": 30,
                    "range": [0, 90],  # [47.0, 48.0], #
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$i$",
                },
                "fwd": {
                    "value": 0.45,
                    "range": [0., 1.],  # [47.0, 48.0], #
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$f_{wd}$",
                },
                "aw": {
                    "value": -1.5,
                    "range": [-2.75, -0.25],  # [47.0, 48.0], #
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$a_{w}$",
                },
                "thetaw": {
                    "value": 30,
                    "range": [25, 50],  # [47.0, 48.0], #
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$\theta_{w}$",
                },
                "thetasig": {
                    "value": 10,
                    "range": [7., 16.],  # [47.0, 48.0], #
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$\theta_{\sigma}$",
                },
                "logL": {
                    "value": 44.,
                    "range": [38., 48.],
                    "type": "c",
                    "vary": True,  # False, #
                    "latex": r"$\mathrm{log}\,L$",
                }
            }
         ),
        ("DL07", {
                "function": "DL07",
                "logumin": {
                    "value": 0.,
                    "range": [-1.0, 1.6], #log[0.1, 25]
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
                    "range": [0.3, 5.0],#10**[-1.0, 0.661],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$q_\mathrm{PAH}$",
                },
                "loggamma": {
                    "value": -2.0,
                    "range": [-5.0, 0.0], #[0.01, 0.03],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\mathrm{log}\,\gamma$",
                },
                "logMd": {
                    "value": 8.9,
                    "range": [6.0, 11.0], #[9.0, 10.0],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\mathrm{log}\,M_\mathrm{d}$",
                }
            }
        ),
    )
)
parTruth = None  #Whether to provide the truth of the model
#modelUnct = True #Whether to consider the model uncertainty in the fitting
unctDict = OrderedDict(
    (
        ("lnf"  , [-10., 2]),
        ("lna"  , [-10., 5]),
        ("lntau", [-10., 0.]),
    )
)

################################################################################
#                                   emcee                                      #
################################################################################

#emcee options#
#-------------#
burninDict = OrderedDict(
    (
        ("sampler"  , "EnsembleSampler"),
        ("ntemps"   , 16), #The number of temperature ladders only for PTSampler.
        ("nwalkers" , 128), #The number of walkers.
        ("iteration", [600, 600]), #The iteration of burn-in run.
        ("thin"     , 1), #To thin the recorded sample.
        ("ball-r"   , 0.1), #The radius of the ball as the fraction of the full range.
    )
)
finalDict = OrderedDict(
    (
        ("sampler"  , "EnsembleSampler"),
        ("ntemps"   , 4),
        ("nwalkers" , 128),
        ("iteration", [1000]),
        ("thin"     , 5),
        ("ball-r"   , 0.01),
    )
)
setupDict = OrderedDict(
    (
        ("threads"  , 4),
        ("printfrac", 0.1),
        ("pslow"    , 16),
        ("pscenter" , 50),
        ("pshigh"   , 84),
    )
)
emceeDict = OrderedDict(
    (
        ("Burnin", burninDict),
        ("Final", finalDict),
        ("Setup", setupDict),
    )
)

#Postprocess#
#-----------#
ppDict = {
    "low": 16,
    "center": 50,
    "high": 84,
    "nuisance": True, #False, #
    "fraction": 0, #The fraction of walkers to be dropped.
    "burn-in": 500,
    "savepath": "examples/results/"
}
