#This config file is for the radio loud sources.
#
import numpy as np
from collections import OrderedDict

################################################################################
#                                    Data                                      #
################################################################################
targname = "PG0003+158"
redshift = 0.45
distance = None #Luminosity distance
sedFile  = "data/{0}_obs.csed".format(targname)
sedName  = "Phot"
spcName  = "IRS"
dataDict = {
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
        ("Hot_Dust", {
                "function": "BlackBody",
                "logOmega": {
                    "value": -17.24,
                    "range": [-25.0, -10.0], #[0.3, 0.4], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\mathrm{log}\,\Omega$",
                },
                "T": {
                    "value": 944.47,
                    "range": [500, 1500], #[846., 847], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$T$",
                }
            }
        ),
        ("CLUMPY", {
                "function": "CLUMPY_intp",
                "logL": {
                    "value": 45.31,
                    "range": [40.0, 50.0], #[6.3, 6.4],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\mathrm{log}\,L_\mathrm{Torus}$",
                },
                "i": {
                    "value": 80.52,
                    "range": [0.0, 90.0], #[47.0, 48.0], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$i$",
                },
                "tv": {
                    "value": 267.41,
                    "range": [10.0, 300.0], #[17, 18], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\tau_\nu$",
                },
                "q": {
                    "value": 1.99,
                    "range": [0.0, 3.0], #[0.6, 0.8], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$q$",
                },
                "N0": {
                    "value": 3.55,
                    "range": [1.0, 15.0], #[6.42, 6.44], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$N_0$",
                },
                "sigma": {
                    "value": 55.81,
                    "range": [15.0, 70.0], #[58.0, 59.0], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\sigma$",
                },
                "Y": {
                    "value": 28.94,
                    "range": [5.0, 100.0], #[29., 31.], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$Y$",
                }
            }
        ),
        ("DL07", {
                "function": "DL07",
                "logumin": {
                    "value": 0.699,
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
                    "vary": True, #False, #
                    "latex": r"$q_\mathrm{PAH}$",
                },
                "gamma": {
                    "value": 0.071,
                    "range": [0.0, 1.0], #[0.01, 0.03],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\gamma$",
                },
                "logMd": {
                    "value": 8.42,
                    "range": [6.0, 11.0], #[9.0, 10.0],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\mathrm{log}\,M_\mathrm{d}$",
                }
            }
        ),
        ("Jet", {
                "function": "Synchrotron",
                "Sn_alpha": {
                    "value": -1.,
                    "range": [0., 5.0], #[0.01, 0.03],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\alpha_\mathrm{S}$",
                },
                "Sn_logsf": {
                    "value": 8.42,
                    "range": [-5.0, 5.0], #[9.0, 10.0],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\mathrm{log}\,f_\mathrm{S}$",
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
        ("lna"  , [-10, 2]),
        ("lntau", [-5, 2.5]),
    )
)

################################################################################
#                                   emcee                                      #
################################################################################

#emcee options#
#-------------#
emceeDict = OrderedDict(
    (
        ("sampler"   , "EnsembleSampler"),
        #("ntemps"    , 3),
        ("nwalkers"  , 100),
        ("iteration" , 3),
        ("iter-step" , 500),
        ("ball-r"    , 0.1),
        ("ball-t"    , 1.0),
        ("run-step"  , 2000),
        ("burn-in"   , 1000),
        ("thin"      , 1),
        ("threads"   , 4),
        ("printfrac" , 0.1),
    )
)

#Postprocess#
#-----------#
ppDict = {
    "low": 16,
    "center": 50,
    "high": 84,
    "nuisance": True, #False, #
    "fraction": 10, #The fraction of walkers to be dropped.
}
