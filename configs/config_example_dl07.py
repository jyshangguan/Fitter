#This config file is for the radio quiet sources.
#
import numpy as np
from collections import OrderedDict

################################################################################
#                                    Data                                      #
################################################################################
targname = "PG0050+124"
redshift = 0.061
sedFile  = "mock/Ref_SED/PG0050+124_rest.tsed"
sedName  = "2M&W&H"
spcName  = None #"IRS"
sedRng   = [0, 13]
spcRng   = [13, None]
spcRebin = 1.
#bandList = ["j", "h", "ks", "w1", "w2",
bandList = ["j", "h", "ks", "w1", "w2", "w3", "w4",
            "PACS_70", "PACS_100", "PACS_160",
            "SPIRE_250", "SPIRE_350", "SPIRE_500"]

################################################################################
#                                   Model                                      #
################################################################################
waveModel = 10**np.linspace(-0.1, 3.0, 1000)
modelDict = OrderedDict(
    (
        ("BC03", {
                "function": "BC03",
                "logMs":{
                    "value": 9.,
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
        ("DL07", {
                "function": "DL07",
                "logumin": {
                    "value": 1.0,
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
                    "value": 3.19, #10**0.504,
                    "range": [0.1, 4.58],#10**[-1.0, 0.661],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$q_\mathrm{PAH}$",
                },
                "gamma": {
                    "value": 0.02,
                    "range": [0.0, 1.0], #[0.01, 0.03],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\gamma$",
                },
                "logMd": {
                    "value": 9.12,
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
modelUnct = True #Whether to consider the model uncertainty in the fitting
unctDict = OrderedDict(
    (
        ("lnf"  , [-10, 10]),
        ("lna"  , [-10, 1]),
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
        ("iteration" , 1),
        ("iter-step" , 1000),
        ("ball-r"    , 0.3),
        ("ball-t"    , 1.0),
        ("run-step"  , 3000),
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
