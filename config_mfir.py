from collections import OrderedDict

################################################################################
#                                    Data                                      #
################################################################################
sedName  = "W&H"
spcName  = "IRS"
sedRng   = [3, 13]
spcRng   = [13, None]
spcRebin = 1.
bandList = ["w1", "w2", "w3", "w4",
            "PACS_70", "PACS_100", "PACS_160",
            "SPIRE_250", "SPIRE_350", "SPIRE_500"]

################################################################################
#                                   Model                                      #
################################################################################
modelDict = OrderedDict(
    (
        ("Hot_Dust", {
                "function": "Modified_BlackBody",
                "logM": {
                    "value": 0.32,
                    "range": [-10.0, 3.0], #[0.3, 0.4], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\mathrm{log}\, M_\mathrm{d}$",
                },
                "beta": {
                    "value": 2.0,
                    "range": [1.5, 2.5], #[1.9, 2.1], #
                    "type": "c",
                    "vary": False, #True, #
                    "latex": r"$\beta$",
                },
                "T": {
                    "value": 846.77,
                    "range": [500, 1300], #[846., 847], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$T$",
                }
            }
        ),
        ("CLUMPY", {
                "function": "CLUMPY_intp",
                "logL": {
                    "value": 46.4,
                    "range": [40.0, 50.0], #[6.3, 6.4],
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\mathrm{log}\,SF_\mathrm{Torus}$",
                },
                "i": {
                    "value": 47.60,
                    "range": [0.0, 90.0], #[47.0, 48.0], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$i$",
                },
                "tv": {
                    "value": 17.53,
                    "range": [10.0, 300.0], #[17, 18], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\tau_\nu$",
                },
                "q": {
                    "value": 0.7,
                    "range": [0.0, 3.0], #[0.6, 0.8], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$q$",
                },
                "N0": {
                    "value": 6.43,
                    "range": [1.0, 15.0], #[6.42, 6.44], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$N_0$",
                },
                "sigma": {
                    "value": 58.14,
                    "range": [15.0, 70.0], #[58.0, 59.0], #
                    "type": "c",
                    "vary": True, #False, #
                    "latex": r"$\sigma$",
                },
                "Y": {
                    "value": 30.0,
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
                    "range": [8.0, 10.0], #[9.0, 10.0],
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

################################################################################
#                                   emcee                                      #
################################################################################

#emcee options#
#-------------#
emceeDict = OrderedDict(
    (
        ("sampler"   , "EnsembleSampler"),
        ("nwalkers"  , 100),
        ("iteration" , 0),
        ("iter-step" , 200),
        ("ball-r"    , 0.1),
        ("ball-t"    , 1.0),
        ("run-step"  , 500),
        ("burn-in"   , 100),
        ("thin"      , 1),
        ("threads"   , 4),
        ("printfrac" , 0.1),
    )
)

#Postprocess#
#-----------#
ppDict = {
    "low": 1, #16,
    "center": 50,
    "high": 99, #84,
    "nuisance" : True, #False, #
}
