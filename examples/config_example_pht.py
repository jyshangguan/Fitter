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
targname = "IRSA13120-5453" # Target name
redshift = 0.03076 # Redshift
distance = 143.6   # Luminosity distance (Mpc). If it is not provided, the
                   # luminosity distance with be calculated with the redshift
                   # assuming Planck (2015) cosmology, FlatLambdaCDM(H0=67.8,
                   # Om0=0.308).
sedFile  = "examples/{0}_obs.csed".format(targname) # The path to the SED data.
dataDict = {
    "phtName": "Phot", # The name of the photometric data. Use "None" if no
                       # photometric data is used.
    "spcName": None,   # The name of the spectral data. Use "None" if no
                       # spectral data is used.
    "bandList_use": [], # The list of the band name to be used. The name should
                        # be consistent what are provided in data file (forth
                        # column). If empty, all of the bands in the data are
                        # used.
    "bandList_ignore":[], # The list of the band name to be ignored. The name
                          # should be consistent what are provided in data file
                          # (4th column). If empty, none of the bands in the
                          # data are ignored.
    "frame": "obs", # Note the frame of the data. The observed frame ("obs") is
                    # extensively used.
}

################################################################################
#                                   Model                                      #
################################################################################
waveModel = 10**np.linspace(-0.5, 3.0, 1000) # The wavelength array to calculate
                                             # model SED.
#parAddDict_all = {}
modelDict = OrderedDict(
    ( # Each element of the dict is one model component. Remove the element to
      # remove the component.
        ("BC03", { # Bruzual & Charlot (2003) SSP model.
                "function": "BC03", # The name of functions from sedfit.models.
                "logMs":{ # The log10 of stellar mass.
                    "value": 11.45, # Model parameter, not used in the fit.
                    "range": [6., 14.], # Model prior range, used to generate
                                        # the uniform prior.
                    "type": "c", # Indicate the parameter to be continuum, not
                                 # used in the fit.
                    "vary": True, # Indicate whether this parameter is free
                                  # (True) or fixed (False).
                    "latex": r"$\mathrm{log}\,M_\mathrm{s}$", # For the
                                                              # auxiliary plot
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
parTruth = None  # Whether to provide the truth of the model.
modelUnct = True # Whether to consider the model uncertainty in the fitting
unctDict = OrderedDict( # Range of the parameters to model the uncertainties.
                        # These parameters will be marginalized in the end.
    (
        ("lnf"  , [-10, 2]), # The prior range of the ln of the ratio between
                             # the uncertainty over the flux.
        ("lna"  , [-10, 5]), # The prior range of the ln of the amplitude of the
                             # covariance.
        ("lntau", [-5, 2.5]), # The prior range of the ln of the correlation
                              # length of the covariance.
    )
)

################################################################################
#                                   emcee                                      #
################################################################################

#emcee options#
#-------------#
burnin = OrderedDict( # The setup for the burnin run using emcee.
    (
        ("sampler"  , "EnsembleSampler"), # The name of the sampler. Better to
                                          # just use "EnsembleSampler".
        ("nwalkers" , 128), # The number of walkers.
        ("iteration", [600, 600]), # The iteration of burn-in run. Reinitiallize
                                   # the walkers around the likelihood peak.
        ("thin"     , 1), # To thin the recorded sample. Not used unless need to
                          # run very long chain.
        ("ball-r"   , 0.1), # The radius of the ball as the fraction of the full
                            # prior range to reinitiallize the walkers.
    )
)
final = OrderedDict( # Same as burnin but a different setup.
    (
        ("sampler"  , "EnsembleSampler"),
        ("ntemps"   , 4),
        ("nwalkers" , 128),
        ("iteration", [1000]), #[1000, 600]),
        ("thin"     , 1),
        ("ball-r"   , 0.01),
    )
)
setup = OrderedDict( # The setup for emcee run.
    (
        ("threads"  , 4), # Number of threads used for emcee sampler.
                          # Not used if MPI is using.
        ("printfrac", 0.1), # The fraction of the lowest likelihood sample to
                            # be thrown away when print the status. On-the-fly
                            # print. Not critial.
        ("pslow"    , 16), # The lower error bar of the posterior. On-the-fly
                           # print. Not critial.
        ("pscenter" , 50), # The median of the posterior. On-the-fly print. Not
                           # critial.
        ("pshigh"   , 84), # The high error bar of the posterior. On-the-fly
                           # print. Not critial.
    )
)
emceeDict = OrderedDict( # Emcee run strategy.
    (
        ("BurnIn", burnin), # Use the "burnin" first.
        ("Final", final), # Use the "final" in the end.
        ("Setup", setup), # Provide more information for emcee.
    )
)

#Postprocess#
#-----------#
ppDict = {
    "burn-in" : 500, # The number of sampled points to throw away.
    "low"     : 16, # The lower error bar of the posterior.
    "center"  : 50, # The median of the posterior.
    "high"    : 84, # The high error bar of the posterior.
    "nuisance": True, # Do not show the nuisance parameters if True.
    "fraction": 0, # The fraction of walkers with the lowest likelihood to be
                   # dropped.
    "savepath": "examples/results/" # The save path.
}
