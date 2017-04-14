from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib_version = eval(matplotlib.__version__.split(".")[0])
if matplotlib_version > 1:
    plt.style.use("classic")
import sys
import types
import numpy as np
import importlib
from time import time
import cPickle as pickle
import sedfit.SED_Toolkit as sedt
from sedfit.fitter import basicclass as bc
from sedfit.mcmc import mcmc_emcee as mcmc
from sedfit import sedclass as sedsc
from sedfit import model_functions as sedmf

#The starter of this module#
#--------------------------#
print("############################")
print("# Galaxy SED Fitter starts #")
print("############################")

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

def fitter(sedData, sedModel, unctDict, parTruth, emceeDict):
    """
    This function is run the SED fitting with the MCMC method.

    Parameters
    ----------
    sedData : SEDClass object
        The data set of SED.
    sedModel : ModelCombiner object
        The combined model. The parameters are set to generate the mock SED.
    unctDict : dict or None, by default
        {
            "lnf" : float, (-inf, lnf_max]
                The ln of f, the imperfectness of the model.
            "lna" : float, (-inf, lnf_max]
                The ln of a, the amplitude of the residual correlation.
            "lntau" : float, (-inf, lnf_max]
                The ln of tau, the scale length of the residual correlation.
        }
    parTruth : bool
        The toggle whether to provide the truth of the model.

    Returns
    -------
    em : EmceeModel object
        The object of EmceeModel.

    Notes
    -----
    None.
    """

    print("emcee setups:")
    for keys in emceeDict["Setup"].keys():
        print("{0}: {1}".format(keys, emceeDict["Setup"][keys]))
    print("#--------------------------------#")
    threads     = emceeDict["Setup"]["threads"]
    printFrac   = emceeDict["Setup"]["printfrac"]
    psLow       = emceeDict["Setup"]["pslow"]
    psCenter    = emceeDict["Setup"]["pscenter"]
    psHigh      = emceeDict["Setup"]["pshigh"]

    #Burn-in runs#
    #------------#
    print( "\n{:*^50}".format(" Burn-in Sampling ") )
    for keys in emceeDict["Burnin"].keys():
        print("{0}: {1}".format(keys, emceeDict["Burnin"][keys]))
    print("#--------------------------------#")

    SamplerType = emceeDict["Burnin"]["sampler"]
    nwalkers    = emceeDict["Burnin"]["nwalkers"]
    iteration   = emceeDict["Burnin"]["iteration"]
    thin        = emceeDict["Burnin"]["thin"]
    ballR       = emceeDict["Burnin"]["ball-r"]

    if unctDict is None:
        modelUnct = False
    else:
        modelUnct = True
    em = mcmc.EmceeModel(sedData, sedModel, modelUnct, unctDict, SamplerType)
    if SamplerType == "EnsembleSampler":
        p0 = [em.from_prior() for i in range(nwalkers)]
        sampler = em.EnsembleSampler(nwalkers, threads=threads)
    elif SamplerType == "PTSampler":
        ntemps = emceeDict["Burnin"]["ntemps"]
        p0 = []
        for i in range(ntemps):
            p0.append([em.from_prior() for i in range(nwalkers)])
        sampler = em.PTSampler(ntemps, nwalkers, threads=threads)

    t0 = time()
    for i in range(len(iteration)):
        steps = iteration[i]
        print( "\n{:*^35}".format(" {0}th Burn-in ".format(i)) )
        em.run_mcmc(p0, iterations=steps, printFrac=printFrac, thin=thin)
        em.diagnose()
        pmax = em.p_logl_max()
        em.print_parameters(truths=parTruth, burnin=0)
        em.plot_lnlike(filename="gsf_temp_lnprob.png", histtype="step")
        print( "**Time ellapse: {0:.3f} hour".format( (time() - t0)/3600. ) )
        em.reset()
        p0 = em.p_ball(pmax, ratio=ballR)
    del(em)

    #Final runs#
    #----------#
    print( "\n{:*^50}".format(" Final Sampling ") )
    for keys in emceeDict["Final"].keys():
        print("{0}: {1}".format(keys, emceeDict["Final"][keys]))
    print("#--------------------------------#")

    SamplerType = emceeDict["Final"]["sampler"]
    nwalkers    = emceeDict["Final"]["nwalkers"]
    iteration   = emceeDict["Final"]["iteration"]
    thin        = emceeDict["Final"]["thin"]
    burnIn      = emceeDict["Final"]["burn-in"]
    ballR       = emceeDict["Final"]["ball-r"]

    em = mcmc.EmceeModel(sedData, sedModel, modelUnct, unctDict, SamplerType)
    if SamplerType == "EnsembleSampler":
        sampler = em.EnsembleSampler(nwalkers, threads=threads)
    elif SamplerType == "PTSampler":
        ntemps = emceeDict["Final"]["ntemps"]
        sampler = em.PTSampler(ntemps, nwalkers, threads=threads)
    for i in range(len(iteration)):
        steps = iteration[i]
        print( "\n{:*^35}".format(" {0}th Final ".format(i)) )
        em.reset()
        p0 = em.p_ball(pmax, ratio=ballR) #->Initialize the walkers
        em.run_mcmc(p0, iterations=steps, printFrac=printFrac, thin=thin)
        em.diagnose()
        pmax = em.p_logl_max()
        em.print_parameters(truths=parTruth, burnin=burnIn, low=psLow, center=psCenter, high=psHigh)
        em.plot_lnlike(filename="gsf_temp_lnprob.png", histtype="step")
        print( "**Time ellapse: {0:.3f} hour".format( (time() - t0)/3600. ) )
    return em


def gsf_fitter(configName, targname=None, redshift=None, distance=None, sedFile=None):
    """
    The wrapper of fitter() function. If the targname, redshift and sedFile are
    provided as arguments, they will be used overriding the values in the config
    file saved in configName. If they are not provided, then, the values in the
    config file will be used.

    Parameters
    ----------
    configName : str
        The full path of the config file.
    targname : str or None by default
        The name of the target.
    redshift : float or None by default
        The redshift of the target.
    distance : float or None by default
        The distance of the source from the Sun.
    sedFile : str or None by default
        The full path of the sed data file.

    Returns
    -------
    None.

    Notes
    -----
    None.
    """
    ############################################################################
    #                                Setup                                     #
    ############################################################################
    config = configImporter(configName)
    if targname is None:
        assert redshift is None
        assert distance is None
        assert sedFile is None
        targname = config.targname
        redshift = config.redshift
        distance = config.distance
        sedFile  = config.sedFile
    else:
        assert not redshift is None
        assert not sedFile is None
    print("#--------------------------------#")
    print("Target:      {0}".format(targname))
    print("Redshift:    {0}".format(redshift))
    print("Distance:    {0}".format(distance))
    print("SED file:    {0}".format(sedFile))
    print("Config file: {0}".format(configName))
    print("#--------------------------------#")

    try:
        silent = config.silent
    except:
        silent = False

    #->Setup the data Data
    dataDict = config.dataDict
    sedPck = sedt.Load_SED(sedFile)
    sedData = sedsc.setSedData(targname, redshift, distance, dataDict, sedPck, silent)

    #->Setup the model
    modelDict = config.modelDict
    print("The model info:")
    parCounter = 0
    for modelName in modelDict.keys():
        print("[{0}]".format(modelName))
        model = modelDict[modelName]
        for parName in model.keys():
            param = model[parName]
            if not isinstance(param, types.DictType):
                continue
            elif param["vary"]:
                print("-- {0}, {1}".format(parName, param["type"]))
                parCounter += 1
            else:
                pass
    print("Varying parameter number: {0}".format(parCounter))
    print("#--------------------------------#")
    funcLib   = sedmf.funcLib
    waveModel = config.waveModel
    try:
        parAddDict_all = config.parAddDict_all
    except:
        parAddDict_all = {}
    parAddDict_all["DL"]    = sedData.dl
    parAddDict_all["z"]     = redshift
    parAddDict_all["frame"] = "rest"
    sedModel  = bc.Model_Generator(modelDict, funcLib, waveModel, parAddDict_all)

    ############################################################################
    #                                   Fit                                    #
    ############################################################################
    modelUnct = config.modelUnct #Whether to consider the model uncertainty in the fitting
    parTruth  = config.parTruth  #Whether to provide the truth of the model
    unctDict = config.unctDict
    emceeDict = config.emceeDict
    em = fitter(sedData, sedModel, unctDict, parTruth, emceeDict)

    ############################################################################
    #                              Post process                                #
    ############################################################################
    try:
        ppDict = config.ppDict
    except:
        print("[gsf] Warning: cannot find ppDict in the configure file!")
        ppDict = {}
    psLow    = ppDict.get("low", 16)
    psCenter = ppDict.get("center", 50)
    psHigh   = ppDict.get("high", 84)
    nuisance = ppDict.get("nuisance", True)
    fraction = ppDict.get("fraction", 0)
    burnIn   = emceeDict["Final"]["burn-in"]

    dataPck = {
        "targname": targname,
        "redshift": redshift,
        "distance": sedData.dl,
        "sedPck": sedPck,
        "dataDict": dataDict
    }
    modelPck = {
        "modelDict": modelDict,
        "waveModel": waveModel,
        "parAddDict_all": parAddDict_all,
        "parTruth": parTruth,
        "modelUnct": modelUnct
    }
    fitrs = {
        "dataPck": dataPck,
        "modelPck": modelPck,
        "ppDict": ppDict,
        "posterior_sample": em.posterior_sample(burnin=burnIn, fraction=fraction),
        "chain": em.sampler.chain,
        "lnprobability": em.sampler.lnprobability
    }
    fp = open("{0}.fitrs".format(targname), "w")
    pickle.dump(fitrs, fp)
    fp.close()
    #->Save the best-fit parameters
    em.Save_BestFit("{0}_bestfit.txt".format(targname), low=psLow, center=psCenter, high=psHigh,
                    burnin=burnIn, fraction=fraction)
    #->Plot the chain of the final run
    em.plot_chain(filename="{0}_chain.png".format(targname), truths=parTruth)
    #->Plot the SED fitting result figure
    sedwave = sedData.get_List("x")
    sedflux = sedData.get_List("y")
    xmin = np.min(sedwave) * 0.9
    xmax = np.max(sedwave) * 1.1
    xlim = [xmin, xmax]
    ymin = np.min(sedflux) * 0.5
    ymax = np.max(sedflux) * 2.0
    ylim = [ymin, ymax]
    if sedData.check_csData():
        fig, axarr = plt.subplots(2, 1)
        fig.set_size_inches(10, 10)
        em.plot_fit_spec(truths=parTruth, FigAx=(fig, axarr[0]), nSamples=100,
                         burnin=burnIn, fraction=fraction)
        em.plot_fit(truths=parTruth, FigAx=(fig, axarr[1]), xlim=xlim, ylim=ylim,
                    nSamples=100, burnin=burnIn, fraction=fraction)
        axarr[0].set_xlabel("")
        axarr[0].set_ylabel("")
        axarr[0].text(0.05, 0.8, targname, transform=axarr[0].transAxes, fontsize=24,
                      verticalalignment='bottom', horizontalalignment='left',
                      bbox=dict(facecolor='white', alpha=0.5, edgecolor="none"))
    else:
        fig = plt.figure(figsize=(7, 7))
        ax = plt.gca()
        em.plot_fit(truths=parTruth, FigAx=(fig, ax), xlim=xlim, ylim=ylim,
                    nSamples=100, burnin=burnIn, fraction=fraction)
        ax.text(0.05, 0.8, targname, transform=ax.transAxes, fontsize=24,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor="none"))
    plt.savefig("{0}_result.png".format(targname), bbox_inches="tight")
    plt.close()
    #->Plot the posterior probability distribution
    em.plot_corner(filename="{0}_triangle.png".format(targname), burnin=burnIn,
                   nuisance=nuisance, truths=parTruth, fraction=fraction,
                   quantiles=[psLow/100., psCenter/100., psHigh/100.], show_titles=True,
                   title_kwargs={"fontsize": 20})
    print("Post-processed!")
