from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import types
import numpy as np
import importlib
from time import time
import matplotlib.pyplot as plt
import cPickle as pickle
import sedfit.SED_Toolkit as sedt
from sedfit.fitter import basicclass as bc
from sedfit.fitter import mcmc_emcee as mcmc
from sedfit import sedclass as sedsc
from sedfit import model_functions as sedmf

#The code starts#
#---------------#
print("############################")
print("# Galaxy SED Fitter starts #")
print("############################")

def fitter(targname, redshift, sedPck, config, Dist=None):
    """
    This function is the main routine to do the SED fitting. The code will
    produce the final fitting results.

    Parameters
    ----------
    targname : str
        The name of the target.
    redshift : float
        The redshift of the target.
    sedPck: dict
        {
            sed : tuple
                (wave, flux, sigma) of SED photometric data.
            spc : tuple
                (wave, flux, sigma) of SED spectral data.
        }
    config : module or class
        The configuration information for the fitting.
    Dist : float (optional)
        The physical (or luminosity) distance of the source. If not provided, the
        value will be estimated from the redshift. Unit: Mpc.

    Returns
    -------
    None.

    Notes
    -----
    None.
    """
    try:
        silent = config.silent
    except:
        silent = False
    ################################################################################
    #                                    Data                                      #
    ################################################################################
    dataDict = config.dataDict
    sed = sedPck["sed"]
    spc = sedPck["spc"]
    #->Settle into the rest frame
    frame = dataDict.get("frame", "rest") #The coordinate frame of the SED; "rest"
                                          #by default.
    if frame == "obs":
        sed = sedt.SED_to_restframe(sed, redshift)
        if not silent:
            print("[gsf]: The input SED is in the observed frame!")
    else:
        if not silent:
            print("[gsf]: The input SED is in the rest frame!")
    #->Select bands
    bandList_use = dataDict.get("bandList_use", []) #The list of bands to incorporate;
                                                    #use all the available bands if empty.
    bandList_ignore = dataDict.get("bandList_ignore", []) #The list of bands to be
                                                          #ignored from the bands to use.
    sed = sedt.SED_select_band(sed, bandList_use, bandList_ignore, silent)
    sedwave  = sed[0]
    sedflux  = sed[1]
    sedsigma = sed[2]
    sedband  = sed[3]
    spcwave  = spc[0]
    spcflux  = spc[1]
    spcsigma = spc[2]
    if not silent:
        print("[gsf]: The incorporated bands are: {0}".format(sedband))
    #->Check data
    chck_sed = np.sum(np.isnan(sedflux)) + np.sum(np.isnan(sedsigma))
    chck_spc = np.sum(np.isnan(spcflux)) + np.sum(np.isnan(spcsigma))
    if chck_sed:
        raise ValueError("The photometry contains bad data!")
    if chck_spc:
        raise ValueError("The spectrum contains bad data!")
    #->Put into the sedData
    sedName  = config.sedName
    spcName  = config.spcName
    if not sedName is None:
        sedflag = np.ones_like(sedwave)
        sedDataType = ["name", "wavelength", "flux", "error", "flag"]
        phtData = {sedName: bc.DiscreteSet(sedband, sedwave, sedflux, sedsigma, sedflag, sedDataType)}
    else:
        phtData = {}
    if not spcName is None:
        spcflag = np.ones_like(spcwave)
        spcDataType = ["wavelength", "flux", "error", "flag"]
        spcData = {"IRS": bc.ContinueSet(spcwave, spcflux, spcsigma, spcflag, spcDataType)}
    else:
        spcData = {}
    if Dist is None:
        try:
            Dist = config.distance
        except:
            Dist = None
    sedData = sedsc.SedClass(targname, redshift, Dist, phtDict=phtData, spcDict=spcData)
    sedData.set_bandpass(sedband, sedwave, silent)

    ################################################################################
    #                                   Model                                      #
    ################################################################################
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

    #Build up the model#
    #------------------#
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
    modelUnct = config.modelUnct #Whether to consider the model uncertainty in the fitting
    parTruth  = config.parTruth  #Whether to provide the truth of the model
    """
    parVary   = sedModel.get_parVaryList()
    if modelUnct:
        nAddPars = 3 #If model the uncertainty, there are 3 additional parameters.
        for loop in range(nAddPars):
            parVary.append(-20) #Supplement the truth values for additional parameters.
    else:
        nAddPars = 0 #Else, there is not additional parameters.
    if parTruth is None:
        pass
    elif (len(parVary) - len(parTruth)) == nAddPars:
        print("\n**Parameter truths are given!\n")
        for loop in range(nAddPars):
            parTruth.append(-20) #Supplement the truth values for additional parameters.
    else:
        raise ValueError("The parameter truth list length is incorrect!")
    """


    ################################################################################
    #                                   emcee                                      #
    ################################################################################
    #Fit with MCMC#
    #-------------#
    emceeDict = config.emceeDict
    imSampler = emceeDict["sampler"]
    nwalkers  = emceeDict["nwalkers"]
    iteration = emceeDict["iteration"]
    iStep     = emceeDict["iter-step"]
    ballR     = emceeDict["ball-r"]
    ballT     = emceeDict["ball-t"]
    rStep     = emceeDict["run-step"]
    burnIn    = emceeDict["burn-in"]
    thin      = emceeDict["thin"]
    threads   = emceeDict["threads"]
    printFrac = emceeDict["printfrac"]
    unctDict = config.unctDict
    ppDict   = config.ppDict
    psLow    = ppDict["low"]
    psCenter = ppDict["center"]
    psHigh   = ppDict["high"]
    nuisance = ppDict["nuisance"]
    fraction = ppDict["fraction"]


    print("emcee Info:")
    for keys in emceeDict.keys():
        print("{0}: {1}".format(keys, emceeDict[keys]))
    print("#--------------------------------#")
    #em = mcmc.EmceeModel(sedData, sedModel, modelUnct, imSampler)
    em = mcmc.EmceeModel(sedData, sedModel, modelUnct, unctDict, imSampler)
    if imSampler == "EnsembleSampler":
        p0 = [em.from_prior() for i in range(nwalkers)]
        sampler = em.EnsembleSampler(nwalkers, threads=threads)
    elif imSampler == "PTSampler":
        ntemps = emceeDict["ntemps"]
        p0 = []
        for i in range(ntemps):
            p0.append([em.from_prior() for i in range(nwalkers)])
        sampler = em.PTSampler(ntemps, nwalkers, threads=threads)

    t0 = time()
    #Burn-in iterations
    for i in range(iteration):
        print( "\n{:*^35}".format(" {0}th iteration ".format(i)) )
        em.run_mcmc(p0, iterations=iStep, printFrac=printFrac, thin=thin)
        em.diagnose()
        pmax = em.p_logl_max()
        em.print_parameters(truths=parTruth, burnin=0)
        em.plot_lnlike(filename="{0}_lnprob.png".format(targname), histtype="step")
        print( "**Burn-in time ellapse: {0:.3f} hour".format( (time() - t0)/3600. ) )
        em.reset()
        ratio = ballR * ballT**i
        print("-- P0 ball radius ratio: {0:.3f}".format(ratio))
        p0 = em.p_ball(pmax, ratio=ratio)

    #Run MCMC
    #t0 = time()
    #print("[gsf test]: {0}".format(np.array(p0).shape))
    print( "\n{:*^35}".format(" Final Sampling ") )
    em.run_mcmc(p0, iterations=rStep, printFrac=printFrac, thin=thin)
    em.diagnose()
    em.print_parameters(truths=parTruth, burnin=burnIn, low=psLow, center=psCenter, high=psHigh)
    em.plot_lnlike(filename="{0}_lnprob.png".format(targname), histtype="step")
    print( "**Fit time ellapse: {0:.3f} hour".format( (time() - t0)/3600. ) )


    ################################################################################
    #                                  Post process                                #
    ################################################################################
    dataPck = {
        "targname": targname,
        "redshift": redshift,
        "sedPck": sedPck,
        "sedband": sedband,
        "sedName": sedName,
        "spcName": spcName,
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
        "posterior_sample": em.posterior_sample(burnin=burnIn, select=True, fraction=fraction),
        "chain": sampler.chain,
        "lnprobability": sampler.lnprobability
    }
    fp = open("{0}.fitrs".format(targname), "w")
    pickle.dump(fitrs, fp)
    fp.close()
    em.Save_BestFit("{0}_bestfit.txt".format(targname), low=psLow, center=psCenter, high=psHigh,
                    burnin=burnIn, select=True, fraction=fraction)
    em.plot_chain(filename="{0}_chain.png".format(targname), truths=parTruth)
    em.plot_corner(filename="{0}_triangle.png".format(targname), burnin=burnIn,
                   nuisance=nuisance, truths=parTruth, select=True, fraction=fraction,
                   quantiles=[psLow/100., psCenter/100., psHigh/100.], show_titles=True,
                   title_kwargs={"fontsize": 20})
    #->Plot the SED fitting result figure
    if sedData.check_csData():
        fig, axarr = plt.subplots(2, 1)
        fig.set_size_inches(10, 10)
        em.plot_fit_spec(truths=parTruth, FigAx=(fig, axarr[0]), nSamples=100,
                         burnin=burnIn, select=True, fraction=fraction)
        em.plot_fit(truths=parTruth, FigAx=(fig, axarr[1]), nSamples=100,
                    burnin=burnIn, select=True, fraction=fraction)
        axarr[0].set_xlabel("")
        axarr[0].set_ylabel("")
        axarr[0].text(0.05, 0.8, targname,
                      verticalalignment='bottom', horizontalalignment='left',
                      transform=axarr[0].transAxes, fontsize=24,
                      bbox=dict(facecolor='white', alpha=0.5, edgecolor="none"))
        plt.savefig("{0}_result.png".format(targname), bbox_inches="tight")
        plt.close()
    else:
        fig = plt.figure(figsize=(7, 7))
        ax = plt.gca()
        em.plot_fit(truths=parTruth, FigAx=(fig, ax), nSamples=100,
                    burnin=burnIn, select=True, fraction=fraction)
        plt.savefig("{0}_result.png".format(targname), bbox_inches="tight")
        plt.close()
    print("Post-processed!")

def gsf_fitter(configName, targname=None, redshift=None, sedFile=None):
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
    sedFile : str or None by default
        The full path of the sed data file.

    Returns
    -------
    None.

    Notes
    -----
    None.
    """
    config = importlib.import_module(configName.split("/")[-1].split(".")[0])
    if targname is None:
        assert redshift is None
        assert sedFile is None
        targname = config.targname
        redshift = config.redshift
        sedFile  = config.sedFile
    else:
        assert not redshift is None
        assert not sedFile is None
    print("#--------------------------------#")
    print("Target: {0}".format(targname))
    print("SED file: {0}".format(sedFile))
    print("Config file: {0}".format(configName))
    print("#--------------------------------#")
    #sedPck = sedt.Load_SED(sedFile, config.sedRng, config.spcRng, config.spcRebin)
    sedPck = sedt.Load_SED(sedFile)
    fitter(targname, redshift, sedPck, config)
