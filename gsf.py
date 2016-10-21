from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import types
import numpy as np
import matplotlib.pyplot as plt
from time import time
import cPickle as pickle
import rel_SED_Toolkit as sedt
from sedfit.fitter import basicclass as bc
from sedfit.fitter import mcmc
from sedfit import sedclass as sedsc
from sedfit import model_functions as sedmf

#The code starts#
#---------------#
print("############################")
print("# Galaxy SED Fitter starts #")
print("############################")

def gsf_run(targname, redshift, sedFile, config):
    """
    This function is the main routine to do the SED fitting. The code will
    produce the final fitting results.

    Parameters
    ----------
    targname : str
        The name of the target.
    redshift : float
        The redshift of the target.
    sedFile : str
        The full path of the SED data file.
    config : module or class
        The configuration information for the fitting.

    Returns
    -------
    None.

    Notes
    -----
    None.
    """
    print("#--------------------------------#")
    print("Target: {0}".format(targname))
    print("SED file: {0}".format(sedFile))
    print("#--------------------------------#")
    ################################################################################
    #                                    Data                                      #
    ################################################################################
    sedRng = config.sedRng
    sedPck = sedt.Load_SED(sedFile, sedRng, config.spcRng, config.spcRebin)
    sed = sedPck["sed_cb"]
    sedwave = sedPck["sed"][0]
    sedflux = sedPck["sed"][1]
    sedsigma = sedPck["sed"][2]
    spcwave = sedPck["spc"][0]
    spcflux = sedPck["spc"][1]
    spcsigma = sedPck["spc"][2]
    ##Check data
    chck_sed = np.sum(np.isnan(sedflux)) + np.sum(np.isnan(sedsigma))
    chck_spc = np.sum(np.isnan(spcflux)) + np.sum(np.isnan(spcsigma))
    if chck_sed:
        raise ValueError("The photometry contains bad data!")
    if chck_spc:
        raise ValueError("The spectrum contains bad data!")

    ## Put into the sedData
    bandList = config.bandList
    sedName  = config.sedName
    spcName  = config.spcName
    if not sedName is None:
        sedflag = np.ones_like(sedwave)
        sedDataType = ["name", "wavelength", "flux", "error", "flag"]
        phtData = {sedName: bc.DiscreteSet(bandList, sedwave, sedflux, sedsigma, sedflag, sedDataType)}
    else:
        phtData = {}
    if not spcName is None:
        spcflag = np.ones_like(spcwave)
        spcDataType = ["wavelength", "flux", "error", "flag"]
        spcData = {"IRS": bc.ContinueSet(spcwave, spcflux, spcsigma, spcflag, spcDataType)}
    else:
        spcData = {}
    sedData = sedsc.SedClass(targname, redshift, phtDict=phtData, spcDict=spcData)
    sedData.set_bandpass(bandList)


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
    parAddDict_all = {
        "DL": sedData.dl,
    }
    funcLib    = sedmf.funcLib
    waveModel = 10**np.linspace(0.0, 3.0, 1000)
    sedModel = bc.Model_Generator(modelDict, funcLib, waveModel, parAddDict_all)
    parTruth = config.parTruth   #Whether to provide the truth of the model
    modelUnct = config.modelUnct #Whether to consider the model uncertainty in the fitting
    parAllList = sedModel.get_parVaryList()
    if modelUnct:
        parAllList.append(-np.inf)
        parAllList.append(-np.inf)
        parAllList.append(-5)


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
    em = mcmc.EmceeModel(sedData, sedModel, modelUnct, unctDict)
    p0 = [em.from_prior() for i in range(nwalkers)]
    sampler = em.EnsembleSampler(nwalkers, threads=threads)

    #Burn-in 1st
    t0 = time()
    print( "\n{:*^35}".format(" {0}th iteration ".format(0)) )
    em.run_mcmc(p0, iterations=iStep, printFrac=printFrac, thin=thin)
    em.diagnose()
    pmax = em.p_logl_max()
    em.print_parameters(truths=parTruth, burnin=0)
    em.plot_lnlike(filename="{0}_lnprob.png".format(targname), histtype="step")
    print( "**Burn-in time ellapse: {0:.3f} hour".format( (time() - t0)/3600. ) )

    #Burn-in rest iteration
    for i in range(iteration-1):
        print( "\n{:*^35}".format(" {0}th iteration ".format(i+1)) )
        em.reset()
        ratio = ballR * ballT**i
        print("-- P1 ball radius ratio: {0:.3f}".format(ratio))
        p1 = em.p_ball(pmax, ratio=ratio)
        em.run_mcmc(p1, iterations=iStep, printFrac=printFrac)
        em.diagnose()
        pmax = em.p_logl_max()
        em.print_parameters(truths=parTruth, burnin=50)
        em.plot_lnlike(filename="{0}_lnprob.png".format(targname), histtype="step")
        print( "**Burn-in time ellapse: {0:.3f} hour".format( (time() - t0)/3600. ) )

    #Run MCMC
    t0 = time()
    print( "\n{:*^35}".format(" Final Sampling ") )
    em.reset()
    ratio = ballR * ballT**i
    print("-- P1 ball radius ratio: {0:.3f}".format(ratio))
    p1 = em.p_ball(pmax, ratio=ratio)
    em.run_mcmc(p1, iterations=rStep, printFrac=printFrac, thin=thin)
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
        "sedFile": sedFile,
        "sedPck": sedPck,
        "bandList": bandList,
        "sedName": sedName,
        "spcName": spcName,
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
    fig, axarr = plt.subplots(2, 1)
    fig.set_size_inches(10, 10)
    em.plot_fit_spec(truths=parTruth, FigAx=(fig, axarr[0]),
                     burnin=burnIn, select=True, fraction=fraction,
                     low=psLow, center=psCenter, high=psHigh)
    em.plot_fit(truths=parTruth, FigAx=(fig, axarr[1]),
                burnin=burnIn, select=True, fraction=fraction,
                low=psLow, center=psCenter, high=psHigh)
    plt.savefig("{0}_result.png".format(targname), bbox_inches="tight")
    plt.close()
    print("Post-processed!")
