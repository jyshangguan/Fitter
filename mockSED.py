import os
import sys
import types
import importlib
import numpy as np
import matplotlib.pyplot as plt
import rel_SED_Toolkit as sedt
from astropy.table import Table
from sedfit.fitter import basicclass as bc
from sedfit import sedclass as sedsc
from sedfit import model_functions as sedmf
from sedfit.fit_functions import logLFunc_gp, logLFunc

def dataPerturb(x, sigma, pert=True):
    """
    Perturb the data assuming it is a Gaussian distribution around the detected
    values with standard deviation as the uncertainties.
    """
    if pert:
        xp = sigma * np.random.randn(len(np.atleast_1d(x))) + x
        counter = 0
        while np.any(xp<=0):
            xp = sigma * np.random.randn(len(np.atleast_1d(x))) + x
            counter += 1
            if counter > 10:
                raise ValueError("The data is too noisy...")
    else:
        xp = x
    return xp

def randomRange(low, high):
    """
    Calculate the random number in range [low, high).

    Parameters
    ----------
    low : float
        The lower boundary.
    high: float
        The upper boundary.

    Returns
    -------
    r : float
        The random number in [low, high).

    Notes
    -----
    None.
    """
    assert high >= low
    rg = high - low
    r = low + rg * np.random.rand()
    return r

def mocker(targname, redshift, sedPck, mockPars, config, Dist=0,
           sysUnc=None, uncModel=[-20, -20, -20], pert=True, plot=False):
    """
    This function is to generate a mock SED according to a given observed SED.
    Basically, the flux densities will be replaced by the model value while the
    wavelength and uncertainties of the data will be kept.

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
    Dist : float
        The physical distance. It should be provided if the redshift is 0.
    sysUnc : dict
        {
            "pht": [([nBgn, nEnd], frac), ([nBgn, nEnd], frac), ...],
            "spc": frac
        }
    pert : bool, default: True
        Perturb the data according to the uncertainty if True.
    plot : bool, default: False
        Plot the SED to visually check if True.

    Returns
    -------
    (mock, lnlike) : (dict, float)
        mock : The dict of mock data, with wave, flux and sigma inside.
        lnlike : The likelihood calculated with the input model and the mock data.

    Notes
    -----
    None.
    """
    ################################################################################
    #                                    Data                                      #
    ################################################################################
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
    sedData = sedsc.SedClass(targname, redshift, Dist=Dist, phtDict=phtData, spcDict=spcData)
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
    try:
        parAddDict_all = config.parAddDict_all
    except:
        parAddDict_all = {}
    parAddDict_all["DL"]    = sedData.dl
    parAddDict_all["z"]     = redshift
    parAddDict_all["frame"] = "rest"
    funcLib    = sedmf.funcLib
    waveModel = config.waveModel
    sedModel = bc.Model_Generator(modelDict, funcLib, waveModel, parAddDict_all)
    sedModel.updateParList(mockPars)
    #print sedModel.get_parVaryNames(latex=False)
    #print sedModel.get_parVaryList()
    #->Generate the mock data
    fluxModel = sedModel.combineResult()
    mockPht0 = np.array(sedData.model_pht(waveModel, fluxModel))
    mockSpc0 = np.array(sedData.model_spc(sedModel.combineResult))
    #->Make sure the perturbed data not likely be too far away, or even negative.
    #For photometric data
    mockPhtSigma = np.array(sedData.get_dsList("e"))
    fltr_sigma = mockPht0 < 3.0*mockPhtSigma #sedsigma
    if np.any(fltr_sigma):
        mockPhtSigma[fltr_sigma] = mockPht0[fltr_sigma] / 3.0
    mockPht = dataPerturb(mockPht0, mockPhtSigma, pert)
    #For spectroscopic data
    mockSpcSigma = np.array(sedData.get_csList("e"))
    fltr_sigma = mockSpc0 < 3.0*mockSpcSigma
    if np.any(fltr_sigma):
        mockSpcSigma[fltr_sigma] = mockSpc0[fltr_sigma] / 3.0
    mockSpc = dataPerturb(mockSpc0, mockSpcSigma, pert)
    #->Systematic uncertainties
    if not sysUnc is None:
        sysSpc = sysUnc["spc"]
        mockSpc = (1 + randomRange(-sysSpc, sysSpc)) * mockSpc
        sysPhtList = sysUnc["pht"]
        for phtRg, frac in sysPhtList:
            #print phtRg, frac, mockPht[phtRg[0]:phtRg[1]]
            mockPht[phtRg[0]:phtRg[1]] = (1 + randomRange(-frac, frac)) * mockPht[phtRg[0]:phtRg[1]]
    #->Add the upperlimits
    #fltr_undct = sedsigma < 0
    #mockPht[fltr_undct] = sedflux[fltr_undct]
    #->Add systematic uncertainty
    mockPhtWave = np.array(sedData.get_dsList("x"))
    mockSpcWave = np.array(sedData.get_csList("x"))
    mockSED = np.concatenate([mockPht, mockSpc])
    mockWav = np.concatenate([mockPhtWave, mockSpcWave])
    mockSig = np.concatenate([mockPhtSigma, mockSpcSigma])
    mock = {
        "sed": mockSED,
        "wave": mockWav,
        "sigma": mockSig
    }
    #->Calculate the lnprob
    if sedData.check_dsData():
        mockPhtFlag = np.ones_like(mockPhtWave)
        phtData = {sedName: bc.DiscreteSet(bandList, mockPhtWave, mockPht, mockPhtSigma, mockPhtFlag, sedDataType)}
    else:
        phtData = {}
    if sedData.check_csData():
        mockSpcFlag = np.ones_like(mockSpcWave)
        spcData = {"IRS": bc.ContinueSet(mockSpcWave, mockSpc, mockSpcSigma, mockSpcFlag, spcDataType)}
    else:
        spcData = {}
    mckData = sedsc.SedClass(targname, redshift, phtDict=phtData, spcDict=spcData)
    mckData.set_bandpass(bandList)
    #lnlike = logLFunc_gp(list(mockPars)+uncModel, mckData, sedModel)
    lnlike = logLFunc(mockPars, mckData, sedModel)

    if plot:
        xmin = np.min(waveModel)
        xmax = np.max(waveModel)
        ymin = np.min(mockSED) / 10.0
        ymax = np.max(mockSED) * 10.0
        FigAx = sedData.plot_sed()
        FigAx = sedModel.plot(FigAx=FigAx)
        if sedData.check_dsData():
            plt.errorbar(mockPhtWave, mockPht, yerr=mockPhtSigma, linestyle="none", marker="s",
                         mfc="none", mec="r", color="r", alpha=0.5)
        if sedData.check_csData():
            plt.errorbar(mockSpcWave, mockSpc, yerr=mockSpcSigma, linestyle="-", color="r", alpha=0.5)
        fig, ax = FigAx
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    return (mock, lnlike)

    #sedModel.plot()

def gsm_mocker(configName, targname=None, redshift=None, sedFile=None,
               mockPars=None, **kwargs):
    """
    The wrapper of mocker() function. If the targname, redshift, sedFile and
    mockPars are provided as arguments, they will be used overriding the values
    in the config file saved in configName. If they are not provided, then, the
    values in the config file will be used.

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
    mockPars : list
        The parameters to generate the mock SED.

    Returns
    -------
    None.

    Notes
    -----
    None.
    """
    config = importlib.import_module(configName.split(".")[0])
    if targname is None:
        assert redshift is None
        assert sedFile is None
        #assert mockPars is None
        targname = config.targname
        redshift = config.redshift
        sedFile  = config.sedFile
    else:
        assert not redshift is None
        assert not sedFile is None
        #assert not mockPars is None
    print("#--------------------------------#")
    print("Target: {0}".format(targname))
    print("SED file: {0}".format(sedFile))
    print("Config file: {0}".format(configName))
    print("#--------------------------------#")
    sedPck = sedt.Load_SED(sedFile, config.sedRng, config.spcRng, config.spcRebin)
    mock = mocker(targname, redshift, sedPck, mockPars, config, **kwargs)
    return mock

if __name__ == "__main__":
    #-->Generate Mock Data
    parTable = Table.read("/Volumes/Transcend/Work/PG_MCMC/pg_sil/compile_sil.ipac", format="ascii.ipac")
    infoTable = Table.read("targlist/targlist_rq.ipac", format="ascii.ipac")
    #print parTable.colnames
    if os.path.isdir("configs"):
        sys.path.append("configs/")
    configName = "config_sil"
    mockSub = "sil"

    parNameList = ['sizeSil', 'T1Sil', 'T2Sil', 'logM1Sil', 'logM2Sil', 'sizeGra',
                   'T1Gra', 'T2Gra', 'R1G2S', 'R2G2S', 'logumin', 'qpah', 'gamma',
                   'logMd']
    comments = """
#This mock SED is created from {0} at redshift {1}.
#The uncertainties of the data are the real uncertainties of the sources.
#The systematics: WISE:{S[0]}, PACS:{S[1]}, SPIRE:{S[2]}, MIPS:{S[3]}.
#The config file in use is {2}.
#lnlikelihood = {3}
#parNames = {4}
#inputPars = {5}
    """
    #->WISE (Jarrett2011), PACS(Balog2014), SPIRE(Pearson2013), Spitzer(MIPS handbook)
    sysUnc = {
        #"spc": 0.05,
        "spc": 0.00,
        #"pht": [([0, 2], 0.03), ([2, 5], 0.05), ([5, 8], 0.05)]
        "pht": [([0, 2], 0.00), ([2, 5], 0.00), ([5, 8], 0.00)] #Check for the accurate case
    }
    #loop_T = 0
    nRuns = len(parTable)
    for loop_T in range(nRuns):
        targname = parTable["Name"][loop_T]
        fltr_Target = infoTable["Name"]==targname
        redshift = infoTable["z"][fltr_Target][0]
        sedFile = infoTable["sed"][fltr_Target][0]
        #Load the mock parameters
        mockPars = []
        for parName in parNameList:
            mockPars.append(parTable["{0}_C".format(parName)][loop_T])
        #print parTable[loop_T]
        mock, lnlike = gsm_mocker(configName, targname, redshift, sedFile, mockPars,
                        sysUnc=sysUnc, uncModel=[-np.inf, -np.inf, -np.inf],
                        pert=False, plot=True)
        print("--------lnlike={0:.5f}".format(lnlike))
        plt.savefig("mock/{0}_mock.png".format(targname))
        plt.close()

        #->Save mock file
        wave = mock["wave"]
        flux = mock["sed"]
        sigma = mock["sigma"]
        data = np.transpose(np.array([wave, flux, sigma]))
        mockName = targname
        #f = open("mock/{0}_{1}.msed".format(mockName, mockSub), "w")
        f = open("mock/{0}_{1}_ex.msed".format(mockName, mockSub), "w")
        f.writelines("wavelength\tflux\tsigma\n")
        np.savetxt(f, data, fmt="%.2f", delimiter="\t")
        suList = [sysUnc["pht"][0][1], sysUnc["pht"][1][1], sysUnc["pht"][2][1], sysUnc["spc"]]
        cmnt = comments.format(targname, redshift, configName, lnlike, parNameList, mockPars, S=suList)
        f.writelines(cmnt)
        f.close()
