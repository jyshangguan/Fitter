import os
import sys
import types
import importlib
import numpy as np
#np.seterr(all="ignore")
import george
from george import kernels
import matplotlib
import matplotlib.pyplot as plt
matplotlib_version = eval(matplotlib.__version__.split(".")[0])
if matplotlib_version > 1:
    plt.style.use("classic")
import sedfit.SED_Toolkit as sedt
from astropy.table import Table
from sedfit.fitter import basicclass as bc
from sedfit import sedclass as sedsc
from sedfit import model_functions as sedmf
from sedfit.fit_functions import logLFunc_gp, logLFunc

def dataPerturb(x, sigma, pert=True, maxIter=10):
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
            if counter > maxIter:
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

def mocker(sedData, sedModel, sysUnc=None, uncModel=None, silent=True,
           pert=True, nonDetect=True):
    """
    This function is to generate a mock SED according to a given observed SED.
    Basically, the flux densities will be replaced by the model value while the
    wavelength and uncertainties of the data will be kept.

    Parameters
    ----------
    sedData : SEDClass object
        The data set of SED.
    sedModel : ModelCombiner object
        The combined model. The parameters are set to generate the mock SED.
    sysUnc : dict or None, by default
        {
            "pht": [([nBgn, nEnd], frac), ([nBgn, nEnd], frac), ...],
            "spc": frac
        }
    uncModel : dict or None, by default
        {
            "lnf" : float, (-inf, 0]
                The ln of f, the imperfectness of the model.
            "lna" : float, (-inf, 1]
                The ln of a, the amplitude of the residual correlation.
            "lntau" : float, (-inf, 1]
                The ln of tau, the scale length of the residual correlation.
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
    #->Generate the mock data
    waveModel = sedModel.get_xList()
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
    mockPhtWave = np.array(sedData.get_dsList("x"))
    mockSpcWave = np.array(sedData.get_csList("x"))
    #->Systematic uncertainties
    if not sysUnc is None:
        sysSpc = sysUnc["spc"]
        mockSpc = (1 + randomRange(-sysSpc, sysSpc)) * mockSpc
        sysPhtList = sysUnc["pht"]
        for phtRg, frac in sysPhtList:
            #print phtRg, frac, mockPht[phtRg[0]:phtRg[1]]
            mockPht[phtRg[0]:phtRg[1]] = (1 + randomRange(-frac, frac)) * mockPht[phtRg[0]:phtRg[1]]
    #->Model imperfectness & spectral residual correlation
    if not uncModel is None:
        e = np.e
        #For the photometric data
        if sedData.check_dsData():
            f = e**uncModel["lnf"]
            mockPht = (1 + randomRange(-f, f)) * mockPht
        else:
            f = 0
        #For the spectral data
        if sedData.check_csData():
            a   = e**uncModel["lna"]
            tau = e**uncModel["lntau"]
            gp = george.GP(a * kernels.ExpSquaredKernel(tau))
            mockSpc = (1 + randomRange(-f, f)) * mockSpc
            mockSpc += gp.sample(mockSpcWave)
    #->Add the upperlimits
    if nonDetect:
        phtflux = np.array(sedData.get_dsList("y"))
        phtsigma = np.array(sedData.get_dsList("e"))
        fltr_undct = phtsigma < 0
        mockPht[fltr_undct] = phtflux[fltr_undct]
    mockSedFlux  = np.concatenate([mockPht, mockSpc])
    mockSedWave  = np.concatenate([mockPhtWave, mockSpcWave])
    mockSedSigma = np.concatenate([mockPhtSigma, mockSpcSigma])
    mockPhtBand = sedData.get_unitNameList()
    mockPck = {
        "sed": (mockSedWave, mockSedFlux, mockSedSigma),
        "pht": (mockPhtWave, mockPht, mockPhtSigma, mockPhtBand),
        "spc": (mockSpcWave, mockSpc, mockSpcSigma),
    }
    return mockPck

def sedLnLike(sedData, sedModel, uncModel):
    """
    Calculate the lnlike of the SED
    """
    mockPars = sedModel.get_parVaryList()
    if uncModel is None:
        lnlike = logLFunc(mockPars, sedData, sedModel)
    else:
        mockPars = list(mockPars)
        mockPars.append(uncModel["lnf"])
        mockPars.append(uncModel["lna"])
        mockPars.append(uncModel["lntau"])
        lnlike = logLFunc_gp(mockPars, sedData, sedModel)
    return lnlike

def PlotMockSED(sedData, mockData, sedModel):
    waveModel = sedModel.get_xList()
    sedPhtFlux = sedData.get_List("y")
    xmin = np.min(waveModel)
    xmax = np.max(waveModel)
    ymin = np.min(sedPhtFlux) / 10.0
    ymax = np.max(sedPhtFlux) * 10.0
    FigAx = sedData.plot_sed()
    FigAx = mockData.plot_sed(FigAx=FigAx, phtColor="r", spcColor="r")
    FigAx = sedModel.plot(FigAx=FigAx)
    fig, ax = FigAx
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    return (fig, ax)

def gsm_mocker(configName, targname=None, redshift=None, distance=None, sedFile=None,
               mockPars=None, uncModel=None, plot=False, **kwargs):
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

    try:
        silent = config.silent
    except:
        silent = False
    ############################################################################
    #                                 Data                                     #
    ############################################################################
    dataDict = config.dataDict
    sedPck = sedt.Load_SED(sedFile)
    sedData = sedsc.setSedData(targname, redshift, distance, dataDict, sedPck, silent)

    ############################################################################
    #                                 Model                                    #
    ############################################################################
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

    #->Build up the model
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
    sedModel.updateParList(mockPars) #Set the model with the desiring parameters.

    ############################################################################
    #                                  Mock                                    #
    ############################################################################
    mockPck = mocker(sedData, sedModel, **kwargs)
    #->Reform the result and switch back to the observed frame
    sed = mockPck["sed"]
    pht = mockPck["pht"]
    spc = mockPck["spc"]
    sed = sedt.SED_to_obsframe(sed, redshift)
    pht = sedt.SED_to_obsframe(pht, redshift)
    spc = sedt.SED_to_obsframe(spc, redshift)
    phtwave = pht[0]
    phtband = pht[3]
    spcwave = spc[0]
    mockSedWave = sed[0]
    mockSedFlux = sed[1]
    mockSedSigma = sed[2]
    mockSedBand = np.concatenate([phtband, np.zeros_like(spcwave, dtype="int")])
    mock = (mockSedWave, mockSedFlux, mockSedSigma, mockSedBand)
    #->Calculate the lnlike
    mockDict = {
        "phtName": "Phot",
        "spcName": "IRS",
        "bandList_use": [],
        "bandList_ignore":["WISE_w3", "WISE_w4"],
        "frame": "obs",
    }
    mockPck = {
        "sed": sed,
        "pht": pht,
        "spc": spc
    }
    mockData = sedsc.setSedData(targname, redshift, distance, mockDict, mockPck, silent)
    lnlike = sedLnLike(mockData, sedModel, uncModel)
    #->Plot
    if plot:
        FigAx = PlotMockSED(sedData, mockData, sedModel)
    return mock, lnlike, FigAx

if __name__ == "__main__":
    #-->Generate Mock Data
    parTable = Table.read("/Volumes/Transcend/Work/PG_MCMC/pg_clu_qpahVar/compile_pg_clu.ipac", format="ascii.ipac")
    infoTable = Table.read("targlist/targlist_rq.ipac", format="ascii.ipac")
    #print parTable.colnames
    if os.path.isdir("configs"):
        sys.path.append("configs/")
    configName = "config_mock_clu"
    mockSub = "test"

    parNameList = ['logMs', 'logOmega', 'T', 'logL', 'i', 'tv', 'q', 'N0', 'sigma', 'Y',
                   'logumin', 'qpah', 'gamma', 'logMd']
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
    nRuns = 1 #len(parTable)
    for loop_T in range(nRuns):
        targname = infoTable["Name"][loop_T]
        redshift = infoTable["z"][loop_T]
        sedFile = infoTable["sed"][loop_T]
        fltr_Target = parTable["Name"]==targname
        #Load the mock parameters
        mockPars = []
        for parName in parNameList:
            mockPars.append(parTable["{0}_C".format(parName)][fltr_Target][0])
        #print parTable[loop_T]
        gsmPck = gsm_mocker(configName, targname, redshift, sedFile=sedFile,
                            mockPars=mockPars, sysUnc=sysUnc, #uncModel=[-np.inf, -np.inf, -np.inf],
                            pert=False, plot=True)
        mock, lnlike, FigAx = gsmPck
        plt.savefig("mock/{0}_mock.png".format(targname))
        plt.close()
        print("--------lnlike={0:.5f}".format(lnlike))

        #->Save mock file
        wave = mock[0]
        flux = mock[1]
        sigma = mock[2]
        band = mock[3]
        mockTable = Table([wave, flux, sigma, band],
                          names=['wavelength', 'flux', 'sigma', "band"])
        mockTable["wavelength"].format = "%.3f"
        mockTable["flux"].format = "%.3f"
        mockTable["sigma"].format = "%.3f"
        mockName = "mock/{0}_{1}.msed".format(targname, mockSub)
        mockTable.write(mockName, format="ascii", delimiter="\t", overwrite=True)
        #f = open("mock/{0}_{1}.msed".format(mockName, mockSub), "w")
        f = open(mockName, "a")
        suList = [sysUnc["pht"][0][1], sysUnc["pht"][1][1], sysUnc["pht"][2][1], sysUnc["spc"]]
        cmnt = comments.format(targname, redshift, configName, lnlike, parNameList, mockPars, S=suList)
        f.writelines(cmnt)
        f.close()
