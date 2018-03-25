#This code is from: Composite_Model_Fit/dl07/dev_SEDClass.ipynb

import types
import numpy as np
import matplotlib.pyplot as plt
from fitter import basicclass as bc
from . import bandfunc as bf
from .dir_list import filter_path
from scipy.interpolate import splrep, splev
from collections import OrderedDict
import SED_Toolkit as sedt
#import sedfit.SED_Toolkit as sedt
__all__ = ["SedClass", "setSedData"]

ls_mic = 2.99792458e14 #micron/s
xlabelDict = {
    "cm": r'$\lambda \, \mathrm{(cm)}$',
    "mm": r'$\lambda \, \mathrm{(mm)}$',
    "micron": r'$\lambda \, \mathrm{(\mu m)}$',
    "angstrom": r'$\lambda \, \mathrm{(\AA)}$',
    "Hz": r'$\nu \, \mathrm{(Hz)}$',
    "MHz": r'$\nu \, \mathrm{(MHz)}$',
    "GHz": r'$\nu \, \mathrm{(GHz)}$',
}
ylabelDict = {
    "fnu": r'$f_\nu \, \mathrm{(mJy)}$',
    "nufnu": r'$\nu f_\nu \, \mathrm{(erg\,s^{-1}\,cm^{-2})}$',
}

class SedClass(bc.DataSet):
    """
    A class represent the SED data. It is a kind of DataSet class.

    Parameters
    ----------
    targetName : str
        The target name of the SED.
    redshift : str
        The redshift of the target.
    H0 : float, default: 67.8
        The Hubble constant assumed for the cosmology.
    Om0 : float, default: 0.308
        The density of the vacuum assumed for the cosmology.
    phtDict : dict, default: {}
        The photometric data packed in a dict. The items should be
        the DiscreteSet().
    spcDict : dict, default: {}
        The spectral data packed in a dict. The items should be the
        ContinueSet().
    """
    def __init__(self, targetName, redshift, Dist=None, H0=67.8, Om0=0.308,
                 phtDict={}, spcDict={}):
        """
        Parameters
        ----------
        targetName : string
            The target name.
        redshift : float
            The redshift of the target.
        Dist : float (optional)
            The physical (or luminosity) distance of the source.
        H0 : float
            The Hubble constant.
        Om0 : float
            The baryonic mass fraction.
        phtDict : dict
            The dict containing the information of the photometric data.
        spcDict : dict
            The dict containing the information of the spectroscopic data.

        Returns
        -------
        None.

        Notes
        -----
        None.
        """
        bc.DataSet.__init__(self, phtDict, spcDict)
        self.targetName = targetName
        self.redshift = redshift
        spc_wave = np.array(self.get_csList("x"))
        spc_flux = np.array(self.get_csList("y"))
        spc_unct = np.array(self.get_csList("e"))
        if bool(spcDict): # If there is spectral data
            self.spc_WaveLength = np.max(spc_wave) - np.min(spc_wave)
            self.spc_FluxMedian = np.sqrt(np.sum((spc_flux / spc_unct)**2) / np.sum(spc_unct**-2))
        self.__bandDict = {}
        if Dist is None:
            if redshift > 1e-2:
                #Calculate the luminosity distance
                from astropy.cosmology import FlatLambdaCDM
                cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
                self.dl = cosmo.luminosity_distance(redshift).value #Luminosity distance in unit Mpc.
            else:
                raise ValueError("The redshift ({0}) is too small to accurately estimate the distance.".format(redshift))
        else:
            self.dl = Dist

    def pht_plotter(self, wave, flux, sigma, flag, FigAx=None, ebDict=None,
                    Quiet=True, xUnits="micron", yUnits="fnu"):
        wave = np.array(wave)
        flux = np.array(flux)
        sigma = np.array(sigma)
        flag = np.array(flag)
        if(len(wave) == 0):
            if Quiet is False:
                print 'There is no data in the SED!'
            return FigAx
        npt = len(wave) # The number of data points
        nup = np.sum(sigma<0) # The number of upperlimits
        if FigAx == None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca()
        else:
            fig = FigAx[0]
            ax = FigAx[1]
            #print 'The ax is provided!'
        if ebDict is None:
            ebDict = {
                "linestyle": "none",
                "ms": 6,
                "mew": 1.5,
                "elinewidth": 1.5,
                "color": "black",
                "fmt": "o",
                "capsize": 0,
                "zorder": 4,
                }
        if yUnits == "fnu": # Default settings, units: mJy
            pass
        elif yUnits == "nufnu": # Convert from mJy to erg s^-1 cm^-2
            y_conv = ls_mic / sedt.WaveToMicron(wave, xUnits) * 1.e-26
            flux   *= y_conv
            sigma  *= y_conv
        else:
            raise ValueError("The yUnits ({0}) is not recognised!".format(yUnits))
        ax.errorbar(wave, flux, yerr=sigma, uplims=flag, **ebDict)
        ax.set_xlabel(xlabelDict[xUnits], fontsize=18)
        ax.set_ylabel(ylabelDict[yUnits], fontsize=18)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=16)
        return (fig, ax)

    def plot_pht(self, FigAx=None, phtLW=1.5, phtColor='k', phtMS=6, zorder=4,
                 **kwargs):
        dataDict = self.get_dsArrays()
        for name in dataDict.keys():
            wave = dataDict[name][0]
            flux = dataDict[name][1]
            sigma = dataDict[name][2]
            flag = dataDict[name][3]
            pht_ebDict = {
                "linestyle": "none",
                "ms": phtMS,
                "mew": phtLW,
                "elinewidth": phtLW,
                "color": phtColor,
                "fmt": "o",
                "capsize": 0,
                "zorder": zorder,
                "label": name,
                }
            FigAx = self.pht_plotter(wave, flux, sigma, flag, FigAx, pht_ebDict,
                                     **kwargs)
        return FigAx

    def spc_plotter(self, wave, flux, sigma, FigAx=None, ebDict=None, Quiet=True,
                    xUnits="micron", yUnits="fnu"):
        wave = np.array(wave)
        flux = np.array(flux)
        sigma = np.array(sigma)
        if(len(wave) == 0):
            if Quiet is False:
                print 'There is no data in the SED!'
            return FigAx
        if FigAx == None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca()
        else:
            fig = FigAx[0]
            ax = FigAx[1]
        if ebDict is None:
            ebDict = {
                "linewidth": 1.,
                "color": "grey",
                "zorder": 4.,
                }
        if yUnits == "fnu": # Default settings, units: mJy
            pass
        elif yUnits == "nufnu": # Convert from mJy to erg s^-1 cm^-2
            y_conv = ls_mic / sedt.WaveToMicron(wave, xUnits) * 1.e-26
            flux  *= y_conv
            sigma *= y_conv
        else:
            raise ValueError("The yUnits ({0}) is not recognised!".format(yUnits))
        ax.step(wave, flux, **ebDict)
        fel = flux - sigma
        feu = flux + sigma
        ebDict["label"] = None
        ax.fill_between(wave, y1=fel, y2=feu, step="pre", alpha=0.4, **ebDict)
        ax.set_xlabel(xlabelDict[xUnits], fontsize=18)
        ax.set_ylabel(ylabelDict[yUnits], fontsize=18)
        ax.set_xscale('log')
        ax.set_yscale('log')
        return (fig, ax)

    def plot_spc(self, FigAx=None, spcLS="-", spcLW=2., spcColor='grey',
                 zorder=4, **kwargs):
        dataDict = self.get_csArrays()
        for name in dataDict.keys():
            wave = dataDict[name][0]
            flux = dataDict[name][1]
            sigma = dataDict[name][2]
            spc_ebDict = {
                "linestyle": spcLS,
                "linewidth": spcLW,
                "color": spcColor,
                "zorder": zorder,
                "label": name,
                }
            FigAx = self.spc_plotter(wave, flux, sigma, FigAx, spc_ebDict,
                                     **kwargs)
        return FigAx

    def plot_sed(self, FigAx=None, **kwargs):
        if self.check_dsData() > 0:
            FigAx = self.plot_pht(FigAx=FigAx, **kwargs)
        if self.check_csData() > 0:
            FigAx = self.plot_spc(FigAx=FigAx, **kwargs)
        return FigAx

    def add_bandpass(self, bandDict):
        for bn in bandDict.keys():
            if isinstance(bandDict[bn], bf.BandPass):
                self.__bandDict[bn] = bandDict[bn]
            else:
                raise ValueError('The bandpass {0} has incorrect type!'.format(bn))

    def set_bandpass(self, bandList, sedwave, silent=True):
        z = self.redshift
        bandDict = OrderedDict()
        for loop in range(len(bandList)):
            bn = bandList[loop]
            bandCenter = sedwave[loop] * (1 + z)
            #Determine how to use the relative spectral response data.
            if bn in bf.monoFilters:
                bandFile = "{0}.dat".format(bn)
                bandPck = np.genfromtxt(filter_path+bandFile)
                bandWave = bandPck[:, 0]
                bandRsr = bandPck[:, 1]
                bandDict[bn] = bf.BandPass(bandWave, bandRsr, bandCenter, "mono", bn, z, silent)
            elif bn in bf.meanFilters:
                bandFile = "{0}.dat".format(bn)
                bandPck = np.genfromtxt(filter_path+bandFile)
                bandWave = bandPck[:, 0]
                bandRsr = bandPck[:, 1]
                bandDict[bn] = bf.BandPass(bandWave, bandRsr, bandCenter, "mean", bn, z, silent)
            else:
                bandDict[bn] = bf.BandPass(bandCenter=bandCenter, bandType="none", bandName=bn, redshift=z, silent=silent)
                if not silent:
                    print("The band {0} is not included in our database!".format(bn))
        self.add_bandpass(bandDict)

    def get_bandpass(self):
        return self.__bandDict

    def filtering(self, bandName, wavelength, flux):
        """
        Calculate the flux density of the input spectrum filtered by
        the specified band. The spectrum is considered at the rest frame.

        Parameters
        ----------
        bandName : str
            The name of the band.
        wavelength : float array
            The wavelength of the input spectrum.
        flux : float array
            The flux of the input spectrum.

        Returns
        -------
        A tuple of the band central wavelength and the flux density after filtered by
        the bandpass.

        Notes
        -----
        None.
        """
        z = self.redshift
        bandpass = self.__bandDict.get(bandName, None)
        if bandpass is None:
            raise AttributeError("The bandpass '{0}' is not found!".format(bandName))
        bandwave = bandpass.get_bandCenter_rest()
        bandflux = bandpass.filtering(wavelength, flux)
        return (bandwave, bandflux)

    def model_pht(self, wavelength, flux):
        """
        Calculate the model flux density of the input spectrum at the wavelengths of
        all the bands of the photometric data of the SED. The spectrum is considered
        at the rest frame.

        Parameters
        ----------
        wavelength : float array
            The wavelength of the input spectrum.
        flux : float array
            The flux of the input spectrum.

        Returns
        -------
        fluxList : list
            The model flux at all the wavelengths of the photometric SED.

        Notes
        -----
        None.
        """
        bandNameList = self.get_unitNameList()
        fluxList = []
        for bandName in bandNameList:
            fluxList.append(self.filtering(bandName, wavelength, flux)[1])
        return fluxList

    def model_spc(self, fluxFunc, cSetName=None):
        """
        Calculate the model flux density of the input spectrum at the wavelengths of
        all the spectra. The input spectrum is considered at the rest frame.

        Parameters
        ----------
        wavelength : float array
            The wavelength of the input spectrum.
        fluxFunc : function
            The the function to return the model fluxes.
        cSetName : str or None by default
            Specify the name of continual set to use.

        Returns
        -------
        fluxList : list
            The model flux at the wavelengths of the spectral SED.

        Notes
        -----
        None.
        """
        if cSetName is None:
            cWaveList = self.get_csList('x')
        else:
            cSet = self.__continueSetDict.get(cSetName, None)
            if cSet is None:
                raise KeyError("The set name '{0}' is not found!".format(cSetName))
            cWaveList = cSet.get_List('x')
        if len(cWaveList) > 0:
            fluxList = list( fluxFunc( np.array(cWaveList) ) )
        else:
            fluxList = []
        return fluxList

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict


def setSedData(targname, redshift, distance, dataDict, sedPck, silent=True):
    """
    Setup sedData.

    Parameters
    ----------
    targname : string
        The name of the target.
    redshift : float
        The redshift of the object.
    distance : float
        The distance of the object.
    dataDict : dict
        The information of the data.
            phtName : the name of the photometric data.
            spcName : the name of the spectroscopic data.
            bandList_use : the list of bands to be used; use all bands if empty.
            bandList_ignore : the list of bands to be ignored.
            frame : the frame of the input data is in; "rest" or "obs".
    sedPck : dict
        The data package of the SED.
            sed : concatenated SED.
            pht : photometric data.
            spc : spectroscopic data.
        All the three items are tuple of ("wave", "flux", "sigma"), with "pht"
        having an extra component "band" in the end.
    silent : (optional) bool
        The toggle not to print some information if True.

    Returns
    -------
    sedData : SEDClass object
        The data set of SED.
    """
    pht = sedPck["pht"]
    spc = sedPck["spc"]
    #->The upperlimit corresponds to how many sigmas
    nSigma = dataDict.get("nSigma", 3.)
    #->Settle into the rest frame
    frame = dataDict.get("frame", "rest") #The coordinate frame of the SED; "rest"
                                          #by default.
    if frame == "obs":
        pht = sedt.SED_to_restframe(pht, redshift)
        spc = sedt.SED_to_restframe(spc, redshift)
        if not silent:
            print("[setSedData]: The input SED is in the observed frame!")
    elif frame == "rest":
        if not silent:
            print("[setSedData]: The input SED is in the rest frame!")
    else:
        if not silent:
            print("[setSedData]: The input SED frame ({0}) is not recognised!".format(frame))
    #->Select bands
    bandList_use = dataDict.get("bandList_use", []) #The list of bands to incorporate;
                                                    #use all the available bands if empty.
    bandList_ignore = dataDict.get("bandList_ignore", []) #The list of bands to be
                                                          #ignored from the bands to use.
    pht = sedt.SED_select_band(pht, bandList_use, bandList_ignore, silent)
    phtwave  = np.array(pht[0])
    phtflux  = np.array(pht[1])
    phtsigma = np.array(pht[2])
    phtband  = np.array(pht[3])
    spcwave  = np.array(spc[0])
    spcflux  = np.array(spc[1])
    spcsigma = np.array(spc[2])
    if not silent:
        print("[setSedData]: The incorporated bands are: {0}".format(phtband))
    #->Check data. If there are nan, raise error.
    chck_pht = np.sum(np.isnan(phtflux)) + np.sum(np.isnan(phtsigma))
    chck_spc = np.sum(np.isnan(spcflux)) + np.sum(np.isnan(spcsigma))
    if chck_pht:
        raise ValueError("The photometry contains bad data!")
    if chck_spc:
        raise ValueError("The spectrum contains bad data!")
    #->Put into the sedData
    phtName = dataDict.get("phtName", None)
    if not phtName is None:
        fltr_uplim = phtsigma < 0 #Find the upperlimits.
        phtsigma[fltr_uplim] = phtflux[fltr_uplim] / nSigma #Restore the uncertainties for the non-detections.
        phtflag = np.zeros_like(phtwave) #Generate the flag for the upperlimits
        phtflag[fltr_uplim] = 1 #Find the position of the non-detections and mark them.
        phtDataType = ["name", "wavelength", "flux", "error", "flag"]
        phtData = {phtName: bc.DiscreteSet(phtband, phtwave, phtflux, phtsigma, phtflag, phtDataType)}
    else:
        phtData = {}
    spcName = dataDict.get("spcName", None)
    if not spcName is None:
        spcflag = np.zeros_like(spcwave)
        spcDataType = ["wavelength", "flux", "error", "flag"]
        spcData = {spcName: bc.ContinueSet(spcwave, spcflux, spcsigma, spcflag, spcDataType)}
    else:
        spcData = {}
    sedData = SedClass(targname, redshift, distance, phtDict=phtData, spcDict=spcData)
    sedData.set_bandpass(phtband, phtwave, silent)
    return sedData
