#This code is from: Composite_Model_Fit/dl07/dev_SEDClass.ipynb

import types
import numpy as np
import matplotlib.pyplot as plt
from fitter import basicclass as bc
from . import bandfunc as bf
from .dir_list import filter_path
from scipy.interpolate import splrep, splev
from collections import OrderedDict

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
    def __init__(self, targetName, redshift, Dist=None, H0=67.8, Om0=0.308, phtDict={}, spcDict={}):
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

    def pht_plotter(self, wave, flux, sigma, flag, FigAx=None, linewidth='1.5',
                    symbolColor='k', symbolSize=6, label=None, Quiet=True):
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
        if(nup == npt): # If there are all upper limits.
            pwavu = wave
            psedu = flux
            psigu = psedu/3.0
            uplims = np.ones(len(psedu), dtype=bool)
            ax.errorbar(pwavu, psedu, yerr=psigu, uplims=uplims, linestyle='none',
                        color=symbolColor, fmt='o', mfc='none', mec=symbolColor,
                        mew=linewidth, elinewidth=linewidth, label=label, ms=symbolSize)
        elif(nup > 0): # If there are some upper limits.
            fltr_upperlimit = (sigma<0)
            fltr_detection = np.logical_not(fltr_upperlimit)
            pwav = wave[fltr_detection]
            psed = flux[fltr_detection]
            psig = sigma[fltr_detection]
            pwavu = wave[fltr_upperlimit]
            psedu = flux[fltr_upperlimit]
            psigu = psedu/3.0
            uplims = np.ones(len(psedu), dtype=bool)
            ax.errorbar(pwav, psed, yerr=psig, linestyle='none', color=symbolColor, fmt='o',
                        mfc='none', mec=symbolColor, mew=linewidth, elinewidth=linewidth, label=label, ms=symbolSize)
            ax.errorbar(pwavu, psedu, yerr=psigu, uplims=uplims, linestyle='none',
                        color=symbolColor, fmt='o', mfc='none', mec=symbolColor,
                        mew=linewidth, elinewidth=linewidth, ms=symbolSize)
        else:
            pwav = wave
            psed = flux
            psig = sigma
            ax.errorbar(pwav, psed, yerr = psig, linestyle='none', color=symbolColor, fmt='o',
                        mfc='none', mec=symbolColor, mew=linewidth, elinewidth=linewidth, label=label, ms=symbolSize)
        str_xlabel = r'$\lambda \, \mathrm{(\mu m)}$'
        ax.set_xlabel(str_xlabel, fontsize=18)
        ax.set_ylabel(r'$f_\nu \, \mathrm{(mJy)}$', fontsize=18)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=16)
        return (fig, ax)

    def plot_pht(self, FigAx=None, linewidth='1.5', symbolColor='k', symbolSize=6, **kwargs):
        dataDict = self.get_dsArrays()
        for name in dataDict.keys():
            wave = dataDict[name][0]
            flux = dataDict[name][1]
            sigma = dataDict[name][2]
            flag = dataDict[name][3]
            FigAx = self.pht_plotter(wave, flux, sigma, flag, FigAx, linewidth,
                                       symbolColor, symbolSize, name)
        return FigAx

    def spc_plotter(self, wave, flux, sigma, FigAx=None, linewidth=1.,
                    color='grey', label=None, Quiet=True):
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
        ax.errorbar(wave, flux, yerr=sigma, color=color, linewidth=linewidth, label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        return (fig, ax)

    def plot_spc(self, FigAx=None, linewidth=1., color='grey', **kwargs):
        dataDict = self.get_csArrays()
        for name in dataDict.keys():
            wave = dataDict[name][0]
            flux = dataDict[name][1]
            sigma = dataDict[name][2]
            FigAx = self.spc_plotter(wave, flux, sigma, FigAx, linewidth, color, name)
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
