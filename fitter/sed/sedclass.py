#This code is from: Composite_Model_Fit/dl07/dev_SEDClass.ipynb

import numpy as np
import matplotlib.pyplot as plt
from .. import basicclass as bc
from scipy.interpolate import interp1d
import types


class BandPass(object):
    """
    A class to represent one filter bandpass of a instrument.

    It contains the bandpass information and can convert the spectra
    into the band flux density.

    Parameters
    ----------
    wavelength : float array
        The wavelength of the relative system response curve.
    rsr : float array
        The relative system response curve.
    """
    def __init__(self, waveList, rsrList, bandCenter=None, bandFunc=None, bandName='None'):
        waveMin = waveList[0]
        waveMax = waveList[-1]
        if waveMin >= waveMax:
            raise ValueError("The waveList sequence is incorrect!")
        if len(waveList) == len(rsrList):
            self.__waveList = waveList
            self.__rsrList = rsrList
            self._bandName = bandName
        else:
            raise ValueError("The inputs are not matched!")
        if (type(bandFunc) == types.FunctionType) or (bandFunc is None):
            self.__bandFunc = bandFunc
        else:
            raise ValueError("The input bandFunc is incorrect!")
        self.__filterFunction = interp1d(waveList, rsrList)
        if bandCenter is None: #The bandCenter is not specified, the effective wavelength will
                               #be used (Eq. A21, Bessell&Murphy 2012), assuming fnu~const.
            self._bandCenter = np.trapz(rsrList, waveList)/np.trapz(rsrList/waveList, waveList)
        else:
            self._bandCenter = bandCenter

    def get_bandpass(self):
        bandInfo = {
            'wave_list': self.__waveList,
            'rsr_list': self.__rsrList,
            'band_name': self._bandName,
            'wave_center': self._bandCenter,
        }
        return bandInfo

    def filtering(self, wavelength, flux, **kwargs):
        """
        Calculate the flux density of the input spectrum filtered
        by the bandpass.

        Parameters
        ----------
        wavelength : float array
            The wavelength of the input spectrum.
        flux : float array
            The flux of the input spectrum.
        kwargs : dict
            The dict for additional parameters.

        Returns
        -------
        fluxFltr : float
            The flux density of the spectrum after filtered by the bandpass.

        Notes
        -----
        None.
        """
        bandCenter = self._bandCenter
        wvMin = wavelength[0]
        wvMax = wavelength[-1]
        if( (bandCenter <= wvMin) or (bandCenter >= wvMax) ):
            raise ValueError("The band center '{0}' is out of the wavelength range '[{1}, {2}]!".format(bandCenter, wvMin, wvMax))
        waveMin = self.__waveList[0]
        waveMax = self.__waveList[-1]
        fltr = (wavelength > waveMin) & (wavelength < waveMax)
        if np.sum(fltr) == 0:
            raise ValueError("The wavelength is not overlapped with the filter!")
        wavelength = wavelength[fltr]
        flux = flux[fltr]
        rsrList = self.__filterFunction(wavelength)
        bandFunc = self.__bandFunc
        if bandFunc is None: #By default, the rsr is photon response and the band flux
                             #is defined as eq. A12 (Bessell&Murphy 2012).
            signal = np.trapz(rsrList/wavelength*flux, x=wavelength)
            norm   = np.trapz(rsrList/wavelength, x=wavelength)
            fluxFltr = signal/norm
        else: #Use the user specified function to get the filtered flux.
            kwargsAdd = {
                'band_center': bandCenter,
                'band_name': self._bandName,
                'wavelength': wavelength,
                'rsr_list': rsrList,
                'flux': flux
            }
            kwargs.update(kwargsAdd)
            fluxFltr = bandFunc(kwargs)
        return fluxFltr


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
    def __init__(self, targetName, redshift, H0=67.8, Om0=0.308, phtDict={}, spcDict={}):
        bc.DataSet.__init__(self, phtDict, spcDict)
        self.targetName = targetName
        self.redshift = redshift
        self.__bandDict = {}
        #Calculate the luminosity distance
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        self.dl = cosmo.luminosity_distance(redshift).value #Luminosity distance in unit Mpc.

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
            print sigma
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
        FigAx = self.plot_pht(FigAx=FigAx, **kwargs)
        FigAx = self.plot_spc(FigAx=FigAx, **kwargs)
        return FigAx

    def set_bandpass(self, bandDict):
        for bn in bandDict.keys():
            if not isinstance(bandDict[bn], BandPass):
                raise ValueError('The bandpass {0} has incorrect type!'.format(bn))
        self.__bandDict = bandDict

    def add_bandpass(self, bandDict):
        for bn in bandDict.keys():
            if isinstance(bandDict[bn], BandPass):
                self.__bandDict[bn] = bandDict[bn]
            else:
                raise ValueError('The bandpass {0} has incorrect type!'.format(bn))

    def get_bandpass(self):
        return self.__bandDict

    def filtering(self, bandName, wavelength, flux, **kwargs):
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
        waveObs = wavelength * (1+z) #Only need to move the wavelength.
        fluxObs = flux# / (1+z) #Do not need to move the flux.
        bandpass = self.__bandDict.get(bandName, None)
        if bandpass is None:
            raise AttributeError("The bandpass '{0}' is not found!".format(bandName))
        bandCenter = bandpass._bandCenter
        bandFlux = bandpass.filtering(waveObs, fluxObs, **kwargs)
        waveRst = bandCenter / (1+z)
        fluxRst = bandFlux# * (1+z)
        return (waveRst, fluxRst)

    def model_pht(self, wavelength, flux, bandKws={}):
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
        bandKws : dict, by default: {}
            All the necessary parameters of each bandpass.

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
            kws = bandKws.get(bandName, {})
            fluxList.append(self.filtering(bandName, wavelength, flux, **kws)[1])
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
        fluxList = list( fluxFunc( np.array(cWaveList) ) )
        return fluxList
