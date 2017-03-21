
# coding: utf-8

# # This page is to release the functions to manipulate the SEDs and spectra
# * The prototype of this page is [SEDToolKit](http://localhost:8888/notebooks/SEDFitting/SEDToolKit.ipynb) in /Users/jinyi/Work/PG_QSO/SEDFitting/

# In[2]:

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import interp1d

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Dec. 13, 2015    #
#   Modified by SGJY, Jul. 15, 2016   #
#-------------------------------------#
#From: /PG_QSO/catalog/Data_SG/wrk_SED_Combine.ipynb
def Plot_SED(wavelength,
             sed,
             sigma,
             TargetName=None,
             RestFrame=False,
             FigAx=None,
             gridon=True,
             symbolColor='black',
             linewidth=2,
             Quiet=False):
    '''
    This function is to conveniently plot the SED. The parameters are:
    wavelength: the array of wavelength.
    sed: the fluxes or upperlimits at the corresponding wavelength.
    sigma: the uncertainties for detecion and -1 for non-detection at the corresponding wavelength.
    TargetName: the name of the target which will be shown in the figure.
    RestFrame: if False, the xlabel would be lambda_obs, while if True, the xlabel would be lambda_rest.
    FigAx: if not None, FigAx should be a tuple with (fig, ax) in it.
    '''
    if((len(wavelength)!=len(sed))|(len(wavelength)!=len(sigma))):
        raise Exception('Array lengths are unequal!')
        return None
    if(len(sed) == 0):
        if Quiet is False:
            print 'There is no data in the SED!'
        return FigAx
    npt = len(sed) # The number of data points
    nup = np.sum(sigma<0) # The number of upperlimits
    #print 'The total points are %d with %d upper limits.' % (npt, nup)
    if FigAx == None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca()
    else:
        fig = FigAx[0]
        ax = FigAx[1]
        #print 'The ax is provided!'
    if(nup == npt): # If there are all upper limits.
        pwavu = wavelength
        psedu = sed
        psigu = psedu/3.0
        uplims = np.ones(len(psedu), dtype=bool)
        ax.errorbar(pwavu, psedu, yerr=psigu, uplims=uplims, linestyle='none',
                    color=symbolColor, fmt='o', mfc='none', mec=symbolColor,
                    mew=linewidth, elinewidth=linewidth)
    elif(nup > 0): # If there are some upper limits.
        fltr_upperlimit = (sigma<0)
        fltr_detection = np.logical_not(fltr_upperlimit)
        pwav = wavelength[fltr_detection]
        psed = sed[fltr_detection]
        psig = sigma[fltr_detection]
        pwavu = wavelength[fltr_upperlimit]
        psedu = sed[fltr_upperlimit]
        psigu = psedu/3.0
        uplims = np.ones(len(psedu), dtype=bool)
        ax.errorbar(pwav, psed, yerr=psig, linestyle='none', color=symbolColor, fmt='o',
                    mfc='none', mec=symbolColor, mew=linewidth, elinewidth=linewidth)
        ax.errorbar(pwavu, psedu, yerr=psigu, uplims=uplims, linestyle='none',
                    color=symbolColor, fmt='o', mfc='none', mec=symbolColor,
                    mew=linewidth, elinewidth=linewidth)
    else:
        pwav = wavelength
        psed = sed
        psig = sigma
        ax.errorbar(pwav, psed, yerr = psig, linestyle='none', color=symbolColor, fmt='o',
                    mfc='none', mec=symbolColor, mew=linewidth, elinewidth=linewidth)
    if(RestFrame == True):
        str_xlabel = r'$\lambda_\mathrm{rest} \, \mathrm{(\mu m)}$'
    else:
        str_xlabel = r'$\lambda_\mathrm{obs} \, \mathrm{(\mu m)}$'
    ax.set_xlabel(str_xlabel, fontsize=18)
    ax.set_ylabel(r'$f_\nu \, \mathrm{(mJy)}$', fontsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if gridon:
        ax.grid(which='both')
    if TargetName is not None:
        ax.text(0.45, 0.9, TargetName,
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontsize=15)
    ax.tick_params(labelsize=16)
    return fig, ax
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Jan. 26, 2016    #
#   Modified by SGJY, Mar. 23, 2016   #
#   Modified by SGJY, May. 4, 2016    #
#-------------------------------------#
def SpectraRebin(bin_rsl, spcwave, spcflux, spcsigma):
    '''
    This function rebin the spectra according the bin resolution provided in
    parameters. The median value in each flux bin are used. The root of square
    mean value are adapted in each uncertainty bin.

    Parameters
    ----------
    bin_rsl : int
        The resolution of each bin. How many data go into one bin.
    spcwave : float array
        The array of spectra wavelength.
    spcflux : float array
        The array of spectra flux.
    spcsigma : float array
        The array of spectra uncertainty.

    Returns
    -------
    A tuple of three arrays (binwave, binflux, binsigma).
    binwave : float array
        The array of binned spectra wavelength.
    binflux : float array
        The array of binned spectra flux.
    binsigma : float array
        The array of binned spectra uncertainty.

    Notes
    -----
    None.
    '''
    if bin_rsl <= 1.0:
        return (spcwave, spcflux, spcsigma)
    len_spc = int(len(spcwave)/bin_rsl)
    binwave = np.zeros(len_spc)
    binflux = np.zeros(len_spc)
    binsigma = np.zeros(len_spc)
    for loop_spc in range(len_spc):
        indx_bng = bin_rsl * loop_spc
        indx_end = bin_rsl * (loop_spc + 1)
        select_wave = spcwave[indx_bng:indx_end]
        select_flux = spcflux[indx_bng:indx_end]
        select_sigma = spcsigma[indx_bng:indx_end]
        fltr = np.logical_not(np.isnan(select_flux))
        binwave[loop_spc] = np.mean(select_wave[fltr])
        binflux[loop_spc] = np.median(select_flux[fltr])
        binsigma[loop_spc] = np.sqrt(np.sum(select_sigma[fltr])**2.0)/len(select_sigma[fltr])

    return (binwave, binflux, binsigma)
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Jan. 24, 2016    #
#-------------------------------------#
def SEDResample(wave, flux, wave_resmp):
    '''
    This function is to resample the SED at given wavelengths.

    Parameters
    ----------
    wave : array like
        The wavelengths of the template.
    flux : array like
        The fluxes of the template.
    wave_resmp : array like
        The wavelengths for resampled SED.

    Returns
    -------
    flux_resmp : array like
        The fluxes for resampled SED.

    Notes
    -----
    None.
    '''
    fsed = interp1d(wave, flux, kind='linear')
    flux_resmp = fsed(wave_resmp)
    return flux_resmp
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Mar. 7, 2016     #
#-------------------------------------#
def Filter_Pass(wave, flux, fltr_wv, fltr_rs):
    '''
    This function is to calculate the flux density of a model spectrum
    passing the given filter. The flux density is calculate as the
    average of the model spectrum weighted by the filter responce curve.

    Parameters
    ----------
    wave : float array
        The model wavelength.
    flux : float array
        The model spectral flux.
    fltr_wv : float array
        The wavelength of the response curve.
    fltr_rs : float array
        The fractional flux density transmission.

    Returns
    -------
    fd : float
        The flux density.

    Notes
    -----
    None.
    '''

    model_func = interp1d(wave, flux, kind='linear')
    model_flux = model_func(fltr_wv)
    fd = np.trapz(model_flux * fltr_rs, fltr_wv)/np.trapz(fltr_rs, fltr_wv)
    return fd
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Mar. 7, 2016     #
#-------------------------------------#
def Herschel_Bands(wave, flux,
                   pacs_70=None,
                   pacs_100=None,
                   pacs_160=None,
                   spire_250=None,
                   spire_350=None,
                   spire_500=None):
    '''
    This function calculate the photometric data points from the model spectrum.

    Parameters
    ----------
    wave : float array
        The model wavelength.
    flux : float array
        The model flux.
    pacs_70 : tuple, (wave, response), default: None
        The wavelength and response fraction of the 70 micron filter.
    pacs_100 : tuple, (wave, response), default: None
        The wavelength and response fraction of the 100 micron filter.
    pacs_160 : tuple, (wave, response), default: None
        The wavelength and response fraction of the 160 micron filter.
    spire_250 : tuple, (wave, response), default: None
        The wavelength and response fraction of the 250 micron filter.
    spire_350 : tuple, (wave, response), default: None
        The wavelength and response fraction of the 350 micron filter.
    spire_500 : tuple, (wave, response), default: None
        The wavelength and response fraction of the 500 micron filter.

    Returns
    -------
    (photo_wave, photo_flxd) : tuple
        The wavelength of the photometric points and the corresponding
        flux density.

    Notes
    -----
    None.
    '''
    if pacs_70 is None:
        fd_70 = np.nan
    else:
        pacs_70_wv = pacs_70[0]
        pacs_70_rs = pacs_70[1]
        fd_70 = Filter_Pass(wave, flux, pacs_70_wv, pacs_70_rs)
    if pacs_100 is None:
        fd_100 = np.nan
    else:
        pacs_100_wv = pacs_100[0]
        pacs_100_rs = pacs_100[1]
        fd_100 = Filter_Pass(wave, flux, pacs_100_wv, pacs_100_rs)
    if pacs_160 is None:
        fd_160 = np.nan
    else:
        pacs_160_wv = pacs_160[0]
        pacs_160_rs = pacs_160[1]
        fd_160 = Filter_Pass(wave, flux, pacs_160_wv, pacs_160_rs)
    if spire_250 is None:
        fd_250 = np.nan
    else:
        spire_250_wv = spire_250[0]
        spire_250_rs = spire_250[1]
        fd_250 = Filter_Pass(wave, flux, spire_250_wv, spire_250_rs)
    if spire_350 is None:
        fd_350 = np.nan
    else:
        spire_350_wv = spire_350[0]
        spire_350_rs = spire_350[1]
        fd_350 = Filter_Pass(wave, flux, spire_350_wv, spire_350_rs)
    if spire_500 is None:
        fd_500 = np.nan
    else:
        spire_500_wv = spire_500[0]
        spire_500_rs = spire_500[1]
        fd_500 = Filter_Pass(wave, flux, spire_500_wv, spire_500_rs)

    photo_wave = np.array([70., 100., 160., 250., 350., 500.])
    photo_flxd = np.array([fd_70, fd_100, fd_160, fd_250, fd_350, fd_500])
    return (photo_wave, photo_flxd)
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Mar. 28, 2016    #
#-------------------------------------#
#From: wrk_Plot_Models.ipynb
def Load_SED_old(sedfile, sed_range=[7, 13], spc_range=[13, None], spc_binrsl=10.):
    '''
    This function is to load the SED data and compile it for use.

    Parameters
    ----------
    sedfile : str
        The full path of sed file.
    sed_range : list, default: [7, 13]
        The min and max+1 index of the sed data points.
    spc_range : list, default: [13, None]
        The min and max+1 index of the spectra data points.
    spc_binrsl : float
        The resolution to rebin the spectra data.

    Returns
    -------
    sed_package : dict
        The dictionary storing:
            sed_cb : combined sed
            sed : photometric data
            spc : spectra data
            spc_raw : unbinned spectra data

    Notes
    -----
    None.
    '''
    sedtb = Table.read(sedfile, format="ascii")
    wave  = sedtb["wavelength"].data
    flux  = sedtb["flux"].data
    sigma = sedtb["sigma"].data
    sedwave = wave[sed_range[0]:sed_range[1]]
    sedflux = flux[sed_range[0]:sed_range[1]]
    sedsigma = sigma[sed_range[0]:sed_range[1]]
    spcwave_raw = wave[spc_range[0]:spc_range[1]]
    spcflux_raw = flux[spc_range[0]:spc_range[1]]
    spcsigma_raw = sigma[spc_range[0]:spc_range[1]]
    spcbin = SpectraRebin(spc_binrsl, spcwave_raw, spcflux_raw, spcsigma_raw)
    spcwave = spcbin[0]
    spcflux = spcbin[1]
    spcsigma = spcbin[2]

    wave_cb = np.concatenate([sedwave, spcwave])
    flux_cb = np.concatenate([sedflux, spcflux])
    sigma_cb = np.concatenate([sedsigma, spcsigma])
    sed_cb = (wave_cb, flux_cb, sigma_cb)
    sed = (sedwave, sedflux, sedsigma)
    spc = (spcwave, spcflux, spcsigma)
    spc_raw = (spcwave_raw, spcflux_raw, spcsigma_raw)
    sed_package = {
        'sed_cb':sed_cb,
        'sed':sed,
        'spc':spc,
        'spc_raw':spc_raw,
    }
    return sed_package
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Jan. 8, 2017     #
#-------------------------------------#
def Load_SED(sedfile):
    '''
    This function is to load the SED data and compile it for use.

    Parameters
    ----------
    sedfile : str
        The full path of sed file.

    Returns
    -------
    sed_package : dict
        The dictionary storing:
            sed_cb : combined sed; (wave, flux, sigma)
            sed : photometric data; (wave, flux, sigma, band)
            spc : spectra data; (wave, flux, sigma)

    Notes
    -----
    The returned SED data are in the lists instead of the numpy.array.
    '''
    sedtb = Table.read(sedfile, format="ascii")
    wave  = sedtb["wavelength"].data
    flux  = sedtb["flux"].data
    sigma = sedtb["sigma"].data
    band  = sedtb["band"].data
    fltr_spc = band == "0"
    fltr_sed = np.logical_not(fltr_spc)
    sedwave  = wave[fltr_sed]
    sedflux  = flux[fltr_sed]
    sedsigma = sigma[fltr_sed]
    sedband  = band[fltr_sed]
    spcwave  = wave[fltr_spc]
    spcflux  = flux[fltr_spc]
    spcsigma = sigma[fltr_spc]

    wave_cb = np.concatenate([sedwave, spcwave])
    flux_cb = np.concatenate([sedflux, spcflux])
    sigma_cb = np.concatenate([sedsigma, spcsigma])
    sed_cb = (list(wave_cb), list(flux_cb), list(sigma_cb))
    sed = (list(sedwave), list(sedflux), list(sedsigma), list(sedband))
    spc = (list(spcwave), list(spcflux), list(spcsigma))
    sed_package = {
        'sed_cb':sed_cb,
        'sed':sed,
        'spc':spc,
    }
    return sed_package
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Jan. 9, 2017     #
#-------------------------------------#
def SED_select_band(sed, bandList_use=[], bandList_ignore=[], silent=True):
    """
    Select the SED from the input band list.

    Parameters
    ----------
    sed : tuple
        The tuple of the photometric SED data; (wave, flux, sigma, band).
    bandList_use : list
        The list of the bands that are used.
    bandList_ignore : list
        The list of the bands that are ignored.
    silent : bool
        Stop printing out information if True, by default.

    Returns
    -------
    sed_select : tuple
        The selected SED data that are used.

    Notes
    -----
    None.
    """
    wave  = []
    flux  = []
    sigma = []
    band  = []
    if not len(bandList_use):
        bandList_use = list(sed[3])
    for bn in bandList_ignore:
        if bn in bandList_use:
            bandList_use.remove(bn)
        else:
            if not silent:
                print("Warning: Band {0} is not included in the SED.".format(bn))
    for bn in bandList_use:
        idx = sed[3].index(bn)
        if np.isnan(sed[1][idx]*sed[2][idx]):
            if not silent:
                print("Warning: Band {0} contains bad data!".format(bn))
        else:
            wave.append(sed[0][idx])
            flux.append(sed[1][idx])
            sigma.append(sed[2][idx])
            band.append(sed[3][idx])
    if len(band) == 0:
        raise RuntimeError("There is no band in use!")
    sed_select = (wave, flux, sigma, band)
    return sed_select
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Feb. 11, 2017    #
#-------------------------------------#
def SED_to_restframe(sed, redshift):
    """
    Transform the input SED into its rest frame according to the given redshift.

    Parameters
    ----------
    sed : tuple
        The input SED data in the observed frame. Assume it consists (wave, flux,
        sigma, ...).
    redshift : float
        The redshift of the SED.

    Returns
    -------
    sed_rest : tuple
        The output SED data in the rest frame. The tuple consists all the same
        data as the input except the first three items.

    Notes
    -----
    None.
    """
    wave  = np.array(sed[0])
    flux  = np.array(sed[1])
    sigma = np.array(sed[2])
    fltr  = sigma == -1
    sigma[fltr] = sigma[fltr] / (1 + redshift)
    sed_rest = list(sed)
    sed_rest[0] = list( wave / (1 + redshift) )
    sed_rest[1] = list( flux * (1 + redshift) )
    sed_rest[2] = list( sigma * (1 + redshift) )
    sed_rest = tuple(sed_rest)
    return sed_rest
#Func_end

#Func_bgn:
#-------------------------------------#
#   Created by SGJY, Feb. 11, 2017    #
#-------------------------------------#
def SED_to_obsframe(sed, redshift):
    """
    Transform the input SED into the observed frame according to the given redshift.

    Parameters
    ----------
    sed : tuple
        The input SED data in the observed frame. Assume it consists (wave, flux,
        sigma, ...).
    redshift : float
        The redshift of the SED.

    Returns
    -------
    sed_obs : tuple
        The output SED data in the observed frame. The tuple consists all the same
        data as the input except the first three items.

    Notes
    -----
    None.
    """
    wave  = np.array(sed[0])
    flux  = np.array(sed[1])
    sigma = np.array(sed[2])
    fltr  = sigma == -1
    sigma[fltr] = sigma[fltr] * (1 + redshift)
    sed_obs = list(sed)
    sed_obs[0] = list( wave * (1 + redshift) )
    sed_obs[1] = list( flux / (1 + redshift) )
    sed_obs[2] = list( sigma / (1 + redshift) )
    sed_obs= tuple(sed_obs)
    return sed_obs
#Func_end

if __name__ == "__main__":
    wave = np.arange(8, dtype="float")
    flux = np.arange(8, dtype="float")
    sigma = np.arange(8, dtype="float")
    band = (np.arange(8)).astype("S")
    flux[4] = np.nan
    print band
    bandList_use = []
    bandList_ignore = ["1", "6"]
    sed = (list(wave), list(flux), list(sigma), list(band))
    sed_select = SED_select_band(sed, bandList_use, bandList_ignore, False)
    sed_rest = SED_to_restframe(sed_select, 1.0)
    print sed
    print sed_select
    print sed_rest
