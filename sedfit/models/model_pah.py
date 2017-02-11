import numpy as np
from scipy.interpolate import interp1d
from ..dir_list import template_path

Mpc = 3.08567758e24 #unit: cm
pi = np.pi

tb = np.genfromtxt(template_path+"PAH.template.dat")
twave = tb[:, 0]
tflux_temp = tb[:, 1]
norm = np.trapz(tflux_temp, twave)
tflux = tflux_temp / norm
tPAH = interp1d(twave, tflux)
waveLim = [np.min(twave), np.max(twave)]

def pah(logLpah, wave, DL, z, frame="rest", t=tPAH, waveLim=waveLim):
    """
    Calculate the model flux density of the PAH template.

    Parameters
    ----------
    logLpah : float
        The log luminosity of the PAH template, unit: erg/s.
    wave : float array
        The wavelength array of the model SED, unit: match the template.
    DL : float
        The luminosity density, unit: Mpc.
    z : float
        The redshift.
    frame : string
        "rest" for the rest frame SED and "obs" for the observed frame.
    t : function
        The interpolated PAH template, which should have been normalised.
    waveLim : list
        The min and max of the wavelength covered by the template.

    Returns
    -------
    flux : float array
        The model flux density, unit: mJy.

    Notes
    -----
    None.
    """
    if frame == "rest":
        idx = 2.0
    elif frame == "obs":
        idx = 1.0
    else:
        raise ValueError("The frame '{0}' is not recognised!".format(frame))
    flux = np.zeros_like(wave)
    fltr = (wave > waveLim[0]) & (wave < waveLim[1])
    if np.sum(fltr) == 0:
        return np.zeros_like(wave)
    f0 = (1 + z)**idx * 10**(logLpah+26) / (4 * pi * (DL * Mpc)**2.) #Convert to mJy unit
    flux[fltr] = f0 * t(wave[fltr])
    return flux

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print waveLim
    wave = 10**np.linspace(0, 3, 1000)
    flux = pah(40, wave, DL=30, z=0)
    plt.plot(wave, flux)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    print np.trapz(flux, wave) * (4 * pi * (30 * Mpc)**2.) / 1e26
