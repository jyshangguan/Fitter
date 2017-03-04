import numpy as np
from astropy.table import Table
if __name__ == "__main__":
    from sedfit.dir_list import template_path
else:
    from ..dir_list import template_path
from scipy.interpolate import interp1d

Mpc = 3.08567758e24 #unit: cm
mJy = 1e26 #unit: erg/s/cm^2/Hz
pi  = np.pi

torus_total_tb  = Table.read(template_path+"torus_templates/torus_total_template.dat",
                             format="ascii.ipac")
torus_clumpy_tb = Table.read(template_path+"torus_templates/torus_clumpy_template.dat",
                             format="ascii.ipac")
wavelength  = torus_total_tb["wavelength"].data
flux_total  = torus_total_tb["flux"].data
flux_clumpy = torus_clumpy_tb["flux"].data

func_total  = interp1d(wavelength, flux_total)
func_clumpy = interp1d(wavelength, flux_clumpy)

waveLim = [1e-1, 1e3]
def Torus_Template(logLtorus, DL, wave, z, frame="rest", t="total", waveLim=waveLim):
    """
    Calculate the torus emission with the dust torus templates.

    Parameters
    ----------
    logLtorus : float
        The log10 of the torus luminosity, unit: erg/s.
    DL : float
        The luminosity distance, unit: Mpc.
    wave : float array
        The wavelength at which we want to calculate the flux, unit: micron.
    frame : string
        "rest" for the rest frame SED and "obs" for the observed frame.
    t : string
        "total" for the CLUMPY+blackbody template and "clumpy" for CLUMPY template only.
    waveLim : list
        The min and max of the wavelength covered by the template.

    Returns
    -------
    flux : array of float
        The flux density (F_nu) from the model, unit: mJy.

    Notes
    -----
    None.
    """
    flux = np.zeros_like(wave)
    fltr = (wave > waveLim[0]) & (wave < waveLim[1])
    if np.sum(fltr) == 0:
        return np.zeros_like(wave)
    if t == "total":
        fluxFunc = func_total
    elif t == "clumpy":
        fluxFunc = func_clumpy
    else:
        raise ValueError("The template type ({0}) is not recognised!".format(t))
    flux[fltr] = fluxFunc(wave[fltr])
    if frame == "rest":
        idx = 2.0
    elif frame == "obs":
        idx = 1.0
    else:
        raise ValueError("The frame '{0}' is not recognised!".format(frame))
    fnu = (1.0 + z)**idx * flux * 10**logLtorus / (4 * pi * (DL * Mpc)**2) * mJy
    return fnu

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    wave = wavelength
    flux = Torus_Template(40, 200, wave, 0)
    plt.plot(wave, flux)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
