import numpy as np
from extinction import calzetti00

waveLim = [0.12, 2.2] # units: Micron
def Calzetti00(Av, wave, Rv=3.1, waveLim=waveLim, QuietMode=True):
    """
    Calculate the extinction that is directly applied to the flux:
        10**(-0.4 * A_lambda).
    For the input wavelength out of the effective range (waveLim), 1 will be
    returned, since this is multiplied on the fluxes.

    Parameters
    ----------
    Av : float
        The Av, V band extinction.
    wave : array
        The wavelength to calculate the extinction, units: micron.
    Rv : float
        Ratio of total to selective extinction, A_V / E(B-V).

    Returns
    -------
    f0 : array
        The ratio of the fluxes after and before the extinction.

    Notes
    -----
    None.
    """
    #-> Check the wavelength coverage.
    fltr = (wave >= waveLim[0]) & (wave <= waveLim[1])
    #-> If the wavelength is fully out of the effective range.
    if np.sum(fltr) == 0:
        if not QuietMode:
            print("Warning (c2000): The input wavelength is out of the effective range!")
        return np.ones_like(wave)
    #-> Calculate the extinction within the effective regime.
    wave_aa = wave[fltr] * 1e4
    av = calzetti00(wave_aa, Av, Rv)
    f0 = np.ones_like(wave)
    f0[fltr] = 10**(-0.4 * av)
    return f0

if __name__ == "__main__":
    import extinction
    import matplotlib.pyplot as plt

    Rv = 3.1
    wave = np.logspace(np.log10(910.), np.log10(30000.), 2000)
    a_lambda = {
        'ccm89': extinction.ccm89(wave, 1.0, Rv),
        'odonnell94': extinction.odonnell94(wave, 1.0, Rv),
        'fitzpatrick99': extinction.fitzpatrick99(wave, 1.0),
        "c00": extinction.calzetti00(wave, 1.0, Rv),
        'fm07': extinction.fm07(wave, 1.0)
    }
    for mn in a_lambda.keys():
        ext = a_lambda[mn]
        plt.plot(wave, ext, label=mn)
    c00_2 = -2.5 * np.log10(Calzetti00(2., wave/1e4, Rv))
    c00_3 = extinction.calzetti00(wave, 2.0, Rv)
    plt.plot(wave, c00_2, color="turquoise")
    plt.plot(wave, c00_3, color="tomato", linestyle=":")
    plt.xscale("log")
    plt.legend(loc="upper right")
    plt.show()
