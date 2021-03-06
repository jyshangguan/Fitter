import numpy as np
import cPickle as pickle
from ..fitter.template import Template
from ..dir_list import template_path

Msun = 1.9891e33 #unit: gram
Mpc = 3.08567758e24 #unit: cm
mJy = 1e26 #unit: erg/s/cm^2/Hz
pi  = np.pi

fp = open(template_path+"bc03_sps_cha_kdt.tmplt")
tp_bc03 = pickle.load(fp)
fp.close()
bc03 = Template(**tp_bc03)
waveLim = [1e-2, 1e3]
def BC03_ref(logMs, logAge, sfh, DL, wave, z, frame="rest", t=bc03, waveLim=waveLim):
    """
    This function call the interpolated BC03 template to generate the stellar
    emission SED with the given parameters.

    Parameters
    ----------
    logMs : float
        The log10 of stellar mass with the unit solar mass.
    logAge : float
        The log10 of the age of the stellar population with the unit Gyr.
    sfh : int
        The code for the SFH, it ranges from 0-5 currently for the following
        SFHs:
          0 - bc03_ssp_z_0.02_chab.model,
          1 - bc03_burst_0.1_z_0.02_chab.model,
          2 - bc03_exp_0.1_z_0.02_chab.model,
          3 - bc03_exp_1.0_z_0.02_chab.model,
          4 - bc03_const_1.0_tV_0.2_z_0.02_chab.model,
          5 - bc03_const_1.0_tV_5.0_z_0.02_chab.model.
        It is not suggested to let this parameter free, since the relation between
        the adjascent models are not continuous.
    DL : float
        The luminosity distance with the unit Mpc.
    wave : float array
        The wavelength of the SED.
    z : float
        The redshift.
    frame : string
        "rest" for the rest frame SED and "obs" for the observed frame.
    t : Template class
        The interpolated BC03 template.
    waveLim : list
        The min and max of the wavelength covered by the template.

    Returns
    -------
    fnu : float array
        The flux density of the calculated SED with the unit erg/s/cm^2/Hz.

    Notes
    -----
    None.
    """
    flux = np.zeros_like(wave)
    fltr = (wave > waveLim[0]) & (wave < waveLim[1])
    if np.sum(fltr) == 0:
        return np.zeros_like(wave)
    age = 10**logAge
    flux[fltr] = t(wave[fltr], [age, sfh])
    if frame == "rest":
        idx = 2.0
    elif frame == "obs":
        idx = 1.0
    else:
        raise ValueError("The frame '{0}' is not recognised!".format(frame))
    fnu = (1.0 + z)**idx * flux * 10**logMs / (4 * pi * (DL * Mpc)**2) * mJy
    return fnu

def BC03_ref_PosPar(logMs, logAge, sfh, t=bc03):
    """
    Find the position of the parameters on the discrete grid.

    Parameters
    ----------
    logMs : float
        The log of the stellar mass, unit: Msun.
    age : float
        The age of the stellar population, unit: Gyr.
    sfh : int
        The code for the SFH.

    Returns
    -------
    parDict : dict
        The dict of the parameters.
    """
    age = 10**logAge
    age_d = t.get_nearestParameters([age, sfh])
    parDict = {
        "logMs": logMs,
        "age": age_d
    }
    return parDict
