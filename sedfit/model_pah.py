import numpy as np
import matplotlib.pyplot as plt

tb = np.genfromtxt("/Users/jinyi/Work/mcmc/Fitter/template/PAH.template.dat")
wave = tb[:, 0]
flux = tb[:, 1]
norm = np.trapz(flux, wave)
flux_norm = flux / norm
plt.plot(wave, flux_norm)
plt.xscale("log")
plt.yscale("log")
plt.show()
print np.trapz(flux_norm, wave)
"""
"""
