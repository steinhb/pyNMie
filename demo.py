import numpy as np
import NMie as nmie
import matplotlib.pyplot as plt

# Define the core-shell particle
#
# 20nm Ag Core with 10nm Au shell immersed in water
nums = 101
wl_sh = 500
wl_lo = 600
wavelength = np.linspace(wl_sh,wl_lo,nums) # Define Wavelength range

dAg = 20 # Diameter of Ag core
nAg = 0.051585+1j*3.9046 # @587.6nm
dAu = 10 # Thickness of Au shell
nAu = 0.27732+1j*2.9278 # @587.6nm
nH2O = 1.3325 # @587.6nm

 # Refractive indices must have same shape as wavelength
nAg*=np.ones(np.shape(wavelength))
nAu*=np.ones(np.shape(wavelength))
nH2O*=np.ones(np.shape(wavelength))

# Pack the diameters and refractive indices
# diameter list has 1(core) + NUMBER_OF_SHELLS elements
# d = [d_core, d_shell1, d_shell2, ..., d_shellN]
dias = [dAg,dAu]
# Refractive index tuple 1(core) + NUMBER_OF_SHELLS + 1(ambient) elements
# n = [n_core, n_shell1, n_shell2, ..., n_shellN, n_ambient]
n = [nAg, nAu, nH2O]

particle = nmie.NMie(diameters = dias, refind = n, wl = wavelength)

fig,axs = plt.subplots(nrows = 1, ncols = 3)
ax = axs[0]
ax.plot(wavelength, particle.qsca, color = 'C0', label = 'Scattering Cross Section')
ax.plot(wavelength, particle.qsca_el[:,0], color = 'C0', label = 'Electric Dipole')
ax.plot(wavelength, particle.qsca_ma[:,0], color = 'C0', label = 'Magnetic Dipole')
