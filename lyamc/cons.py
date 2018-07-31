"""

This list of constans is copied from Teja's package.

"""

import astropy.constants as const
import astropy.units as u
import numpy as np

NU21 = 1420.40575177e+6  # Frequency of the 1s hyperfine transition (Hz)
# Wavelength of Lyman-alpha in cm, exact value is not super important since
# we work with offsets
LLYA = 1.215668e-5
NULYA = const.c.to('cm/s').value / LLYA  # Lyman alpha frequency (Hz)
ALYA = 6.2648e+8            # Einstein A coefficient of Lyman-alpha lines (Hz)
GLYA = ALYA/(4.0 * np.pi)   # HWHM for Lyman lines, in Hz
# Mass of H-atom in Kelvin
MHK = (const.m_p * const.c**2 / const.k_B).to('K').value
# Mass of H-atom in GeV
MHGEV = (const.m_p * const.c**2).to('GeV').value
# Mass of electron in GeV
MEGEV = (const.m_e * const.c**2).to('GeV').value
# Reduced mass of electron in GeV
MUEGEV = MEGEV * MHGEV / (MEGEV + MHGEV)
# Speed of light in cm/s
CCMS = const.c.to('cm/s').value
# Thomson cross-section in cm^2
SIGMAT = const.sigma_T.to('cm2').value
# Radiation constant in erg/cm^3/K^4
ARAD = (4.0 * const.sigma_sb / const.c).to('erg/cm3/K4').value
NURYD = (const.Ryd*const.c).to('Hz').value  # Lyman limit (Hz)
# Planck constant in erg-s
PLH = (const.h).to('erg s').value
# Fine structure constant
ALPHA = const.alpha.value

# Megaparsec in cm
MPCCM = u.Mpc.to('cm')
# Solar mass in units of proton mass
MSUNMP = u.M_sun.to('M_p')

# Conversion factor from Kelvin to Hz = 20.84E+9
K2HZ = (const.k_B * u.Kelvin / const.h).to('Hz').value
# Conversion factor from g to GeV
G2GEV = (u.g * const.c**2).to('GeV').value
# Conversion factor from GeV to K
GEV2K = MHK / MHGEV
# Conversion factor from GeV to Hz
GEV2HZ = GEV2K * K2HZ
# Conversion factor from erg to Kelvin
ERG2K = (u.erg/const.k_B).to('K').value
# Conversion factor from MeV to erg
MEV2ERG = (u.MeV).to('erg')

# Hyperfine transition energy in Kelvin = 68.16 mK
T21 = NU21 / K2HZ
# Lyman-alpha energy in Kelvin = 1.18E+5 K
TLYA = NULYA / K2HZ
# 1 Rydberg in Kelvin
TRYD = NURYD / K2HZ
# Lyman-alpha energy as a fraction of hydrogen mass
FLYA = TLYA / MHK
