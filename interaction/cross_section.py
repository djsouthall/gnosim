"""
Nucleon interaction cross sections for high-energy neutrinos from 10^4 GeV to 10^12 GeV. 
Both charged current (CC) and neutral current (NC) cross sections for neutrinos and anti-neutrinos are implemented.

Source: Connolly et al. 2011, arXiv:1102.0691
"""

import numpy
import pylab

import gnosim.utils.constants

pylab.ion()

############################################################

mode_dict = {'nc': [-1.826, -17.31, -6.448, 1.431, -18.61],
             'cc': [-1.826, -17.31, -6.406, 1.431, -17.91],
             'nc_anti': [-1.033, -15.95, -7.296, 1.569, -18.30],
             'cc_anti': [-1.033, -15.95, -7.247, 1.569, -17.72]}

############################################################

def crossSection(energy_neutrino, mode):
    """
    energy_neutrino = neutrino energy (GeV)
    mode = specifies the neutrino type (nu or anti-nu) and interaction type (charged current or neutral current)
    
    Returns:
    interaction cross section (m^2)
    """
    epsilon = numpy.log10(energy_neutrino) # Neutrino energy in GeV
    c_0, c_1, c_2, c_3, c_4 = mode_dict[mode]
    log_cross_section = c_1 \
                        + c_2 * numpy.log(epsilon - c_0) \
                        + c_3 * numpy.log(epsilon - c_0)**2 \
                        + c_4 / numpy.log(epsilon - c_0)
    return 10**log_cross_section * gnosim.utils.constants.cm_to_m**2 # Convert from cm^2 to m^2

############################################################

def totalCrossSection(energy_neutrino, anti=False):
    """
    energy_neutrino = neutrino energy (GeV)
    anti = True -> anti-neutrinos
    
    Returns:
    total (charged current + neutral current) interaction cross section (m^2)
    """
    if anti:
        return crossSection(energy_neutrino, 'cc_anti') + crossSection(energy_neutrino, 'nc_anti')
    else:
        return crossSection(energy_neutrino, 'cc') + crossSection(energy_neutrino, 'nc')

############################################################

if __name__ == "__main__":
    energy_neutrino = 10**numpy.linspace(4, 12, 100)
    cross_section_cc = crossSection(energy_neutrino, 'cc') * gnosim.utils.constants.cm_to_m**-2 # Convert from m^2 to cm^2
    cross_section_nc = crossSection(energy_neutrino, 'nc') * gnosim.utils.constants.cm_to_m**-2 # Convert from m^2 to cm^2
    cross_section_cc_anti = crossSection(energy_neutrino, 'cc_anti') * gnosim.utils.constants.cm_to_m**-2 # Convert from m^2 to cm^2
    cross_section_nc_anti = crossSection(energy_neutrino, 'nc_anti') * gnosim.utils.constants.cm_to_m**-2 # Convert from m^2 to cm^2

    pylab.figure()
    pylab.plot(energy_neutrino, cross_section_cc, label=r'$\nu$ CC', color='red', linestyle='-')
    pylab.plot(energy_neutrino, cross_section_nc, label=r'$\nu$ NC', color='red', linestyle='--')
    pylab.plot(energy_neutrino, cross_section_cc_anti, label=r'$\bar{\nu}$ CC', color='blue', linestyle='-')
    pylab.plot(energy_neutrino, cross_section_nc_anti, label=r'$\bar{\nu}$ NC', color='blue', linestyle='--')
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.legend(loc='lower right')
    pylab.title('Neutrino-Nucleon Cross Sections')
    pylab.xlabel('Energy Neutrino (GeV)')
    pylab.ylabel(r'Cross Section (cm$^{2}$)')

############################################################
