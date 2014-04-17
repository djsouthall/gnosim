"""
Askaryan Emission

Source: Lehtinen et al. 2003, arXiv:0309656
"""

import numpy
import pylab

import gnosim.utils.constants
import gnosim.interaction.inelasticity

pylab.ion()

############################################################

def electricField(frequency, d, angle, energy_neutrino, mode, index_of_refraction):
    """
    frequency = frequency (GHz)
    d = observation distance (m)
    angle = observation angle relative to shower axis (deg)
    energy neutrino = neutrino energy (GeV)
    Return electric field (V m^-1 GHz^-1)
    """
    omega = 2. * numpy.pi * frequency
    mu = 1.
    length = 1.5 # m
    energy_shower = gnosim.interaction.inelasticity.inelasticity(energy_neutrino, mode) * energy_neutrino # GeV
    q = 5.5e-20 * energy_shower # C
    k = index_of_refraction * omega / gnosim.utils.constants.speed_light

    #print 'last'
    #print numpy.exp(-0.5 * (k * length)**2 * (numpy.cos(numpy.radians(angle)) - index_of_refraction**-1)**2)

    # Note that units for time need to be looked at in detail

    return (mu * gnosim.utils.constants.mu_0 * q * length * (gnosim.utils.constants.GHz_to_Hz * omega) / (d * numpy.sqrt(2. * numpy.pi))) \
        * numpy.sin(numpy.radians(angle)) \
        * numpy.exp(-0.5 * (k * length)**2 * (numpy.cos(numpy.radians(angle)) - index_of_refraction**-1)**2) \
        * gnosim.utils.constants.GHz_to_Hz # V m^-1 GHz^-1
    
############################################################

if __name__ == "__main__":

    energy_neutrino = 1.e9 # GeV
    mode = 'cc'
    d = 1000. # m
    angle = numpy.linspace(0., 90., 1000)
    index_of_refraction = 1.8
    #omega = 0.5 # GHz
    
    """
    omega_array = numpy.arange(0.1, 1., 0.05)
    electric_field_array = []
    for omega in omega_array:
        electric_field_array.append(electricField(omega, d, angle, energy_neutrino, mode, index_of_refraction))
    pylab.figure()
    for ii, omega in enumerate(omega_array):
        pylab.plot(angle, electric_field_array[ii])
    """
    #
    """
    angle_array = numpy.arange(45., 65, 2.5)
    omega = numpy.linspace(0.1, 1., 1000)
    electric_field_array = []
    for angle in angle_array:
        electric_field_array.append(electricField(omega, d, angle, energy_neutrino, mode, index_of_refraction))
    pylab.figure()
    for ii, angle in enumerate(angle_array):
        pylab.plot(omega, electric_field_array[ii])
    """
    
    frequency_mesh, angle_mesh = numpy.meshgrid(numpy.linspace(0.01, 1.5, 100), numpy.arange(50., 65. + 1.e-10, 0.1))
    electric_field = electricField(frequency_mesh, d, angle_mesh, energy_neutrino, mode, index_of_refraction)
    pylab.figure()
    #pylab.scatter(frequency_mesh, angle_mesh, c=electric_field, edgecolors='none')
    pylab.pcolormesh(frequency_mesh, angle_mesh, numpy.roll(electric_field, 0, 0))
    colorbar = pylab.colorbar()
    colorbar.set_label(r'V m$^{-1}$ GHz$^{-1}$')
    pylab.title(r'E$_{\nu}$ = %.1e GeV; d = %.1f m'%(energy_neutrino, d))
    pylab.xlabel('Frequency (GHz)')
    pylab.ylabel('Observation Angle (deg)')
    pylab.xlim([numpy.min(frequency_mesh), numpy.max(frequency_mesh)])
    pylab.ylim([numpy.min(angle_mesh), numpy.max(angle_mesh)])

############################################################
