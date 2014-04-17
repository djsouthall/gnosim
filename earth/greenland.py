"""
Properties of the Greenlandic ice.
"""

import numpy
import pylab

import gnosim.utils.constants

pylab.ion()

############################################################

def density(z):
    """
    z = elevation (m)
    
    Returns:
    density (nuclei m^-3)
    
    Source: http://www.iceandclimate.nbi.ku.dk/research/flowofice/densification/
    """
    density_surface = 50. # km m^-3
    density_deep = 917. # kg m^-3
    scale_depth = 30. # m
    
    # Require that neutrino interacts in upper 3 km of ice
    density = (z <= 0.) * (z >= -3000.) \
              * (density_deep - (density_deep - density_surface) * numpy.exp(z / scale_depth)) # kg m^-3
    
    return density / gnosim.utils.constants.mass_proton # convert from kg m^-3 to nucleons m^-3

############################################################

def indexOfRefraction(z):
    """
    z = elevation (m)

    Returns:
    index of refraction

    Source: Pawlowicz 1972
    """
    n_infinity = 1.831 # Index of refraction deep in the ice
    n_0 = 1.29 # Index of refraction at surface
    n_air = 1.000293
    a = 10 # m
    return (z <= 0.) * (n_0 - (n_infinity * z / a)) / (1. - (z / a)) + (z > 0.) * n_air

############################################################

def attenuationLength(z, frequency):
    """
    z = elevation (m)
    frequency = radio frequency (GHz)

    Returns:
    attenuation length, i.e., distance at which electric field is reduced by e (m)

    Source: ARA insturment paper Allison et al. 2011, arXiv:1105.2854
    """
    # Treat as constant with respect to depth and frequency until add real data points
    frequency_0 = 0.3 # GHz
    if numpy.isscalar(z):
        return (z <= 0.) * 820. + (z > 0.) * 1.e20 # m
    else:
        return (z <= 0.) * 820. * numpy.ones(len(z)) + (z > 0.) * 1.e20 * numpy.ones(len(z)) # m

############################################################

if __name__ == "__main__":
    z = numpy.linspace(-3000., 0., 1000) # Array of elevations (m)

    pylab.figure()
    pylab.plot(z, density(z) * gnosim.utils.constants.mass_proton) # Convert from nucleons m^-3 to kg m^-3
    pylab.xlabel('Elevation (m)')
    pylab.ylabel(r'Density (kg m$^{-3}$)')

    pylab.figure()
    pylab.plot(z, indexOfRefraction(z))
    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Index of Refraction')

    pylab.figure()
    frequency = 0.3 # GHz
    pylab.plot(z, attenuationLength(z, frequency=frequency))
    pylab.title('Frequency = %i MHz'%(frequency * gnosim.utils.constants.MHz_to_GHz))
    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Electric Field Attenuation Length (m)')

############################################################
