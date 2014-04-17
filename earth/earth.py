"""
Attenuation of neutrinos passing through the Earth.
"""

import numpy
import pylab

import gnosim.earth.prem
import gnosim.interaction.cross_section
import gnosim.utils.constants

pylab.ion()

############################################################

def interactionLength(density_input, energy_neutrino_array, plot=True):
    """
    Tool for quickly plotting interaction length as a function of neutrino energy for a fixed density medium.

    density_input = density (kg m^-3)
    energy_neutrino_array = energy of neutrino (GeV)
    """
    #density = numpy.array(density_input) \
    #          * gnosim.utils.constants.cm_to_m**3 / gnosim.utils.constants.mass_proton # convert from kg m^-3 to nucleons cm^-3
    density = numpy.array(density_input) / gnosim.utils.constants.mass_proton

    interaction_length_array = numpy.zeros(len(energy_neutrino_array))
    interaction_length_anti_array = numpy.zeros(len(energy_neutrino_array))
    for ii, energy_neutrino in enumerate(energy_neutrino_array):
        total_cross_section_anti = gnosim.interaction.cross_section.totalCrossSection(energy_neutrino, anti=True) # m^2
        total_cross_section = gnosim.interaction.cross_section.totalCrossSection(energy_neutrino, anti=False) # m^2
        interaction_length_anti_array[ii] = (density * total_cross_section_anti)**-1 # m
        interaction_length_array[ii] = (density * total_cross_section)**-1 # m

    if plot:
        pylab.figure()
        pylab.plot(energy_neutrino_array, interaction_length_array, color='red', label=r'$\nu$')
        pylab.plot(energy_neutrino_array, interaction_length_anti_array, color='blue', label=r'$\bar{\nu}$')
        pylab.xscale('log')
        pylab.yscale('log')
        pylab.title(r'Density = %.1f kg m$^{-3}$'%(density_input))
        pylab.xlabel('Energy Neutrino (GeV)')
        pylab.ylabel('Interaction Length (m)')
        pylab.legend(loc='upper right')

    return interaction_length_array, interaction_length_anti_array

############################################################

def probSurvival(energy_neutrino, theta, elevation=0., anti=False, plot=False):
    """
    energy_neutrino = neutrino energy (GeV)
    theta = zenith angle (deg)

    Returns:
    Survival probability of neutrino passing through the Earth
    """

    # Original
    #if theta <= 90.:
    #    return 1.

    total_cross_section = gnosim.interaction.cross_section.totalCrossSection(energy_neutrino, anti=anti) # m^2

    # Steps going out from the detector chord through the Earth
    # Line below was original
    #distance_chord = 2. * gnosim.utils.constants.radius_earth * numpy.cos(numpy.radians(180. - theta)) # m
    # Line below is improved version which takes into account the depth of the interaction
    distance_chord = chordLength(theta, elevation) # m
    if distance_chord <= 1.:
        return 1.
    d = numpy.linspace(0., distance_chord, 1000) # m
    delta_d = d[1] - d[0] # m

    # Neutrino trajectory in Earth coordinates where z_axis goes through the detector located at (0, 0, EARTH_RADIUS)
    phi = 0.
    x_earth = d * numpy.sin(numpy.radians(theta)) * numpy.cos(numpy.radians(phi))
    y_earth = d * numpy.sin(numpy.radians(theta)) * numpy.sin(numpy.radians(phi))
    z_earth = gnosim.utils.constants.radius_earth + (d * numpy.cos(numpy.radians(theta)))

    r_earth = numpy.sqrt(x_earth**2 + y_earth**2 + z_earth**2) # m

    f_density = gnosim.earth.prem.prem()
    density = f_density(r_earth) # nucleons m^-3

    # OLD
    #interaction_length = (density * total_cross_section)**-1 # m
    #transmission = numpy.exp(-1. * delta_d / interaction_length)
    # OLD

    # ORIGINAL
    #transmission = numpy.exp(-1. * delta_d * density * total_cross_section)
    #survival = numpy.cumprod(transmission[1:])
    # ORIGINAL

    # More numerically stable
    optical_depth = numpy.cumsum(delta_d * density * total_cross_section)
    survival = numpy.exp(-1. * optical_depth[1:])

    if plot:
        cut = density > 0.
        pylab.figure()
        pylab.scatter(x_earth[cut], z_earth[cut], c=survival[cut], edgecolors='none')
        pylab.colorbar()

    return survival[-1]

############################################################

def chordLength(theta, elevation=0.):
    """
    theta = zenith angle (deg)
    elevation = elevation from surface, negative is below surface (m)

    Return chord length through the Earth (m)
    """
    depth = -1. * elevation
    if elevation > 0.:
        return 0.
    a = 1.
    b = 2. * (gnosim.utils.constants.radius_earth - depth) * numpy.cos(numpy.radians(theta))
    c = depth**2 - (2. * gnosim.utils.constants.radius_earth * depth)
    return (numpy.sqrt(b**2 - (4. * a * c)) - b) / (2. * a)

############################################################

def probInteract(energy_neutrino, elevation, anti=False):
    """
    energy_neutrino = neutrino energy (GeV)
    elevation = elevation of neutrino interaction (m)
    anti = True -> anti-neutrino

    Returns:
    probability to interact in a sphere containing a cubic meter of ice
    """
    volume_sphere = 1. # m^3
    radius_sphere = (3. * volume_sphere / (4. * numpy.pi))**(1. / 3.)
    characteristic_length = (4. / 3.) * radius_sphere

    total_cross_section = gnosim.interaction.cross_section.totalCrossSection(energy_neutrino, anti=anti) # m^2
    density = gnosim.earth.greenland.density(elevation) # nucleons m^-3
    return 1. - numpy.exp(-1. * characteristic_length * density * total_cross_section)

############################################################

if __name__ == "__main__":
    energy_neutrino_array = 10**numpy.arange(4., 12. + 1.e-10, 1.) # GeV
    #theta_array = numpy.linspace(0, 180 + 1.e-10, 1000)
    cos_theta_array = numpy.linspace(1., -1., 1000)
    theta_array = numpy.degrees(numpy.arccos(cos_theta_array))

    pylab.figure()
    survival_array = []
    for energy_neutrino in energy_neutrino_array:
        print 'Energy neutrino = %e'%(energy_neutrino)
        survival_array.append([])
        for theta in theta_array:
            survival_array[-1].append(probSurvival(energy_neutrino, theta))
        pylab.plot(cos_theta_array, survival_array[-1])
    pylab.title('Neutrinos')
    pylab.xlabel('Cos(Zenith Angle)')
    pylab.ylabel('Survival Probability')

    pylab.figure()
    survival_anti_array = []
    for energy_neutrino in energy_neutrino_array:
        print 'Energy neutrino = %e'%(energy_neutrino)
        survival_anti_array.append([])
        for theta in theta_array:
            survival_anti_array[-1].append(probSurvival(energy_neutrino, theta, anti=True))
        pylab.plot(cos_theta_array, survival_anti_array[-1])
    pylab.title('Anti-Neutrinos')
    pylab.xlabel('Cos(Zenith Angle)')
    pylab.ylabel('Survival Probability')
    
############################################################
