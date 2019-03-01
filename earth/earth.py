"""
Attenuation of neutrinos passing through the Earth.
Geometry of the Earth.
"""

import numpy
import pylab

import gnosim.earth.prem
import gnosim.interaction.cross_section
import gnosim.utils.constants
import gnosim.earth.ice

pylab.ion()

############################################################

def interactionLength(density_input, energy_neutrino_array, plot=False):
    """
    Tool for quickly plotting interaction length as a function of neutrino energy for a fixed density medium.

    density_input = density (kg m^-3)
    energy_neutrino_array = energy of neutrino (GeV)
    """
    
    if numpy.isscalar(energy_neutrino_array):
        energy_neutrino_array = numpy.array([energy_neutrino_array])

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

    if len(energy_neutrino_array) == 1:
        return interaction_length_array[0], interaction_length_anti_array[0] # m, m
    else:
        return interaction_length_array, interaction_length_anti_array # m, m

############################################################

def probSurvival(energy_neutrino, theta, ice, elevation=0., anti=False, plot=False):
    """
    energy_neutrino = neutrino energy (GeV)
    theta = zenith angle (deg)
    ice should be an Ice object from the gnosim.earth.ice file.
    Returns:
    Survival probability of neutrino passing through the Earth
    """

    # Original
    #if theta <= 90.:
    #    return 1.

    total_cross_section = gnosim.interaction.cross_section.totalCrossSection(energy_neutrino, anti=anti) # m^2 #No randomness

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

    f_density = gnosim.earth.prem.prem(ice) #no randomness
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

def horizon(elevation):
    """
    elevation = elevation from surface, negative is below surface (m)
    
    Return distance to horizon (m), angle below horizontal (deg), 
    and distance along curved Earth surface to horizon intercept (m)
    """
    # Distance from antenna to the horizon taking into account Earth's curvature
    distance = numpy.sqrt((2. * gnosim.utils.constants.radius_earth * elevation) + elevation**2) # m
    
    # Angle below horizontal where the horizon is located
    angle = numpy.degrees(numpy.arccos(gnosim.utils.constants.radius_earth / (gnosim.utils.constants.radius_earth + elevation))) # deg
    
    # Distance along surface of the Earth to the horizon
    theta = 90. + (angle + 1.e-10)
    x_curve = curvature(elevation, theta)[2] # m

    return distance, angle, x_curve

############################################################

def curvature(elevation, theta):
    """
    elevation = elevation from surface, negative is below surface (m)
    theta = zenith angle of out-going ray from antenna

    Return distance from antenna to surface (m), angle by which the angle of incidence is increased (deg),
    and distance along curve of Earth's surface to the intercept (m)
    """
    # Distance from the antenna to the surface, taking into account the Earth's curvature
    a = 1.
    b = 2. * (gnosim.utils.constants.radius_earth + elevation) * numpy.cos(numpy.radians(180. - theta))
    c = (gnosim.utils.constants.radius_earth + elevation)**2 - gnosim.utils.constants.radius_earth**2
    d = -1. * (numpy.sqrt(b**2 - (4. * a * c)) - b) / (2. * a) # m
    
    # Angle by which the angle of incidence is increased
    aa = d
    bb = gnosim.utils.constants.radius_earth
    cc = gnosim.utils.constants.radius_earth + elevation
    angle = numpy.degrees(numpy.arccos((bb**2 + cc**2 - aa**2) / (2. * bb * cc))) # deg
    
    # Distance along Earth's curved surface to the intercept 
    x_curve = gnosim.utils.constants.radius_earth \
              * numpy.arcsin(d * numpy.sin(numpy.radians(180. - theta)) / gnosim.utils.constants.radius_earth) # m

    return d, angle, x_curve

############################################################

def curvatureToTheta(elevation, x_curve):
    """
    elevation = elevation from surface, negative is below surface (m)
    x_curve = distance along Earth's curved surface (m)

    Returns zenith angle to intercept (deg)
    """
    alpha_radians = x_curve / gnosim.utils.constants.radius_earth # radians
    theta = 180. - numpy.degrees(numpy.arctan2(gnosim.utils.constants.radius_earth * numpy.sin(alpha_radians),
                                               elevation + gnosim.utils.constants.radius_earth * (1. - numpy.cos(alpha_radians)))) # deg
    return theta

############################################################

def plotCurvature(elevation, n=10000):
    """
    elevation = elevation from surface, negative is below surface (m)
    """
    theta_array = numpy.linspace(180. - 1.e-10, 90. + horizon(elevation)[1], n)
    
    x_flat = numpy.zeros(n)
    x_curve = numpy.zeros(n)
    distance_flat = numpy.zeros(n)
    distance_curve = numpy.zeros(n)
    
    for ii in range(0, n):
        x_flat[ii] = -1. * elevation * numpy.tan(numpy.radians(theta_array[ii]))
        x_curve[ii] = curvature(elevation, theta_array[ii])[2]
        distance_flat[ii] = numpy.sqrt(elevation**2 + x_flat[ii]**2)
        distance_curve[ii] = curvature(elevation, theta_array[ii])[0]
    
    """
    pylab.figure()
    #pylab.scatter(distance_flat, distance_curve / distance_flat, c=theta_array, edgecolors='none')
    pylab.plot(distance_flat, distance_curve / distance_flat)
    #pylab.scatter(x_flat, x_curve / x_flat, c=theta_array, edgecolors='none')
    pylab.plot(x_flat, x_curve / x_flat) 
    pylab.title('Curvature of the Earth')
    pylab.xlabel('x_flat')
    pylab.ylabel('x_curve / x_flat')
    #pylab.colorbar()
    """

    pylab.figure()
    theta_circle_radians = numpy.linspace(0, 2. * numpy.pi, 100000)
    r_circle = gnosim.utils.constants.radius_earth * numpy.cos(theta_circle_radians)
    z_circle = gnosim.utils.constants.radius_earth * (numpy.sin(theta_circle_radians) - 1.)
    pylab.plot(r_circle, z_circle, c='black')
    r = distance_curve * numpy.sin(numpy.radians(theta_array))
    z = elevation + (distance_curve * numpy.cos(numpy.radians(theta_array)))
    pylab.scatter(r, z, c=x_curve, edgecolors='none')
    pylab.colorbar()
    pylab.xlim([0., gnosim.utils.constants.radius_earth])
    pylab.ylim([-1. * gnosim.utils.constants.radius_earth, 0.])

############################################################

def plotSampling(detector_volume_radius, n_events=1000000, detector_volume_depth=3000.):
    """
    Sanity check function to compare radial event sampling and geometric acceptance factors between
    flat Earth and curved Earth cases.

    detector volume radius (m)
    detector volume depth (m)
    """
    
    r_flat = numpy.random.triangular(0., detector_volume_radius, detector_volume_radius, size=n_events) # m

    alpha_max_radians = detector_volume_radius / gnosim.utils.constants.radius_earth # radians                                                           
    alpha = numpy.arccos(numpy.random.uniform(1., numpy.cos(alpha_max_radians), size=n_events)) # radians                                                
    r_curve = gnosim.utils.constants.radius_earth * alpha # m

    pylab.figure()
    bins = numpy.linspace(0, detector_volume_radius, 41)
    pylab.hist(r_flat, bins=bins, alpha=0.5, label='Flat')
    pylab.hist(r_curve, bins=bins, alpha=0.5, label='Curve')
    pylab.legend(loc='upper left')
    pylab.xlabel('Distance Along Earth Surface (m)')
    pylab.ylabel('Number of Events')

    detector_volume_radius = numpy.linspace(0, detector_volume_radius, 10000)
    geometric_factor_flat = (4. * numpy.pi) * (numpy.pi * detector_volume_radius**2 * detector_volume_depth) # m^3 sr
    geometric_factor_curve = (4. * numpy.pi) \
                             * (2. * numpy.pi * gnosim.utils.constants.radius_earth**2 \
                                * (1. - numpy.cos(detector_volume_radius / gnosim.utils.constants.radius_earth))\
                                * detector_volume_depth) # m^3 sr
    
    pylab.figure()
    pylab.plot(detector_volume_radius, geometric_factor_flat, label='Flat')
    pylab.plot(detector_volume_radius, geometric_factor_curve, label='Curve')
    pylab.legend(loc='upper left')
    pylab.xlabel('Distance Along Earth Surface (m)')
    pylab.ylabel(r'Geometric Factor (m$^{3}$ sr)')

############################################################

def probInteract(energy_neutrino, density, anti=False):
    """
    energy_neutrino = neutrino energy (GeV)
    density = density at elevation of neutrino interaction in nucleons m^-3
    anti = True -> anti-neutrino

    Returns: 
    probability to interact in a sphere containing a cubic meter of ice
    """
    volume_sphere = 1. # m^3 
    radius_sphere = (3. * volume_sphere / (4. * numpy.pi))**(1. / 3.)
    characteristic_length = (4. / 3.) * radius_sphere

    total_cross_section = gnosim.interaction.cross_section.totalCrossSection(energy_neutrino, anti=anti) # m^2
    #Look into python 1- exponential function, because this has large error 
    return 1. - numpy.exp(-1. * characteristic_length * density * total_cross_section) 

############################################################

if __name__ == "__main__":
    energy_neutrino_array = 10**numpy.arange(4., 12. + 1.e-10, 1.) # GeV
    #theta_array = numpy.linspace(0, 180 + 1.e-10, 1000)
    cos_theta_array = numpy.linspace(1., -1., 1000)
    theta_array = numpy.degrees(numpy.arccos(cos_theta_array))
    ice_model = 'antarctica'
    ice = gnosim.earth.ice.Ice(ice_model)
    pylab.figure()
    survival_array = []
    for energy_neutrino in energy_neutrino_array:
        print ('Energy neutrino = %e'%(energy_neutrino))
        survival_array.append([])
        for theta in theta_array:
            survival_array[-1].append(probSurvival(energy_neutrino, theta,ice))
        pylab.plot(cos_theta_array, survival_array[-1], c='blue')
    pylab.title('Neutrinos')
    pylab.xlabel('Cos(Zenith Angle)')
    pylab.ylabel('Survival Probability')

    pylab.figure()
    survival_anti_array = []
    for energy_neutrino in energy_neutrino_array:
        print ('Energy neutrino = %e'%(energy_neutrino))
        survival_anti_array.append([])
        for theta in theta_array:
            survival_anti_array[-1].append(probSurvival(energy_neutrino, theta, ice, anti=True))
        pylab.plot(cos_theta_array, survival_anti_array[-1], c='blue')
    pylab.title('Anti-Neutrinos')
    pylab.xlabel('Cos(Zenith Angle)')
    pylab.ylabel('Survival Probability')
    
############################################################
