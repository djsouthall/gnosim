"""
Properties of the Greenlandic ice.
"""

import numpy
import pylab
import scipy.interpolate

import gnosim.utils.constants

pylab.ion()

############################################################

# From
# Density (Mg m^-3)
# Elevation, negative is down (m)
profile_deep = numpy.array([#[0.87, -63.09751434034433],
                            #[0.8798869143780291, -86.0420650095607],
                            #[0.8898707592891761, -103.2504780114723],
                            [0.8998546042003231, -114.72275334608048],
                            #[0.9101292407108239, -137.6673040152964],
                            [0.911583198707593, -137.6673040152964],
                            [0.9132310177705978, -143.40344168260026],
                            [0.9142003231017771, -149.13957934990412],
                            [0.9149757673667206, -160.6118546845123],
                            [0.9156542810985461, -177.8202676864248],
                            [0.9163327948303716, -195.0286806883364],
                            [0.9169143780290792, -217.97323135755323],
                            [0.9173990306946689, -246.653919694073],
                            [0.9178836833602585, -281.0707456978971],
                            [0.9182714054927302, -332.69598470363326],
                            [0.918659127625202, -390.0573613766737],
                            [0.9190468497576737, -476.0994263862335],
                            [0.9193376413570276, -573.613766730402],
                            [0.9197253634894992, -757.1701720841302],
                            [0.9201130856219709, -952.198852772467],
                            [0.9203069466882068, -1101.3384321223714],
                            [0.9205008077544427, -1319.311663479924],
                            [0.9205977382875606, -1462.715105162524],
                            [0.9206946688206785, -1806.8833652007647],
                            [0.9206946688206785, -2042.0650095602296],
                            [0.9205977382875606, -2214.1491395793496],
                            [0.9201130856219709, -2334.6080305927344],
                            [0.9196284329563813, -2466.5391969407265],
                            [0.9188529886914378, -2621.414913957935],
                            [0.9175928917609047, -2845.1242829827916],
                            [0.9167205169628433, -3000], 
                            [0.9167205169628433, -1.e10]]) # Filler


elevation_deep = profile_deep.transpose()[1]
density_deep = profile_deep.transpose()[0] * 1.e3 / gnosim.utils.constants.mass_proton # nucleons m^-3

# From 
# Depth, positive is down (m)
# Density (kg m^-3)
profile_firn = numpy.array([[-1.e10, 0.], # Filler
                            [0, 256.9169960474308],
                            [0.46728971962616633, 270.75098814229256],
                            [0.778816199376947, 292.49011857707507],
                            [1.5576323987538898, 322.1343873517786],
                            [2.4922118380062313, 341.89723320158106],
                            [3.4267912772585656, 371.5415019762846],
                            [4.8286604361370715, 401.1857707509881],
                            [6.230529595015572, 428.85375494071144],
                            [7.788161993769471, 458.498023715415],
                            [9.501557632398756, 494.0711462450593],
                            [11.682242990654204, 521.7391304347826],
                            [14.01869158878504, 541.501976284585],
                            [17.289719626168225, 565.2173913043478],
                            [21.028037383177566, 584.9802371541502],
                            [23.98753894080996, 596.8379446640316],
                            [27.258566978193137, 616.6007905138341],
                            [31.152647975077876, 644.2687747035573],
                            [37.22741433021806, 662.0553359683794],
                            [42.83489096573209, 691.699604743083],
                            [50.155763239875384, 719.3675889328063],
                            [56.54205607476635, 745.0592885375495],
                            [65.42056074766354, 776.6798418972331],
                            [73.05295950155762, 796.4426877470355],
                            [79.12772585669781, 820.1581027667984],
                            [87.85046728971962, 847.8260869565217],
                            [95.4828660436137, 861.6600790513833]])
                            #[100, 867.5889328063241]])

elevation_firn = -1. * profile_firn.transpose()[0]
density_firn = profile_firn.transpose()[1] / gnosim.utils.constants.mass_proton # nucleons m^-3

f_density = scipy.interpolate.interp1d(numpy.concatenate([elevation_deep[::-1], elevation_firn[::-1]]), 
                                       numpy.concatenate([density_deep[::-1], density_firn[::-1]]), 
                                       kind='linear', bounds_error=False, fill_value=0.)

############################################################

def density(z, mode='parametric'):
    """
    z = elevation (m)
    
    Returns:
    density (nuclei m^-3)
    
    Source: http://www.iceandclimate.nbi.ku.dk/research/flowofice/densification/
    """
    if mode == 'parametric':
        density_surface = 50. # km m^-3
        density_deep = 917. # kg m^-3
        scale_depth = 30. # m
    
        # Require that neutrino interacts in upper 3 km of ice
        density = (z <= 0.) * (z >= -3000.) \
                  * (density_deep - (density_deep - density_surface) * numpy.exp(z / scale_depth)) # kg m^-3
    
        return density / gnosim.utils.constants.mass_proton # convert from kg m^-3 to nucleons m^-3
    elif mode == 'empirical':
        return f_density(z) # nucleons m^-3
    else:
        print 'WARNING'
        return -999

############################################################

def indexOfRefraction(z, mode='parametric'):
    """
    z = elevation (m)

    Returns:
    index of refraction

    Source: Pawlowicz 1972
    """
    if mode == 'parametric':
        n_infinity = 1.831 # Index of refraction deep in the ice
        n_0 = 1.29 # Index of refraction at surface
        n_air = 1.000293
        a = 10 # m
        return (z <= 0.) * (n_0 - (n_infinity * z / a)) / (1. - (z / a)) + (z > 0.) * n_air
    elif mode == 'empirical':
        k = 0.86 * 1.e-3 # kg^-1 m^3                                                                                                                        
        return 1. + (k * f_density(z) * gnosim.utils.constants.mass_proton)
    else:
        print 'WARNING'
        return -999

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
    pylab.scatter(elevation_firn[1:-1], density_firn[1:-1] * gnosim.utils.constants.mass_proton, c='red')
    pylab.scatter(elevation_deep[1:-1], density_deep[1:-1] * gnosim.utils.constants.mass_proton, c='black')
    pylab.plot(z, f_density(z) * gnosim.utils.constants.mass_proton, c='green')

    pylab.figure()
    pylab.plot(z, indexOfRefraction(z))
    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Index of Refraction')

    k = 0.86 * 1.e-3 # kg^-1 m^3
    pylab.plot(z, 1. + (k * f_density(z) * gnosim.utils.constants.mass_proton), c='green')

    """
    pylab.figure()
    frequency = 0.3 # GHz
    pylab.plot(z, attenuationLength(z, frequency=frequency))
    pylab.title('Frequency = %i MHz'%(frequency * gnosim.utils.constants.MHz_to_GHz))
    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Electric Field Attenuation Length (m)')
    """

############################################################
