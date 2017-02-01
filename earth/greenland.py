"""
Properties of the Greenlandic ice.
"""

import numpy
import pylab
import scipy.interpolate

import gnosim.utils.constants

pylab.ion()

############################################################

ice_model_default = 'arthern'

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
                            [-1.e-10, 0.], # Filler
                            [0., 256.9169960474308],
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

# Temperature
"""
z_data, t_data = zip(*[[0., -31.5],                                                                                                                      
                       [16.170300866426658, -31.474526746720628],                                                                                        
                       [217.25899825777185, -31.519352016062875],                                                                                        
                       [308.6673906419683, -31.515445674507994],                                                                                         
                       [500.6543122104426, -31.346984694953782],                                                                                         
                       [605.7739634522686, -31.34249240216567],                                                                                          
                       [1003.3516410540874, -31.592595920216873],                                                                                        
                       [1186.10006484527, -31.958717782448016],                                                                                          
                       [1396.2710063517115, -32.3236677422127],                                                                                          
                       [1496.839769682102, -32.21253232497635],                                                                                          
                       [1597.4085330124926, -32.10139690774001],                                                                                         
                       [1794.0440010312743, -31.505386845004164],                                                                                        
                       [1995.4452057470103, -29.840797049930842],                                                                                        
                       [2288.7137979796407, -25.66159753744227],                                                                                         
                       [2513.3772666546874, -21.752423884934778],                                                                                        
                       [2696.7897685102, -18.486038735282833],                                                                                           
                       [3000., -13.024191973249344]])                                                                                                    
"""

# Johnsen et al. 1995
t_data, z_data = zip(*[[-31.753821080914417, 0.],
                       [-31.753821080914417, -96.02695678985128],
                       [-31.687750517552747, -185.0240056380212],
                       [-31.55983790688455, -261.3222922080781],
                       [-31.43086816720257, -363.0445315597058],
                       [-31.301634145267144, -471.1227591067259],
                       [-31.296877064705104, -585.5305466237946],
                       [-31.354490595956477, -699.9251200281897],
                       [-31.41316125622164, -788.8957406510153],
                       [-31.533938246046777, -884.2091353565606],
                       [-31.714707307404307, -1036.7132097079682],
                       [-31.897326344535962, -1144.7253666916267],
                       [-32.14020173545346, -1303.572215125754],
                       [-32.261243007532045, -1392.5296216359068],
                       [-32.25727877373034, -1487.869444566797],
                       [-32.191736774875565, -1564.1545170241816],
                       [-32.000396423380174, -1665.8899704884818],
                       [-31.68510769501828, -1748.5838875919483],
                       [-31.245077743029555, -1831.3042329207594],
                       [-30.116592520812226, -1971.3738272475002],
                       [-29.239704003876138, -2060.5426595604104],
                       [-28.301237721886974, -2130.6567413998146],
                       [-27.175131039950664, -2213.5224419680217],
                       [-25.049244593225566, -2341.0914857067346],
                       [-23.36153812271506, -2430.4321014843845],
                       [-19.612165793067, -2602.8366295203277],
                       [-15.3638285689116, -2775.3468704576485],
                       [-10.928379509315953, -2947.8967537329863],
                       [-8.741972426551555, -3030.9870942166235]])

#z_data = -1. * numpy.array(z_data)[::-1]                                                                                                                
z_data = numpy.array(z_data)[::-1]
t_data = numpy.array(t_data)[::-1]

f_temperature = scipy.interpolate.interp1d(z_data, t_data, bounds_error=False, fill_value=-50.)

############################################################

def density(z, ice_model=ice_model_default):
    """
    z = elevation (m)
    
    Returns:
    density (nuclei m^-3)
    
    Source: http://www.iceandclimate.nbi.ku.dk/research/flowofice/densification/
    """
    if ice_model == 'parametric':
        density_surface = 50. # kg m^-3
        density_deep = 917. # kg m^-3
        scale_depth = 30. # m
    
        # Require that neutrino interacts in upper 3 km of ice
        density = (z <= 0.) * (z >= -3000.) \
                  * (density_deep - (density_deep - density_surface) * numpy.exp(z / scale_depth)) # kg m^-3
    
        return density / gnosim.utils.constants.mass_proton # convert from kg m^-3 to nucleons m^-3
    elif ice_model == 'empirical':
        return f_density(z) # nucleons m^-3
    elif ice_model == 'ross':
        return (z >= -500.) * (z <= 0.) * f_density(z) # nucleons m^-3
    elif ice_model in ['test_shallow', 'test_mid', 'test_deep']:
        density_surface = 257. # kg m^-3
        density_deep = 920. # kg m^-3
        scale_depth = 40. # m
        # Require that neutrino interacts in upper 3 km of ice
        density = (z >= -3000.) * (z <= 0.) \
                  * (density_deep - (density_deep - density_surface) * numpy.exp(z / scale_depth)) # kg m^-3
        return density / gnosim.utils.constants.mass_proton # convert from kg m^-3 to nucleons m^-3
    elif ice_model == 'arthern':
        density_surface = 280. # kg m^-3
        density_critical = 550. # kg m^-3
        density_deep = 917. # kg m^-3
        
        scale_1 = 27. # m
        scale_2 = 42. # m
        z_critical = -1. * scale_1 * numpy.log((density_deep - density_surface) / (density_deep - density_critical)) # m

        density = (z <= 0.) * (z >= z_critical) \
                  * (density_deep - (density_deep - density_surface) * numpy.exp(z / scale_1)) \
                  + (z < z_critical) * (z >= -3000.) \
                  * (density_deep - (density_deep - density_critical) * numpy.exp((z - z_critical) / scale_2)) # kg m^-3
        return density / gnosim.utils.constants.mass_proton # convert from kg m^-3 to nucleons m^-3
    else:
        print ('WARNING')
        return -999

############################################################

def indexOfRefraction(z, ice_model=ice_model_default):
    """
    z = elevation (m)

    Returns:
    index of refraction

    Source: Pawlowicz 1972
    """
    if numpy.isscalar(z):
        scalar = True
        z = numpy.array([z])
    else:
        scalar = False

    if ice_model == 'parametric':
        n_infinity = 1.831 # Index of refraction deep in the ice
        n_0 = 1.29 # Index of refraction at surface
        n_air = 1.000293
        a = 10 # m

        value = numpy.zeros(z.shape)
        cut_1 = z <= 0.
        cut_2 = z > 0.
        value[cut_1] = (n_0 - (n_infinity * z[cut_1] / a)) / (1. - (z[cut_1] / a))
        value[cut_2] = n_air
    elif ice_model == 'empirical':
        n_air = 1.000293
        k = 0.86 * 1.e-3 # kg^-1 m^3                                                                                                                   
        
        value = numpy.zeros(z.shape)
        cut_1 = z <= 0.
        cut_2 = z > 0.
        value[cut_1] = 1. + (k * f_density(z[cut_1]) * gnosim.utils.constants.mass_proton)
        value[cut_2] = n_air
    elif ice_model == 'ross':
        n_air = 1.000293
        n_water = 1.33
        k = 0.86 * 1.e-3 # kg^-1 m^3
        
        value = numpy.zeros(z.shape)
        cut_1 = z < -500.
        cut_2 = numpy.logical_and(z >= -500., z <= 0.)
        cut_3 = z > 0.
        value[cut_1] = n_water
        value[cut_2] = 1. + (k * f_density(z[cut_2]) * gnosim.utils.constants.mass_proton)
        value[cut_3] = n_air
    elif ice_model == 'test_shallow':
        n_air = 1.000293
        n_water = 1.33
        k = 0.86 * 1.e-3 # kg^-1 m^3                                                                                                                        
        
        value = numpy.zeros(z.shape)
        cut_1 = z < -500.
        cut_2 = numpy.logical_and(z >= -500., z <= 0.)
        cut_3 = z > 0.
        value[cut_1] = n_water
        value[cut_2] = 1. + (k * density(z[cut_2], ice_model=ice_model) * gnosim.utils.constants.mass_proton)
        value[cut_3] = n_air
    elif ice_model == 'test_mid':
        n_air = 1.000293
        n_water = 1.33
        k = 0.86 * 1.e-3 # kg^-1 m^3                                                                                                                        
        
        value = numpy.zeros(z.shape)
        cut_1 = z < -1500.
        cut_2 = numpy.logical_and(z >= -1500., z <= 0.)
        cut_3 = z > 0.
        value[cut_1] = n_water
        value[cut_2] = 1. + (k * density(z[cut_2], ice_model=ice_model) * gnosim.utils.constants.mass_proton)
        value[cut_3] = n_air
    elif ice_model == 'test_deep':
        n_air = 1.000293
        n_water = 1.33
        k = 0.86 * 1.e-3 # kg^-1 m^3                                                                                                                        

        value = numpy.zeros(z.shape)
        cut_1 = z < -3000.
        cut_2 = numpy.logical_and(z >= -3000., z <= 0.)
        cut_3 = z > 0.
        value[cut_1] = n_water
        value[cut_2] = 1. + (k * density(z[cut_2], ice_model=ice_model) * gnosim.utils.constants.mass_proton)
        value[cut_3] = n_air
    elif ice_model == 'arthern':
        n_air = 1.000293
        n_water = 1.33
        k = 0.845 * 1.e-3 # kg^-1 m^3

        value = numpy.zeros(z.shape)
        cut_1 = z < -3000.
        cut_2 = numpy.logical_and(z >= -3000., z <= 0.)
        cut_3 = z > 0.
        value[cut_1] = n_water
        value[cut_2] = 1. + (k * density(z[cut_2], ice_model=ice_model) * gnosim.utils.constants.mass_proton)
        value[cut_3] = n_air
    else:
        print ('WARNING')
        return -999

    if scalar:
        return value[0]
    else:
        return value

############################################################

def deltaIndexOfRefraction(z, ice_model=ice_model_default, delta_z=2.):
    """
    Change in index of refraction
    """
    return numpy.max([numpy.fabs(indexOfRefraction(z) - indexOfRefraction(z - delta_z)),
                      numpy.fabs(indexOfRefraction(z) - indexOfRefraction(z + delta_z))], axis=0) / delta_z

############################################################

def temperature(z):
    """
    z = elevation (m)

    Returns temperature (C)
    """
    return f_temperature(z) 

############################################################
    
def attenuationLengthBase(temperature, frequency):
    """
    http://icecube.wisc.edu/~mnewcomb/radio/
    
    temperature = Temperature (C)
    frequency = Frequency (GHz)

    Returns attenuation length (m)
    """
    a = 5.03097 * numpy.exp(0.134806 * temperature)
    b = (0.172082 * temperature) + 10.629
    c = (-0.00199175 * temperature) - 0.703323
    return 2. * 1.e3 * 1.701 / (a + (b * frequency**(c + 1.))) # WARNING: The factor 2 is a kludge!

############################################################

def attenuationLength(z, frequency, ice_model=ice_model_default):
    """
    z = elevation (m)
    frequency = radio frequency (GHz)

    Returns:
    attenuation length, i.e., distance at which electric field is reduced by e (m)

    Source: ARA insturment paper Allison et al. 2011, arXiv:1105.2854

    Question: What is radio attenuation lenth in air??
    """
    # Treat as constant with respect to depth and frequency until add real data points
    frequency_0 = 0.3 # GHz

    if ice_model in ['greenland', 'empirical', 'test_deep', 'test_mid']:
        attenuation_length = 1000. # m
    elif ice_model in ['ross', 'test_shallow']:
        attenuation_length = 400. # m
    elif ice_model in ['arthern']:
        attenuation_length = attenuationLengthBase(temperature(z), frequency) # m
    else:
        print ('WARNING')
        attenuation_length = 1000. # m

    if numpy.isscalar(z):
        return (z <= 0.) * attenuation_length + (z > 0.) * 1.e20 # m
    else:
        return (z <= 0.) * attenuation_length * numpy.ones(len(z)) + (z > 0.) * 1.e20 * numpy.ones(len(z)) # m

############################################################

if __name__ == "__main__":
    z = numpy.linspace(-3000., 100., 10000) # Array of elevations (m)

    pylab.figure()
    #pylab.plot(z, density(z, ice_model='parametric') * gnosim.utils.constants.mass_proton, label='Parametric') # Convert from nucleons m^-3 to kg m^-3
    pylab.plot(z, density(z, ice_model='empirical') * gnosim.utils.constants.mass_proton, label='Empirical') # Convert from nucleons m^-3 to kg m^-3
    #pylab.plot(z, density(z, ice_model='ross') * gnosim.utils.constants.mass_proton, label='Ross') # Convert from nucleons m^-3 to kg m^-3
    #pylab.plot(z, density(z, ice_model='test_shallow') * gnosim.utils.constants.mass_proton, label='Test Shallow') # Convert from nucleons m^-3 to kg m^-3
    #pylab.plot(z, density(z, ice_model='test_mid') * gnosim.utils.constants.mass_proton, label='Test Mid') # Convert from nucleons m^-3 to kg m^-3
    #pylab.plot(z, density(z, ice_model='test_deep') * gnosim.utils.constants.mass_proton, label='Test Deep') # Convert from nucleons m^-3 to kg m^-3
    pylab.plot(z, density(z, ice_model='arthern') * gnosim.utils.constants.mass_proton, label='Arthern') # Convert from nucleons m^-3 to kg m^-3
    pylab.xlabel('Elevation (m)')
    pylab.ylabel(r'Density (kg m$^{-3}$)')
    pylab.legend(loc='lower center')
    #pylab.scatter(elevation_firn[1:-1], density_firn[1:-1] * gnosim.utils.constants.mass_proton, c='red')
    #pylab.scatter(elevation_deep[1:-1], density_deep[1:-1] * gnosim.utils.constants.mass_proton, c='black')
    #pylab.xlim([-600., 100.])
    pylab.xlim([-3100., 100.])

    z = numpy.linspace(-3100., 100., 2000) # Array of elevations (m)

    pylab.figure()
    #pylab.plot(z, indexOfRefraction(z, ice_model='parametric'), label='Parametric')
    pylab.plot(z, indexOfRefraction(z, ice_model='empirical'), label='Empirical')
    #pylab.plot(z, indexOfRefraction(z, ice_model='ross'), label='Ross')
    #pylab.plot(z, indexOfRefraction(z, ice_model='test_shallow'), label='Test Shallow')
    #pylab.plot(z, indexOfRefraction(z, ice_model='test_mid'), label='Test Mid')
    #pylab.plot(z, indexOfRefraction(z, ice_model='test_deep'), label='Test Deep')
    pylab.plot(z, indexOfRefraction(z, ice_model='arthern'), label='Arthern')
    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Index of Refraction')
    pylab.legend(loc='lower center')
    #k = 0.86 * 1.e-3 # kg^-1 m^3
    #pylab.plot(z, 1. + (k * f_density(z) * gnosim.utils.constants.mass_proton), c='green')
    pylab.xlim([-3100., 100.])
    
    z = numpy.linspace(-3000., 0., 2000) # Array of elevations (m)
    pylab.figure()
    frequency = 0.3 # GHz
    pylab.plot(z, attenuationLength(z, frequency=frequency, ice_model='arthern'))
    pylab.title('Frequency = %i MHz'%(frequency / gnosim.utils.constants.MHz_to_GHz))
    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Electric Field Attenuation Length (m)')
    pylab.xlim([-3000., 0.])
    
############################################################
