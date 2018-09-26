"""
Properties of the Antarctic ice.
"""

import numpy
import pylab
import scipy.interpolate

import gnosim.utils.constants

pylab.ion()

############################################################

ice_model_default = 'polar'

# From South Pole Ice Characteristics, http://icecube.wisc.edu/~mnewcomb/radio/#icerefraction
# Density (Mg m^-3)
# Elevation, negative is down (m)
profile_deep = numpy.array([[0.7230343614334508, -63.09751434034433],
                                                 [0.7864140997923716, -86.0420650095607],
                                                 [0.8212633137439305, -103.2504780114723],
                                                 [0.8398757018985611, -114.72275334608048],
                                                 [0.8685699534228556, -137.6673040152964],
                                                 [0.8742983634188128, -143.40344168260026],
                                                 [0.8795439563972824, -149.13957934990412],
                                                 [0.88874603766566, -160.6118546845123],
                                                 [0.8998398145522166, -177.8202676864248],
                                                 [0.9083582711221764, -195.0286806883364],
                                                 [0.9167228067791977, -217.97323135755323],
                                                 [0.9237783972356667, -246.653919694073],
                                                 [0.9290136117978872, -281.0707456978971],
                                                 [0.933129801265018, -332.69598470363326],
                                                 [0.9351232837375564, -390.0573613766737],
                                                 [0.9361581618621482, -476.0994263862335],
                                                 [0.936450650352632, -573.613766730402],
                                                 [0.9365299601324286, -757.1701720841302],
                                                 [0.9365347474915189, -952.198852772467],
                                                 [0.936534974411263, -1101.3384321223714],
                                                 [0.9365349990985374, -1319.311663479924],
                                                 [0.9365349999002391, -1462.715105162524],
                                                 [0.9365349999994935, -1806.8833652007647],
                                                 [0.9365349999999864, -2042.0650095602296],
                                                 [0.936534999999999, -2214.1491395793496],
                                                 [0.9365349999999999, -2334.6080305927344],
                                                 [0.936535, -2466.5391969407265],
                                                 [0.936535, -2621.414913957935],
                                                 [0.936535, -2845.1242829827916],
                                                 [0.936535, -3000],
                                                 [0.936535, -10000000000.0]]) #Filler

'''
before removing duplicate point [0.8685699534228556, -137.6673040152964]
profile_deep = numpy.array([[0.7230343614334508, -63.09751434034433],
                                                 [0.7864140997923716, -86.0420650095607],
                                                 [0.8212633137439305, -103.2504780114723],
                                                 [0.8398757018985611, -114.72275334608048],
                                                 [0.8685699534228556, -137.6673040152964],
                                                 [0.8685699534228556, -137.6673040152964],
                                                 [0.8742983634188128, -143.40344168260026],
                                                 [0.8795439563972824, -149.13957934990412],
                                                 [0.88874603766566, -160.6118546845123],
                                                 [0.8998398145522166, -177.8202676864248],
                                                 [0.9083582711221764, -195.0286806883364],
                                                 [0.9167228067791977, -217.97323135755323],
                                                 [0.9237783972356667, -246.653919694073],
                                                 [0.9290136117978872, -281.0707456978971],
                                                 [0.933129801265018, -332.69598470363326],
                                                 [0.9351232837375564, -390.0573613766737],
                                                 [0.9361581618621482, -476.0994263862335],
                                                 [0.936450650352632, -573.613766730402],
                                                 [0.9365299601324286, -757.1701720841302],
                                                 [0.9365347474915189, -952.198852772467],
                                                 [0.936534974411263, -1101.3384321223714],
                                                 [0.9365349990985374, -1319.311663479924],
                                                 [0.9365349999002391, -1462.715105162524],
                                                 [0.9365349999994935, -1806.8833652007647],
                                                 [0.9365349999999864, -2042.0650095602296],
                                                 [0.936534999999999, -2214.1491395793496],
                                                 [0.9365349999999999, -2334.6080305927344],
                                                 [0.936535, -2466.5391969407265],
                                                 [0.936535, -2621.414913957935],
                                                 [0.936535, -2845.1242829827916],
                                                 [0.936535, -3000],
                                                 [0.936535, -10000000000.0]]) #Filler
'''

elevation_deep = profile_deep.transpose()[1]
density_deep = profile_deep.transpose()[0] * 1.e3 / gnosim.utils.constants.mass_proton # nucleons m^-3


# From 
# Depth, positive is down (m)
# Density (kg m^-3)
"""
profile_firn = numpy.array([[0.00037415, 0.0],
                            [0.00037415, 0.0],
                            [0.0280870065738, 256.916996047],
                            [0.0349534776539, 270.750988142],
                            [0.0491701639089, 292.490118577],
                            [0.078043431808, 322.134387352],
                            [0.106034053131, 341.897233202],
                            [0.167674323108, 371.541501976],
                            [0.264833916194, 401.185770751],
                            [0.405462908974, 428.853754941],
                            [0.639644770391, 458.498023715],
                            [1.10497311129, 494.071146245],
                            [1.69015088017, 521.739130435],
                            [2.28947485275, 541.501976285],
                            [3.29523965819, 565.217391304],
                            [4.4634095545, 584.980237154],
                            [5.35464107285, 596.837944664],
                            [7.25266598785, 616.600790514],
                            [11.0908158357, 644.268774704],
                            [14.5728652401, 662.055335968],
                            [22.9708098687, 691.699604743],
                            [35.1260120164, 719.367588933],
                            [52.1079858882, 745.059288538],
                            [84.6648585386, 776.679841897],
                            [114.670546116, 796.442687747],
                            [165.025055378, 820.158102767],
                            [252.346422072, 847.826086957],
                            [312.047469776, 861.660079051]])


elevation_firn = -1. * profile_firn.transpose()[0]
density_firn = profile_firn.transpose()[1] / gnosim.utils.constants.mass_proton # nucleons m^-3
"""
#f_density = scipy.interpolate.interp1d(elevation_deep[::-1], density_deep[::-1], kind='linear', bounds_error=False, fill_value=0.)
f_density = scipy.interpolate.interp1d( elevation_deep[::-1] , density_deep[::-1], kind='cubic', bounds_error = None, fill_value= 'extrapolate' ) #extroplate appears to do a good job here to 0m - DS

# Berkeley Group (antarctic)
t_data, z_data = zip(*[[-51.0696, 0.0],
                       [-50.8110708872, -96.0269567899],
                       [-50.5632416462, -185.024005638],
                       [-50.3384290394, -261.322292208],
                       [-50.0121098059, -363.04453156],
                       [-49.6202011626, -471.122759107],
                       [-49.1394644123, -585.530546624],
                       [-48.5748721854, -699.925120028],
                       [-48.0672072572, -788.895740651],
                       [-47.4471795627, -884.209135357],
                       [-46.2678790842, -1036.71320971],
                       [-45.2748570915, -1144.72536669],
                       [-43.5441984447, -1303.57221513],
                       [-42.4200531809, -1392.52962164],
                       [-39.902500389, -1564.15451702],
                       [-38.1748023155, -1665.88997049],
                       [-36.6314481824, -1748.58388759],
                       [-34.9561656655, -1831.30423292],
                       [-31.802185966, -1971.37382725],
                       [-29.5748535016, -2060.54265956],
                       [-27.697431353, -2130.6567414],
                       [-25.3298479683, -2213.52244197],
                       [-21.3562903244, -2341.09148571],
                       [-18.3255166962, -2430.43210148],
                       [-11.8672562746, -2602.83662952],
                       [-4.55379399077, -2775.34687046],
                       [3.66965515649, -2947.89675373],
                       [7.97033030664, -3030.98709422]])

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
    elif ice_model == 'polar':
        return f_density(z) # nucleons m^-3
    else:
        print ('WARNING')
        return -999

############################################################

def indexOfRefraction(z, ice_model=ice_model_default):
    """
    z = elevation (m)

    Returns:
    index of refraction

   
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
    
    elif ice_model == 'polar':
        n_air = 1.000293
        n_water = 1.33
        k = 0.845 * 1.e-3 # kg^-1 m^3

        value = numpy.zeros(z.shape)
        cut_1 = z < -3000.
        cut_2 = numpy.logical_and(z >= -3000., z <= 0.)
        cut_3 = z > 0.
        value[cut_1] = n_water
        value[cut_2] = 1.325 + 0.463 * (1.0-numpy.exp(0.0140*z[cut_2]))
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
    return 2. * 1.e3 * 1.701 / (a + (b * frequency**(c + 1.))) # WARNING: The factor 2 is a kludge! do we need it??
########################################################################################################

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
    elif ice_model in ['polar']:
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
    #pylab.plot(z, density(z, ice_model='arthern') * gnosim.utils.constants.mass_proton, label='Arthern') # Convert from nucleons m^-3 to kg m^-3
    pylab.plot(z, density(z, ice_model='polar') * gnosim.utils.constants.mass_proton, label='Polar') # Convert from nucleons m^-3 to kg m^-3
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
    pylab.plot(z, indexOfRefraction(z, ice_model='polar'), label='Polar')

    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Index of Refraction')
    pylab.legend(loc='lower center')
    #k = 0.86 * 1.e-3 # kg^-1 m^3
    #pylab.plot(z, 1. + (k * f_density(z) * gnosim.utils.constants.mass_proton), c='green')
    pylab.xlim([-3100., 100.])
    
    z = numpy.linspace(-3000., 0., 2000) # Array of elevations (m)
    pylab.figure()
    frequency = 0.3 # GHz
    pylab.plot(z, attenuationLength(z, frequency=frequency, ice_model='polar'))
    pylab.title('Frequency = %i MHz'%(frequency / gnosim.utils.constants.MHz_to_GHz))
    pylab.xlabel('Elevation (m)')
    pylab.ylabel('Electric Field Attenuation Length (m)')
    pylab.xlim([-3000., 0.])
    
    input("press any key to exit")
############################################################

