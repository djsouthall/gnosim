import glob
import numpy
import scipy.interpolate
import scipy.spatial
import h5py
import pylab

import gnosim.utils.constants
import gnosim.utils.rf
import gnosim.utils.plane
import gnosim.earth.earth
import gnosim.earth.greenland

pylab.ion()

############################################################

def fresnel(n_1, n_2, incidence_angle, mode):
    """
    Reflected or transmitted power for different polarizations.

    n_1 = index of refraction of current medium
    n_2 = index of refraction of encountered medium
    incidence_angle = incidence angle to surface (deg)
    mode = 'reflection' or 'transmission'

    Need to return to this function to make it more numpy compliant eventually.
    """
    c = numpy.cos(numpy.radians(incidence_angle))
    s = numpy.sin(numpy.radians(incidence_angle))

    # s-polarized light (i.e., electric field perpendicular to the plane of reflection and transmission)
    if 1. - (n_1 * s /  n_2)**2 < 0.:
        numerator_s = numpy.complex(n_1 * c, 
                                    -1. * n_2 * numpy.sqrt((n_1 * s / n_2)**2 - 1.))
        denominator_s = numpy.complex(n_1 * c, 
                                    n_2 * numpy.sqrt((n_1 * s / n_2)**2 - 1.))
    else:
        numerator_s = numpy.complex(n_1 * c - n_2 * numpy.sqrt(1. - (n_1 * s / n_2)**2), 
                                    0.)
        denominator_s = numpy.complex(n_1 * c + n_2 * numpy.sqrt(1. - (n_1 * s / n_2)**2), 
                                      0.)
    f_s = numerator_s / denominator_s
    r_s = numpy.real(f_s * numpy.conj(f_s))
    t_s = 1. - r_s

    # p-polarized light (i.e., electric field parallel to the plane of reflection and transmission)
    if 1. - (n_1 * s / n_2)**2 < 0.:
        numerator_p = numpy.complex(-1. *  n_2 * c, 
                                    n_1 * numpy.sqrt((n_1 * s / n_2)**2 - 1.))
        denominator_p = numpy.complex(n_2 * c, 
                                      n_1 * numpy.sqrt((n_1 * s / n_2)**2 -1.))
    else:
        numerator_p = numpy.complex(n_1 * numpy.sqrt(1. - (n_1 * s / n_2)**2) - n_2 * c, 
                                    0.)
        denominator_p = numpy.complex(n_1 * numpy.sqrt(1. - (n_1 * s / n_2)**2) + n_2 * c, 
                                      0.)
    f_p = numerator_p / denominator_p
    r_p = numpy.real(f_p * numpy.conj(f_p))
    t_p = 1. - r_p

    if mode == 'reflection':
        return r_s, r_p
    elif mode == 'transmission':
        return t_s, t_p
    else:
        print 'WARNING: mode %s not recognized'%(mode)

############################################################

def testFresnel(n_low=1., n_high=1.5):
    """
    Simple test for Fresnel coefficients.
    """

    incidence_angle_array = numpy.arange(0., 90. + 1.e-10, 1.)
    
    r_s = numpy.zeros(len(incidence_angle_array))
    r_p = numpy.zeros(len(incidence_angle_array))
    t_s = numpy.zeros(len(incidence_angle_array))
    t_p = numpy.zeros(len(incidence_angle_array))
    for ii, incidence_angle in enumerate(incidence_angle_array):
        r_s[ii], r_p[ii] = fresnel(n_low, n_high, incidence_angle, mode='reflection')
        t_s[ii], t_p[ii] = fresnel(n_low, n_high, incidence_angle, mode='transmission')
    pylab.figure()
    pylab.plot(incidence_angle_array, r_s, label='R_s', c='red', linestyle='--')
    pylab.plot(incidence_angle_array, r_p, label='R_p', c='red', linestyle='-')
    pylab.plot(incidence_angle_array, t_s, label='T_s', c='blue', linestyle='--')
    pylab.plot(incidence_angle_array, t_p, label='T_p', c='blue', linestyle='-')
    pylab.legend(loc='center left')
    pylab.xlabel('Incidence Angle (deg)')
    pylab.ylabel('Reflection or Transmission Fraction')
    pylab.title('n_1 = %.2f and n_2 = %.2f'%(n_low, n_high))
    pylab.ylim([-0.1, 1.1])

    r_s = numpy.zeros(len(incidence_angle_array))
    r_p = numpy.zeros(len(incidence_angle_array))
    t_s = numpy.zeros(len(incidence_angle_array))
    t_p = numpy.zeros(len(incidence_angle_array))
    for ii, incidence_angle in enumerate(incidence_angle_array):
        r_s[ii], r_p[ii] = fresnel(n_high, n_low, incidence_angle, mode='reflection')
        t_s[ii], t_p[ii] = fresnel(n_high, n_low, incidence_angle, mode='transmission')
    pylab.figure()
    pylab.plot(incidence_angle_array, r_s, label='R_s', c='red', linestyle='--')
    pylab.plot(incidence_angle_array, r_p, label='R_p', c='red', linestyle='-')
    pylab.plot(incidence_angle_array, t_s, label='T_s', c='blue', linestyle='--')
    pylab.plot(incidence_angle_array, t_p, label='T_p', c='blue', linestyle='-')
    pylab.legend(loc='center left')
    pylab.xlabel('Incidence Angle (deg)')
    pylab.ylabel('Reflection or Transmission Fraction')
    pylab.title('n_1 = %.2f and n_2 = %.2f'%(n_high, n_low))
    pylab.ylim([-0.1, 1.1])

############################################################

def rayTrace(origin, phi_0, theta_0, t_max=50000., t_step=1.): # t_max=40000, t_max=1000 (testing)
    """
    z_0 = initial elevation (m)
    t_max = max time (ns)
    t_step = time step (ns)
    """

    # IDEA: let t_step grow when the index of refraction isn't changing fast, i.e.,
    #       a dynamic t_step 

    # ORIGINAL
    n_steps =  int(t_max / t_step)
    # ORIGINAL
    # NEW
    #n_steps = 100000 # 1000000, Arbitrary large number, but might be too big...
    # NEW
    
    # ORIGINAL
    #t_array = numpy.arange(0., t_step * (n_steps + 1), t_step) # Time (ns)
    # ORIGINAL
    # NEW
    t_array = numpy.zeros(n_steps + 1) # m
    # NEW
    x_array = numpy.zeros(n_steps + 1) # m
    y_array = numpy.zeros(n_steps + 1) # m
    z_array = numpy.zeros(n_steps + 1) # m
    a_v_array = numpy.ones(n_steps + 1) # Amplitude of vertically polarized electric field (m)
    a_h_array = numpy.ones(n_steps + 1) # Amplitude of horizontally polarized electric field (m)
    d_array = numpy.zeros(n_steps + 1) # m

    phi_array = phi_0 * numpy.ones(n_steps + 1) # deg
    theta_array = numpy.zeros(n_steps + 1) # deg

    r_s = 1.
    r_p = 1.

    x_array[0] = origin[0]
    y_array[0] = origin[1]
    z_array[0] = origin[2]
    theta_array[0] = theta_0

    index_reflect = 0 # Latch to prevent multiple reflections at interface
    index_reflect_air = 0 # Stores index of reflection event at ice-air interface
    index_reflect_water = 0 # Stores index of reflection event at ice-water interface

    for ii in range(0, n_steps):

        # Dynamic time step depending on how fast the index of refraction is changing
        if gnosim.earth.greenland.deltaIndexOfRefraction(z_array[ii]) > 1.e-4 or z_array[ii] >= 0.:
            t_step = 1.
        else:
            t_step = 5.
        t_array[ii + 1] = t_array[ii] + t_step

        if ii == 0 and z_array[ii] > 1.:
            # Rather than take many steps to the ice, take one large jump

            """
            # THIS VERSION WORKS, BUT DOES NOT FULLY TAKE INTO ACCOUNT EARTH CURVATURE
            r = (z_array[ii] + 0.01) * numpy.tan(numpy.radians(theta_array[ii])) # 0.01 meter into ground (i.e., below surface)
            d = numpy.sqrt(r**2 + z_array[ii]**2)
            t = d * gnosim.earth.greenland.indexOfRefraction(z_array[ii]) / gnosim.utils.constants.speed_light

            t_array[1:] += (t - t_step)
            x_step = d * numpy.sin(numpy.radians(theta_array[ii])) * numpy.cos(numpy.radians(phi_array[ii]))
            y_step = d * numpy.sin(numpy.radians(theta_array[ii])) * numpy.sin(numpy.radians(phi_array[ii]))
            z_step = d * numpy.cos(numpy.radians(theta_array[ii]))
            # THIS VERSION WORKS, BUT DOES NOT FULLY TAKE INTO ACCOUNT EARTH CURVATURE
            """
            
            # Curvature corrections
            distance_with_curvature, angle_curvature, x_curvature = gnosim.earth.earth.curvature(z_array[ii], theta_array[ii]) 
            
            x_step = x_curvature * numpy.cos(numpy.radians(phi_array[ii]))
            y_step = x_curvature * numpy.sin(numpy.radians(phi_array[ii]))
            z_step = -1. * (z_array[ii] + 0.01) # Go 0.01 m into ground (i.e., below surface)

            # Correct time
            t = distance_with_curvature * gnosim.earth.greenland.indexOfRefraction(z_array[ii]) / gnosim.utils.constants.speed_light
            #t_array[1:] += (t - t_step)
            t_array[ii + 1] = t_array[ii] + t

            # Incidence angle is increased due to curvature of Earth
            theta_array[ii] = theta_array[ii] - angle_curvature
            if theta_array[ii] < 90. or numpy.isnan(theta_array[ii]):
                print 'WARNING: Ray does not intersect Earth'
                raw_input('CONTINUE?')
                theta_array[ii] = 90. + 1.e10
            
            # Distance corrected for curvature of the Earth
            d_step = distance_with_curvature
        else:
            x_step = (t_step * gnosim.utils.constants.speed_light * numpy.sin(numpy.radians(theta_array[ii])) * numpy.cos(numpy.radians(phi_array[ii])) \
                      / gnosim.earth.greenland.indexOfRefraction(z_array[ii]))
            y_step = (t_step * gnosim.utils.constants.speed_light * numpy.sin(numpy.radians(theta_array[ii])) * numpy.sin(numpy.radians(phi_array[ii])) \
                      / gnosim.earth.greenland.indexOfRefraction(z_array[ii]))
            z_step = (t_step * gnosim.utils.constants.speed_light * numpy.cos(numpy.radians(theta_array[ii])) \
                      / gnosim.earth.greenland.indexOfRefraction(z_array[ii]))
            
            d_step = numpy.sqrt(x_step**2 + y_step**2 + z_step**2) # Distance traveled during step
        
        d_array[ii + 1] = d_step
        
        attenuation_length = gnosim.earth.greenland.attenuationLength(z_array[ii], 0.3) # m, Transmission
        if attenuation_length > 1.e10:
            a_v_array[ii + 1] = 1.
            a_h_array[ii + 1] = 1.
        else:
            a_v_array[ii + 1] = numpy.exp(-1. * d_array[ii + 1] / gnosim.earth.greenland.attenuationLength(z_array[ii], 0.3)) # Transmission 
            a_h_array[ii + 1] = numpy.exp(-1. * d_array[ii + 1] / gnosim.earth.greenland.attenuationLength(z_array[ii], 0.3)) # Transmission

        x_array[ii + 1] = x_array[ii] + x_step
        y_array[ii + 1] = y_array[ii] + y_step
        z_array[ii + 1] = z_array[ii] + z_step

        # Test whether the ray is refracted from upward going to downward going
        value = (gnosim.earth.greenland.indexOfRefraction(z_array[ii]) / gnosim.earth.greenland.indexOfRefraction(z_array[ii + 1])) \
                    * numpy.sin(numpy.radians(theta_array[ii]))
        
        # Hit ice-air interface at surface?
        delta_index_of_refraction = gnosim.earth.greenland.indexOfRefraction(z_array[ii + 1]) - gnosim.earth.greenland.indexOfRefraction(z_array[ii])
        
        if ii == index_reflect and ii > 0:
            #print 'SKIP', ii
            theta_array[ii + 1] = theta_array[ii]
            index_reflect = 0
        elif delta_index_of_refraction < -0.1 and theta_array[ii] < 90.:
            # Ray going from ice to air (for a ray going backwards from antenna to interaction vertex)
            # Compute reflection coefficients (power which is reflected)
            incidence_angle = theta_array[ii]
            r_s, r_p = fresnel(gnosim.earth.greenland.indexOfRefraction(z_array[ii]),
                               gnosim.earth.greenland.indexOfRefraction(z_array[ii + 1]),
                               incidence_angle, mode='reflection')
            a_v_array[ii + 1] *= numpy.sqrt(r_p)
            a_h_array[ii + 1] *= numpy.sqrt(r_s)
            #print 'ICE -> AIR', ii, r_p, r_s
            theta_array[ii + 1] = 180. - theta_array[ii]
            index_reflect = ii + 1
            if index_reflect_air == 0 and index_reflect_water == 0:
                # Do not allow this index to be set twice
                index_reflect_air = ii + 1
        elif delta_index_of_refraction > 0.1 and theta_array[ii] > 90.:
            # Ray going from air to ice (for a ray going backwards from antenna to interaction vertex)
            # Compute reflection coefficients (power which is transmitted)
            theta_array[ii + 1] = 180. - numpy.degrees(numpy.arcsin(value))
            incidence_angle = 180. - theta_array[ii + 1]
            t_s, t_p = fresnel(gnosim.earth.greenland.indexOfRefraction(z_array[ii + 1]),
                               gnosim.earth.greenland.indexOfRefraction(z_array[ii]),
                               incidence_angle, mode='transmission')
            a_v_array[ii + 1] *= numpy.sqrt(t_p)
            a_h_array[ii + 1] *= numpy.sqrt(t_s)
            #print 'AIR -> ICE', ii, t_p, t_s
        elif delta_index_of_refraction < -0.1 and theta_array[ii] > 90.:
            # Ray going from ice to water
            # Compute reflection coefficients (power which is reflected)
            incidence_angle = 180. - theta_array[ii]
            r_s, r_p = 1., 1. # Water is conductor, so complete reflection
            #r_s, r_p = fresnel(gnosim.earth.greenland.indexOfRefraction(z_array[ii]),
            #                   gnosim.earth.greenland.indexOfRefraction(z_array[ii + 1]),
            #                   incidence_angle, mode='reflection')
            a_v_array[ii + 1] *= numpy.sqrt(r_p)
            a_h_array[ii + 1] *= numpy.sqrt(r_s)
            reflection_water = True
            #print 'ICE -> WATER', ii, r_p, r_s
            theta_array[ii + 1] = 180. - theta_array[ii]
            index_reflect = ii + 1
            index_reflect_water = ii + 1
        elif value >= 1.:
            # Total internal reflection
            theta_array[ii + 1] = 180. - theta_array[ii]
        else:
            # Refraction
            theta_array[ii + 1] = numpy.degrees(numpy.arcsin(value))
            if theta_array[ii] > 90.:
                theta_array[ii + 1] = 180. - theta_array[ii + 1]
        
        #if numpy.fabs(theta_array[ii + 1] - theta_array[ii]) > 5.:
        #    #print theta_array[ii], theta_array[ii + 1]
        #    #raw_input('WAIT')

        # Define a stop condition
        if index_reflect_water > 0 and theta_array[ii + 1] > 90.:
            n_steps = ii + 1
            #print 'STOP', n_steps
            break

    # Convert to total distance
    d_array = numpy.cumsum(d_array)

    # Convert from transmission at each step to cumulative transmission
    a_v_array = numpy.cumprod(a_v_array)
    a_h_array = numpy.cumprod(a_h_array)

    return (x_array[0: n_steps], y_array[0: n_steps], z_array[0: n_steps], \
        t_array[0: n_steps], d_array[0: n_steps], phi_array[0: n_steps], \
        theta_array[0: n_steps], a_v_array[0: n_steps], a_h_array[0: n_steps], \
        index_reflect_air, index_reflect_water)

############################################################

def makeLibrary(z_0, theta_ray_array, save=True, library_dir='library'):
    x_0 = 0.
    y_0 = 0.
    phi_0 = 0.

    x_array = []
    y_array = []
    z_array = []
    t_array = []
    d_array = []
    phi_array = []
    theta_array = []
    a_v_array = []
    a_h_array = []
    reflection = False
    
    theta_0_array = []

    for ii in range(0, len(theta_ray_array)):
        print '(%i/%i) theta_0 = %.4f'%(ii, len(theta_ray_array), theta_ray_array[ii])
        x, y, z, t, d, phi, theta, a_v, a_h, index_reflect_air, index_reflect_water = rayTrace([x_0, y_0, z_0], phi_0, theta_ray_array[ii])
        x_array.append(x)
        y_array.append(y)
        z_array.append(z)
        t_array.append(t)
        d_array.append(d)
        phi_array.append(phi)
        theta_array.append(theta)
        a_v_array.append(a_v)
        a_h_array.append(a_h)

        if save:
            n_points = len(t)
            outfile = '%s/z0_%.2f_theta_%.4f_n_%i.h5'%(library_dir, z_0, theta_ray_array[ii], n_points)
            file = h5py.File(outfile, 'w')
            
            file.attrs['index_reflect_air'] = index_reflect_air
            file.attrs['index_reflect_water'] = index_reflect_water
            file.attrs['z_0'] = z_0
            file.attrs['theta_0'] = theta_ray_array[ii]
            file.attrs['ice_model'] = gnosim.earth.greenland.ice_model_default

            file.create_dataset('r', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('z', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('t', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('d', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('theta', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('theta_0', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('a_v', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('a_h', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
        
            file['r'][...] = x
            file['z'][...] = z
            file['t'][...] = t
            file['d'][...] = d
            file['theta'][...] = theta
            file['theta_0'][...] = theta_ray_array[ii] * numpy.ones(n_points)
            file['a_v'][...] = a_v
            file['a_h'][...] = a_h

            file.close()
    
    x_array = numpy.concatenate([x_array])
    y_array = numpy.concatenate([y_array])
    z_array = numpy.concatenate([z_array])
    t_array = numpy.concatenate([t_array])
    d_array = numpy.concatenate([d_array])
    phi_array = numpy.concatenate([phi_array])
    theta_array = numpy.concatenate([theta_array])
    a_v_array = numpy.concatenate([a_v_array])
    a_h_array = numpy.concatenate([a_h_array])

    """
    pylab.figure()
    pylab.scatter(x_array, z_array, c=numpy.clip(gnosim.utils.rf.decibel(a_v_array), -30., 0.), edgecolors='none')
    colorbar = pylab.colorbar()
    colorbar.set_label('VPOL Attenuation (dB)')
    pylab.xlabel('Radius (m)')
    pylab.ylabel('Elevation (m)')
    """

############################################################

class RefractionLibrary:

    def __init__(self, search):
        self.infiles = glob.glob(search)
    
        # List attributes of interest
        self.solutions = ['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2']
        self.keys = ['r', 'z', 't', 'd', 'theta', 'theta_0', 'a_v', 'a_h']

        # Dictionary to store data
        self.data = {}
        for solution in self.solutions:
            self.data[solution] = {}
            for key in self.keys:
                self.data[solution][key] = []

        # Ice model
        self.ice_model = None
        
        # Open data files and begin sorting solutions
        for infile in self.infiles:
            print infile
            reader = h5py.File(infile, 'r')
            n = len(reader['t'])

            if self.ice_model is None:
                self.ice_model = reader.attrs['ice_model']
            else:
                if self.ice_model != reader.attrs['ice_model']:
                    print 'WARNING: Ice models used in ray-tracing libraries do not match, e.g., %s != %s'%(self.ice_model, 
                                                                                                            reader.attrs['ice_model'])

            if reader.attrs['index_reflect_air'] > 0:
                # Rays with reflections off ice-air interface
                cut = numpy.zeros(n, bool)
                cut[0: reader.attrs['index_reflect_air']] = True
                for key in self.keys:
                    #Upward-going rays prior to reflection off ice-air interface 
                    self.data['direct'][key].append(reader[key][cut])
                if reader.attrs['index_reflect_water'] > 0:
                    # Additional reflection off ice-water interface
                    cut = numpy.zeros(n, bool)
                    cut[reader.attrs['index_reflect_air']: reader.attrs['index_reflect_water']] = True
                    for key in self.keys:
                        # Downward-going rays from reflection off ice-air interface 
                        self.data['reflect'][key].append(reader[key][cut])
                    cut = numpy.zeros(n, bool)
                    cut[reader.attrs['index_reflect_water']:] = True
                    for key in self.keys:
                        # Upward-going rays from reflection off ice-water interface 
                        self.data['reflect_2'][key].append(reader[key][cut])
                else:
                    # No reflection off ice-water interface
                    cut = numpy.zeros(n, bool)
                    cut[reader.attrs['index_reflect_air']:] = True
                    for key in self.keys:
                        # Downward-going rays from reflection off ice-air interface 
                        self.data['reflect'][key].append(reader[key][cut])
            else:
                # Rays without reflections off ice-air interface
                if reader.attrs['index_reflect_water'] > 0:
                    # Reflection off ice-water interface
                    cut = numpy.zeros(n, bool)
                    cut[0: reader.attrs['index_reflect_water']] = True
                    for key in self.keys:
                        # Downward-going rays before reflection off ice-water interface 
                        self.data['direct'][key].append(reader[key][cut])
                    cut = numpy.zeros(n, bool)
                    cut[reader.attrs['index_reflect_water']:] = True
                    for key in self.keys:
                        # Upward-going rays after reflection off ice-water interface 
                        self.data['direct_2'][key].append(reader[key][cut])
                else:
                    # No reflection off ice-water interface
                    cut = numpy.ones(n, bool)
                    for key in self.keys:
                        # Downward-going rays from reflection off ice-air interface 
                        self.data['direct'][key].append(reader[key][cut])
            
            reader.close()

        for solution in self.solutions:
            for key in self.keys:
                if len(self.data[solution][key]) > 0:
                    self.data[solution][key] = numpy.concatenate(self.data[solution][key])
            print solution, len(self.data[solution]['t'])

        print 'Intersections...'
        # Find intersections
        dic_direct, dic_cross = self.intersect(self.data['direct'])
        self.data['direct'] = dic_direct
        self.data['cross'] = dic_cross

        print 'Sort solutions...'

        if len(self.data['direct_2']['t']) > 0:
            theta_0_divide = self.data['direct']['theta_0'][numpy.argmax(self.data['direct']['r'])]
            cut = self.data['direct_2']['theta_0'] < theta_0_divide
            for key in self.keys:
                self.data['cross_2'][key] = self.data['direct_2'][key][cut]
            cut = numpy.logical_not(cut)
            for key in self.keys:
                self.data['direct_2'][key] = self.data['direct_2'][key][cut]
        
        self.exists = {}
        self.hull = {}
        self.envelope = {}
        for solution in self.solutions:
            print solution, len(self.data[solution]['t'])
            if len(self.data[solution]['t']) > 0:
                self.exists[solution] = True
                r, z = self.makeHull(self.data[solution])
                self.hull[solution] = {'r': r,
                                       'z': z}
                envelope_low, envelope_high = self.makeEnvelope(r, z, 
                                                                self.data[solution]['r'], 
                                                                self.data[solution]['z'], 
                                                                self.data[solution]['theta_0'],)
                self.envelope[solution] = {'low': envelope_low,
                                           'high': envelope_high}
            else:
                self.exists[solution] = False
        
        # Check density of ray traces to determine optimal interpolation method
        if numpy.max(self.data['direct']['r']) / len(numpy.unique(self.data['direct']['theta_0'])) < 1000.:
            self.dense_rays = True
            print 'Dense ray traces'
        else:
            self.dense_rays = False
            print 'Sparse ray traces'

    def makeHull(self, dic):
        hull = scipy.spatial.ConvexHull(zip(dic['r'], dic['z']))
        return dic['r'][hull.vertices], dic['z'][hull.vertices]

    def getValue(self, dic, r, z):
        if self.dense_rays:
            # This scheme works well when the rays are densely packed, 
            # but completely fails when the rays are spread far apart.
            distance = numpy.sqrt((r - dic['r'])**2 + (z - dic['z'])**2)
            index_1 = numpy.argmin(distance)
            weight_1 = distance[index_1]**(-1)
            
            distance[dic['theta_0'] == dic['theta_0'][index_1]] = 1.e10
            index_2 = numpy.argmin(distance)
            weight_2 = distance[index_2]**(-1)
        else:
            # This scheme works much better when the rays are spread
            # far apart.
            distance = numpy.sqrt((r - dic['r'])**2 + (z - dic['z'])**2)
            theta_0_1 = dic['theta_0'][numpy.argmin(distance)]
            cut_1 = dic['theta_0'] == theta_0_1
            index_1 = numpy.nonzero(cut_1)[0][numpy.argmin(numpy.fabs(dic['z'][cut_1] - z))]
            
            distance[cut_1] = 1.e10
            theta_0_2 = dic['theta_0'][numpy.argmin(distance)]
            cut_2 = dic['theta_0'] == theta_0_2
            index_2 = numpy.nonzero(cut_2)[0][numpy.argmin(numpy.fabs(dic['z'][cut_2] - z))]
            
            weight_1 = 1. / numpy.fabs(r - dic['r'][index_1])
            weight_2 = 1. / numpy.fabs(r - dic['r'][index_2])

        #print '%10.2f %10.2f'%(r, z)
        #print '%10.2f %10.2f'%(dic['r'][index_1], dic['z'][index_1])
        #print '%10.2f %10.2f'%(dic['r'][index_2], dic['z'][index_2])
        #raw_input('WAIT')

        val_dic = {}
        for key in self.keys:
            val_dic[key] = ((weight_1 * dic[key][index_1]) + (weight_2 * dic[key][index_2])) / (weight_1 + weight_2)
        return val_dic

    def query(self, r_query, z_query):
        """
        Check to see which solutions exist.
        r_query = radius (m)
        z_query = elevation (m)
        Returns:
        flag_direct, flag_crossover, flag_reflect
        """
        flag_array = []
        for solution in self.solutions:
            if self.exists[solution]:
                flag_array.append(numpy.logical_and(z_query > self.envelope[solution]['low'](r_query), 
                                                    z_query < self.envelope[solution]['high'](r_query)))
            else:
                flag_array.append(False)
        return flag_array

    def event(self, r, z):
        flag_array = self.query(r, z)
        dic_array = []
        for ii, solution in enumerate(self.solutions):
            if flag_array[ii]:
                dic_array.append(self.getValue(self.data[solution], r, z))
            else:
                dic_array.append({})
        return flag_array, dic_array

    def test(self, field, n_trials=10, solution='direct'):
        r_min, r_max = numpy.min(self.data[solution]['r']), numpy.max(self.data[solution]['r'])
        z_min, z_max = numpy.min(self.data[solution]['z']), numpy.max(self.data[solution]['z'])

        if z_max > 0.:
            z_max = 0.

        r = numpy.zeros(n_trials)
        z = numpy.zeros(n_trials)
        val = numpy.zeros(n_trials)

        #pylab.figure()

        for ii in range(0, n_trials):
            r[ii], z[ii] = numpy.random.uniform(r_min, r_max), numpy.random.uniform(z_min, z_max)
            while True:                                                                                                                                     
                r[ii], z[ii] = numpy.random.uniform(r_min, r_max), numpy.random.uniform(z_min, z_max)                                                       
                if self.query(r[ii], z[ii])[0]:                                                                                                             
                    break
            """
            # Begin simple procedure
            theta_0_unique = numpy.unique(self.data[solution]['theta_0'])
            distance_array = numpy.zeros(len(theta_0_unique))
            val_array = numpy.zeros(len(theta_0_unique))
            for jj in range(0, len(theta_0_unique)):
                cut = (self.data[solution]['theta_0'] == theta_0_unique[jj])
                r_cut = self.data[solution]['r'][cut]
                z_cut = self.data[solution]['z'][cut]
                val_cut = self.data[solution][field][cut]
                distance = numpy.sqrt((r[ii] - r_cut)**2 + (z[ii] - z_cut)**2)
                index = numpy.argmin(distance)
                distance_array[jj] = distance[index]
                val_array[jj] = val_cut[index]

            index = numpy.argsort(distance_array)
            weight_1 = distance_array[index[0]]**(-1)
            val_1 = val_array[index[0]]
            weight_2 = distance_array[index[1]]**(-1)
            val_2 = val_array[index[1]]
            val_simple = ((weight_1 * val_1) + (weight_2 * val_2)) / (weight_1 + weight_2) 
            # End simple procedure
            """
            
            """
            # Begin simple procedure fast
            distance = numpy.sqrt((r[ii] - self.data[solution]['r'])**2 + (z[ii] - self.data[solution]['z'])**2)
            index = numpy.argmin(distance)
            weight_1 = distance[index]**(-1)
            val_1 = self.data[solution][field][index]
            distance[self.data[solution]['theta_0'] == self.data[solution]['theta_0'][index]] = 1.e10
            index = numpy.argmin(distance)
            weight_2 = distance[index]**(-1)
            val_2 = self.data[solution][field][index]
            val_simple = ((weight_1 * val_1) + (weight_2 * val_2)) / (weight_1 + weight_2)
            # End simple procedure fast
            """
            
            val_dic = self.getValue(self.data[solution], r[ii], z[ii])
            val_simple = val_dic[field]
            """
            r_interp_array = []
            z_interp_array = []
            theta_0_interp_array = []
            val_interp_array = []

            condition_set = [[False, False],
                             [False, True],
                             [True, False],
                             [True, True]]
            for condition_1, condition_2 in condition_set:
                cut_1 = r[ii] < self.data[solution]['r']
                cut_2 = z[ii] < self.data[solution]['z']
                cut = numpy.logical_and(cut_1 == condition_1, cut_2 == condition_2)
                if not numpy.any(cut):
                    print 'WARNING'

                r_cut = self.data[solution]['r'][cut]
                z_cut = self.data[solution]['z'][cut]
                theta_0_cut = self.data[solution]['theta_0'][cut]
                val_cut = self.data[solution][field][cut]
                distance = numpy.sqrt((r[ii] - r_cut)**2 + (z[ii] - z_cut)**2)
                index = numpy.argsort(distance)
                index_max = min(100, numpy.sum(cut))
                r_interp_array.append(r_cut[index[0:index_max]])
                z_interp_array.append(z_cut[index[0:index_max]])
                theta_0_interp_array.append(theta_0_cut[index[0:index_max]])
                val_interp_array.append(val_cut[index[0:index_max]])

                #print ii, numpy.sum(cut), len(self.data[solution]['r'])

            r_interp_array = numpy.concatenate(r_interp_array)
            z_interp_array = numpy.concatenate(z_interp_array)
            theta_0_interp_array = numpy.concatenate(theta_0_interp_array)
            val_interp_array = numpy.concatenate(val_interp_array)
            """

            """
            pylab.clf()
            pylab.scatter(r_interp_array, z_interp_array, c=val_interp_array, edgecolors='none')
            pylab.colorbar()
            pylab.scatter(r[ii], z[ii], c='gray', edgecolors='none')
            """

            #weight = ((r[ii] - r_interp_array)**2 + (z[ii] - z_interp_array)**2)**(-1)
            #val_weight = numpy.sum(weight * val_interp_array) / numpy.sum(weight)
            
            #f = scipy.interpolate.interp2d(r_interp_array, z_interp_array, val_interp_array, kind='cubic')
            #val_interp = f(r[ii], z[ii])

            #pylab.title('%f'%(val[ii]))

            #print val_weight, val_interp, val_simple
            val[ii] = val_simple
            
            #raw_input('WAIT')
        
        pylab.figure()
        pylab.scatter(r, z, c=val, edgecolors='none')
        pylab.colorbar()

        pylab.figure()
        pylab.scatter(self.data[solution]['r'], self.data[solution]['z'], c=self.data[solution][field], edgecolors='none')
        pylab.colorbar()

        if field == 'z':
            pylab.figure()
            pylab.hist(val - z, bins=50)

        return r, z, val
        

    def test2(self, field, n_trials=10):

        r_min, r_max = numpy.min(self.direct['r']), numpy.max(self.direct['r'])
        z_min, z_max = numpy.min(self.direct['z']), numpy.max(self.direct['z'])

        r = numpy.zeros(n_trials)
        z = numpy.zeros(n_trials)
        val = numpy.zeros(n_trials)

        for ii in range(0, n_trials):
            r[ii], z[ii] = numpy.random.uniform(r_min, r_max), numpy.random.uniform(z_min, z_max)
            while True:                                                                                                                                     
                r[ii], z[ii] = numpy.random.uniform(r_min, r_max), numpy.random.uniform(z_min, z_max)                                                       
                if self.query(r[ii], z[ii])[0]:                                                                                                             
                    break
            
            d = numpy.sqrt((r[ii] - self.direct['r'])**2 + (z[ii] - self.direct['z'])**2)
            index = numpy.argsort(d)
            #print ii, r[ii], z[ii]
            #print index, d[index[0:3]]
            r_select = self.direct['r'][index[0:3]]
            z_select = self.direct['z'][index[0:3]]
            val_select = self.direct[field][index[0:3]]
            #print r_select, z_select, val_select
            p = gnosim.utils.plane.Plane(r_select, z_select, val_select)
            val[ii] = p(r[ii], z[ii])
            #print val[ii]
        
        pylab.figure()
        pylab.scatter(r, z, c=val, edgecolors='none')
        pylab.colorbar()

        pylab.figure()
        pylab.scatter(self.direct['r'], self.direct['z'], c=self.direct[field], edgecolors='none')
        pylab.colorbar()

        if field == 'z':
            pylab.figure()
            pylab.hist(val - z, bins=50)

        return r, z, val
        
    def plotHull(self):
        """
        Tool for visualizing regions of (r, z) space where different solutions exist.
        """
        pylab.figure()
        #pylab.gca().add_patch(pylab.Polygon(zip(self.hull_direct_r, self.hull_direct_z), 
        #                                    closed=True, fill=False, linestyle='dashed', color='black'))
        #pylab.gca().add_patch(pylab.Polygon(zip(self.hull_crossover_r, self.hull_crossover_z),
        #                                    closed=True, fill=False, linestyle='dashed', color='red'))
        #pylab.gca().add_patch(pylab.Polygon(zip(self.hull_reflect_r, self.hull_reflect_z),
        #                                    closed=True, fill=False, linestyle='dashed', color='green'))
        #pylab.plot(self.hull_direct_r, self.hull_direct_z, c='black', linestyle='-', label='Direct Rays')
        #pylab.plot(self.hull_crossover_r, self.hull_crossover_z, c='red', linestyle='-', label='Crossover Rays')
        #pylab.plot(self.hull_reflect_r, self.hull_reflect_z, c='blue', linestyle='-', label='Reflected Rays')
        #pylab.legend(loc='upper right')

        for solution, color, label in [['direct', 'red', 'Direct'],
                                       ['cross', 'green', 'Cross'],
                                       ['reflect', 'blue', 'Reflect'],
                                       ['direct_2', 'black', 'Direct (Bottom)'],
                                       ['cross_2', 'orange', 'Cross (Bottom)'],
                                       ['reflect_2', 'magenta', 'Reflect (Bottom)']]:
            if not self.exists[solution]:
                continue
            r_direct = numpy.linspace(numpy.min(self.hull[solution]['r']), numpy.max(self.hull[solution]['r']), 10000)
            pylab.plot(r_direct, self.envelope[solution]['low'](r_direct), c=color, label=label)
            pylab.plot(r_direct, self.envelope[solution]['high'](r_direct), c=color)
            pylab.scatter(self.hull[solution]['r'], self.hull[solution]['z'], c=color, edgecolors='none')

        pylab.legend(loc='lower right')
        pylab.xlabel('Radius (m)')
        pylab.ylabel('Elevation (m)')

    def makeEnvelope(self, r, z, r_full=None, z_full=None, theta_0_full=None):
        """
        Define the regions (r, z) where ray-tracing solutions exist. Normally only need
        to supply the (r, z) coordinates of the convex hull, but for regions that are 
        expected to be concave in shape (e.g., crossover solutions) should supply the 
        full array of (r, z, theta_0) coordinates.
        """

        index_0 = numpy.argmin(r)
        r_0 = r[index_0] 
        z_0 = z[index_0]

        index_1 = numpy.argmax(r)
        r_1 = r[index_1]
        z_1 = z[index_1]

        slope = (z_1 - z_0) / (r_1 - r_0)
        cut_low = (z <= (z_0 + (slope * (r - r_0))))
        cut_high = (z >= (z_0 + (slope * (r - r_0))))
        
        # Make sure to get the endpoints
        cut_low[index_0] = True
        cut_low[index_1] = True
        cut_high[index_0] = True
        cut_high[index_1] = True

        r_low = r[cut_low]
        z_low = z[cut_low]
        index = numpy.argsort(r_low)
        r_low = r_low[index]
        z_low = z_low[index]
        f_low = scipy.interpolate.interp1d(r_low, z_low, bounds_error=False, fill_value=numpy.max(z_low))

        r_high = r[cut_high]
        z_high = z[cut_high]
        index = numpy.argsort(r_high)
        r_high = r_high[index]
        z_high = z_high[index]
        f_high = scipy.interpolate.interp1d(r_high, z_high, bounds_error=False, fill_value=numpy.min(z_high))

        #pylab.figure()
        #r_interp = numpy.linspace(r_0, r_1, 10000)
        #pylab.scatter(r_full, z_full, c=theta_0_full, edgecolors='none')
        #pylab.plot(r_interp, f_low(r_interp), c='blue')
        #pylab.plot(r_interp, f_high(r_interp), c='red')
        #pylab.scatter(r, z, c='black', edgecolors='none')

        theta_0_unique = numpy.unique(theta_0_full)
        for ii in [0, -1]:
            # Only one of these solutions will be the right one, but which?
            cut_theta_0 = theta_0_full == theta_0_unique[ii]
            r_cut_theta_0 = r_full[cut_theta_0]
            z_cut_theta_0 = z_full[cut_theta_0]
            cut_select = z_cut_theta_0 > f_low(r_cut_theta_0)
            r_select = r_cut_theta_0[cut_select]
            z_select = z_cut_theta_0[cut_select]
            
            if numpy.all(r_select < 1.):
                continue

            if numpy.mean(numpy.fabs(z_select - f_low(r_cut_theta_0)[cut_select])) \
                          > numpy.mean(numpy.fabs(z_select - f_high(r_cut_theta_0)[cut_select])):
                continue
            
            #pylab.scatter(r_select, z_select, c='gray', edgecolors='none')
            #if numpy.sum(cut_select) > 0:
            #    print ii, numpy.mean(z_select - f_low(r_cut_theta_0)[cut_select])
            
            if numpy.any(cut_select):
                r_low_final = numpy.concatenate([f_low.x, r_select])
                z_low_final = numpy.concatenate([f_low.y, z_select])
                index_sort = numpy.argsort(r_low_final)
                f_low = scipy.interpolate.interp1d(r_low_final[index_sort], z_low_final[index_sort], 
                                                   bounds_error=False, fill_value=numpy.max(z_low))
                #pylab.plot(r_interp, f_low(r_interp), c='green')

        return f_low, f_high
            
    def intersect(self, dic):
        """
        Find intersection between rays to separate the "direct" and "cross" solutions.
        """
        select_cross = []
        theta_0_unique = numpy.unique(dic['theta_0'])
        r_intersect = []
        z_intersect = []
        for ii in range(0, len(theta_0_unique) - 1):
            for jj in range(ii + 1, len(theta_0_unique)):
                cut_1 = numpy.logical_and(dic['theta_0'] == theta_0_unique[ii], dic['z'] < 1.)
                cut_2 = numpy.logical_and(dic['theta_0'] == theta_0_unique[jj], dic['z'] < 1.)
                if numpy.fabs(dic['r'][cut_1][0] - dic['r'][cut_1][-1]) < 1. \
                   or numpy.fabs(dic['r'][cut_2][0] - dic['r'][cut_2][-1]) < 1.:
                    continue
                r_min = max(numpy.min(dic['r'][cut_1]), numpy.min(dic['r'][cut_2]))
                r_max = min(numpy.max(dic['r'][cut_1]), numpy.max(dic['r'][cut_2]))
                #r = numpy.arange(r_min, r_max, 0.1)
                if (r_max - r_min) < 1.:
                    continue
                r = numpy.linspace(r_min, r_max, 10000)
                #if len(r) < 2:
                #    continue

                f_1 = scipy.interpolate.interp1d(dic['r'][cut_1], dic['z'][cut_1])
                f_2 = scipy.interpolate.interp1d(dic['r'][cut_2], dic['z'][cut_2])
                diff = f_1(r) - f_2(r)
                
                if diff[1] * diff[-1] < 0.:
                    index = numpy.nonzero(diff[1] * diff < 0.)[0][0]
                    r_intersect.append(r[index])
                    z_intersect.append(f_1(r[index]))
                    
                    selection_1 = numpy.logical_and(dic['theta_0'] == theta_0_unique[ii], dic['r'] > r_intersect[-1])
                    selection_2 = numpy.logical_and(dic['theta_0'] == theta_0_unique[jj], dic['r'] > r_intersect[-1])
                    
                    if dic['theta'][selection_1][0] > dic['theta'][selection_2][0]:                                                                       
                        select_cross.append(selection_1)                                                                                               
                    else:
                        select_cross.append(selection_2)

        r_intersect = numpy.array(r_intersect)
        z_intersect = numpy.array(z_intersect)
        dic_direct = {}
        dic_cross = {}
        if len(r_intersect) == 0:
            # No intersections found, so do nothing
            dic_direct = dic
            dic_cross = {}
            for key in self.keys:
                dic_cross[key] = numpy.array([])
        else:
            # Intersections found, so partition the traces
            select_cross = numpy.any(select_cross, axis=0)
            
            # Apply cut for to select cross points
            for key in self.keys:
                dic_cross[key] = dic[key][select_cross]
            # Then select direct points
            for key in self.keys:
                dic_direct[key] = dic[key][numpy.logical_not(select_cross)]

        #pylab.figure()
        #pylab.scatter(dic_direct['r'], dic_direct['z'], c='blue', edgecolors='none')
        #pylab.scatter(dic_cross['r'], dic_cross['z'], c='red', edgecolors='none')
        #pylab.scatter(r_intersect, z_intersect, c='black', edgecolors='none')

        return dic_direct, dic_cross
    
    def plot(self, field, solution, cmap='summer'):
        """
        field (t, d, theta, theta_0, a_v, a_h)
        mode (direct, crossover, reflect)
        """

        colorbar_dict = {'t': 'Time (ns)',
                         'd': 'Distance (m)',
                         'theta': 'Zenith Angle (deg)',
                         'theta_0': 'Zenith Angle at Antenna (deg)',
                         'a_v': 'VPOL Attenuation (dB)',
                         'a_h': 'HPOL Attenuation (dB)'}

        title = solution
        #dic = self.data[solution]
        #dic['a_v'] = gnosim.utils.rf.decibel(dic['a_v'])
        #dic['a_h'] = gnosim.utils.rf.decibel(dic['a_h'])
        
        pylab.figure()
        if field in ['a_v', 'a_h']:
            pylab.scatter(self.data[solution]['r'], self.data[solution]['z'], c=gnosim.utils.rf.decibel(self.data[solution][field]), 
                          edgecolors='none', cmap=cmap) # cmap='autumn'
        else:
            pylab.scatter(self.data[solution]['r'], self.data[solution]['z'], c=self.data[solution][field],
                          edgecolors='none', cmap=cmap) # cmap='autumn'
        colorbar = pylab.colorbar()
        colorbar.set_label(colorbar_dict[field])

        r_interp = numpy.arange(numpy.min(self.data[solution]['r']), numpy.max(self.data[solution]['r']), 0.1)
        pylab.plot(r_interp, self.envelope[solution]['low'](r_interp), color='black', linestyle='--')
        pylab.plot(r_interp, self.envelope[solution]['high'](r_interp), color='black', linestyle='--')
        
        #pylab.plot(hull_r, hull_z, )
        #pylab.gca().add_patch(pylab.Polygon(zip(hull_r, hull_z), closed=True, fill=False, linestyle='dashed', color='black'))

        #pylab.scatter(self.r_intersect, self.z_intersect, c='red')

        pylab.title(title)
        pylab.xlabel('Radius (m)')
        pylab.ylabel('Elevation (m)')

    def plotRays(self, s=10):
        r = []
        z = []
        theta_0 = []
        theta = []
        #for dic in [self.direct, self.crossover, self.reflect]:
        for solution in self.solutions:
            dic = self.data[solution]
            r.append(dic['r'])
            z.append(dic['z'])
            theta_0.append(dic['theta_0'])
            theta.append(dic['theta'])
        r = numpy.concatenate(r)
        z = numpy.concatenate(z)
        theta_0 = numpy.concatenate(theta_0)
        theta = numpy.concatenate(theta)

        pylab.figure()
        pylab.scatter(r, z, c=theta_0, s=s, edgecolors='none', marker='.')
        pylab.xlabel('Radius (m)')
        pylab.ylabel('Elevation (m)')
        colorbar = pylab.colorbar()
        colorbar.set_label('Zenith Angle Antenna (deg)')
        #pylab.ylim([-3000., 100.])

        pylab.figure()  
        pylab.scatter(r, z, c=theta, s=s, edgecolors='none', marker='.')
        pylab.xlabel('Radius (m)')
        pylab.ylabel('Elevation (m)')
        colorbar = pylab.colorbar()
        colorbar.set_label('Zenith Angle Ray (deg)')
        #pylab.ylim([-3000., 100.])
        
############################################################

if __name__ == '__main__':
    z_0 = -100. # -2, -30, -100, 0, 10, 1000, 5000, 6000, 38000
    #library_dir = 'library_-100_deep'
    #library_dir = 'library_-30_deep'
    #library_dir = 'library_-2_deep'
    #library_dir = 'library_0_ross'
    #library_dir = 'library_1000_ross'
    #library_dir = 'library_5000_ross'
    #library_dir = 'library_38000_deep'
    #library_dir = 'library_38000_mid'
    #library_dir = 'library_-1500_arthern'
    #library_dir = 'library_-100_arthern'
    #library_dir = 'library_-30_arthern'
    #library_dir = 'library_-2_arthern'
    #library_dir = 'library_6000_ross'
    #library_dir = 'library_6000_mid'
    #library_dir = 'library_-30_arthern_steph'
    #library_dir = 'library_-75_arthern_steph'
    library_dir = 'library_-100_arthern_steph'
    print 'library dir = %s'%(library_dir)
    print 'z_0 = %.2f'%(z_0)
    print 'ice model = %s'%(gnosim.earth.greenland.ice_model_default)
   
    #theta_array = numpy.degrees(numpy.arccos(numpy.linspace(-1, 0, 20)))
    #theta_array = numpy.linspace(10., 170., 20)
    #theta_array = numpy.linspace(0., 180., 60) # 60, THIS IS THE USUAL FOR SUBTERRANEAN CONFIGURATIONS
    #theta_array = numpy.linspace(90.01, 180., 30) # TESTING ROSS ICE SHELF AND LIGHTPOST IDEA
    theta_array = 90. - numpy.array([10., 20., 30., 40.]) # For Steph, switching from elevation angle to zenith angle

    """
    # WANT TO TAKE INTO ACCOUNT CURVATURE OF THE EARTH HERE
    #theta_array = numpy.degrees(numpy.arctan2(numpy.linspace(0., 50000., 120), -1000.)) # MINNA BLUFF
    theta_horizon = 90. + gnosim.earth.earth.horizon(z_0)[1] + 1.e-3 # deg
    x_horizon = -1. * z_0 * numpy.tan(numpy.radians(theta_horizon)) # m
    #theta_array = numpy.degrees(numpy.arctan2(numpy.linspace(0., x_horizon, 120), -1. * z_0)) # MINNA BLUFF
    theta_array = numpy.degrees(numpy.arctan2(numpy.linspace(0., x_horizon, 120), -1. * z_0)) # MINNA BLUFF
    """

    # FULLY ACCOUNTING FOR EARTH CURVATURE, USE THIS FOR HIGH-ALTITUDE CONFIGURATIONS
    #theta_array = gnosim.earth.earth.curvatureToTheta(z_0, numpy.linspace(0., gnosim.earth.earth.horizon(z_0)[2] - 1., 60)) # 30, 60
    #print theta_array
    #import sys
    #sys.exit('DONE')

    #theta_array = numpy.linspace(80., 100., 20)
    #theta_array = numpy.array([68.9473684211])
    #theta_array = numpy.array([30.])
    
    makeLibrary(z_0, theta_array, save=True, library_dir=library_dir)

############################################################


