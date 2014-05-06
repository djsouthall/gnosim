import glob
import numpy
import scipy.interpolate
import scipy.spatial
import h5py
import pylab

import gnosim.utils.constants
import gnosim.utils.rf
import gnosim.utils.plane
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

def rayTrace(origin, phi_0, theta_0, t_max=10000., t_step=1.): # t_max=40000, t_max=1000 (testing)
    """
    z_0 = initial elevation (m)
    t_max = max time (ns)
    t_step = time step (ns)
    """
    n_steps =  int(t_max / t_step)

    t_array = numpy.arange(0, t_step * (n_steps + 1), t_step) # Time (ns)
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
    reflection = False

    x_array[0] = origin[0]
    y_array[0] = origin[1]
    z_array[0] = origin[2]
    theta_array[0] = theta_0
    
    for ii in range(0, n_steps):

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
            a_v_array[ii + 1] = numpy.exp(-1. * d_step / gnosim.earth.greenland.attenuationLength(z_array[ii], 0.3)) # Transmission 
            a_h_array[ii + 1] = numpy.exp(-1. * d_step / gnosim.earth.greenland.attenuationLength(z_array[ii], 0.3)) # Transmission

        x_array[ii + 1] = x_array[ii] + x_step
        y_array[ii + 1] = y_array[ii] + y_step
        z_array[ii + 1] = z_array[ii] + z_step

        # Test whether the ray is refracted from upward going to downward going
        value = (gnosim.earth.greenland.indexOfRefraction(z_array[ii]) / gnosim.earth.greenland.indexOfRefraction(z_array[ii + 1])) \
                    * numpy.sin(numpy.radians(theta_array[ii]))
        
        # Hit ice-air interface at surface?
        delta_index_of_refraction = gnosim.earth.greenland.indexOfRefraction(z_array[ii + 1]) - gnosim.earth.greenland.indexOfRefraction(z_array[ii])
        
        # Ray going from ice to air
        if delta_index_of_refraction < -0.1 and theta_array[ii] <= 90.:
            # Compute reflection coefficients (power which is reflected)
            incidence_angle = theta_array[ii]
            r_s, r_p = fresnel(gnosim.earth.greenland.indexOfRefraction(z_array[ii]),
                               gnosim.earth.greenland.indexOfRefraction(z_array[ii + 1]),
                               incidence_angle, mode='reflection')
            a_v_array[ii + 1] *= numpy.sqrt(r_p)
            a_h_array[ii + 1] *= numpy.sqrt(r_s)
            reflection = True

        # Ray going from air to ice
        if delta_index_of_refraction > 0.1 and theta_array[ii] >= 90.:
            # Compute reflection coefficients (power which is transmitted)
            incidence_angle = 180. - theta_array[ii]
            t_s, t_p = fresnel(gnosim.earth.greenland.indexOfRefraction(z_array[ii]),
                               gnosim.earth.greenland.indexOfRefraction(z_array[ii + 1]),
                               incidence_angle, mode='transmission')

            #print 'GOING INTO ICE', gnosim.earth.greenland.indexOfRefraction(z_array[ii]), gnosim.earth.greenland.indexOfRefraction(z_array[ii + 1]), \
            #    incidence_angle, t_s, t_p

            a_v_array[ii + 1] *= numpy.sqrt(t_p)
            a_h_array[ii + 1] *= numpy.sqrt(t_s)
            #reflection = True

        #if value >= 1. or numpy.fabs(delta_index_of_refraction) > 0.1: #ORIGINAL
        if value >= 1. or delta_index_of_refraction < -0.1:
            # Reflection
            theta_array[ii + 1] = 180. - theta_array[ii]
        else:
            # Refraction
            theta_array[ii + 1] = numpy.degrees(numpy.arcsin(value))

        # Ensure that downward-going rays continue going down 
        if theta_array[ii] > 90.:
            theta_array[ii + 1] = 180. - theta_array[ii + 1]

    # Convert to total distance
    d_array = numpy.cumsum(d_array)

    # Convert from transmission at each step to cumulative transmission
    a_v_array = numpy.cumprod(a_v_array)
    a_h_array = numpy.cumprod(a_h_array)

    return x_array, y_array, z_array, t_array, d_array, phi_array, theta_array, a_v_array, a_h_array, reflection

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
        print theta_ray_array[ii]
        x, y, z, t, d, phi, theta, a_v, a_h, reflection = rayTrace([x_0, y_0, z_0], phi_0, theta_ray_array[ii])
        x_array.append(x)
        y_array.append(y)
        z_array.append(z)
        t_array.append(t)
        d_array.append(d)
        phi_array.append(phi)
        theta_array.append(theta)
        a_v_array.append(a_v)
        a_h_array.append(a_h)
        
        # Count as reflection if initially upward going
        #if theta[0] < 90.:
        #    reflection = True

        if save:
            n_points = len(t)
            outfile = '%s/z0_%.2f_theta_%.2f_n_%i.h5'%(library_dir, z_0, theta_ray_array[ii], n_points)
            file = h5py.File(outfile, 'w')
            
            file.attrs['reflection'] = reflection
            file.attrs['z_0'] = z_0
            file.attrs['theta_0'] = theta_ray_array[ii]

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
        self.keys = ['r', 'z', 't', 'd', 'theta', 'theta_0', 'a_v', 'a_h']

        # Dictionaries to store data
        self.direct = {}
        self.crossover = {}
        self.reflect ={}
        for key in self.keys:
            self.direct[key] = []
            self.crossover[key] = []
            self.reflect[key] = []

        for infile in self.infiles:
            print infile
            reader = h5py.File(infile, 'r')
            if reader.attrs['reflection']:
                # Rays with reflections off ice-air interface
                cut = reader['theta'][...] < 90.
                for key in self.keys:
                    # Upward-going rays from the antenna towards surface
                    self.direct[key].append(reader[key][cut])
                    # Downward-going rays from the surface to deep ice
                    self.reflect[key].append(reader[key][numpy.logical_not(cut)])
            else:
                # Rays which only refract
                for key in self.keys:
                    self.direct[key].append(reader[key][...])

            reader.close()

        for key in self.keys:
            self.direct[key] = numpy.concatenate(self.direct[key])
            if len(self.reflect[key]) > 0:
                self.reflect[key] = numpy.concatenate(self.reflect[key])
        
        print len(self.direct['t'])
        print len(self.crossover['t'])
        print len(self.reflect['t'])

        self.intersect()
        
        self.exists_direct = len(self.direct['t']) > 0
        self.exists_crossover = len(self.crossover['t']) > 0
        self.exists_reflect = len(self.reflect['t']) > 0

        print self.direct['a_v']
        raw_input('WAIT')

        # Trim
        print 'BEFORE', len(self.direct['t'])
        #self.direct = self.trim(self.direct)
        print 'AFTER', len(self.direct['t'])
        #self.crossover = self.trim(self.crossover)
        print 'AFTER', len(self.crossover['t'])
        #self.reflect = self.trim(self.reflect)
        print 'AFTER', len(self.reflect['t'])
        
        if self.exists_direct:
            self.hull_direct_r, self.hull_direct_z = self.makeHull(self.direct)
            self.envelope_direct_low, self.envelope_direct_high = self.makeEnvelope(self.hull_direct_r, self.hull_direct_z)
        if self.exists_crossover:
            self.hull_crossover_r, self.hull_crossover_z = self.makeHull(self.crossover)
            self.envelope_crossover_low, self.envelope_crossover_high = self.makeEnvelope(self.hull_crossover_r, self.hull_crossover_z,
                                                                                          self.crossover['r'], self.crossover['z'], 
                                                                                          self.crossover['theta_0'])
        if self.exists_reflect:
            self.hull_reflect_r, self.hull_reflect_z = self.makeHull(self.reflect)
            self.envelope_reflect_low, self.envelope_reflect_high = self.makeEnvelope(self.hull_reflect_r, self.hull_reflect_z)
        

    def makeHull(self, dic):
        hull = scipy.spatial.ConvexHull(zip(dic['r'], dic['z']))
        return dic['r'][hull.vertices], dic['z'][hull.vertices]

    def getValue(self, dic, r, z):
        distance = numpy.sqrt((r - dic['r'])**2 + (z - dic['z'])**2)
        index_1 = numpy.argmin(distance)
        weight_1 = distance[index_1]**(-1)
        #val_1 = dic[field][index]
        distance[dic['theta_0'] == dic['theta_0'][index_1]] = 1.e10
        index_2 = numpy.argmin(distance)
        weight_2 = distance[index_2]**(-1)
        #val_2 = dic[field][index]
        #val_simple = ((weight_1 * val_1) + (weight_2 * val_2)) / (weight_1 + weight_2)
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
        if self.exists_direct:
            flag_direct = numpy.logical_and(z_query > self.envelope_direct_low(r_query), z_query < self.envelope_direct_high(r_query))
        else:
            flag_direct = False
        if self.exists_crossover:
            flag_crossover = numpy.logical_and(z_query > self.envelope_crossover_low(r_query), z_query < self.envelope_crossover_high(r_query))
        else:
            flag_crossover = False
        if self.exists_reflect:
            flag_reflect = numpy.logical_and(z_query > self.envelope_reflect_low(r_query), z_query < self.envelope_reflect_high(r_query))
        else:
            flag_reflect = False
        return flag_direct, flag_crossover, flag_reflect

    def event(self, r, z):
        flag_direct, flag_crossover, flag_reflect = self.query(r, z)

        if flag_direct:
            dic_direct = self.getValue(self.direct, r, z)
        else:
            dic_direct = {}

        if flag_crossover:
            dic_crossover = self.getValue(self.crossover, r, z)
        else:
            dic_crossover = {}

        if flag_reflect:
            dic_reflect = self.getValue(self.reflect, r, z)
        else:
            dic_reflect = {}

        return flag_direct, flag_crossover, flag_reflect, dic_direct, dic_crossover, dic_reflect

    def trim(self, dic):
        """
        Create a reduced dictionary which only includes the necessary points. Currently not used.
        """
        theta_0_unique = numpy.unique(dic['theta_0'])
        #pylab.figure()
        index_array = []
        for ii in range(0, len(theta_0_unique)):
            # Select only the points along a given ray
            cut = dic['theta_0'] == theta_0_unique[ii]
            bins = numpy.arange(numpy.min(dic['theta'][cut]), numpy.max(dic['theta'][cut]), 0.2)
            index = numpy.nonzero(numpy.fabs(numpy.diff(numpy.digitize(dic['theta'][cut], bins=bins))) > 0)[0]
            if 0 not in index:
                index = numpy.insert(index, 0, 0)
            if numpy.sum(cut) - 1 not in index:
                index = numpy.append(index, numpy.sum(cut) - 1)
            
            r = dic['r'][cut][index]
            z = dic['z'][cut][index]

            #pylab.clf()
            #pylab.plot(dic['r'][cut], dic['z'][cut], c='black')
            #pylab.scatter(r, z)
            #print '%i %i %i'%(len(index), numpy.sum(cut), len(bins))
            #raw_input('%i %i %i'%(len(index), numpy.sum(cut), len(bins)))
            
            index_array.append(numpy.nonzero(cut)[0][index])

        index_array = numpy.concatenate(index_array)
        new_dic = {}
        for key in dic.keys():
            new_dic[key] = dic[key][index_array]
        return new_dic

    def test(self, field, n_trials=10):
        r_min, r_max = numpy.min(self.direct['r']), numpy.max(self.direct['r'])
        z_min, z_max = numpy.min(self.direct['z']), numpy.max(self.direct['z'])

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
            theta_0_unique = numpy.unique(self.direct['theta_0'])
            distance_array = numpy.zeros(len(theta_0_unique))
            val_array = numpy.zeros(len(theta_0_unique))
            for jj in range(0, len(theta_0_unique)):
                cut = (self.direct['theta_0'] == theta_0_unique[jj])
                r_cut = self.direct['r'][cut]
                z_cut = self.direct['z'][cut]
                val_cut = self.direct[field][cut]
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
            distance = numpy.sqrt((r[ii] - self.direct['r'])**2 + (z[ii] - self.direct['z'])**2)
            index = numpy.argmin(distance)
            weight_1 = distance[index]**(-1)
            val_1 = self.direct[field][index]
            distance[self.direct['theta_0'] == self.direct['theta_0'][index]] = 1.e10
            index = numpy.argmin(distance)
            weight_2 = distance[index]**(-1)
            val_2 = self.direct[field][index]
            val_simple = ((weight_1 * val_1) + (weight_2 * val_2)) / (weight_1 + weight_2)
            # End simple procedure fast
            """
            
            val_dic = self.getValue(self.direct, r[ii], z[ii])
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
                cut_1 = r[ii] < self.direct['r']
                cut_2 = z[ii] < self.direct['z']
                cut = numpy.logical_and(cut_1 == condition_1, cut_2 == condition_2)
                if not numpy.any(cut):
                    print 'WARNING'

                r_cut = self.direct['r'][cut]
                z_cut = self.direct['z'][cut]
                theta_0_cut = self.direct['theta_0'][cut]
                val_cut = self.direct[field][cut]
                distance = numpy.sqrt((r[ii] - r_cut)**2 + (z[ii] - z_cut)**2)
                index = numpy.argsort(distance)
                index_max = min(100, numpy.sum(cut))
                r_interp_array.append(r_cut[index[0:index_max]])
                z_interp_array.append(z_cut[index[0:index_max]])
                theta_0_interp_array.append(theta_0_cut[index[0:index_max]])
                val_interp_array.append(val_cut[index[0:index_max]])

                #print ii, numpy.sum(cut), len(self.direct['r'])

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
        pylab.scatter(self.direct['r'], self.direct['z'], c=self.direct[field], edgecolors='none')
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
        
        if self.exists_direct:
            r_direct = numpy.linspace(numpy.min(self.hull_direct_r), numpy.max(self.hull_direct_r), 10000)
            pylab.plot(r_direct, self.envelope_direct_low(r_direct), c='red', label='Direct Rays')
            pylab.plot(r_direct, self.envelope_direct_high(r_direct), c='red')
            pylab.scatter(self.hull_direct_r, self.hull_direct_z, c='red', edgecolors='none')

        if self.exists_crossover:
            r_crossover = numpy.linspace(numpy.min(self.hull_crossover_r), numpy.max(self.hull_crossover_r), 10000)
            pylab.plot(r_crossover, self.envelope_crossover_low(r_crossover), c='green', label='Crossover Rays')
            pylab.plot(r_crossover, self.envelope_crossover_high(r_crossover), c='green')
            pylab.scatter(self.hull_crossover_r, self.hull_crossover_z, c='green', edgecolors='none')

        if self.exists_reflect:
            r_reflect = numpy.linspace(numpy.min(self.hull_reflect_r), numpy.max(self.hull_reflect_r), 10000)
            pylab.plot(r_reflect, self.envelope_reflect_low(r_reflect), c='blue', label='Reflected Rays')
            pylab.plot(r_reflect, self.envelope_reflect_high(r_reflect), c='blue')
            pylab.scatter(self.hull_reflect_r, self.hull_reflect_z, c='blue', edgecolors='none')

        pylab.legend(loc='upper right')
        
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
        
        # If region is expected to be concave in shape examine all points, not only convex hull points
        if r_full is not None and z_full is not None and theta_0_full is not None:
            r_interp = numpy.linspace(r_0, r_1, 10000)
            
            theta_0_unique = numpy.unique(theta_0_full)
            f_array = []
            z_array = []
            for ii in range(0, len(theta_0_unique)):
                cut = theta_0_full == theta_0_unique[ii]
                f_array.append(scipy.interpolate.interp1d(r_full[cut], z_full[cut], bounds_error=False, fill_value=numpy.max(z_full))) 
                z_array.append(f_array[-1](r_interp))

            z_interp = numpy.min(numpy.array(z_array), axis=0)
            f_theta_0 = scipy.interpolate.interp1d(r_interp, z_interp)
            
            r_minimum = r_full[numpy.argmin(z_full)]
            z_final = numpy.concatenate([numpy.max([f_low(r_interp),
                                                    f_theta_0(r_interp)], axis=0)[r_interp < r_minimum],
                                         numpy.min([f_low(r_interp),
                                                    f_theta_0(r_interp)], axis=0)[r_interp > r_minimum]])

            f_low = scipy.interpolate.interp1d(r_interp, z_final, bounds_error=False, fill_value=numpy.max(z_low))
        
        return f_low, f_high
            
    def intersect(self):
        """
        Find intersection between rays to separate the "direct" and "crossover" solutions.
        """

        select_crossover = []

        theta_0_unique = numpy.unique(self.direct['theta_0'])
        r_intersect = []
        z_intersect = []
        for ii in range(0, len(theta_0_unique) - 1):
            for jj in range(ii + 1, len(theta_0_unique)):
                cut_1 = self.direct['theta_0'] == theta_0_unique[ii]
                cut_2 = self.direct['theta_0'] == theta_0_unique[jj]
                r_max = min(numpy.max(self.direct['r'][cut_1]), numpy.max(self.direct['r'][cut_2]))
                r = numpy.arange(0., r_max, 1.)
                if len(r) < 2:
                    continue
                
                f_1 = scipy.interpolate.interp1d(self.direct['r'][cut_1], self.direct['z'][cut_1])
                f_2 = scipy.interpolate.interp1d(self.direct['r'][cut_2], self.direct['z'][cut_2])
                diff = f_1(r) - f_2(r)
                
                if diff[1] * diff[-1] < 0.:
                    index = numpy.nonzero(diff[1] * diff < 0.)[0][0]
                    r_intersect.append(r[index])
                    z_intersect.append(f_1(r[index]))
                    
                    selection_1 = numpy.logical_and(self.direct['theta_0'] == theta_0_unique[ii], self.direct['r'] > r_intersect[-1])
                    selection_2 = numpy.logical_and(self.direct['theta_0'] == theta_0_unique[jj], self.direct['r'] > r_intersect[-1])
                    
                    if self.direct['theta'][selection_1][0] > self.direct['theta'][selection_2][0]:                                                                       
                        select_crossover.append(selection_1)                                                                                               
                    else:
                        select_crossover.append(selection_2)                                                                                               
        
        self.r_intersect = numpy.array(r_intersect)
        self.z_intersect = numpy.array(z_intersect)

        if len(self.r_intersect) == 0:
            # No intersections found, so do nothing
            pass
        else:
            # Intersections found, so partition the traces
            select_crossover = numpy.any(select_crossover, axis=0)

            # Apply cut for to select crossover points
            for key in self.keys:
                self.crossover[key] = self.direct[key][select_crossover]
            # Then to select direct points
            for key in self.keys:
                self.direct[key] = self.direct[key][numpy.logical_not(select_crossover)]
        
    
    def plot(self, field, mode, cmap='summer'):
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

        if mode == 'direct':
            title = 'Direct Rays'
            dic = self.direct
            envelope_low = self.envelope_direct_low
            envelope_high = self.envelope_direct_high
            hull_r, hull_z = self.hull_direct_r, self.hull_direct_z
        elif mode == 'crossover':
            title = 'Crossover Rays'
            dic = self.crossover
            envelope_low = self.envelope_crossover_low
            envelope_high = self.envelope_crossover_high
            hull_r, hull_z = self.hull_crossover_r, self.hull_crossover_z
        elif mode == 'reflect':
            title = 'Reflected Rays'
            dic = self.reflect
            envelope_low = self.envelope_reflect_low
            envelope_high = self.envelope_reflect_high
            hull_r, hull_z = self.hull_reflect_r, self.hull_reflect_z
        else:
            print 'WARNING!!'

        dic['a_v'] = gnosim.utils.rf.decibel(dic['a_v'])
        dic['a_h'] = gnosim.utils.rf.decibel(dic['a_h'])
        
        pylab.figure()
        pylab.scatter(dic['r'], dic['z'], c=dic[field], edgecolors='none', cmap=cmap) # cmap='autumn'
        colorbar = pylab.colorbar()
        colorbar.set_label(colorbar_dict[field])

        r_interp = numpy.arange(numpy.min(dic['r']), numpy.max(dic['r']), 0.1)
        pylab.plot(r_interp, envelope_low(r_interp), color='black', linestyle='--')
        pylab.plot(r_interp, envelope_high(r_interp), color='black', linestyle='--')
        
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
        for dic in [self.direct, self.crossover, self.reflect]:
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
        pylab.ylim([-3000., 100.])

        pylab.figure()  
        pylab.scatter(r, z, c=theta, s=s, edgecolors='none', marker='.')
        pylab.xlabel('Radius (m)')
        pylab.ylabel('Elevation (m)')
        colorbar = pylab.colorbar()
        colorbar.set_label('Zenith Angle Ray (deg)')
        pylab.ylim([-3000., 100.])

    def interp(self, field, r, z):
        pass
        
############################################################

if __name__ == '__main__':
    z_0 = -2. # -2, -30, -100, 10
    #theta_array = numpy.degrees(numpy.arccos(numpy.linspace(-1, 0, 20)))
    #theta_array = numpy.linspace(10., 170., 20)
    theta_array = numpy.linspace(0., 180., 60) # 30 THIS IS THE USUAL
    #theta_array = numpy.linspace(91., 180., 30) # Testing lightpost idea
    #theta_array = numpy.linspace(80., 100., 20)
    #theta_array = numpy.array([68.9473684211])
    #theta_array = numpy.array([30.])
    makeLibrary(z_0, theta_array, save=True, library_dir='library_-2_empirical')

############################################################
# CODE SCRAPS
############################################################

#def indexOfRefraction(z):
#    """
#    z = elevation (m)
#    """
#    n_infinity = 1.831
#    n_0 = 1.29
#    a = 10 # m
#    return (n_0 - (n_infinity * z / a)) / (1. - (z / a))

############################################################
