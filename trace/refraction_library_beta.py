#!/usr/bin/env python

#refractionlibrarynew
import sys
import os
import glob
import numpy
import scipy.interpolate
import scipy.spatial
import h5py
import pylab
import os
sys.path.append('/home/dsouthall/Projects/GNOSim/')
import gnosim.utils.constants
import gnosim.utils.rf
import gnosim.utils.plane
import gnosim.earth.earth
import gnosim.earth.ice

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
        print ('WARNING: mode %s not recognized'%(mode))

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
    input("press any key to exit")

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



def rayTrace(origin, phi_0, theta_ant, ice, t_max=50000., t_step=1., r_limit = None): # t_max=40000, t_max=1000 (testing)
    """
    z_0 = initial elevation (m)
    t_max = max time (ns)
    t_step = time step (ns)
    """
    # IDEA: let t_step grow when the index of refraction isn't changing fast, i.e.,
    #       a dynamic t_step 

    # ORIGINAL
    t_step_in = t_step
    n_steps =  int(t_max / t_step)
    #print(n_steps,' with t_step ', t_step)
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
    theta_array[0] = theta_ant

    index_reflect = 0 # Latch to prevent multiple reflections at interface
    index_reflect_air = 0 # Stores index of reflection event at ice-air interface
    index_reflect_water = 0 # Stores index of reflection event at ice-water interface
    max_ii = None
    #Finding surfaces:
    z_locate = numpy.linspace(1000,-10000,10000000)
    dndz = numpy.divide(numpy.diff(ice.indexOfRefraction(z_locate)),numpy.diff(z_locate))
    z_upward_surface = ((z_locate[1:] + z_locate[:-1]) / 2.0)[numpy.equal(dndz,numpy.amin(dndz))]
    z_downward_surface = ((z_locate[1:] + z_locate[:-1]) / 2.0)[numpy.equal(dndz,numpy.amax(dndz))]
    dndz = None
    z_locate = None
    #print('What is going on')
    for ii in range(0, n_steps):
        #print('On Event',ii)
        # Dynamic time step depending on how fast the index of refraction is changing
        if ice.deltaIndexOfRefraction(z_array[ii]) > 1.e-4 or z_array[ii] >= 0.:
            t_step = 1.*t_step_in
            #print(t_step)
        else:
            t_step = 5.*t_step_in
            #print(t_step)
        
        
        
        potential_z_step = (t_step * gnosim.utils.constants.speed_light * numpy.cos(numpy.radians(theta_array[ii])) \
                      / ice.indexOfRefraction(z_array[ii]))
        if numpy.logical_or( (numpy.fabs(z_array[ii] - z_upward_surface) < 5.0),(numpy.fabs(z_array[ii] - z_downward_surface) < 5.0)):
            
            if (numpy.logical_or( (z_array[ii] + potential_z_step) > z_upward_surface , (z_array[ii] + potential_z_step) < z_downward_surface)):
                #print_new = True
                #print('Within 5m of a boundary at z_array[%i] = %0.2f'%(ii,z_array[ii]))
                t_step = t_array[ii] - t_array[ii-1]
                #print('Initial t_step is:', t_step)
            else:
                print_new = False
            while (numpy.logical_or( (z_array[ii] + potential_z_step) > z_upward_surface , (z_array[ii] + potential_z_step) < z_downward_surface)):
                t_step = t_step/2.0
                potential_z_step = (t_step * gnosim.utils.constants.speed_light * numpy.cos(numpy.radians(theta_array[ii])) \
                                   / ice.indexOfRefraction(z_array[ii]))
            #if print_new:
            #    print('t_step changed to' , t_step)
            
        
        t_array[ii + 1] = t_array[ii] + t_step

        if ii == 0 and z_array[ii] > 1.:
            # Rather than take many steps to the ice, take one large jump

            """
            # THIS VERSION WORKS, BUT DOES NOT FULLY TAKE INTO ACCOUNT EARTH CURVATURE
            r = (z_array[ii] + 0.01) * numpy.tan(numpy.radians(theta_array[ii])) # 0.01 meter into ground (i.e., below surface)
            d = numpy.sqrt(r**2 + z_array[ii]**2)
            t = d * ice.indexOfRefraction(z_array[ii]) / gnosim.utils.constants.speed_light

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
            t = distance_with_curvature * ice.indexOfRefraction(z_array[ii]) / gnosim.utils.constants.speed_light
            #t_array[1:] += (t - t_step)
            t_array[ii + 1] = t_array[ii] + t

            # Incidence angle is increased due to curvature of Earth
            theta_array[ii] = theta_array[ii] - angle_curvature
            if theta_array[ii] < 90. or numpy.isnan(theta_array[ii]):
                print ('WARNING: Ray does not intersect Earth')
                raw_input('CONTINUE?')
                theta_array[ii] = 90. + 1.e10
            
            # Distance corrected for curvature of the Earth
            d_step = distance_with_curvature
        else:
            x_step = (t_step * gnosim.utils.constants.speed_light * numpy.sin(numpy.radians(theta_array[ii])) * numpy.cos(numpy.radians(phi_array[ii])) \
                      / ice.indexOfRefraction(z_array[ii]))
            y_step = (t_step * gnosim.utils.constants.speed_light * numpy.sin(numpy.radians(theta_array[ii])) * numpy.sin(numpy.radians(phi_array[ii])) \
                      / ice.indexOfRefraction(z_array[ii]))
            z_step = (t_step * gnosim.utils.constants.speed_light * numpy.cos(numpy.radians(theta_array[ii])) \
                      / ice.indexOfRefraction(z_array[ii]))
            
            d_step = numpy.sqrt(x_step**2 + y_step**2 + z_step**2) # Distance traveled during step
        
        d_array[ii + 1] = d_step
        
        attenuation_length = ice.attenuationLength(z_array[ii], 0.3) # m, Transmission
        if attenuation_length > 1.e10:
            a_v_array[ii + 1] = 1.
            a_h_array[ii + 1] = 1.
        else:
            a_v_array[ii + 1] = numpy.exp(-1. * d_array[ii + 1] / ice.attenuationLength(z_array[ii], 0.3)) # Transmission 
            a_h_array[ii + 1] = numpy.exp(-1. * d_array[ii + 1] / ice.attenuationLength(z_array[ii], 0.3)) # Transmission

        x_array[ii + 1] = x_array[ii] + x_step
        y_array[ii + 1] = y_array[ii] + y_step
        z_array[ii + 1] = z_array[ii] + z_step
        if r_limit != None:
            if x_array[ii + 1]**2 + y_array[ii + 1]**2 > r_limit**2:
                max_ii = ii
                break

        # Test whether the ray is refracted from upward going to downward going
        value = (ice.indexOfRefraction(z_array[ii]) / ice.indexOfRefraction(z_array[ii + 1])) \
                    * numpy.sin(numpy.radians(theta_array[ii]))
        
        # Hit ice-air interface at surface?
        delta_index_of_refraction = ice.indexOfRefraction(z_array[ii + 1]) - ice.indexOfRefraction(z_array[ii])
        
        if ii == index_reflect and ii > 0:
            #print ('SKIP', ii)
            theta_array[ii + 1] = theta_array[ii]
            index_reflect = 0
        elif delta_index_of_refraction < -0.1 and theta_array[ii] < 90.:
            # Ray going from ice to air (for a ray going backwards from antenna to interaction vertex)
            # Compute reflection coefficients (power which is reflected)
            incidence_angle = theta_array[ii]
            r_s, r_p = fresnel(ice.indexOfRefraction(z_array[ii]),
                               ice.indexOfRefraction(z_array[ii + 1]),
                               incidence_angle, mode='reflection')
            a_v_array[ii + 1] *= numpy.sqrt(r_p)
            a_h_array[ii + 1] *= numpy.sqrt(r_s)
            #print ('ICE -> AIR', ii, r_p, r_s)
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
            t_s, t_p = fresnel(ice.indexOfRefraction(z_array[ii + 1]),
                               ice.indexOfRefraction(z_array[ii]),
                               incidence_angle, mode='transmission')
            a_v_array[ii + 1] *= numpy.sqrt(t_p)
            a_h_array[ii + 1] *= numpy.sqrt(t_s)
            #print ('AIR -> ICE', ii, t_p, t_s)
        elif delta_index_of_refraction < -0.1 and theta_array[ii] > 90.:
            # Ray going from ice to water
            # Compute reflection coefficients (power which is reflected)
            incidence_angle = 180. - theta_array[ii]
            r_s, r_p = 1., 1. # Water is conductor, so complete reflection
            #r_s, r_p = fresnel(ice.indexOfRefraction(z_array[ii]),
            #                   ice.indexOfRefraction(z_array[ii + 1]),
            #                   incidence_angle, mode='reflection')
            a_v_array[ii + 1] *= numpy.sqrt(r_p)
            a_h_array[ii + 1] *= numpy.sqrt(r_s)
            reflection_water = True
            #print ('ICE -> WATER', ii, r_p, r_s)
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
        #    #print (theta_array[ii], theta_array[ii + 1])
        #    #raw_input('WAIT')

        # Define a stop condition
        if index_reflect_water > 0 and theta_array[ii + 1] > 90.:
            n_steps = ii + 1
            #print ('STOP', n_steps)
            break

    # Convert to total distance
    d_array = numpy.cumsum(d_array)

    # Convert from transmission at each step to cumulative transmission
    a_v_array = numpy.cumprod(a_v_array)
    a_h_array = numpy.cumprod(a_h_array)
    if r_limit != None:
        if max_ii != None:
            n_steps = max_ii
                
    return (x_array[0: n_steps], y_array[0: n_steps], z_array[0: n_steps], \
        t_array[0: n_steps], d_array[0: n_steps], phi_array[0: n_steps], \
        theta_array[0: n_steps], a_v_array[0: n_steps], a_h_array[0: n_steps], \
        index_reflect_air, index_reflect_water)

def plotGeometry(config,neutrino_loc,phi_0,info,ice):
    '''
    origin shuold be of the form [[x_0, y_0, z_0],[x_1, y_1, z_1],[x_2, y_2, z_2]]
    neutrino_loc should be similar but for neutrino location but [x, y, z]
    
    '''
    '''
    if len(numpy.shape(origin)) == 1:
        origin = [origin] #this is sloppy and might break
    #might want to do some check here to make sure values are all floats
    '''
    if type(config) == dict:
        linestyle_dict = {'direct':(0, ()),'cross':(0, (3, 1, 1, 1)),'reflect':(0, (1, 1)),'direct_2':(0, (5, 1)),'cross_2':(0, (3, 5, 1, 5)),'reflect_2':(0, (1, 5))}
        neutrino_loc_r = numpy.sqrt(neutrino_loc[0]**2 + neutrino_loc[1]**2)
        if len(numpy.unique(info['eventid'])) == 1:
            eventid = numpy.unique(info['eventid'])[0]
            fig = pylab.figure(figsize=(16.,11.2)) #my screensize
            pylab.title('Event %i'%(eventid))
            sub_info = numpy.unique(info[info['eventid'] == eventid])
            for index_station in numpy.unique(sub_info['station']):
                station_cut = sub_info['station'] == index_station
                station_loc = numpy.array(config['stations']['positions'][index_station])
                colormap = pylab.cm.gist_ncar #nipy_spectral, Set1,Paired   
                antenna_colors = [colormap(i) for i in numpy.linspace(0, 1,len(numpy.unique(sub_info['antenna']))+1)]
                for i, index_antenna in enumerate(numpy.unique(sub_info['antenna'])):
                    antenna_cut = sub_info['antenna'] == index_antenna
                    #cut = numpy.logical_and(station_cut,antenna_cut)
                    antenna_loc = numpy.add(numpy.array(config['antennas']['positions'][index_antenna]),station_loc)
                    origin_r = numpy.sqrt(antenna_loc[0]**2 + antenna_loc[1]**2)
                    ssub_info = sub_info[numpy.logical_and(station_cut,antenna_cut)]
                    for index_solution, solution in enumerate(ssub_info['solution']):
                        x, y, z, t, d, phi, theta, a_v, a_h, index_reflect_air, index_reflect_water = rayTrace(antenna_loc, phi_0, ssub_info['theta_ant'][ssub_info['solution'] == solution],ice, r_limit = 1.001*neutrino_loc_r)
                        r = numpy.sqrt(x**2 + y**2)
                        label = 'S%iA%i %s'%(index_station,index_antenna,solution.decode())
                        if numpy.isin(solution.decode(),list(linestyle_dict.keys())):
                            style = linestyle_dict[solution.decode()]
                        else:
                            style = '-'
                        pylab.plot(r,z,label=label,color=antenna_colors[i],linestyle = style)
                    pylab.scatter(origin_r,antenna_loc[2],label='Antenna %i'%(index_antenna),color=antenna_colors[i],marker = 'd',s = 100)
            pylab.scatter(neutrino_loc_r,neutrino_loc[2],label='Neutrino Loc',marker = '*',color = 'k',s = 100)
            pylab.legend(fontsize = 14)
            pylab.xlabel('r(m)',fontsize=20)
            pylab.ylabel('z(m)',fontsize=20)
            ax = pylab.gca()
            ax.tick_params(axis = 'both',labelsize = 14)
            pylab.subplots_adjust(left = 0.07, bottom = 0.06, right = 0.97, top = 0.97, wspace = 0.20, hspace = 0.20)
            #fig.patch.set_alpha(0.)
            #ax.patch.set_alpha(0.)
            return fig
        else:
            print('Info should be input with only one event, otherwise this breaks')
            return
    else:
        print('Input \'config\' in plotGeometry is not a dict, breaking')
        return
############################################################

def makeLibrary(z_0, theta_ray_array, ice_model, save=True, library_dir='library',r_limit = None):
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
    
    theta_ant_array = []

    ice = gnosim.earth.ice.Ice(ice_model)

    for ii in range(0, len(theta_ray_array)):
        print ('(%i/%i) theta_ant = %.4f'%(ii, len(theta_ray_array), theta_ray_array[ii]))
        x, y, z, t, d, phi, theta, a_v, a_h, index_reflect_air, index_reflect_water = rayTrace([x_0, y_0, z_0], phi_0, theta_ray_array[ii],ice)
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
            file.attrs['theta_ant'] = theta_ray_array[ii]
            file.attrs['ice_model'] = ice.ice_model

            file.create_dataset('r', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('z', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('t', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('d', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('theta', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('theta_ant', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('a_v', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('a_h', (n_points,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
        
            file['r'][...] = x
            file['z'][...] = z
            file['t'][...] = t
            file['d'][...] = d
            file['theta'][...] = theta
            file['theta_ant'][...] = theta_ray_array[ii] * numpy.ones(n_points)
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
    def __init__(self, search, solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2']), pre_split = True):
        self.infiles = glob.glob(search)
        default_solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2'])
        # List attributes of interest
        self.solutions = default_solutions[numpy.isin(default_solutions,solutions)]
        if len(self.solutions) == 0:
            print('Selection of solution types did not match predefined values.  Using default types.')
            self.solutions = default_solutions
        elif numpy.logical_and(pre_split == False,numpy.logical_not(len(self.solutions) == len(default_solutions))):
            print('Limiting Solution Types Currently only works for pre_split = True, using default solution types.')
            self.solutions = default_solutions
            
        self.keys = ['r', 'z', 't', 'd', 'theta', 'theta_ant', 'a_v', 'a_h']

        # Dictionary to store data
        self.data = {}
        for solution in self.solutions:
            self.data[solution] = {}
            for key in self.keys:
                self.data[solution][key] = []

        # Ice model
        self.ice_model = None
        
        #Checking if pre_split is possible:
        if pre_split == True:
            for solution in self.solutions:
                if (os.path.isdir(search.replace('*.h5',solution + '/')) == False):
                    print('WARNING! No directory' , search.replace('*.h5',solution + '/'))
                    pre_split = False
                    self.solutions = default_solutions
            if pre_split == False:
                print('Cannot run pre_split library, running unsorted library')
            
        if (pre_split == False):
            # Open data files and begin sorting solutions
            for infile in self.infiles:
                print (infile)
                reader = h5py.File(infile, 'r')
                n = len(reader['t'])
                current_model = reader.attrs['ice_model']
                if current_model == 'arthern':
                    current_model = 'greenland'
                elif current_model == 'polar':
                    current_model = 'antarctica'

                if self.ice_model is None:
                    self.ice_model = current_model
                else:
                    if self.ice_model != current_model:
                        print ('WARNING: Ice models used in ray-tracing libraries do not match, e.g., %s != %s'%(self.ice_model, 
                                                                                                                current_model))

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
                print (solution, len(self.data[solution]['t']))

            print ('Intersections...')
            # Find intersections
            self.data['direct'], self.data['cross'] = self.intersect(self.data['direct'])
            #self.data['direct'] = dic_direct
            #self.data['cross'] = dic_cross

            print ('Sort solutions...')

            if len(self.data['direct_2']['t']) > 0:
                theta_ant_divide = self.data['direct']['theta_ant'][numpy.argmax(self.data['direct']['r'])]
                cut = self.data['direct_2']['theta_ant'] < theta_ant_divide
                for key in self.keys:
                    self.data['cross_2'][key] = self.data['direct_2'][key][cut]
                cut = numpy.logical_not(cut)
                for key in self.keys:
                    self.data['direct_2'][key] = self.data['direct_2'][key][cut]
            
        else:
            #Below is what happens if the solution types are already sorted into subfolders
            print('Loading data from pre split directories')
            for infile in self.infiles:
                #print (infile)
                for solution in self.solutions:
                    solutionfile = infile.replace(infile.split('/')[-1], solution + '/' + infile.split('/')[-1])
                    reader = h5py.File(solutionfile, 'r')
                    n = len(reader['t'])
                    if self.ice_model is None:
                        self.ice_model = reader.attrs['ice_model']
                    else:
                        if self.ice_model != reader.attrs['ice_model']:
                            print ('WARNING: Ice models used in ray-tracing libraries do not match, e.g., %s != %s'%(self.ice_model, reader.attrs['ice_model']))
                    for key in self.keys:
                        self.data[solution][key] = numpy.append( self.data[solution][key] , reader[key][...] )
            

    def saveEnvelope(self, out_dir, solution_list = None,verbose = False, plot_hulls = False):
        '''
        Should save the necessary information for a libraries hull so it can be
        used to create a hull later
        
        out_dir should be of the form '/dir1/dir2'
        i.e. it shouldn't end with a /
        '''
        out_dir = out_dir + '/concave_hull'
        os.mkdir(out_dir)
        legend_locs = {'direct':'upper right','cross':'upper right','reflect':'upper right','direct_2':'lower right','cross_2':'lower right','reflect_2':'lower right'}        
        concave_hull = {'direct':{'n_bins':1000},'cross':{'n_bins':1500},'reflect':{'n_bins':2000},'direct_2':{'n_bins':500},'cross_2':{'n_bins':400},'reflect_2':{'n_bins':2000}} # worked for 120 rays
        if solution_list == None:
            solution_list = ['direct','cross','reflect','direct_2','cross_2','reflect_2']
            
        for solution in solution_list:
            if verbose:
                print('\tSolution Type: %10s \tNumber of points: %i'%( solution , len(self.data[solution]['z'])))
            if (len(self.data[solution]['z']) == 0):
                print('\tNot enough points, returning 0 value hull')
                z_out = [0]
                r_out = [0]
                z_in = [0]
                r_in = [0]
                concave_hull[solution]['z_inner_r_bound'] = z_in
                concave_hull[solution]['r_inner_r_bound'] = r_in
                concave_hull[solution]['z_outer_r_bound'] = z_out
                concave_hull[solution]['r_outer_r_bound'] = r_out
                concave_hull[solution]['z_min'] = 0
                concave_hull[solution]['z_max'] = 0
            else:
                test_z = self.data[solution]['z']
                test_r = self.data[solution]['r']
                
                z_bins = numpy.linspace(min(test_z),max(test_z),concave_hull[solution]['n_bins'])
                z_out = numpy.zeros(concave_hull[solution]['n_bins']-1)
                r_out = numpy.zeros(concave_hull[solution]['n_bins']-1)
                z_in = numpy.zeros(concave_hull[solution]['n_bins']-1)
                r_in = numpy.zeros(concave_hull[solution]['n_bins']-1)
                
                #tz = numpy.tile(test_z,(len(z_bins)-1,1))
                #bz = numpy.tile(z_bins, (len(test_z),1)).T
                #cut = numpy.logical_and(numpy.greater_equal(tz, bz[0:-1]),numpy.less_equal(tz , bz[1:]))
                #use_in_interp = numpy.where(numpy.sum(cut,axis=1)!=0)[0]
                use_in_interp = numpy.zeros(concave_hull[solution]['n_bins']-1,dtype=bool)               
                for bin in range(1,concave_hull[solution]['n_bins']):
                    cut = numpy.logical_and(numpy.greater_equal(test_z, z_bins[bin-1]),numpy.less_equal(test_z, z_bins[bin]))
                    use_in_interp[bin-1] = numpy.any(cut)
                    if use_in_interp[bin-1] == True:
                        r_out[bin-1] = max(test_r[cut])
                        r_in[bin-1] = min(test_r[cut])
                        z_in[bin-1] = max(numpy.unique(test_z[cut][ numpy.where(test_r[cut] == r_in[bin-1])]))
                        z_out[bin-1] = max(numpy.unique(test_z[cut][ numpy.where(test_r[cut] == r_out[bin-1])]))
                        if (bin == concave_hull[solution]['n_bins']-1):
                            #shallowest
                            #print('Trying to adjust shallowest')
                            r_out[bin-1] = max(test_r[numpy.isclose(test_z,max(test_z),atol = 0.5)])
                            z_out[bin-1] = max(test_z)
                #These could be calculated and stored in the original h5 file, then called to make interp1d and max/min within the library, this would save time.
                z_out = z_out[use_in_interp]
                r_out = r_out[use_in_interp]
                z_in = z_in[use_in_interp]
                r_in = r_in[use_in_interp]
                
                #concave_hull[solution]['f_inner_r_bound'] = scipy.interpolate.interp1d(z_in,r_in,bounds_error=False,fill_value = (r_in[0],r_in[-1])) #fill_value=max(r_in))#,kind='cubic') #given z, give r, want big value for fill, because this is region where solution shouldn't exist, so a test of is this > f_in then solution should be false
                #concave_hull[solution]['f_outer_r_bound'] = scipy.interpolate.interp1d(z_out,r_out,bounds_error=False,fill_value = (r_out[0],r_out[-1]))# fill_value=min(r_out))#,kind='cubic') These make boundaries weird but I think are a necessary evil?  Unless I match each with an z_min, z_max?  Could do....,  I can give interp1d two fill values so it fits well up to min/max z
                concave_hull[solution]['z_inner_r_bound'] = z_in
                concave_hull[solution]['r_inner_r_bound'] = r_in
                concave_hull[solution]['z_outer_r_bound'] = z_out
                concave_hull[solution]['r_outer_r_bound'] = r_out
                
                concave_hull[solution]['z_min'] = min(z_in[0],z_out[0])
                concave_hull[solution]['z_max'] = max(z_in[-1],z_out[-1])
                
            if plot_hulls:
                fig1, ax1 = pylab.subplots()
                if numpy.logical_and((len(self.data[solution]['r']) != 0),(len(self.data[solution]['z']) != 0)):
                    pylab.scatter(self.data[solution]['r'],self.data[solution]['z'],c='k',s=1,label='Trace Library Points')
                pylab.xlabel('r(m)',fontsize=20)
                pylab.ylabel('z(m)',fontsize=20)
                pylab.title('Convex Hull for %s'%(solution),fontsize=20)
                pylab.scatter(concave_hull[solution]['r_outer_r_bound'],concave_hull[solution]['z_outer_r_bound'],c='r',label = 'Points Used To\nCreate Outter Bound' )
                pylab.scatter(concave_hull[solution]['r_inner_r_bound'],concave_hull[solution]['z_inner_r_bound'],c='b',label = 'Points Used To\nCreate Inner Bound')
                lines = pylab.hlines([concave_hull[solution]['z_min'],concave_hull[solution]['z_max']],ax1.get_xlim()[0],ax1.get_xlim()[1],lw='1.5',colors='r',linestyles = 'dotted',label='Accepted Depth Window')
                
                pylab.legend(loc = legend_locs[solution],fontsize=16)
            
            #save data into outdir
            outname = out_dir + '/concave_hull_data_' + solution + '.h5'
            outfile  = h5py.File(outname, 'w')
            outfile.attrs['z_min'] = concave_hull[solution]['z_min']
            outfile.attrs['z_max'] = concave_hull[solution]['z_max']
            
            outfile.create_dataset('z_inner_r_bound', (len(z_in),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            outfile.create_dataset('r_inner_r_bound', (len(r_in),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            outfile.create_dataset('z_outer_r_bound', (len(z_out),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            outfile.create_dataset('r_outer_r_bound', (len(r_out),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            
            outfile['z_inner_r_bound'][...] = concave_hull[solution]['z_inner_r_bound']
            outfile['r_inner_r_bound'][...] = concave_hull[solution]['r_inner_r_bound']
            outfile['z_outer_r_bound'][...] = concave_hull[solution]['z_outer_r_bound']
            outfile['r_outer_r_bound'][...] = concave_hull[solution]['r_outer_r_bound']
            
            outfile.close()

    def loadEnvelope(self, indir,store_fit_data = False):
        '''
        Should load the necessary information for a libraries hull so it can be
        used
        
        indir should be of the form '/dir1/concave_hull'
        i.e. it shouldn't end with a /
        where dir2 contains the files '/concave_hull_data_[solution].h5'
        '''
        solution_list = numpy.array(['direct','cross','reflect','direct_2','cross_2','reflect_2'])
        concave_hull = {}
        infiles = glob.glob(indir + '/*.h5')
        infile_list = []
        
        for infile in infiles:
            solution = (infile.split('concave_hull_data_')[-1]).replace('.h5','')
            concave_hull[solution] ={}
            reader = h5py.File(infile, 'r')
            if store_fit_data:
                for key in list(reader.keys()):
                    concave_hull[solution][key] = reader[key][...]
            for attr in list(reader.attrs.keys()):
                concave_hull[solution][attr] = reader.attrs[attr]
            concave_hull[solution]['f_inner_r_bound'] = scipy.interpolate.interp1d(reader['z_inner_r_bound'][...],reader['r_inner_r_bound'][...],bounds_error=False,fill_value = ((reader['r_inner_r_bound'][...])[0],(reader['r_inner_r_bound'][...])[-1])) #fill_value=max(r_in))#,kind='cubic') #given z, give r, want big value for fill, because this is region where solution shouldn't exist, so a test of is this > f_in then solution should be false
            concave_hull[solution]['f_outer_r_bound'] = scipy.interpolate.interp1d(reader['z_outer_r_bound'][...],reader['r_outer_r_bound'][...],bounds_error=False,fill_value = ((reader['r_outer_r_bound'][...])[0],(reader['r_outer_r_bound'][...])[-1]))# fill_value=min(r_out))#,kind='cubic') These make boundaries weird but I think are a necessary evil?  Unless I match each with an z_min, z_max?  Could do....,  I can give interp1d two fill values so it fits well up to min/max z
        return concave_hull    
        

    def intersect(self, dic):
        """
        Find intersection between rays to separate the "direct" and "cross" solutions.
        """
        select_cross = numpy.array([],dtype=int)
        theta_ant_unique = numpy.unique(dic['theta_ant'])
        r_intersect = []
        z_intersect = []
        for ii in range(0, len(theta_ant_unique) - 1):
            cut_1 = numpy.logical_and(dic['theta_ant'] == theta_ant_unique[ii], dic['z'] < 1.)
            for jj in range(ii + 1, len(theta_ant_unique)):
                cut_2 = numpy.logical_and(dic['theta_ant'] == theta_ant_unique[jj], dic['z'] < 1.)
                if numpy.logical_or(numpy.fabs(dic['r'][cut_1][0] - dic['r'][cut_1][-1]) < 1. , numpy.fabs(dic['r'][cut_2][0] - dic['r'][cut_2][-1]) < 1.):
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
                    
                    selection_1 = numpy.logical_and(dic['theta_ant'] == theta_ant_unique[ii], dic['r'] > r_intersect[-1])
                    selection_2 = numpy.logical_and(dic['theta_ant'] == theta_ant_unique[jj], dic['r'] > r_intersect[-1])
                    
                    if (numpy.sum(selection_1) == 0):
                        continue
                    if (numpy.sum(selection_2) == 0):
                        continue
                
                    if  dic['theta'][selection_1][0] > dic['theta'][selection_2][0]:
                        select_cross = numpy.unique(numpy.append(select_cross,numpy.where(selection_1)[0]))
                    else:
                        select_cross = numpy.unique(numpy.append(select_cross,numpy.where(selection_2)[0]))
        r_intersect = numpy.array(r_intersect)
        z_intersect = numpy.array(z_intersect)
        dic_direct = {}
        dic_cross = {}
        cross_cut = numpy.zeros(len(dic[self.keys[0]][...]),dtype=bool)
        cross_cut[select_cross] = True
        if len(r_intersect) == 0:
            # No intersections found, so do nothing
            dic_direct = dic
            dic_cross = {}
            for key in self.keys:
                dic_cross[key] = numpy.array([])
        else:
            # Intersections found, so partition the traces
            # Apply cut for to select cross points
            for key in self.keys:
                dic_cross[key] = dic[key][cross_cut]
            # Then select direct points
            for key in self.keys:
                dic_direct[key] = dic[key][numpy.logical_not(cross_cut)]
        """       
        pylab.figure()
        pylab.scatter(dic_direct['r'], dic_direct['z'], c='blue', edgecolors='none')
        pylab.scatter(dic_cross['r'], dic_cross['z'], c='red', edgecolors='none')
        pylab.scatter(r_intersect, z_intersect, c='black', edgecolors='none')
        """
        return dic_direct, dic_cross


    def plotRays(self, s=10):
        r = []
        z = []
        theta_ant = []
        theta = []
        #for dic in [self.direct, self.crossover, self.reflect]:
        for solution in self.solutions:
            dic = self.data[solution]
            r.append(dic['r'])
            z.append(dic['z'])
            theta_ant.append(dic['theta_ant'])
            theta.append(dic['theta'])
        r = numpy.concatenate(r)
        z = numpy.concatenate(z)
        theta_ant = numpy.concatenate(theta_ant)
        theta = numpy.concatenate(theta)

        pylab.figure()
        pylab.scatter(r, z, c=theta_ant, s=s, edgecolors='none', marker='.')
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
    make_library = True
    split_library = True
    plot_library = True
    save_envelope = True
    plot_envelope = True
    z_array = [-100]#[-175,-176,-177,-178,-179,-181,-183]#[-200.,-201.,-202.,-203.,-204.,-205.,-206.,-207.]
    n_rays = 18
    r_limit = None #Note if this is NOT None, then all thrown rays will quit once they read this particular radius.  Use with care.  If you want a simulation with r = 6300m, it might be advisable to make r_limit = 7000 so the boundaries of hulls are still well defined
    ice_model = 'antarctica'

    ############################################


    ice_model = gnosim.earth.ice.checkIceModel(ice_model) #Checks if ice model exists and replaces as necessary.
    for z_0 in z_array:
        library_dir = 'library_%i_%s_%i_rays'%(int(z_0),ice_model,n_rays)
        if os.path.isdir(library_dir):
            print('Output directory Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(library_dir))
            while os.path.isdir(library_dir):
                library_dir = library_dir + '_new'
        print ('library dir = %s'%(library_dir))
        print ('z_0 = %.2f'%(z_0))
        #print ('ice model = %s'%(ice_model))
       
        theta_array = numpy.linspace(0., 180., n_rays) # 60, Trying double the density
        #Below is an attemopt to enure more rays are thrown in the region surrounding the reflect/cross hull partition.
        #theta_reflect = numpy.rad2deg(numpy.arcsin(ice.indexOfRefraction(-0.01)/ice.indexOfRefraction(z_0)))
        #theta_array = numpy.append(theta_array,0.99*theta_reflect)
        #theta_array = numpy.append(theta_array,1.01*theta_reflect)
        
        #solution_list = numpy.array(['direct','cross','reflect','direct_2','cross_2','reflect_2'])
        solution_list = numpy.array(['direct','cross','reflect'])
        if make_library == True:
            os.mkdir(library_dir)
            makeLibrary(z_0, theta_array,ice_model, save=True, library_dir=library_dir,r_limit = r_limit)
            
        if numpy.any([plot_library == True,split_library == True,save_envelope == True]):
            print('\n'+library_dir+'/*.h5\n')
            pre_split = (split_library == False)
            test_lib = RefractionLibrary(library_dir+'/*.h5',pre_split = pre_split)
            
        if split_library == True:
            def sortTraceFileByAng(infiles):
                infile_theta = numpy.zeros(len(infiles))
                for index, f in enumerate(infiles):
                    infile_theta[index] = f.split('theta_')[-1].split('_n')[-2]
                sorted_infiles = numpy.zeros_like(infiles)
                for new,old in enumerate(numpy.argsort(infile_theta)):
                    sorted_infiles[new] = infiles[old]
                return sorted_infiles , numpy.sort(infile_theta)

            infiles, infile_angles = sortTraceFileByAng(glob.glob(library_dir + '/*.h5'))
            in_path = infiles[0].replace(infiles[0].split('/')[-1],'')
            in_dir = infiles[0].split('/')[-2] 

            for solution in list(test_lib.data.keys()):
                sub_dir = in_dir + '/' + solution
                while os.path.isdir(in_path.replace(in_dir,sub_dir)):
                    print('Outfile path:' , in_path.replace(in_dir,sub_dir),'exists, appending _new to path')
                    sub_dir = sub_dir + '_new'
                os.mkdir(in_path.replace(in_dir,sub_dir) + '/')
                
                for ang_index, infile in enumerate(infiles):
                    print('Saving Split:\t' , infile)
                    reader = h5py.File(infile, 'r')
                    file = h5py.File(in_path.replace(in_dir,sub_dir) + '/' + infile.split('/')[-1], 'w')
                    theta_ant_cut = numpy.where(numpy.isclose(test_lib.data[solution]['theta_ant'],infile_angles[ang_index], atol = 0.99*min(numpy.diff(infile_angles))))[0]
                    file.attrs['solution_type'] = solution.encode('utf8')
                    for attr in list(reader.attrs):
                        file.attrs[attr] = reader.attrs[attr]
                    print('Saving',len(theta_ant_cut),'values for solution type:',solution,'\ttheta_ant = ',infile_angles[ang_index])
                    for key in list(test_lib.data[solution].keys()):
                        file.create_dataset(key, (len(theta_ant_cut),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                        if len(theta_ant_cut != 0):
                            file[key][...] = test_lib.data[solution][key][theta_ant_cut]

                    file.close()

        if plot_library == True: 
            color_key = {'direct':'r', 'cross':'darkgreen', 'reflect':'blue', 'direct_2':'gold', 'cross_2':'lawngreen', 'reflect_2':'purple'}           
            pylab.figure()
            for solution in solution_list:
                if numpy.logical_and(len(test_lib.data[solution]['r']) > 0, len(test_lib.data[solution]['z']) > 0):
                    pylab.scatter( test_lib.data[solution]['r'], test_lib.data[solution]['z'],label=solution,color=color_key[solution],s=1)
            pylab.legend(loc='upper right')
            pylab.xlabel('r(m)',fontsize=16)
            pylab.ylabel('z(m)',fontsize=16)
            pylab.ylim(-3010,10)
            pylab.xlim(-10,6310)
            
        if save_envelope == True:
            test_lib.saveEnvelope(library_dir,solution_list = None,verbose = True, plot_hulls = False)
            
        if plot_envelope == True:
            color_key = {'direct':'r', 'cross':'darkgreen', 'reflect':'blue', 'direct_2':'gold', 'cross_2':'lawngreen', 'reflect_2':'purple'}
            concave_hull = {}
            infiles = glob.glob('./'+library_dir+'/concave_hull/*.h5')
            if len(infiles) == 0:
                print('Error loading concave hull files.  Ensure they are saved and in dir\n./'+library_dir+'/concave_hull/')
                continue
            infile_list = []
            fig,ax = pylab.subplots()
            for infile in infiles:
                solution = (infile.split('concave_hull_data_')[-1]).replace('.h5','')
                if numpy.isin(solution,solution_list):
                    concave_hull[solution] ={}
                    reader = h5py.File(infile, 'r')
                    z = numpy.linspace(reader.attrs['z_min'],reader.attrs['z_max'],3000)
                    f_in = scipy.interpolate.interp1d(reader['z_inner_r_bound'][...],reader['r_inner_r_bound'][...],bounds_error=False,fill_value = ((reader['r_inner_r_bound'][...])[0],(reader['r_inner_r_bound'][...])[-1]),kind='cubic') 
                    f_out = scipy.interpolate.interp1d(reader['z_outer_r_bound'][...],reader['r_outer_r_bound'][...],bounds_error=False,fill_value = ((reader['r_outer_r_bound'][...])[0],(reader['r_outer_r_bound'][...])[-1]),kind='cubic')
                    pylab.scatter(f_in(z),z,label = solution+' inner',color=color_key[solution],marker = '<')
                    pylab.scatter(f_out(z),z,label = solution+' outer',color=color_key[solution],marker = '>')
                    ax.fill_betweenx(z,f_in(z),f_out(z),color=color_key[solution],alpha=0.2)
            pylab.legend(loc='upper right')
            pylab.ylim(-3010,10)
            pylab.xlim(-10,6310)
            
            pylab.show()


"""
#From before DS worked on the code
# WANT TO TAKE INTO ACCOUNT CURVATURE OF THE EARTH HERE
#theta_array = numpy.degrees(numpy.arctan2(numpy.linspace(0., 50000., 120), -1000.)) # MINNA BLUFF
theta_horizon = 90. + gnosim.earth.earth.horizon(z_0)[1] + 1.e-3 # deg
x_horizon = -1. * z_0 * numpy.tan(numpy.radians(theta_horizon)) # m
#theta_array = numpy.degrees(numpy.arctan2(numpy.linspace(0., x_horizon, 120), -1. * z_0)) # MINNA BLUFF
theta_array = numpy.degrees(numpy.arctan2(numpy.linspace(0., x_horizon, 120), -1. * z_0)) # MINNA BLUFF


# FULLY ACCOUNTING FOR EARTH CURVATURE, USE THIS FOR HIGH-ALTITUDE CONFIGURATIONS
#theta_array = gnosim.earth.earth.curvatureToTheta(z_0, numpy.linspace(0., gnosim.earth.earth.horizon(z_0)[2] - 1., 60)) # 30, 60
#print (theta_array)
#import sys
#sys.exit('DONE')
"""




