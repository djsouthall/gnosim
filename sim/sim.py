#!/usr/bin/env python

"""
Simulation
"""

import sys
import numpy
import h5py
import pylab

import gnosim.utils.quat
import gnosim.earth.earth
import gnosim.earth.greenland
import gnosim.trace.refraction_library
import gnosim.interaction.askaryan
import gnosim.sim.detector

pylab.ion()

############################################################

class Sim:

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = eval(''.join(open(config_file).readlines()))
        self.detector()

        # List attributes of interest
        self.keys = ['t', 'd', 'theta', 'theta_0', 'a_v', 'a_h']

    def detector(self):
        self.lib = {}
        for key in self.config['antenna_definitions'].keys():
            self.lib[key] = gnosim.trace.refraction_library.RefractionLibrary(self.config['antenna_definitions'][key]['lib'])

        self.stations = []
        for ii in range(0, self.config['stations']['n']):
            x_station, y_station, z_station = self.config['stations']['positions'][ii]
            self.stations.append(gnosim.sim.detector.Station(x_station, y_station, z_station))
            for jj in range(0, self.config['antennas']['n']):
                antenna_type = self.config['antennas']['types'][jj]
                x_antenna, y_antenna, z_antenna = self.config['antennas']['positions'][jj]
                antenna = gnosim.sim.detector.Antenna(x_antenna + x_station, y_antenna + y_station, z_antenna + z_station, 
                                                      self.lib[self.config['antennas']['types'][jj]],
                                                      self.config['antenna_definitions'][antenna_type]['frequency_low'],
                                                      self.config['antenna_definitions'][antenna_type]['frequency_high'])
                self.stations[ii].antennas.append(antenna)

    def event(self, energy_neutrino, phi_0, theta_0, x_0, y_0, z_0, anti=False):
        # Probability for neutrino to interact in small target volume of ice
        p_interact = gnosim.earth.earth.probInteract(energy_neutrino, z_0, anti=anti)
    
        # Probability for neutrino to make it through the Earth
        p_earth = gnosim.earth.earth.probSurvival(energy_neutrino, theta_0, elevation=z_0, anti=anti)

        p_detect = 0.

        # Placeholders
        inelasticity = -999.
        electric_field_max = 0.
        dic_max = {}
        observation_angle_max = -999.
        solution_max = -999.
        index_station_max = -999.
        index_antenna_max = -999.

        # Loop over stations
        for index_station in range(0, len(self.stations)):
            # Loop over station antennas
            for index_antenna in range(0, len(self.stations[index_station].antennas)):
                x_antenna = self.stations[index_station].antennas[index_antenna].x
                y_antenna = self.stations[index_station].antennas[index_antenna].y
                z_antenna = self.stations[index_station].antennas[index_antenna].z
                r = numpy.sqrt((x_0 - x_antenna)**2 + (y_0 - y_antenna)**2)
                flag_array, dic_array = self.stations[index_station].antennas[index_antenna].lib.event(r, z_0)                                              
                
                if numpy.any(flag_array):
                    p_detect = 1.
                    inelasticity = gnosim.interaction.inelasticity.inelasticity(energy_neutrino, mode='cc') # GENERALIZE THIS LATER
                    frequency = numpy.linspace(self.stations[index_station].antennas[index_antenna].frequency_low,
                                               self.stations[index_station].antennas[index_antenna].frequency_high,
                                               100) # GHz
                    index_of_refraction = gnosim.earth.greenland.indexOfRefraction(z_0, ice_model=self.config['detector_volume']['ice_model'])
                    vector_neutrino = gnosim.utils.quat.angToVec(phi_0, theta_0) # Direction neutrino came from
                    phi_ray = numpy.degrees(numpy.arctan2(y_0 - y_antenna, x_0 - x_antenna)) % 360. # deg
                    
                    for ii, solution in enumerate(self.stations[index_station].antennas[index_antenna].lib.solutions):
                        if flag_array[ii]:
                            # Direction of outgoing ray from antenna to interaction vertex
                            vector_ray = gnosim.utils.quat.angToVec(phi_ray, dic_array[ii]['theta'])
                            observation_angle = gnosim.utils.quat.angTwoVec(vector_neutrino, vector_ray) # deg
                            d = dic_array[ii]['d'] # m
                            electric_field \
                                = gnosim.interaction.askaryan.electricField(frequency, d, observation_angle,
                                                                            energy_neutrino, inelasticity, 
                                                                            'cc', index_of_refraction) # V m^-1 GHz^-1, dimensionless
                            electric_field *= dic_array[ii]['a_v'] # COME BACK TO GENERALIZE THIS
                            electric_field = self.stations[index_station].antennas[index_antenna].electricField(frequency, electric_field) # V m^-1
                            
                            if electric_field > electric_field_max:
                                electric_field_max = electric_field
                                dic_max = dic_array[ii]
                                observation_angle_max = observation_angle
                                solution_max = ii
                                index_station_max = index_station
                                index_antenna_max = index_antenna
                                
        return p_interact, p_earth, p_detect, inelasticity, electric_field_max, dic_max, observation_angle_max, solution_max, index_station_max, index_antenna_max
        #return p_interact, p_earth, p_detect, electric_field_direct, electric_field_crossover, electric_field_reflect,  dic_direct, dic_crossover, dic_reflect

    def throw(self, energy_neutrino=1.e9, 
              theta_0=None, phi_0=None, x_0=None, y_0=None, z_0=None, 
              anti=False, n_events=10000, detector_volume_radius=6000., detector_volume_depth=3000., 
              outfile=None):

        energy_neutrino = energy_neutrino * numpy.ones(n_events)
    
        phi_0 = numpy.random.uniform(0., 360., size=n_events) # deg
        theta_0 = numpy.degrees(numpy.arccos(numpy.random.uniform(1., -1., size=n_events))) # deg
        
        # ORIGINAL 28 MAY 2014
        #r = numpy.random.triangular(0., detector_volume_radius, detector_volume_radius, size=n_events) # m
        # ORIGINAL 20 MAY 2014
        # NEW CURVATURE
        alpha_max_radians = detector_volume_radius / gnosim.utils.constants.radius_earth # radians
        alpha = numpy.arccos(numpy.random.uniform(1., numpy.cos(alpha_max_radians), size=n_events)) # radians
        r = gnosim.utils.constants.radius_earth * alpha # m
        # NEW CURVATURE

        phi_vertex = numpy.random.uniform(0., 360., size=n_events) # deg
    
        x_0 = r * numpy.cos(numpy.radians(phi_vertex))
        y_0 = r * numpy.sin(numpy.radians(phi_vertex))
        z_0 = numpy.random.uniform(-1. * detector_volume_depth, 0., size=n_events) # m

        """
        pylab.figure()
        pylab.scatter(x, y)
        pylab.figure()
        pylab.hist(r, bins=40)
        pylab.figure()
        pylab.hist(z, bins=40)
        pylab.figure()
        pylab.hist(phi, bins=40)
        pylab.figure()
        pylab.hist(theta, bins=40)
        """

        p_interact = numpy.zeros(n_events)
        p_earth = numpy.zeros(n_events)
        p_detect = numpy.zeros(n_events)
        inelasticity = numpy.zeros(n_events)
        electric_field_max = numpy.zeros(n_events)
        observation_angle_max = numpy.zeros(n_events)
        solution_max = numpy.zeros(n_events)
        index_station_max = numpy.zeros(n_events)
        index_antenna_max = numpy.zeros(n_events)
        t_max = numpy.zeros(n_events)
        d_max = numpy.zeros(n_events)
        theta_ray_max = numpy.zeros(n_events)
        theta_ant_max = numpy.zeros(n_events)
        a_v_max = numpy.zeros(n_events)
        a_h_max = numpy.zeros(n_events)

        #electric_field_direct = numpy.zeros(n_events)
        #electric_field_crossover = numpy.zeros(n_events)
        #electric_field_reflect = numpy.zeros(n_events)
        #flag_direct = numpy.zeros(n_events)
        #flag_crossover = numpy.zeros(n_events)
        #flag_reflect = numpy.zeros(events)

        for ii in range(0, n_events):
            print 'Event (%i/%i)'%(ii, n_events)
            #p_interact[ii], p_earth[ii], p_detect[ii], electric_field_direct[ii], electric_field_crossover[ii], electric_field_reflect[ii], flag_direct[ii], flag_crossover[ii], flag_reflect[ii], dic_direct, dic_crossover, dic_reflect \
            #    = self.event(energy_neutrino[ii], phi[ii], theta[ii], x[ii], y[ii], z[ii], anti=anti)
            p_interact[ii], p_earth[ii], p_detect[ii], inelasticity[ii], electric_field_max[ii], dic_max, observation_angle_max[ii], solution_max[ii], index_station_max[ii], index_antenna_max[ii] \
                = self.event(energy_neutrino[ii], phi_0[ii], theta_0[ii], x_0[ii], y_0[ii], z_0[ii], anti=anti)

            if p_detect[ii] == 1.:
                t_max[ii] = dic_max['t']
                d_max[ii] = dic_max['d']
                theta_ray_max[ii] = dic_max['theta']
                theta_ant_max[ii] = dic_max['theta_0']
                a_v_max[ii] = dic_max['a_v']
                a_h_max[ii] = dic_max['a_h']

        for ii in range(0, n_events):
            if p_detect[ii]:
                print r[ii], z_0[ii], electric_field_max[ii]

        cut = (p_detect == 1.)
        cut_inv = numpy.logical_not(cut)
        
        """
        pylab.figure()
        pylab.scatter(r[cut_inv], z[cut_inv], c='gray', edgecolors='none')
        pylab.scatter(r[cut], z[cut], c=numpy.log10(electric_field[cut]), edgecolors='none')
        pylab.colorbar()
        """
        """
        pylab.figure()
        pylab.scatter(theta[cut], numpy.log10(electric_field[cut]), c=p_earth[cut], edgecolors='none')
        pylab.colorbar()
        """
        
        if outfile:
            file = h5py.File(outfile, 'w')
            # ORIGINAL 28 MAY 2014
            #file.attrs['geometric_factor'] = (4. * numpy.pi) * (numpy.pi * detector_volume_radius**2 * detector_volume_depth) # m^3 sr
            # ORIGINAL 28 MAY 2014
            # NEW CURVATURE
            file.attrs['geometric_factor'] = (4. * numpy.pi) \
                                             * (2. * numpy.pi * gnosim.utils.constants.radius_earth**2 \
                                                * (1. - numpy.cos(detector_volume_radius / gnosim.utils.constants.radius_earth))\
                                                * detector_volume_depth) # m^3 sr
            # NEW CURVATURE

            file.attrs['config'] = self.config_file
            #file.attrs['ice_model'] = gnosim.earth.greenland.ice_model_default
            file.attrs['ice_model'] = self.config['detector_volume']['ice_model']

            file.create_dataset('energy_neutrino', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('inelasticity', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('x_0', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('y_0', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('z_0', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('theta_0', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('phi_0', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)

            file.create_dataset('p_interact', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('p_earth', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('p_detect', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            
            file.create_dataset('index_station', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('index_antenna', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)

            file.create_dataset('electric_field', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('observation_angle', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('solution', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('t', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('d', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('theta_ray', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('theta_ant', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('a_v', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('a_h', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)

            file['energy_neutrino'][...] = energy_neutrino
            file['inelasticity'][...] = inelasticity
            file['x_0'][...] = x_0
            file['y_0'][...] = y_0
            file['z_0'][...] = z_0
            file['theta_0'][...] = theta_0
            file['phi_0'][...] = phi_0

            file['p_interact'][...] = p_interact
            file['p_earth'][...] = p_earth
            file['p_detect'][...] = p_detect

            file['index_station'][...] = index_station_max
            file['index_antenna'][...] = index_antenna_max

            file['electric_field'][...] = electric_field_max
            file['observation_angle'][...] = observation_angle_max
            file['solution'][...] = solution_max
            file['t'][...] = t_max
            file['d'][...] = d_max
            file['theta_ray'][...] = theta_ray_max
            file['theta_ant'][...] = theta_ant_max
            file['a_v'][...] = a_v_max
            file['a_h'][...] = a_h_max

            file.close()

############################################################

if __name__ == "__main__":

    config_file = sys.argv[1]
    energy_neutrino = float(sys.argv[2]) # GeV
    n_events = int(sys.argv[3])
    index = int(sys.argv[4])
    #detector_volume_radius = float(sys.argv[5]) # m, 1200 for Ross surface, 51000 for Minna bluff, >6000 for subterranean
    #detector_volume_depth = float(sys.argv[6]) # m, 500 for Ross and Minna, 3000 for subterranean

    outfile = 'results_2014_aug_7/%s_%.2e_GeV_%i_events_%i.h5'%(config_file.replace('.py', ''),
                                                                energy_neutrino,
                                                                n_events,
                                                                index)
    
    my_sim = Sim(config_file)
    my_sim.throw(energy_neutrino, n_events=n_events, 
                 detector_volume_radius=my_sim.config['detector_volume']['radius'],
                 detector_volume_depth=my_sim.config['detector_volume']['depth'],
                 outfile=outfile)

############################################################
