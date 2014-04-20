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
        flag_direct, flag_crossover, flag_reflect, dic_direct, dic_crossover, dic_reflect = 0, 0, 0, {}, {}, {}
        #electric_field_direct = 0.
        #electric_field_crossover = 0.
        #electric_field_reflect = 0.

        inelasticity = -999.
        electric_field_max = 0.
        dic_max = {}
        path_max = -999.
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
                flag_direct, flag_crossover, flag_reflect, dic_direct, dic_crossover, dic_reflect \
                    = self.stations[index_station].antennas[index_antenna].lib.event(r, z_0)

                if numpy.any([flag_direct, flag_crossover, flag_direct]):
                    p_detect = 1.
                    inelasticity = gnosim.interaction.inelasticity.inelasticity(energy_neutrino, mode='cc') # GENERALIZE THIS LATER
                    frequency = numpy.linspace(self.stations[index_station].antennas[index_antenna].frequency_low,
                                               self.stations[index_station].antennas[index_antenna].frequency_high,
                                               100) # GHz
                    index_of_refraction = gnosim.earth.greenland.indexOfRefraction(z_0)
                    vector_neutrino = gnosim.utils.quat.angToVec(phi_0, theta_0) # Direction neutrino came from
                    phi_ray = numpy.degrees(numpy.arctan2(y_0 - y_antenna, x_0 - x_antenna)) % 360. # deg
                    
                    if flag_direct:
                        vector_ray = gnosim.utils.quat.angToVec(phi_ray, dic_direct['theta']) # Direction of outgoing ray from antenna to interaction vertex
                        observation_angle = gnosim.utils.quat.angTwoVec(vector_neutrino, vector_ray)
                        d = dic_direct['d'] # m
                        # COME BACK TO GENERALIZE THIS
                        electric_field \
                            = gnosim.interaction.askaryan.electricField(frequency, d, observation_angle, 
                                                                        energy_neutrino, inelasticity, 'cc', index_of_refraction) # V m^-1 GHz^-1, dimensionless
                        electric_field *= dic_direct['a_v'] # COME BACK TO GENERALIZE THIS
                        electric_field = self.stations[index_station].antennas[index_antenna].electricField(frequency, electric_field) # V m^-1
                        #electric_field_direct = electric_field
                        if electric_field > electric_field_max:
                            electric_field_max = electric_field
                            dic_max = dic_direct
                            path_max = 0
                            index_station_max = index_station
                            index_antenna_max = index_antenna

                    if flag_crossover:
                        vector_ray = gnosim.utils.quat.angToVec(phi_ray, dic_crossover['theta']) # Direction of outgoing ray from antenna to interaction vertex
                        observation_angle = gnosim.utils.quat.angTwoVec(vector_neutrino, vector_ray)
                        d = dic_crossover['d'] # m
                        # COME BACK TO GENERALIZE THIS
                        electric_field \
                            = gnosim.interaction.askaryan.electricField(frequency, d, observation_angle, 
                                                                        energy_neutrino, inelasticity, 'cc', index_of_refraction) # V m^-1 GHz^-1, dimensionless
                        electric_field *= dic_crossover['a_v'] # COME BACK TO GENERALIZE THIS
                        electric_field = self.stations[index_station].antennas[index_antenna].electricField(frequency, electric_field) # V m^-1
                        #electric_field_crossover = electric_field
                        if electric_field > electric_field_max:
                            electric_field_max = electric_field
                            dic_max = dic_crossover
                            path_max = 1
                            index_station_max = index_station
                            index_antenna_max = index_antenna

                    if flag_reflect:
                        vector_ray = gnosim.utils.quat.angToVec(phi_ray, dic_reflect['theta']) # Direction of outgoing ray from antenna to interaction vertex
                        observation_angle = gnosim.utils.quat.angTwoVec(vector_neutrino, vector_ray)
                        d = dic_reflect['d'] # m
                        # COME BACK TO GENERALIZE THIS
                        electric_field \
                            = gnosim.interaction.askaryan.electricField(frequency, d, observation_angle, 
                                                                        energy_neutrino, inelasticity, 'cc', index_of_refraction) # V m^-1 GHz^-1, dimensionless
                        electric_field *= dic_reflect['a_v'] # COME BACK TO GENERALIZE THIS
                        electric_field = self.stations[index_station].antennas[index_antenna].electricField(frequency, electric_field) # V m^-1
                        #electric_field_reflect = electric_field
                        if electric_field > electric_field_max:
                            electric_field_max = electric_field
                            dic_max = dic_reflect
                            path_max = 2
                            index_station_max = index_station
                            index_antenna_max = index_antenna

        return p_interact, p_earth, p_detect, inelasticity, electric_field_max, dic_max, path_max, index_station_max, index_antenna_max
        #return p_interact, p_earth, p_detect, electric_field_direct, electric_field_crossover, electric_field_reflect,  dic_direct, dic_crossover, dic_reflect

    def throw(self, energy_neutrino=1.e9, 
              theta_0=None, phi_0=None, x_0=None, y_0=None, z_0=None, 
              anti=False, n_events=10000, detector_volume_radius=6000., detector_volume_depth=3000., 
              outfile=None):

        energy_neutrino = energy_neutrino * numpy.ones(n_events)
    
        phi_0 = numpy.random.uniform(0., 360., size=n_events) # deg
        theta_0 = numpy.degrees(numpy.arccos(numpy.random.uniform(1., -1., size=n_events))) # deg
        r = numpy.random.triangular(0., detector_volume_radius, detector_volume_radius, size=n_events) # m
        
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
        path_max = numpy.zeros(n_events)
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
            p_interact[ii], p_earth[ii], p_detect[ii], inelasticity[ii], electric_field_max[ii], dic_max, path_max[ii], index_station_max[ii], index_antenna_max[ii] \
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
            file.attrs['geometric_factor'] = (4. * numpy.pi) * (numpy.pi * detector_volume_radius**2 * detector_volume_depth) # m^3 sr
            file.attrs['config'] = self.config_file

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
            file.create_dataset('path', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
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
            file['path'][...] = path_max
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
    energy_neutrino = float(sys.argv[2])
    n_events = int(sys.argv[3])

    outfile = 'results_empirical/%s_%.2e_GeV_%i_events.h5'%(config_file.replace('.py', ''),
                                                            energy_neutrino,
                                                            n_events)

    my_sim = Sim(config_file)
    my_sim.throw(energy_neutrino, n_events=n_events, outfile=outfile)

############################################################
