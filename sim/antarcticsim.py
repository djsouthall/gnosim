#!/usr/bin/env python

"""
Simulation
"""

import sys
import numpy
import h5py
import pylab
import json
import yaml
import os
import os.path
import glob
import scipy
import scipy.signal

sys.path.append("/home/dsouthall/Projects/GNOSim/")
import gnosim.utils.quat
import gnosim.earth.earth
import gnosim.earth.antarctic
import gnosim.trace.refraction_library_beta
from gnosim.trace.refraction_library_beta import *
import gnosim.interaction.askaryan
import gnosim.sim.detector

pylab.ion()

############################################################


class Sim:

    def __init__(self, config_file,solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2']),pre_split = False):
        #pre_split False unless using a library already sorted into different
        #directories by solution type.
        self.config_file = config_file
        self.pre_split = pre_split
        #self.config = eval(''.join(open(config_file).readlines()))
        self.config = yaml.load(open(config_file))
        self.detector()     
        self.info_dtype = numpy.dtype([('eventid','i'),('station','i'),('antenna','i'),('has_solution','i'),('solution','S10'),('time','f'),('distance','f'),('theta_ant','f'),('observation_angle','f'),('electric_field','f'),('weighted_freq','f'),('a_h','f'),('a_v','f')])
        # List attributes of interest
        self.keys = ['t', 'd', 'theta', 'theta_0', 'a_v', 'a_h']
        self.n_antenna = sum([len(self.stations[s].antennas) for s in range(len(self.stations))])
        print('Total Number of Antennas = ', self.n_antenna)

    def detector(self):
        self.lib = {}
        for key in self.config['antenna_definitions'].keys():
            self.lib[key] = gnosim.trace.refraction_library_beta.RefractionLibrary(self.config['antenna_definitions'][key]['lib'],solutions=solutions,pre_split = self.pre_split)
            #if library is split, I want to invoke it here with some flag.  I will need an altered version of RefractionLibrary
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
                                                      self.config['antenna_definitions'][antenna_type]['frequency_high'],
                                                      self.config)
                self.stations[ii].antennas.append(antenna)

    def event(self, energy_neutrino, phi_0, theta_0, x_0, y_0, z_0, eventid, anti=False,electricFieldDomain = 'freq',plot=False,summed_signals=False,save_fig=False):
        p_interact = gnosim.earth.earth.probInteract(energy_neutrino, z_0, anti=anti)
    
        # Probability for neutrino to make it through the Earth
        p_earth = gnosim.earth.earth.probSurvival(energy_neutrino, theta_0, elevation=z_0, anti=anti)

        # Placeholders
        inelasticity = -999.
        #electric_field_max = 0. Now per antenna, not event
        dic_max = {}
        observation_angle_max = -999.
        solution_max = -999.
        index_station_max = -999.
        index_antenna_max = -999.
        
        info = numpy.zeros(  self.n_antenna  , dtype = self.info_dtype) #was using numpy.empy, switch to 0s
        #Old indexing -> (len(self.stations)*len(self.stations[0].antennas))
        
        # Loop over stations
        if electricFieldDomain == 'time':
            u_signals = {}
            E_signals = {}
        for index_station in range(0, len(self.stations)):
            # Loop over station antennas
            station_label = 'station'+str(index_station)
            if electricFieldDomain == 'time':
                u_signals[station_label] = {}
                E_signals[station_label] = {}
            for index_antenna in range(0, len(self.stations[index_station].antennas)):
                antenna_label = self.config['antennas']['types'][index_antenna]
                if electricFieldDomain == 'time':
                    u_signals[station_label][antenna_label] = {}
                    E_signals[station_label][antenna_label] = {}
                    for solution in self.stations[index_station].antennas[index_antenna].lib.solutions:
                        u_signals[station_label][antenna_label][solution] = []
                        E_signals[station_label][antenna_label][solution] = []
                electric_field_max = 0. #This clearly makes no sense.  Reset every antenna??  Why?  So bizarre.  My info object outputs more info. 
                x_antenna = self.stations[index_station].antennas[index_antenna].x
                y_antenna = self.stations[index_station].antennas[index_antenna].y
                z_antenna = self.stations[index_station].antennas[index_antenna].z
                r = numpy.sqrt((x_0 - x_antenna)**2 + (y_0 - y_antenna)**2)
                
                flag_array = []
                for solution in self.stations[index_station].antennas[index_antenna].lib.solutions:
                    flag_array.append(self.in_flag_array[station_label][antenna_label][solution][eventid])
                    
                #flag_array = [self.in_flag_array[station_label][antenna_label]['direct'][eventid],self.in_flag_array[station_label][antenna_label]['cross'][eventid],self.in_flag_array[station_label][antenna_label]['reflect'][eventid],self.in_flag_array[station_label][antenna_label]['direct_2'][eventid],self.in_flag_array[station_label][antenna_label]['cross_2'][eventid],self.in_flag_array[station_label][antenna_label]['reflect_2'][eventid]]
                #flag_array, dic_array = self.stations[index_station].antennas[index_antenna].lib.event(r, z_0)                                              
                temporary_info = numpy.zeros(  numpy.size(self.stations[index_station].antennas[index_antenna].lib.solutions)  , dtype = self.info_dtype)
                #print('len(temporary_info)',len(temporary_info))
                for i in range(len(temporary_info)):
                    temporary_info[i] = numpy.array([(eventid,index_station,index_antenna,0,'',-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0)],dtype = self.info_dtype)
                #temporary_info['has_solution'] = 0
                if numpy.any(flag_array):
                    has_solution = 1
                    #p_detect = 1.  #This essentially flags if any detector sees any solution type.
                    #temporary_info['has_solution'] = 1
                    #want an antenna specific p_detect that flags if a particular detector sees any solution type.  
                    inelasticity = gnosim.interaction.inelasticity.inelasticity(energy_neutrino, mode='cc') # GENERALIZE THIS LATER #DS not sure why this is in the loop.  Does not depend on any loop variables, only need to calculate once.  Maybe because it was going to be 'generalized'?
                    frequency = numpy.linspace(self.stations[index_station].antennas[index_antenna].frequency_low,
                                               self.stations[index_station].antennas[index_antenna].frequency_high,
                                               100) # GHz
                    index_of_refraction = gnosim.earth.antarctic.indexOfRefraction(z_0, ice_model=self.config['detector_volume']['ice_model'])
                    vector_neutrino = gnosim.utils.quat.angToVec(phi_0, theta_0) # Direction neutrino came from
                    phi_ray = numpy.degrees(numpy.arctan2(y_0 - y_antenna, x_0 - x_antenna)) % 360. # deg
                    
                    for ii, solution in enumerate(self.stations[index_station].antennas[index_antenna].lib.solutions):
                        #if flag_array[ii]:
                        if self.in_flag_array[station_label][antenna_label][solution][eventid]:
                            # Direction of outgoing ray from antenna to interaction vertex
                            
                            #vector_ray = gnosim.utils.quat.angToVec(phi_ray, dic_array[ii]['theta'])
                            vector_ray = gnosim.utils.quat.angToVec(phi_ray, self.in_dic_array[station_label][antenna_label][solution]['theta'][eventid])
                            observation_angle = gnosim.utils.quat.angTwoVec(vector_neutrino, vector_ray) # deg
                            d = self.in_dic_array[station_label][antenna_label][solution]['d'][eventid] # m
                            
                            if electricFieldDomain == 'time':
                                #this must output a single value for the electric field in the same way old does
                                #must also output a weighted frequency of sorts.  i.e. the most dominant frequency.
                                #might need to account for polarity swap on reflection.  Not sure if this is an effect.  
                                
                                #up_sample_factor dictates the timestep that the entire calculation is done at.
                                #the time resolution required for an accurent vector potential calculation varies,
                                #but likely less than 0.05ns is preferred.  If up_sample_factor = 0 then the 
                                #time step is closer to 0.32ns.  Near cone the required timing resolution increases. 
                                cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./index_of_refraction))
                                if abs(observation_angle-cherenkov_angle_deg) < 0.5:
                                    up_sample_factor = 30 #it might be fine to do this at 20
                                elif abs(observation_angle-cherenkov_angle_deg) < 10:
                                    up_sample_factor = 20
                                else:
                                    up_sample_factor = 10 #maybe could reduce for reaaally far off cone.  
                                electric_array, u, weighted_freq = gnosim.interaction.askaryan.electricFieldTimeDomainSignal(numpy.deg2rad(observation_angle),self.in_dic_array[station_label][antenna_label][solution]['d'][eventid],energy_neutrino*inelasticity,index_of_refraction,h_fft=self.h_fft,sys_fft=self.sys_fft,freqs=self.freqs_response,return_pos = True,out_dom_freq=True,plot=False,up_sample_factor=up_sample_factor,deriv_mode = 'time')
                                #note that electric_array here should be measured voltage if the response function is correct (ignoring attenuation done below). 
                                u = u + self.in_dic_array[station_label][antenna_label][solution]['t'][eventid]
                                electric_array *= self.in_dic_array[station_label][antenna_label][solution]['a_v'][eventid]
                                electric_field = max(numpy.abs(scipy.signal.hilbert(electric_array))) #Right now this takes the max value of the hilbert envelope
                                E_signals[station_label][antenna_label][solution] = electric_array
                                u_signals[station_label][antenna_label][solution] = u
                            else:
                                if electricFieldDomain != 'freq':
                                    print('Electric field domain selection did not fit one of the\ntwo expected values.  Defaulting to freq.')
                                    
                                electric_field \
                                    = gnosim.interaction.askaryan.electricFieldFrequencyDomainRaw(frequency, d, observation_angle,
                                                                                energy_neutrino, inelasticity, 
                                                                                'cc', index_of_refraction) # V m^-1 GHz^-1, dimensionless
                                electric_field *= self.in_dic_array[station_label][antenna_label][solution]['a_v'][eventid] # COME BACK TO GENERALIZE THIS
                                electric_array, electric_field, weighted_freq = self.stations[index_station].antennas[index_antenna].totalElectricField(frequency, electric_field, self.in_dic_array[station_label][antenna_label][solution]['theta_ant'][eventid]) # V m^-1 #THIS WAS CHANGED THETA WAS ADDED
                            
                            temporary_info[ii] = numpy.array([(eventid,index_station,index_antenna,has_solution,solution,self.in_dic_array[station_label][antenna_label][solution]['t'][eventid],self.in_dic_array[station_label][antenna_label][solution]['d'][eventid],self.in_dic_array[station_label][antenna_label][solution]['theta_ant'][eventid],observation_angle,electric_field,weighted_freq,self.in_dic_array[station_label][antenna_label][solution]['a_h'][eventid],self.in_dic_array[station_label][antenna_label][solution]['a_v'][eventid])],dtype = self.info_dtype)
                            if electric_field >= electric_field_max:
                                electric_field_max = electric_field
                                observation_angle_max = observation_angle
                                solution_max = ii
                                solution_type_max = solution
                                max_eventid = eventid
                                index_station_max = index_station
                                index_antenna_max = index_antenna
                    
                    if plot == True:
                        origin = []
                        for index_antenna in temporary_info[temporary_info['has_solution'] == 1]['antenna']:
                            origin.append([self.stations[index_station].antennas[index_antenna].x,self.stations[index_station].antennas[index_antenna].y,self.stations[index_station].antennas[index_antenna].z])
                
                        #origin = [x_antenna,y_antenna,z_antenna]
                        neutrino_loc = [x_0, y_0, z_0]
                        if len(temporary_info[temporary_info['has_solution'] == 1]) > 0:
                            fig = plotGeometry(origin,neutrino_loc,phi_0,temporary_info[temporary_info['has_solution'] == 1])
                            if save_fig == True:
                                try:
                                    fig.savefig('./%s_antenna%i_all_solutions-event%i.pdf'%(self.outfile.split('/')[-1].replace('.h5',''),index_antenna,eventid),bbox_inches='tight')
                                    pylab.close(fig)
                                except:
                                    print('Failed to save image ./%s_antenna%i_all_solutions-event%i.pdf'%(self.outfile.split('/')[-1].replace('.h5',''),index_antenna,eventid))
                            else:
                                print('Attempted to plot, but no signals for this event are present')
                    #NEED WAY TO HANDLE SCENARIO WHERE NONE!!! ARE A SOLUTION.  PICK UP ON THIS.
                    dic_max['d'] = self.in_dic_array[station_label][antenna_label][solution_type_max]['d'][max_eventid]
                    dic_max['r'] = self.in_dic_array[station_label][antenna_label][solution_type_max]['r'][max_eventid]
                    dic_max['t'] = self.in_dic_array[station_label][antenna_label][solution_type_max]['t'][max_eventid]
                    dic_max['theta'] = self.in_dic_array[station_label][antenna_label][solution_type_max]['theta'][max_eventid]
                    dic_max['theta_ant'] = self.in_dic_array[station_label][antenna_label][solution_type_max]['theta_ant'][max_eventid]
                    dic_max['a_h'] = self.in_dic_array[station_label][antenna_label][solution_type_max]['a_h'][max_eventid]
                    dic_max['a_v'] = self.in_dic_array[station_label][antenna_label][solution_type_max]['a_v'][max_eventid]
                    dic_max['z'] = self.in_dic_array[station_label][antenna_label][solution_type_max]['z'][max_eventid]
                    info[ sum([len(self.stations[s].antennas) for s in range(0,index_station)]) + index_antenna] = temporary_info[numpy.argmax(temporary_info['electric_field'])]
                #self.in_flag_array[station_label][antenna_label][solution] #Want a way to include this in info.  I.e. a binary has_solution flag for each antenna for each event
                else:
                    #This event has no solution for this antenna
                    has_solution = 0
                    info[ sum([len(self.stations[s].antennas) for s in range(0,index_station)]) + index_antenna] = numpy.array([(eventid,index_station,index_antenna,has_solution,'',-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0)],dtype = self.info_dtype)
            if plot == True:
                origin = []
                for index_antenna in info[info['has_solution'] == 1]['antenna']:
                    origin.append([self.stations[index_station].antennas[index_antenna].x,self.stations[index_station].antennas[index_antenna].y,self.stations[index_station].antennas[index_antenna].z])
                
                #x_antenna = self.stations[index_station].antennas[index_antenna].x
                neutrino_loc = [x_0, y_0, z_0]
                if len(info[info['has_solution'] == 1]) > 0:
                    fig = plotGeometry(origin,neutrino_loc,phi_0,info[info['has_solution'] == 1])
                    if save_fig == True:
                        try:
                            fig.savefig('./%s_all_antennas-event%i.pdf'%(self.outfile.split('/')[-1].replace('.h5',''),eventid),bbox_inches='tight')
                            pylab.close(fig)
                        except:
                            print('Failed to save image ./%s_antenna%i_all_solutions-event%i.pdf'%(self.outfile.split('/')[-1].replace('.h5',''),index_antenna,eventid))
                    else:
                        print('Attempted to plot, but no signals for this event are present')        
        p_detect = numpy.any(info['has_solution'])
        
        if electricFieldDomain == 'time':
            summed_signals = False
            for index_station in range(0, len(self.stations)):
                station_label = 'station'+str(index_station)
                for index_antenna in range(0, len(self.stations[index_station].antennas)):
                    antenna_label = self.config['antennas']['types'][index_antenna]                    
                    u_in = []
                    E_in = []
                    if summed_signals == False:
                        max_E_in_val = 0
                        max_E_val_solution_type = ''
                    for solution in self.stations[index_station].antennas[index_antenna].lib.solutions:
                        u_in.append(u_signals[station_label][antenna_label][solution])
                        E_in.append(E_signals[station_label][antenna_label][solution])
                        if summed_signals == False:
                            if self.in_flag_array[station_label][antenna_label][solution][eventid]:
                                if max(numpy.fabs(E_signals[station_label][antenna_label][solution])) >= max_E_in_val:
                                    max_E_in_val = max(abs(E_signals[station_label][antenna_label][solution]))
                                    max_E_val_solution_type = solution
                    #u_in/E_in should be an array of times with dimensions (n_signal , n_timestep )
                    if numpy.size(u_in) != 0:
                        #print(numpy.size(u_in))
                        #print(numpy.shape(u_in))
                        #print(u_in)
                        #print(numpy.size(E_in))
                        #print(numpy.shape(E_in))
                        #print(E_in)
                        if summed_signals == True:
                            E_out, u_out = gnosim.interaction.askaryan.addSignals(u_in,E_in,plot=False)
                            u_signals[station_label][antenna_label] = u_out
                            E_signals[station_label][antenna_label] = E_out
                        else:
                            E_in = numpy.array(E_in)
                            u_in = numpy.array(u_in)
                            
                            u_signals[station_label][antenna_label] = u_signals[station_label][antenna_label][max_E_val_solution_type]
                            E_signals[station_label][antenna_label] = E_signals[station_label][antenna_label][max_E_val_solution_type]
                    else:
                        u_signals[station_label][antenna_label] = []
                        E_signals[station_label][antenna_label] = []
                if plot == True:
                    #might need to account for when signals are not present in certain detectors
                    if len(self.stations[index_station].antennas) > 4:
                        ax = pylab.subplot((len(self.stations[index_station].antennas)+1)//2,2,1)
                    else:
                        ax = pylab.subplot(len(self.stations[index_station].antennas),1,1)
                    pylab.title('Event %i'%(eventid))
                    u_min = 1e20
                    u_max = -1e20
                    E_min = 1e20
                    E_max = -1e20
                    for index_antenna in range(0, len(self.stations[index_station].antennas)):
                        if numpy.size(u_signals[station_label][antenna_label]) != 0:
                            u_min = numpy.min([u_min,min(u_signals[station_label][antenna_label])])
                            u_max = numpy.max([u_max,max(u_signals[station_label][antenna_label])])
                            E_min = numpy.min([E_min,min(E_signals[station_label][antenna_label])])
                            E_max = numpy.max([E_max,max(E_signals[station_label][antenna_label])])
                    if numpy.logical_and(u_min != 1e20, u_max != -1e20) == True:
                        pylab.xlim(u_min,u_max)
                        pylab.ylim(min(E_min*0.99,E_min*1.01),max(E_max*0.99,E_max*1.01))
                        for index_antenna in range(0, len(self.stations[index_station].antennas)):
                            pylab.subplot(len(self.stations[index_station].antennas),1,index_antenna+1,sharex=ax,sharey=ax)
                            pylab.plot(u_signals[station_label][antenna_label],E_signals[station_label][antenna_label],label='s%ia%i'%(index_station,index_antenna),linewidth=0.1)
                            pylab.ylabel('E$_{%i}$ (V/m)'%(eventid),fontsize=12)
                            pylab.legend(fontsize=8)
                        pylab.xlabel('t-t_emit (ns)',fontsize=12)
                        #pylab.show(block=True)
                        if save_fig == True:
                            try:
                                pylab.savefig('./%s-event%i.pdf'%(self.outfile.split('/')[-1].replace('.h5',''),eventid),bbox_inches='tight')
                            except:
                                print('Failed to save image ./%s-event%i.pdf'%(self.outfile,eventid))
                        else:
                            print('Attempted to plot, but no signals for this event are present')
            #By this point the signals for each antenna should be selected.  Here is where they could be plotted
            #or where phasing could take place. 
        
        return p_interact, p_earth, p_detect, inelasticity, electric_field_max, dic_max, observation_angle_max, solution_max, index_station_max, index_antenna_max, info
        #return p_interact, p_earth, p_detect, electric_field_direct, electric_field_crossover, electric_field_reflect,  dic_direct, dic_crossover, dic_reflect
    
    def griddata_Event(self, x_query, y_query , z_query, method = 'cubic'):
        '''
        This function takes a set of x,y,z coordinates and determines whether each set
        is within the set of solutions.  First it checks the points against the
        self.concave_hull bounding functions, and the locates which triangle each
        point is within referencing the self.delaunay grid created elsewhere.  
        Using barycentric weighting of the 3 corners of the triangle, an average
        value is calculated to estimate the coresponding information about the pair.
        
        Right now this expects r_query,z_query to be centered coordinates
        '''
        
        self.in_dic_array = {}
        self.in_flag_array = {}
        for index_station in range(0, len(self.stations)):
            station_label = 'station'+str(index_station)
            self.in_dic_array[station_label] = {}
            self.in_flag_array[station_label] = {}
            for index_antenna, antenna_label in enumerate(self.lib.keys()):
                #antenna_label = self.lib.keys()[index_antenna]
                print('Running Events Through Griddata Interpolation for:', antenna_label)
                x_antenna = self.stations[index_station].antennas[index_antenna].x
                y_antenna = self.stations[index_station].antennas[index_antenna].y
                #z_antenna = self.stations[index_station].antennas[index_antenna].z
                r_query = numpy.sqrt((x_query - x_antenna)**2 + (y_query - y_antenna)**2)
                
                if ((type(r_query) != numpy.ndarray) or (type(z_query) != numpy.ndarray)):
                    if ((type(r_query) != list) or (type(z_query) != list)):
                        r_query = numpy.array([r_query])
                        z_query = numpy.array([z_query])
                    else:
                        r_query = numpy.array(r_query)
                        z_query = numpy.array(z_query)
                
                self.in_dic_array[station_label][antenna_label] = {}
                self.in_flag_array[station_label][antenna_label] = {}
                for solution in self.stations[index_station].antennas[index_antenna].lib.solutions:
                    print('\tSolution Type:', solution)
                    self.in_dic_array[station_label][antenna_label][solution] = {}

                    

                    in_bound = numpy.logical_and((z_query >= self.concave_hull[antenna_label][solution]['z_min']),z_query <= self.concave_hull[antenna_label][solution]['z_max'])
                    r_in_hull = numpy.logical_and((r_query >= self.concave_hull[antenna_label][solution]['f_inner_r_bound'](z_query)),(r_query <= self.concave_hull[antenna_label][solution]['f_outer_r_bound'](z_query)))
                    has_solution = numpy.logical_and(in_bound,r_in_hull)
                    
                    if numpy.all( has_solution == False ):
                        print('No solutions found for', antenna_label, solution)
                        self.in_flag_array[station_label][antenna_label][solution] = numpy.array([False]*len(r_query))
                        for key in self.lib[antenna_label].data[solution].keys():
                            self.in_dic_array[station_label][antenna_label][solution][key] = []
                        continue

                    zm_query = numpy.ma.masked_array( z_query, mask = ~has_solution)
                    rm_query = numpy.ma.masked_array( r_query, mask = ~has_solution)
                    no_solution_index = numpy.where(~has_solution)
                    #no_solution_index = numpy.where(simplices_index == -1) #make sure this is as I want.  
                    self.in_flag_array[station_label][antenna_label][solution] = has_solution
                    
                    
                    events_per_calc = 100000
                    left_event = 0
                    
                    for key in self.lib[antenna_label].data[solution].keys():
                        self.in_dic_array[station_label][antenna_label][solution][key] = numpy.zeros_like(z_query)
                    
                    while left_event < len(z_query):
                        cut = numpy.arange(left_event,min(left_event+events_per_calc,len(z_query)))
                        for key in self.lib[antenna_label].data[solution].keys():
                            self.in_dic_array[station_label][antenna_label][solution][key][cut] = numpy.ma.filled(numpy.ma.masked_array(scipy.interpolate.griddata((self.lib[antenna_label].data[solution]['r'],self.lib[antenna_label].data[solution]['z']),self.lib[antenna_label].data[solution][key],(rm_query[cut], zm_query[cut]),method=method,fill_value=-999.0),mask = ~has_solution[cut],fill_value = -999.0))
                        print('\t\t%s : %i/%i'%(solution,min(left_event+events_per_calc,len(z_query)),len(z_query)))
                        left_event += events_per_calc
                    
                        #Currently can poduce nan values if griddata thinks it is extrapolating but hull says it is in region 
                    
                    
                    '''
                    for key in self.lib[antenna_label].data[solution].keys():
                        self.in_dic_array[station_label][antenna_label][solution][key] = numpy.ma.filled(numpy.ma.masked_array(scipy.interpolate.griddata((self.lib[antenna_label].data[solution]['r'],self.lib[antenna_label].data[solution]['z']),self.lib[antenna_label].data[solution][key],(rm_query, zm_query),method=method,fill_value=-999.0),mask = ~has_solution,fill_value = -999.0))
                        #Currently can poduce nan values if griddata thinks it is extrapolating but hull says it is in region 
                    '''
        print('Finished griddata_Event')
        #return self.in_dic_array[station_label], self.in_flag_array

    def throw(self, energy_neutrino=1.e9, 
              theta_0=None, phi_0=None, x_0=None, y_0=None, z_0=None, 
              anti=False, n_events=10000, detector_volume_radius=6000., detector_volume_depth=3000., 
              outfile=None,seed = None,method = 'cubic',electricFieldDomain = 'freq'):
        '''
        electricFieldDomain should be either freq or time.  The freq domain uses
        the older electric field calculation, while the 'time' uses the new one.
        '''
        self.outfile = outfile
        #seed for testing purposes (if want replicated data)
        if (seed != None):
            numpy.random.seed(seed)
            
        #Loading Hulls (or creating if hulls have not been previously determined in the necessary folder)
        self.concave_hull = {}
        for lkey in self.lib.keys():
            print('Loading Hull For:', lkey)
            self.concave_hull[lkey] = {}
            indir = self.config['antenna_definitions'][lkey]['lib'].replace('*.h5','concave_hull')
            if os.path.isdir(indir) == False:
                print('Hull not previously generated, calculating now.')
                (self.lib[lkey]).saveEnvelope( self.config['antenna_definitions'][lkey]['lib'].replace('*.h5','') )
            chull = (self.lib[lkey]).loadEnvelope( indir ,store_fit_data = False)
            for dkey in self.lib[lkey].data.keys():
                self.concave_hull[lkey][dkey] = {}
                self.concave_hull[lkey][dkey]['z_min'] = chull[dkey]['z_min']
                self.concave_hull[lkey][dkey]['z_max'] = chull[dkey]['z_max']
                self.concave_hull[lkey][dkey]['f_inner_r_bound'] = chull[dkey]['f_inner_r_bound']
                self.concave_hull[lkey][dkey]['f_outer_r_bound'] = chull[dkey]['f_outer_r_bound']
        
        energy_neutrino = energy_neutrino * numpy.ones(n_events)
    
        #Randomizing direction neutrino came from (characterized by phi_0, theta_0
        phi_0 = numpy.random.uniform(0., 360., size=n_events) # deg
        theta_0 = numpy.degrees(numpy.arccos(numpy.random.uniform(1., -1., size=n_events))) # deg
        
        # Producing neutrino randomly in approximate detector volume
        # NEW CURVATURE
        alpha_max_radians = detector_volume_radius / gnosim.utils.constants.radius_earth # radians
        alpha = numpy.arccos(numpy.random.uniform(1., numpy.cos(alpha_max_radians), size=n_events)) # radians
        r = gnosim.utils.constants.radius_earth * alpha # m #Seems to be using small angle formula
        # NEW CURVATURE

        phi_vertex = numpy.random.uniform(0., 360., size=n_events) # deg
    
        x_0 = r * numpy.cos(numpy.radians(phi_vertex))
        y_0 = r * numpy.sin(numpy.radians(phi_vertex))
        z_0 = numpy.random.uniform(-1. * detector_volume_depth, 0., size=n_events) # m #maybe something to double check later, make sure doesn't give solutions outside of earth
        #TESTING FOR ERROR SCENARIO
        z_0[0] = 1000.0
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
        info = numpy.empty(n_events * self.n_antenna , dtype = self.info_dtype)

        
        print('About to run griddata_Event:')
        ############################################################################
        #Interpolating values from using griddata:
        ############################################################################
        
        #If memory becomes an issue this might need to be adapted to run for chunks of events 
        self.griddata_Event(x_0, y_0, z_0, method = method)   
        print('Succesfully ran griddata_Event:')
        if electricFieldDomain == 'time':
            print('Loading Response Functions')
            self.h_fft,self.sys_fft,self.freqs_response = gnosim.interaction.askaryan.loadSignalResponse()
            
        ############################################################################
        #Using interpolated values for further calculations on an event/event basis:
        ############################################################################
        for ii in range(0, n_events):
            if(ii%(n_events/100) == 0):
                print ('Event (%i/%i)'%(ii, n_events)) #might want to comment out these print statements to run faster and spew less
            if (ii%(n_events/20) == 0):
                plot = True  
                #right now this is a quick way to check some events.  In the future 
                #might want to save triggered events only?  Will need to figure out 
                #a better trigger method to do that.  For large simulations would 
                #risk saving crazy amounts of figures.
            else:
                plot = False
            p_interact[ii], p_earth[ii], p_detect[ii], inelasticity[ii], electric_field_max[ii], dic_max, observation_angle_max[ii], solution_max[ii], index_station_max[ii], index_antenna_max[ii], info[(ii * self.n_antenna ):((ii+1) * self.n_antenna )] \
                = self.event(energy_neutrino[ii], phi_0[ii], theta_0[ii], x_0[ii], y_0[ii], z_0[ii], ii, anti=anti,electricFieldDomain = electricFieldDomain,plot=plot,save_fig=plot)

            if p_detect[ii] == 1.:
                t_max[ii] = dic_max['t']
                d_max[ii] = dic_max['d']
                theta_ray_max[ii] = dic_max['theta']
                theta_ant_max[ii] = dic_max['theta_ant']
                a_v_max[ii] = dic_max['a_v']
                a_h_max[ii] = dic_max['a_h']

        cut = (p_detect == 1.)
        cut_inv = numpy.logical_not(cut)
        
        
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
            #file.attrs['ice_model'] = gnosim.earth.antarctic.ice_model_default
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
            file.create_dataset('info', ( n_events * self.n_antenna , ) , dtype=self.info_dtype, compression='gzip', compression_opts=9, shuffle=True)


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
            file['info'][...] = info

            file.close()

############################################################

if __name__ == "__main__":

    config_file = sys.argv[1]
    energy_neutrino = float(sys.argv[2]) # GeV
    n_events = int(sys.argv[3])
    index = int(sys.argv[4])
    #solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2'])
    solutions = numpy.array(['direct', 'cross', 'reflect'])
    #detector_volume_radius = float(sys.argv[5]) # m, 1200 for Ross surface, 51000 for Minna bluff, >6000 for subterranean
    #detector_volume_depth = float(sys.argv[6]) # m, 500 for Ross and Minna, 3000 for subterranean

    #SEED FOR TESTNG:
    seed = None
    config_file_fix = config_file.replace('/home/dsouthall/Projects/GNOSim/','')
    config_file_fix = config_file_fix.replace('gnosim/sim/ConfigFiles/Config_dsouthall/','')
    config_file_fix = config_file_fix.replace('./','')
    if (seed != None):
        outfile = '/home/dsouthall/Projects/GNOSim/Output/results_2018_Oct_%s_%.2e_GeV_%i_events_%i_seed_%i.h5'%(config_file_fix.replace('.py', ''),
                                                                    energy_neutrino,
                                                                    n_events,
                                                                    seed,
                                                                    index)
        print('!!!Using Seed!!! Seed: ', seed, '\nOutfile Name: \n', outfile)
    else:
        outfile = '/home/dsouthall/Projects/GNOSim/Output/results_2018_Oct_%s_%.2e_GeV_%i_events_%i.h5'%(config_file_fix.replace('.py', ''),
                                                                energy_neutrino,
                                                                n_events,
                                                                index)
        print('Outfile Name: \n', outfile)
    if os.path.isfile(outfile):
        print('Outfile Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(outfile))
        outfile = './' + outfile.split('/')[-1]
        while os.path.isfile(outfile):
            outfile = outfile.replace('.h5','_new.h5')
            
    #f = h5py.File(outfile, 'w')
   
    my_sim = Sim(config_file, solutions=solutions,pre_split = True)
    my_sim.throw(energy_neutrino, n_events=n_events, 
                 detector_volume_radius=my_sim.config['detector_volume']['radius'],
                 detector_volume_depth=my_sim.config['detector_volume']['depth'],
                 outfile=outfile,seed=seed,electricFieldDomain = 'time')

    #python /home/dsouthall/Projects/GNOSim/sim/antarcticsim.py config energy n_events index 
    #python /home/dsouthall/Projects/GNOSim/gnosim/sim/antarcticsim.py /home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_octo_-200_polar_120_rays.py 1.0e8 50000 1 
    #f.close()

############################################################


