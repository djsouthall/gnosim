#!/usr/bin/env python

"""
Simulation
"""

import sys
import numpy
import h5py
import matplotlib
#matplotlib.use('Agg') #Use so it doesn't popup plots during the running of the sime
import pylab
#pylab.ioff() #Use so it doesn't popup plots during the running of the sime
import json
import yaml
import os
import os.path
import glob
import scipy
import scipy.signal
import math
from matplotlib import gridspec
import pandas
import time
import multiprocessing
import concurrent.futures

from multiprocessing import cpu_count
import threading

sys.path.append("/home/dsouthall/Projects/GNOSim/")
import gnosim.utils.quat
import gnosim.earth.earth
import gnosim.earth.antarctic
import gnosim.trace.refraction_library_beta
#from gnosim.trace.refraction_library_beta import *
import gnosim.interaction.askaryan
import gnosim.sim.detector
import gnosim.sim.fpga
pylab.ion() #this turns interactive mode on.  I should test with this off

############################################################

import cProfile, pstats, io

def profile(fnc):
    """
    A decorator that uses cProfile to profile a function
    This is lifted from https://osf.io/upav8/
    
    Required imports:
    import cProfile, pstats, io
    
    To use, decorate function of interest by putting @profile above
    its definition.
    
    Meanings:
    ncalls  - for the number of calls.  When there are two numbers (for example 3/1), 
              it means that the function recursed. The second value is the number 
              of primitive calls and the former is the total number of calls. Note 
              that when the function does not recurse, these two values are the same, 
              and only the single figure is printed.
    tottime - for the total time spent in the given function (and excluding time made 
              in calls to sub-functions)
    percall - is the quotient of tottime divided by ncalls
    cumtime - is the cumulative time spent in this and all subfunctions (from invocation 
              till exit). This figure is accurate even for recursive functions.
    percall - is the quotient of cumtime divided by primitive calls
    filename:lineno(function) - provides the respective data of each function
    """
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s)
        ps.strip_dirs().sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


class Sim:

    def __init__(self, config_file,solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2']),pre_split = False):
        #pre_split False unless using a library already sorted into different
        #directories by solution type.
        self.config_file = config_file
        self.pre_split = pre_split
        #self.config = eval(''.join(open(config_file).readlines()))
        self.config = yaml.load(open(config_file))
        self.detector()
        #self.info_dtype = numpy.dtype([('eventid','i'),('station','i'),('antenna','i'),('has_solution','i'),('pre_triggered','i'),('triggered','i'),('solution','S10'),('time','f'),('distance','f'),('theta_ant','f'),('observation_angle','f'),('electric_field','f'),('electric_field_digitized','f'),('dominant_freq','f'),('a_h','f'),('a_v','f'),('SNR','f'),('beam_pattern_factor','f'),('fpga_max','i'),('seed','i')])
        # List attributes of interest
        self.keys = ['t', 'd', 'theta', 'theta_0', 'a_v', 'a_h']
        self.n_antenna = sum([len(self.stations[s].antennas) for s in range(len(self.stations))])
        self.solutions = solutions
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
                if numpy.isin('orientations',list(self.config['antennas'].keys())) == True:                                   
                    antenna.alpha_deg, antenna.beta_deg, antenna.gamma_deg =  self.config['antennas']['orientations'][jj]
                else:
                    antenna.alpha_deg, antenna.beta_deg, antenna.gamma_deg = [0.0,0.0,0.0]
                
                default_sampling_rate = 1.5 #GHz
                default_sampling_bits = 7
                
                if numpy.isin('DAQ',list(self.config.keys())) == True:
                    if numpy.isin('sampling_rate_GHz',list(self.config['DAQ'].keys())) == True:
                        self.sampling_rate = self.config['DAQ']['sampling_rate_GHz']
                    else:
                        print('Sampling rate not given in config file, setting to default')
                        self.sampling_rate = default_sampling_rate
                    self.digital_sampling_period = 1.0  /  self.sampling_rate
                    if numpy.isin('sampling_bits',list(self.config['DAQ'].keys())) == True:
                        self.sampling_bits = self.config['DAQ']['sampling_bits']
                    else:
                        print('Sampling bits not given in config file, setting to default')
                        self.sampling_bits = default_sampling_bits
                else:
                    print('Config file is not using most recent format and does not include DAQ catagory, setting default values for sampling rate and sampling bits')
                    self.sampling_rate = default_sampling_rate
                    self.digital_sampling_period = 1.0  /  self.sampling_rate 
                    self.sampling_bits = default_sampling_bits

                antenna.R = gnosim.sim.detector.eulerRotationMatrix(numpy.deg2rad(antenna.alpha_deg), numpy.deg2rad(antenna.beta_deg), numpy.deg2rad(antenna.gamma_deg))
                antenna.R_inv = numpy.linalg.inv(antenna.R)
                self.stations[ii].antennas.append(antenna)
    #@profile 
    def event(self, energy_neutrino, phi_0, theta_0, x_0, y_0, z_0, eventid, inelasticity, anti=False,
        electricFieldDomain = 'freq',include_noise = False,plot_signals=False,plot_geometry=False,summed_signals=False,
        trigger_threshold = 0,trigger_threshold_units = 'V',plot_filetype_extension = 'svg',image_path = './', 
        random_time_offset = 0, dc_offset = 0, do_beamforming = False, output_all_solutions = False, pre_trigger_angle = None,
        event_seed = None):
        '''
        Note that the freq domain option is outdated and does not just do the same thing differently.  It does 
        what older version of the code attempted did.  Does not have a lot of the newer additions such as noise.  
        '''
        effective_energy_neutrino = energy_neutrino*inelasticity
        triggered = False
        
        #Calculations required for pretrigger
        #Pretrigger should be on observation angle to check how on-cone each solution will be
        #Eventually hope to add a dependance on E/d
        
        info = numpy.zeros(  self.n_antenna  , dtype = self.info_dtype) 
        temporary_info = numpy.zeros(  len(self.solutions)*self.n_antenna  , dtype = self.info_dtype)
        
        x_antennas         = numpy.zeros( len(self.solutions)*self.n_antenna )
        y_antennas         = numpy.zeros( len(self.solutions)*self.n_antenna )
        z_antennas         = numpy.zeros( len(self.solutions)*self.n_antenna )
        thetas_event_deg   = numpy.zeros( len(self.solutions)*self.n_antenna )
        has_solution_array = numpy.zeros( len(self.solutions)*self.n_antenna )
        station_wide_solution_index = -1 #add one to it for each antenna, used for indexing temporary array
        for index_station in range(0, len(self.stations)):
            # Loop over station antennas
            station_label = 'station'+str(index_station)
            for index_antenna in range(0, len(self.stations[index_station].antennas)):
                antenna_label = self.config['antennas']['types'][index_antenna]
                for solution in self.stations[index_station].antennas[index_antenna].lib.solutions:
                    station_wide_solution_index += 1
                    has_solution_array[station_wide_solution_index] = self.in_flag_array[station_label][antenna_label][solution][eventid]
                    x_antennas[station_wide_solution_index] = self.stations[index_station].antennas[index_antenna].x
                    y_antennas[station_wide_solution_index] = self.stations[index_station].antennas[index_antenna].y
                    z_antennas[station_wide_solution_index] = self.stations[index_station].antennas[index_antenna].z
                    thetas_event_deg[station_wide_solution_index] = self.in_dic_array[station_label][antenna_label][solution]['theta'][eventid] #values that fail has_solution_array should already be filled with a -999.0 value from griddata
                    temporary_info[station_wide_solution_index] = numpy.array([(eventid,index_station,index_antenna,has_solution_array[station_wide_solution_index],0,0,solution,self.in_dic_array[station_label][antenna_label][solution]['t'][eventid],self.in_dic_array[station_label][antenna_label][solution]['d'][eventid],self.in_dic_array[station_label][antenna_label][solution]['theta_ant'][eventid],-999.0,-999.0,-999.0,-999.0,self.in_dic_array[station_label][antenna_label][solution]['a_h'][eventid],self.in_dic_array[station_label][antenna_label][solution]['a_v'][eventid],-999.0,-999.0,-999,event_seed)],dtype = self.info_dtype)
                    
        index_of_refraction_at_neutrino = gnosim.earth.antarctic.indexOfRefraction(z_0, ice_model=self.config['detector_volume']['ice_model'])
        cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./index_of_refraction_at_neutrino))
        rs = numpy.sqrt((x_0 - x_antennas)**2 + (y_0 - y_antennas)**2)
        phi_rays = numpy.degrees(numpy.arctan2(y_0 - y_antennas, x_0 - x_antennas)) % 360. # deg
        vector_rays = gnosim.utils.quat.angToVec(phi_rays, thetas_event_deg) #at neutrino event
        vector_neutrinos = numpy.tile(gnosim.utils.quat.angToVec(phi_0, theta_0),(len(phi_rays),1)) # Direction neutrino came from
        observation_angles = gnosim.utils.quat.angTwoVec(vector_neutrinos, vector_rays) # deg
        observation_angles[~has_solution_array.astype(bool)] = -999.0
        if pre_trigger_angle == None:
            #Pre trigger passes for everything with solution
            pre_triggers = has_solution_array 
        else:
            #Pre trigger passes within angular window
            pre_triggers = numpy.abs(observation_angles - cherenkov_angle_deg) < pre_trigger_angle #True for each solution which has observation within tolerance
            pre_triggers[~has_solution_array.astype(bool)] = 0
        temporary_info['pre_triggered'] = pre_triggers
        temporary_info['observation_angle'] = observation_angles
        
        #Calculations not required for pretrigger
        
        
        event_electric_field_max = 0.
        dic_max = {}
        observation_angle_max = -999.
        solution_max = -999.
        index_station_max = -999.
        index_antenna_max = -999.
        event_observation_angle_max = -999.
        event_solution_max = -999.
        event_index_station_max = 0
        event_index_antenna_max = 0
        SNR_max = -999.
        signals_out = {}
        #Note p_interact has no random component
        p_interact = gnosim.earth.earth.probInteract(energy_neutrino, z_0, anti=anti) #Should not use effective_energy_neutrino, as that only becomes effective AFTER interaction, these test if there is an interaction at all
        
        # Probability for neutrino to make it through the Earth
        #note p_earth has no random components
        p_earth = gnosim.earth.earth.probSurvival(energy_neutrino, theta_0, elevation=z_0, anti=anti) #Should not use effective_energy_neutrino, as that only becomes effective AFTER interaction, these test if there is an interaction at all

        
        #Only do triggering if any pass pre_trigger
        if numpy.any(temporary_info['pre_triggered'] == True):
            #Set event seed:
            #Seperate RandomState object used for each event to futur proof for multithreading
            #to avoid issues with reproducability with a global RandomState
            random_local = numpy.random.RandomState(seed = event_seed)
            #random_local.normal() etc.,  would need to pass random_local to any function that uses random
            
            if electricFieldDomain == 'time':
                time_analog = {}
                V_analog = {}
                time_digital = {}
                V_digital = {}
                if numpy.logical_and(include_noise == True,summed_signals == True):
                    V_just_noise = {}
                    
                #The below calculation is the form factor portion fp of the askaryan calculation
                #It only depends on the times you are calculating it and the Energy, and this can
                #be calculated in advance.  It should be calculated with the energy including the
                #inelasticity etc, and thus can't be done once per sim as the energy is not just one
                #value.
                askryan_fp = gnosim.interaction.askaryan.F_p(effective_energy_neutrino,self.signal_times,index_of_refraction_at_neutrino,1.0)
                askryan_fp_fft = numpy.fft.rfft(askryan_fp)
                    
            #Calculating the times to digitize the signal in advance, should be done after a pretrigger
            minimum_time = 1e20
            maximum_time = -1e20
            #can probably be done quicker from temporary_info['time'][temporary_info['has_solution'] == 1]
            # Loop over stations
            '''
            for index_station in range(0, len(self.stations)):
                # Loop over station antennas
                station_label = 'station'+str(index_station)
                for index_antenna in range(0, len(self.stations[index_station].antennas)):
                    antenna_label = self.config['antennas']['types'][index_antenna]
                    for solution in self.stations[index_station].antennas[index_antenna].lib.solutions:
                        if self.in_flag_array[station_label][antenna_label][solution][eventid]:
                            if minimum_time > self.signal_times[0] + self.in_dic_array[station_label][antenna_label][solution]['t'][eventid]:
                                minimum_time = self.signal_times[0] + self.in_dic_array[station_label][antenna_label][solution]['t'][eventid]
                            if maximum_time < self.signal_times[-1] + self.in_dic_array[station_label][antenna_label][solution]['t'][eventid]:
                                maximum_time = self.signal_times[-1] + self.in_dic_array[station_label][antenna_label][solution]['t'][eventid]
                            #minimum_time = numpy.min([minimum_time,self.signal_times[0] + self.in_dic_array[station_label][antenna_label][solution]['t'][eventid]])
                            #maximum_time = numpy.max([maximum_time,self.signal_times[-1] + self.in_dic_array[station_label][antenna_label][solution]['t'][eventid]])
            '''
            event_times = numpy.sort(temporary_info[temporary_info['has_solution'] == True]['time'])
            if minimum_time > self.signal_times[0] + event_times[0]:
                minimum_time = self.signal_times[0] + event_times[0]
            if maximum_time < self.signal_times[-1] + event_times[-1]:
                maximum_time = self.signal_times[-1] + event_times[-1]
            
            if minimum_time == 1e20:
                minimum_time = self.signal_times[0]
            if maximum_time == -1e20:
                maximum_time = self.signal_times[-1]
            digital_sample_times = numpy.arange(minimum_time,maximum_time,self.digital_sampling_period) + random_time_offset #these + random_time_offset #these
            
            # Loop over stations
            station_wide_antenna_index = -1 #add one to it for each antenna, used for indexing temporary array
            for index_station in range(0, len(self.stations)):
                # Loop over station antennas
                station_label = 'station'+str(index_station)
                if electricFieldDomain == 'time':
                    time_analog[station_label] = {}
                    V_analog[station_label] = {}
                    time_digital[station_label] = {}
                    V_digital[station_label] = {}
                    if numpy.logical_and(include_noise == True,summed_signals == True):
                        V_just_noise[station_label] = {}
                for index_antenna in range(0, len(self.stations[index_station].antennas)):
                    antenna_label = self.config['antennas']['types'][index_antenna]
                    station_wide_antenna_index += 1
                    first_solution_index = station_wide_antenna_index * len (self.stations[index_station].antennas[index_antenna].lib.solutions) #useful for things that are antenna specific but calculated for each solution
                    if electricFieldDomain == 'time':
                        time_analog[station_label][antenna_label] = {}
                        V_analog[station_label][antenna_label] = {}
                        if numpy.logical_and(include_noise == True,summed_signals == True):
                            V_just_noise[station_label][antenna_label] = {}
                        #time_digital[station_label][antenna_label] = {}
                        #V_digital[station_label][antenna_label] = {}
                        for solution in self.stations[index_station].antennas[index_antenna].lib.solutions:
                            time_analog[station_label][antenna_label][solution] = []
                            V_analog[station_label][antenna_label][solution] = []
                            if numpy.logical_and(include_noise == True,summed_signals == True):
                                V_just_noise[station_label][antenna_label][solution] = []
                            #time_digital[station_label][antenna_label][solution] = []
                            #V_digital[station_label][antenna_label][solution] = []
                    electric_field_max = 0. 
                    #x_antenna = self.stations[index_station].antennas[index_antenna].x
                    #y_antenna = self.stations[index_station].antennas[index_antenna].y
                    #z_antenna = self.stations[index_station].antennas[index_antenna].z
                    #r = numpy.sqrt((x_0 - x_antenna)**2 + (y_0 - y_antenna)**2)
                    r = rs[first_solution_index]
                    '''
                    flag_array = []
                    for index_solution,solution in enumerate(self.stations[index_station].antennas[index_antenna].lib.solutions):
                        flag_array.append(self.in_flag_array[station_label][antenna_label][solution][eventid])
                        #self.info_dtype = numpy.dtype([('eventid','i'),('station','i'),('antenna','i'),('has_solution','i'),('pre_triggered','i'),('triggered','i'),('solution','S10'),('time','f'),('distance','f'),('theta_ant','f'),('observation_angle','f'),('electric_field','f'),('electric_field_digitized','f'),('dominant_freq','f'),('a_h','f'),('a_v','f'),('SNR','f'),('beam_pattern_factor','f'),('fpga_max','i'),('seed','i')])
                        temporary_info[station_wide_antenna_index * len (self.stations[index_station].antennas[index_antenna].lib.solutions) + index_solution] = numpy.array([(eventid,index_station,index_antenna,0,0,0,solution,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999,event_seed)],dtype = self.info_dtype)
                    '''
                    
                    #flag_array = temporary_info[ numpy.logical_and(temporary_info['station'] == index_station, temporary_info['antenna'] == index_antenna)]['has_solution']
                    flag_array = temporary_info[ numpy.logical_and(temporary_info['station'] == index_station, temporary_info['antenna'] == index_antenna)]['has_solution']
                    if numpy.any(flag_array):
                        has_solution = 1
                        #index_of_refraction = gnosim.earth.antarctic.indexOfRefraction(z_0, ice_model=self.config['detector_volume']['ice_model'])
                        #vector_neutrino = gnosim.utils.quat.angToVec(phi_0, theta_0) # Direction neutrino came from
                        #phi_ray = numpy.degrees(numpy.arctan2(y_0 - y_antenna, x_0 - x_antenna)) % 360. # deg
                        #theta_ant_array = []
                        for ii, solution in enumerate(self.stations[index_station].antennas[index_antenna].lib.solutions):
                            #theta_ant_array.append(self.in_dic_array[station_label][antenna_label][solution]['theta_ant'][eventid])
                            total_solution_index = station_wide_antenna_index * len (self.stations[index_station].antennas[index_antenna].lib.solutions) + ii
                            #if self.in_flag_array[station_label][antenna_label][solution][eventid]:
                            if temporary_info[total_solution_index]['has_solution'] == True:
                                #So now it will always get to the above trigger only if ANY pass pre trigger, bit will still only calculate the ones that have solutions regardless if they were the solutions that pretriggered
                                # Direction of outgoing ray from antenna to interaction vertex
                                
                                #vector_ray = gnosim.utils.quat.angToVec(phi_ray, self.in_dic_array[station_label][antenna_label][solution]['theta'][eventid]) #at neutrino event
                                #observation_angle = gnosim.utils.quat.angTwoVec(vector_neutrino, vector_ray) # deg
                                #print(temporary_info)
                                solution_cut = numpy.logical_and(temporary_info['solution'] == solution.encode(),numpy.logical_and(temporary_info['station'] == index_station, temporary_info['antenna'] == index_antenna))
                                observation_angle = temporary_info[ solution_cut ]['observation_angle']
                                #print(observation_angle)
                                #d = self.in_dic_array[station_label][antenna_label][solution]['d'][eventid] # m
                                #self.in_dic_array[station_label][antenna_label][solution]['theta_ant'][eventid]
                                #self.info_dtype = numpy.dtype([('eventid','i'),('station','i'),('antenna','i'),('has_solution','i'),('pre_triggered','i'),('triggered','i'),('solution','S10'),('time','f'),('distance','f'),('theta_ant','f'),('observation_angle','f'),('electric_field','f'),('electric_field_digitized','f'),('dominant_freq','f'),('a_h','f'),('a_v','f'),('SNR','f'),('beam_pattern_factor','f'),('fpga_max','i'),('seed','i')])
                                
                                d = temporary_info[ solution_cut ]['distance'] #m
                                theta_ant_deg = temporary_info[ solution_cut ]['theta_ant'] #deg
                                
                                if self.config['antenna_type']['antenna_type'] == 'dipole': #this is dumb but that's how it is currently set up
                                    #The below must be normalized such that r = 1 (cylindrical coordinates r, not polar)
                                    ray_x = numpy.sin(numpy.deg2rad(theta_ant_deg)) * numpy.cos(numpy.deg2rad(phi_rays[total_solution_index]))
                                    ray_y = numpy.sin(numpy.deg2rad(theta_ant_deg)) * numpy.sin(numpy.deg2rad(phi_rays[total_solution_index]))
                                    ray_z = numpy.cos(numpy.deg2rad(theta_ant_deg))
                                    ray_vector_cartesian_ice_frame = numpy.array([ray_x,ray_y,ray_z]) #returns xyz, but r should be 1
                                    antenna_frame_coefficients = gnosim.sim.detector.antennaFrameCoefficients(self.stations[index_station].antennas[index_antenna].R_inv, ray_vector_cartesian_ice_frame, pre_inv = True) 
                                    
                                    #Calculating beam pattern from theta
                                    #below is the standard explicit way to do this
                                    #antenna_frame_theta_rad = numpy.arccos(antenna_frame_coefficients[2])# Typically it is arccos(z/r) but r is 1 for unit vector
                                    #beam_pattern_factor = numpy.sin(antenna_frame_theta_rad)**2
                                    #Below is a slightly faster way to do this using more geometry
                                    beam_pattern_factor = 1.0 - antenna_frame_coefficients[2]**2 #where r is assumed to be 1 because working with unit vectors
                                else:
                                    beam_pattern_factor = 1.0
                                
                                if electricFieldDomain == 'time':
                                    #this must output a single value for the electric field in the same way old does
                                    #must also output a weighted frequency of sorts.  i.e. the most dominant frequency.
                                    #might need to account for polarity swap on reflection.  Not sure if this is an effect.  
                                    
                                    #up_sample_factor dictates the timestep that the entire calculation is done at.
                                    #the time resolution required for an accurent vector potential calculation varies,
                                    #but likely less than 0.05ns is preferred.  If up_sample_factor = 0 then the 
                                    #time step is closer to 0.32ns.  Near cone the required timing resolution increases. 
                                    #As far as I can tell from profiling, the time of execution of quickSignalSingle is linear
                                    #in upsample.  
                                    
                                    if include_noise == True:
                                        V_noiseless, u , dominant_freq, V_noise, SNR = gnosim.interaction.askaryan.quickSignalSingle( numpy.deg2rad(observation_angle),\
                                          temporary_info[ solution_cut ]['distance'],effective_energy_neutrino,index_of_refraction_at_neutrino,\
                                          self.in_dic_array[station_label][antenna_label][solution]['t'][eventid],self.in_dic_array[station_label][antenna_label][solution]['a_v'][eventid],\
                                          beam_pattern_factor,self.signal_times,self.h_fft,self.sys_fft,self.freqs_response,fp_fft = askryan_fp_fft,plot_signals=False,plot_spectrum=False,plot_potential = False,\
                                          include_noise = True, resistance = 50.0, temperature = 320.0,random_local = random_local)  #expects ovbservation_angle to be in radians (hence the deg2rad on input)
                                        
                                        if summed_signals == True:
                                            V_just_noise[station_label][antenna_label][solution] = numpy.add(V_noise,-V_noiseless) #subtracting away raw signal from noisy signal to get just the noise
                                        electric_array = V_noise
                                    else:
                                        V_noiseless, u , dominant_freq = gnosim.interaction.askaryan.quickSignalSingle( numpy.deg2rad(observation_angle),\
                                          temporary_info[ solution_cut ]['distance'],effective_energy_neutrino,index_of_refraction_at_neutrino,\
                                          self.in_dic_array[station_label][antenna_label][solution]['t'][eventid],self.in_dic_array[station_label][antenna_label][solution]['a_v'][eventid],\
                                          beam_pattern_factor,self.signal_times,self.h_fft,self.sys_fft,self.freqs_response,fp_fft = askryan_fp_fft,plot_signals=False,plot_spectrum=False,plot_potential = False,\
                                          include_noise = False, resistance = 50.0, temperature = 320.0,random_local = random_local)  #expects ovbservation_angle to be in radians (hence the deg2rad on input)
                                        
                                        SNR = -999.
                                        electric_array = V_noiseless
                                    
                                    electric_field = numpy.max(numpy.abs(electric_array))#maybe shouldn't be abs.  Need to change some lines that comapre to this if made not abs here.  
                                    electric_field_digitized = gnosim.sim.fpga.digitizeSignal([0,1],numpy.array([electric_field,electric_field]),[0.5],self.sampling_bits,self.noise_rms,self.scale_noise_to, dc_offset = 0, plot = False)[0] #I don't like this workaround.... :(
                                    
                                    V_analog[station_label][antenna_label][solution] = electric_array
                                    time_analog[station_label][antenna_label][solution] = u
                                else:
                                    if electricFieldDomain != 'freq':
                                        print('Electric field domain selection did not fit one of the\ntwo expected values.  Defaulting to freq.')
                                        sys.stdout.flush()
                                    frequency = numpy.linspace(self.stations[index_station].antennas[index_antenna].frequency_low,
                                               self.stations[index_station].antennas[index_antenna].frequency_high,
                                               100) # GHz
                                    #Note below doesn't use effective_energy_neutrino because it has inelasticity as an input parameter and does that calculation internally.  This is the problem with trying to carry along old code 
                                    electric_field \
                                        = gnosim.interaction.askaryan.electricFieldFrequencyDomainRaw(frequency, temporary_info[ solution_cut ]['distance'], observation_angle,
                                                                                    energy_neutrino, inelasticity, 
                                                                                    'cc', index_of_refraction_at_neutrino) # V m^-1 GHz^-1, dimensionless, expects observation_angle to be in degrees
                                    electric_field *= self.in_dic_array[station_label][antenna_label][solution]['a_v'][eventid] # COME BACK TO GENERALIZE THIS
                                    electric_array, electric_field, dominant_freq = self.stations[index_station].antennas[index_antenna].totalElectricField(frequency, electric_field, theta_ant_deg) # V m^-1 #THIS WAS CHANGED THETA WAS ADDED
                                    SNR = -999.
                                    electric_field_digitized = -999.
                                #temporary_info[total_solution_index] = numpy.array([(temporary_info[total_solution_index]['eventid'],temporary_info[total_solution_index]['station'],temporary_info[total_solution_index]['antenna'],temporary_info[total_solution_index]['has_solution'],temporary_info[total_solution_index]['pre_triggered'],0,solution,self.in_dic_array[station_label][antenna_label][solution]['t'][eventid],self.in_dic_array[station_label][antenna_label][solution]['d'][eventid],theta_ant_deg,observation_angle,electric_field,electric_field_digitized,dominant_freq,self.in_dic_array[station_label][antenna_label][solution]['a_h'][eventid],self.in_dic_array[station_label][antenna_label][solution]['a_v'][eventid],SNR,beam_pattern_factor),('fpga_max','i'),('seed','i')],dtype = self.info_dtype)
                                #temporary_info[station_wide_solution_index] = numpy.array([(eventid,index_station,index_antenna,has_solution_array[station_wide_solution_index],0,0,solution,self.in_dic_array[station_label][antenna_label][solution]['t'][eventid],self.in_dic_array[station_label][antenna_label][solution]['d'][eventid],self.in_dic_array[station_label][antenna_label][solution]['theta_ant'][eventid],-999.0,-999.0,-999.0,-999.0,self.in_dic_array[station_label][antenna_label][solution]['a_h'][eventid],self.in_dic_array[station_label][antenna_label][solution]['a_v'][eventid],-999.0,-999.0,-999,event_seed)],dtype = self.info_dtype)
                                
                                temporary_info[total_solution_index]['electric_field']           = electric_field
                                temporary_info[total_solution_index]['electric_field_digitized'] = electric_field_digitized
                                temporary_info[total_solution_index]['dominant_freq']            = dominant_freq
                                temporary_info[total_solution_index]['SNR']                      = SNR
                                temporary_info[total_solution_index]['beam_pattern_factor']      = beam_pattern_factor
                                
                                if electric_field >= electric_field_max:
                                    electric_field_max = electric_field
                                    observation_angle_max = observation_angle
                                    solution_max = ii
                                    solution_type_max = solution
                                    index_station_max = index_station
                                    index_antenna_max = index_antenna
                                    SNR_max = SNR
                                    
                        if electric_field_max >= event_electric_field_max:
                            event_electric_field_max = electric_field_max
                            event_electric_field_max = electric_field_max
                            event_observation_angle_max = observation_angle_max
                            event_solution_max = solution_max
                            event_index_station_max = index_station_max
                            event_index_antenna_max = index_antenna_max
                            event_SNR_max = SNR_max
                        
                        #theta_ant_array = numpy.array(theta_ant_array)
                        
                        #info[ sum([len(self.stations[s].antennas) for s in range(0,index_station)]) + index_antenna] = temporary_info[numpy.logical_and(numpy.logical_and(temporary_info['station'] == index_station,temporary_info['antenna'] == index_antenna),temporary_info['solution'] == solution_type_max.encode() )] #numpy.array(temporary_info['solution']).astype(str) == solution_type_max
                    else:
                        #This event has no solution for this antenna
                        #has_solution = 0
                        solution_type_max = self.stations[index_station].antennas[index_antenna].lib.solutions[0]
                        #info[ sum([len(self.stations[s].antennas) for s in range(0,index_station)]) + index_antenna] = numpy.array([(eventid,index_station,index_antenna,has_solution,0,0,'',-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999,event_seed)],dtype = self.info_dtype)
                    
                    #Temporary_info but only the max solution type.  Should defult to the first solution type, with values already filled out previously to be filler values
                    info[ sum([len(self.stations[s].antennas) for s in range(0,index_station)]) + index_antenna] = temporary_info[numpy.logical_and(numpy.logical_and(temporary_info['station'] == index_station,temporary_info['antenna'] == index_antenna),temporary_info['solution'] == solution_type_max.encode() )] #numpy.array(temporary_info['solution']).astype(str) == solution_type_max
                    max_solution_cut = numpy.logical_and(temporary_info['solution'] == solution_type_max.encode(),numpy.logical_and(temporary_info['station'] == index_station, temporary_info['antenna'] == index_antenna))
                    dic_max['d'] = temporary_info[ max_solution_cut ]['distance']
                    dic_max['r'] = rs[max_solution_cut]
                    dic_max['t'] = temporary_info[ max_solution_cut ]['time']
                    dic_max['theta'] = thetas_event_deg[max_solution_cut]
                    dic_max['theta_ant'] = temporary_info[ max_solution_cut ]['theta_ant']
                    dic_max['a_h'] = temporary_info[ max_solution_cut ]['a_h']
                    dic_max['a_v'] = temporary_info[ max_solution_cut ]['a_v']
                    dic_max['z'] = z_0#self.in_dic_array[station_label][antenna_label][solution_type_max]['z'][eventid]
                
                
                #Only do triggering if any pass pre_trigger
                if numpy.any(temporary_info[temporary_info['station'] == index_station]['pre_triggered'] == True):
                    # Triggering Code below:
                    signals_out[station_label] = numpy.array([])
                    if numpy.any(info['has_solution']) == True:
                        if electricFieldDomain == 'time':
                            for index_antenna in range(0, len(self.stations[index_station].antennas)):
                                antenna_label = self.config['antennas']['types'][index_antenna]
                                u_in = []
                                V_in = []
                                if numpy.logical_and(include_noise == True,summed_signals == True):
                                    V_just_noise_in = []
                                if summed_signals == False:
                                    max_V_in_val = 0
                                    max_E_val_solution_type = ''
                                for solution in self.stations[index_station].antennas[index_antenna].lib.solutions:
                                    if self.in_flag_array[station_label][antenna_label][solution][eventid]:
                                        u_in.append(time_analog[station_label][antenna_label][solution])
                                        V_in.append(V_analog[station_label][antenna_label][solution])
                                        
                                        if numpy.logical_and(include_noise == True,summed_signals == True):
                                            V_just_noise_in.append(V_just_noise[station_label][antenna_label][solution])
                                        if summed_signals == False:
                                            current_max = max(numpy.fabs(V_analog[station_label][antenna_label][solution]))
                                            if current_max > max_V_in_val:
                                                max_V_in_val = current_max
                                                max_E_val_solution_type = solution
                                
                                u_in = numpy.array(u_in)
                                V_in = numpy.array(V_in)
                                if numpy.logical_and(include_noise == True,summed_signals == True):
                                    V_just_noise_in = numpy.array(V_just_noise_in)
                                
                                if numpy.size(u_in) != 0:
                                    if summed_signals == True:
                                        if include_noise == True:
                                            V_out, u_out = gnosim.interaction.askaryan.addSignals(u_in,V_in,plot=False,V_noise_in = V_just_noise_in, remove_noise_overlap = True)
                                        else:
                                            V_out, u_out = gnosim.interaction.askaryan.addSignals(u_in,V_in,plot=False)
                                    else:
                                        u_out = numpy.array(time_analog[station_label][antenna_label][max_E_val_solution_type])
                                        V_out = numpy.array(V_analog[station_label][antenna_label][max_E_val_solution_type])
                                    Vd_out, ud_out = gnosim.sim.fpga.digitizeSignal(u_out,V_out,digital_sample_times,self.sampling_bits,self.noise_rms,self.scale_noise_to, dc_offset = dc_offset, plot = False)
                                    
                                else:
                                    V_out = numpy.array([])
                                    u_out = numpy.array([])
                                    Vd_out = numpy.array([])
                                    ud_out = numpy.array([])
                            
                                time_analog[station_label][antenna_label] = u_out
                                V_analog[station_label][antenna_label] = V_out
                                time_digital[station_label][antenna_label] = ud_out
                                V_digital[station_label][antenna_label] = Vd_out
                            
                            
                            min_time = digital_sample_times[0]
                            max_time = digital_sample_times[-1]
                            dt = digital_sample_times[1] - digital_sample_times[0]
                        
                        #Triggering
                        if do_beamforming == True:
                            #Here is where I perform the beamforming algorithms. 
                            
                            Vd_out_sync, ud_out_sync  = gnosim.sim.fpga.syncSignals(time_digital[station_label],V_digital[station_label], min_time, max_time, dt)
                            formed_beam_powers, beam_powersums = gnosim.sim.fpga.fpgaBeamForming(ud_out_sync, Vd_out_sync, self.beam_dict , self.config, plot1 = False, plot2 = False, save_figs = False)
                            #Getting max values
                            keep_top = 3
                            
                            beam_label_list = numpy.array(list(beam_powersums.keys()))
                            stacked_beams = numpy.zeros((len(beam_label_list),len(beam_powersums[beam_label_list[0]])))
                            for beam_index, beam_label in enumerate(beam_label_list):
                                stacked_beams[beam_index,:] = beam_powersums[beam_label]
                            max_vals = numpy.max(stacked_beams,axis=1)
                            top_val_indices = numpy.argsort(max_vals)[-numpy.arange(1,keep_top+1)]
                            top_vals = max_vals[top_val_indices] #descending order
                            top_val_beams = beam_label_list[top_val_indices]
                            top_val_theta_ant = numpy.array([self.beam_dict['theta_ant'][beam_label] for beam_label in top_val_beams])
                            #Currently don't know what to do with these values.  They will be written out as I progress but
                            #right now I am just testing that they can be calculate without breaking the simulation.
                            #Right now I am only storing the 3 highest values.  It is likely that I want to store every beam
                            #that satisfies the trigger condiditon?
                            temporary_info['fpga_max'] = top_vals[0]
                            info['fpga_max'] = top_vals[0]
                        
                        if trigger_threshold_units == 'adu':
                            if numpy.size(V_out) > 0:
                                if numpy.any(Vd_out > trigger_threshold):
                                    triggered = True
                        elif trigger_threshold_units == 'fpga':
                            #DO FPGA CODE
                            if top_vals[0] > trigger_threshold:
                                triggered = True
                        else:
                            if numpy.size(V_out) > 0:
                                if numpy.any(V_out > trigger_threshold):
                                    triggered = True
                        if numpy.logical_and(do_beamforming == False, triggered == True):
                            Vd_out_sync, ud_out_sync  = gnosim.sim.fpga.syncSignals(time_digital[station_label],V_digital[station_label], min_time, max_time, dt)
                    else:
                        triggered = False    
                    #RIGHT NOW I AM SWAPPING ORDERING OF WHEN I ADD SIGNALS.  DOING IT BEFORE I DIGITIZE
                    
                    
                            
                    
                            
                    if triggered == True:
                        info['triggered'] = True
                        temporary_info['triggered'] = True
                        signals_out[station_label] = numpy.vstack((Vd_out_sync, ud_out_sync[0,:]))
                        
                        print('Triggered on event %i at Time: %0.3f'%(eventid,time.time() - self.throw_start_time))
                        sys.stdout.flush()
                        if plot_geometry == True:
                            '''
                            origin = []
                            
                            for index_antenna in info[info['has_solution'] == True]['antenna']:
                                origin.append([self.stations[index_station].antennas[index_antenna].x,self.stations[index_station].antennas[index_antenna].y,self.stations[index_station].antennas[index_antenna].z])
                            '''
                            neutrino_loc = [x_0, y_0, z_0]
                            if len(info[info['has_solution'] == True]) > 0:
                                with self.lock:
                                    fig = gnosim.trace.refraction_library_beta.plotGeometry(self.config,neutrino_loc,phi_0,temporary_info[numpy.logical_and(temporary_info['has_solution'] == True,temporary_info['station'] == index_station)])
                                    try:
                                        fig.savefig('%s%s_all_antennas-event%i.%s'%(image_path,self.outfile.split('/')[-1].replace('.h5',''),eventid,plot_filetype_extension),bbox_inches='tight')
                                        pylab.close(fig)
                                    except:
                                        print('Failed to save image %s%s_all_antennas-event%i.%s'%(image_path,self.outfile.split('/')[-1].replace('.h5',''),eventid,plot_filetype_extension))
                                        sys.stdout.flush()
                        
                        
                        if numpy.logical_and(electricFieldDomain == 'time',plot_signals == True):
                            #might need to account for when signals are not present in certain detectors
                            #print('Attempting to plot', eventid)
                            with self.lock:
                                fig = pylab.figure(figsize=(16.,11.2)) #my screensize
                                
                                n_rows = len(self.stations[index_station].antennas)
                                ntables = 4 #With below lines is 5 for beamforming == True
                                height_ratios = [2,2,n_rows+1,n_rows+1]
                                if do_beamforming == True:
                                    ntables += 1
                                    height_ratios.append(0.9*sum(height_ratios))
                                    
                                gs_left = gridspec.GridSpec(n_rows, 2, width_ratios=[3, 2]) #should only call left plots.  pylab.subplot(gs_left[0]),pylab.subplot(gs_left[2]),...
                                gs_right = gridspec.GridSpec(ntables, 2, width_ratios=[3, 2], height_ratios=height_ratios) #should only call odd tables pylab.subplot(gs_right[1])
                                #if do_beamforming == True:
                                #    gs_beam_forming = gridspec.GridSpec(ntables, 3, width_ratios=[3, 1,5], height_ratios=height_ratios)
                                    
                                #ax = pylab.subplot(gs_left[0])
                                
                                first_in_loop = True
                                axis2 = []
                                max_ax1_range = numpy.array([1e20,-1e20])
                                for index_antenna in range(0, n_rows):
                                    antenna_label = self.config['antennas']['types'][index_antenna]
                                    if first_in_loop == True:
                                        first_in_loop = False
                                        ax = pylab.subplot(gs_left[2*index_antenna])
                                    
                                    ax1 = pylab.subplot(gs_left[2*index_antenna],sharex = ax,sharey = ax)
                                    ax2 = ax1.twinx() #this is not perfect and can be janky with zooming.   
                                    axis2.append(ax2)   
                                    c1 = 'b'
                                    c2 = 'r'
                                    #pylab.subplot(n_rows,1,index_antenna+1,sharex=ax,sharey=ax)
                                    if index_antenna == 0:
                                        boolstring = ['False','True']
                                        pylab.title('Event %i, summed_signals = %s'%(eventid,boolstring[int(summed_signals)])) 
                                    ax1.plot(time_analog[station_label][antenna_label],V_analog[station_label][antenna_label],label='s%ia%i'%(index_station,index_antenna),linewidth=0.6,c = c1)
                                    ax2.plot(time_digital[station_label][antenna_label],V_digital[station_label][antenna_label],label='s%ia%i'%(index_station,index_antenna),linewidth=0.4,c = c2)
                                    
                                    if ( n_rows // 2 == index_antenna):
                                        ax1.set_ylabel('V$_{%i}$ (V)'%(eventid),fontsize=12, color=c1)
                                        ax2.set_ylabel('adu',fontsize=12, color=c2)
                                        
                                    ax1.legend(fontsize=8,framealpha=0.0,loc='upper left')
                                    ax1.tick_params('y', colors=c1)
                                    
                                    ax2.legend(fontsize=8,framealpha=0.0,loc='upper right')
                                    ax2.tick_params('y', colors=c2)
                                    ax1_ylim = numpy.array(ax1.get_ylim())
                                    
                                    if ax1_ylim[0] < max_ax1_range[0]:
                                        max_ax1_range[0] = ax1_ylim[0]
                                    if ax1_ylim[1] > max_ax1_range[1]:
                                        max_ax1_range[1] = ax1_ylim[1]
                                        
                                for ax2 in axis2:
                                    ax2.set_ylim(max_ax1_range * self.scale_noise_to / self.noise_rms)
                                    
                                pylab.xlabel('t-t_emit (ns)',fontsize=12)
                                
                                #Making Tables
                                #TABLE 1: Making position table
                                table_fig = pylab.subplot(gs_right[1])
                                
                                table_ax = pylab.gca()
                                table_fig.patch.set_visible(False)
                                table_ax.axis('off')
                                table_ax.axis('tight')
                                x_neutrino = x_0
                                y_neutrino = y_0
                                z_neutrino = z_0
                                r_neutrino = numpy.sqrt(x_neutrino**2 + y_neutrino**2)
                                phi_neutrino = phi_0
                                df = pandas.DataFrame({'x(m)':[ x_neutrino ] , 'y(m)':[ y_neutrino ] , 'z(m)':[ z_neutrino ] , 'r(m)':[ r_neutrino ] , '$\phi_0$(deg)':[ phi_neutrino ] })
                                table = pylab.table(cellText = df.values.round(2), colLabels = df.columns, loc = 'center')
                                table.auto_set_font_size(False)
                                table.set_fontsize(10)
                                pylab.title('Event Info')
                                
                                #TABLE 2: Making Neutrino Energetics table 
                                '''
                                >>> list(reader.keys())
                                ['a_h', 'a_v', 'd', 'electric_field', 'energy_neutrino', 'index_antenna', 
                                'index_station', 'inelasticity', 'info', 'observation_angle', 'p_detect', 
                                'p_earth', 'p_interact', 'phi_0', 'solution', 't', 'theta_0', 'theta_ant', 
                                'theta_ray', 'x_0', 'y_0', 'z_0']
                                
                                event(self, energy_neutrino, phi_0, theta_0, x_0, y_0, z_0, eventid, inelasticity, anti=False,
                                electricFieldDomain = 'freq',include_noise = False,plot_signals=False,plot_geometry=False,summed_signals=False,
                                trigger_threshold = 0,plot_filetype_extension = 'svg',image_path = './'):
                                '''
                                table_fig = pylab.subplot(gs_right[3])
                                
                                table_ax = pylab.gca()
                                table_fig.patch.set_visible(False)
                                table_ax.axis('off')
                                table_ax.axis('tight')
                                
                                df = pandas.DataFrame({'E$_\\nu$ (GeV)':'%0.4g'%(effective_energy_neutrino) , 'Inelasticity':'%0.4g'%inelasticity , 'p_interact':'%0.4g'%p_interact, 'p_earth':'%0.4g'%p_earth},index=[0])
                                #decimals = pandas.Series([3,3,3,3],index = df.columns)
                                table = pylab.table(cellText = df.values , colLabels = df.columns, loc = 'center')
                                table.auto_set_font_size(False)
                                table.set_fontsize(10)
                                
                                
                                
                                #TABLE 3: Making observed angles and attenuations table
                                table_fig = pylab.subplot(gs_right[5])
                                
                                table_ax = pylab.gca()
                                table_fig.patch.set_visible(False)
                                table_ax.axis('off')
                                table_ax.axis('tight')
                                antenna =           ['%i'%i for i in info['antenna'].astype(int)]
                                observation_angle = ['%0.4g'%i for i in info['observation_angle'].astype(float)]
                                theta_ant =         ['%0.4g'%i for i in info['theta_ant'].astype(float)]
                                distance =          ['%0.3g'%i for i in info['distance'].astype(float)]
                                beam_factor =       ['%0.3g'%i for i in info['beam_pattern_factor']]
                                df = pandas.DataFrame({'antenna':antenna , '$\\theta_\mathrm{ant}$ (deg)':theta_ant , '$\\theta_\mathrm{emit}$ (deg)':observation_angle,'d$_\mathrm{path}$ (m)':distance, 'Beam Factor':beam_factor})
                                table = pylab.table(cellText = df.values, colLabels = df.columns, loc = 'center')
                                table.auto_set_font_size(False)
                                table.set_fontsize(10)
                                
                                
                                #TABLE 4: Max Voltage and SNR per Antenna
                                '''
                                >>> list(reader.keys())
                                ['a_h', 'a_v', 'd', 'electric_field', 'energy_neutrino', 'index_antenna', 
                                'index_station', 'inelasticity', 'info', 'observation_angle', 'p_detect', 
                                'p_earth', 'p_interact', 'phi_0', 'solution', 't', 'theta_0', 'theta_ant', 
                                'theta_ray', 'x_0', 'y_0', 'z_0']
                                
                                event(self, energy_neutrino, phi_0, theta_0, x_0, y_0, z_0, eventid, inelasticity, anti=False,
                                electricFieldDomain = 'freq',include_noise = False,plot_signals=False,plot_geometry=False,summed_signals=False,
                                trigger_threshold = 0,plot_filetype_extension = 'svg',image_path = './'):
                                '''
                                table_fig = pylab.subplot(gs_right[7])
                                
                                table_ax = pylab.gca()
                                table_fig.patch.set_visible(False)
                                table_ax.axis('off')
                                table_ax.axis('tight')
                                antenna =           ['%i'%i for i in info['antenna'].astype(int)]
                                electric_field =    ['%0.3g'%i for i in info['electric_field'].astype(float)]
                                dom_freqs =         ['%0.3g'%i for i in (info['dominant_freq']/1e6).astype(float)]
                                SNRs =              ['%0.3g'%i for i in info['SNR'].astype(float)]
                                df = pandas.DataFrame({'antenna':antenna , '$V_\mathrm{max}$ (V)':electric_field , 'SNR':SNRs, '$f_\mathrm{max}$ (MHz)':dom_freqs})
                                table = pylab.table(cellText = df.values , colLabels = df.columns, loc = 'center')
                                table.auto_set_font_size(False)
                                table.set_fontsize(10)
                                
                                #TABLE 5: THE TABLE THAT'S ACTUALLY A PLOT AND ONLY SOMETIMES SHOWS UP DEPENDING ON SETTINGS :D
                                
                                if do_beamforming == True:
                                    
                                    gs_beam_forming = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[9], wspace=0.1, hspace=0.1, width_ratios=[1,6,3])
                                    #table_fig = pylab.subplot(gs_beam_forming[13])
                                    table_fig = pylab.subplot(gs_beam_forming[1])
                                    #table_fig = pylab.subplot(gs_right[9])
                                    table_ax = pylab.gca()
                                    table_fig.patch.set_visible(True)
                                    
                                    for beam_index, beam_label in enumerate(self.beam_dict['beams'].keys()):
                                        table_ax.plot(beam_powersums[beam_label],label = '%s, $\\theta_{ant} = $ %0.2f'%(beam_label,self.beam_dict['theta_ant'][beam_label]),color = self.beam_colors[beam_index])
                                        #print(beam_powersums[beam_label])
                                    #for line_index,line in enumerate(table_ax.lines):
                                    #    line.set_color(self.beam_colors[line_index])
                                    #xlim = table_ax.get_xlim()
                                    #table_ax.set_xlim( ( xlim[0] , xlim[1]*1.5 ) )
                                    
                                    #box = table_ax.get_position()
                                    #table_ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                                    pylab.yticks(rotation=45)
                                    table_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                    #pylab.legend(loc='upper right', bbox_to_anchor=(1.05, 0.5))
                                    #table_ax.axis('tight')
                                    
                                pylab.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.99, top = 0.97, wspace = 0.15, hspace = 0.28)
                                #pylab.show(block=True)
                                try:
                                    pylab.savefig('%s%s-event%i.%s'%(image_path,self.outfile.split('/')[-1].replace('.h5',''),eventid,plot_filetype_extension),bbox_inches='tight')
                                    pylab.close(fig)
                                    #print('Saved image %s%s-event%i.%s'%(image_path,self.outfile,eventid,plot_filetype_extension))
                                except:
                                    print('Failed to save image %s%s-event%i.%s'%(image_path,self.outfile,eventid,plot_filetype_extension))
                                    sys.stdout.flush()
                    else:
                        info['triggered'] = False
                        temporary_info['triggered'] = False
        else:
            dic_max['d'] = temporary_info[0]['distance']
            dic_max['r'] = rs[0]
            dic_max['t'] = temporary_info[0]['time']
            dic_max['theta'] = thetas_event_deg[0]
            dic_max['theta_ant'] = temporary_info[0]['theta_ant']
            dic_max['a_h'] = temporary_info[0]['a_h']
            dic_max['a_v'] = temporary_info[0]['a_v']
            dic_max['z'] = z_0#self.in_dic_array[station_label][antenna_label][solution_type_max]['z'][eventid]
        if output_all_solutions == True:
            info = temporary_info
        p_detect = numpy.any(info['has_solution'])
        return eventid, p_interact, p_earth, p_detect, event_electric_field_max, dic_max, event_observation_angle_max, event_solution_max, event_index_station_max, event_index_antenna_max, info, triggered, signals_out
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
        griddata_initate_time = time.time()
        self.in_dic_array = {}
        self.in_flag_array = {}
        for index_station in range(0, len(self.stations)):
            station_label = 'station'+str(index_station)
            self.in_dic_array[station_label] = {}
            self.in_flag_array[station_label] = {}
            for index_antenna, antenna_label in enumerate(self.lib.keys()):
                #I might be able to multithread of multiprocess each antenna through griddata.
                #They are each checking independant grids/libraries and saving to dictionary seperately
                
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
        print('Finished griddata_Event in ', time.time() - griddata_initate_time, 's')
        #return self.in_dic_array[station_label], self.in_flag_array
    
    def singleAntennaGridDataEvent(self, r_query , z_query, index_station, station_label, index_antenna, antenna_label, method = 'cubic'):
            '''
            This function takes a set of x,y,z coordinates and determines whether each set
            is within the set of solutions.  First it checks the points against the
            self.concave_hull bounding functions, and the locates which triangle each
            point is within referencing the self.delaunay grid created elsewhere.  
            Using barycentric weighting of the 3 corners of the triangle, an average
            value is calculated to estimate the coresponding information about the pair.
            
            Right now this expects r_query,z_query to be centered coordinates
            
            This will do what griddata_Event does but for a single antenna at time. 
            This is intended to be used during multiprocessing. 
            '''
            print('Running Events Through Griddata Interpolation for:', antenna_label)
            if ((type(r_query) != numpy.ndarray) or (type(z_query) != numpy.ndarray)):
                if ((type(r_query) != list) or (type(z_query) != list)):
                    r_query = numpy.array([r_query])
                    z_query = numpy.array([z_query])
                else:
                    r_query = numpy.array(r_query)
                    z_query = numpy.array(z_query)

            out_dic_array = {}
            out_flag_array = {}
            
            for solution in self.stations[index_station].antennas[index_antenna].lib.solutions:
                out_dic_array[solution] = {}
                out_flag_array[solution] = {}
                

                in_bound = numpy.logical_and((z_query >= self.concave_hull[antenna_label][solution]['z_min']),z_query <= self.concave_hull[antenna_label][solution]['z_max'])
                r_in_hull = numpy.logical_and((r_query >= self.concave_hull[antenna_label][solution]['f_inner_r_bound'](z_query)),(r_query <= self.concave_hull[antenna_label][solution]['f_outer_r_bound'](z_query)))
                has_solution = numpy.logical_and(in_bound,r_in_hull)
                
                if numpy.all( has_solution == False ):
                    print('No solutions found for', antenna_label, solution)
                    out_flag_array[solution] = numpy.array([False]*len(r_query))
                    for key in self.lib[antenna_label].data[solution].keys():
                        out_dic_array[solution][key] = []
                    continue

                zm_query = numpy.ma.masked_array( z_query, mask = ~has_solution)
                rm_query = numpy.ma.masked_array( r_query, mask = ~has_solution)
                no_solution_index = numpy.where(~has_solution)
                out_flag_array[solution] = has_solution
                
                
                events_per_calc = 1000000
                left_event = 0
                
                for key in self.lib[antenna_label].data[solution].keys():
                    out_dic_array[solution][key] = numpy.zeros_like(z_query)
                
                while left_event < len(z_query):
                    cut = numpy.arange(left_event,min(left_event+events_per_calc,len(z_query)))
                    for key in self.lib[antenna_label].data[solution].keys():
                        out_dic_array[solution][key][cut] = numpy.ma.filled(numpy.ma.masked_array(scipy.interpolate.griddata((self.lib[antenna_label].data[solution]['r'],self.lib[antenna_label].data[solution]['z']),self.lib[antenna_label].data[solution][key],(rm_query[cut], zm_query[cut]),method=method,fill_value=-999.0),mask = ~has_solution[cut],fill_value = -999.0))
                    print('\t\t%s : %i/%i'%(solution,min(left_event+events_per_calc,len(z_query)),len(z_query)))
                    left_event += events_per_calc
            print('Done Interpolation for:', antenna_label)
            sys.stdout.flush()
            return out_dic_array, out_flag_array
    
    def multiThreadGridDataEvent(self, x_query, y_query , z_query, method = 'cubic',n_cores = 4):
        '''
        This function takes a set of x,y,z coordinates and determines whether each set
        is within the set of solutions.  First it checks the points against the
        self.concave_hull bounding functions, and the locates which triangle each
        point is within referencing the self.delaunay grid created elsewhere.  
        Using barycentric weighting of the 3 corners of the triangle, an average
        value is calculated to estimate the coresponding information about the pair.
        
        Right now this expects r_query,z_query to be centered coordinates
        '''
        griddata_initate_time = time.time()
        
        #initiate threads
        print('Submitting threads')
        thread_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers = n_cores) as executor:
            for index_station in range(0, len(self.stations)):
                station_label = 'station'+str(index_station)
                thread_results[station_label] = {}
                for index_antenna, antenna_label in enumerate(self.lib.keys()):
                    x_antenna = self.stations[index_station].antennas[index_antenna].x
                    y_antenna = self.stations[index_station].antennas[index_antenna].y
                    r_query = numpy.sqrt((x_query - x_antenna)**2 + (y_query - y_antenna)**2)
                    thread_results[station_label][antenna_label] = executor.submit(self.singleAntennaGridDataEvent, r_query , z_query, index_station, station_label, index_antenna, antenna_label, method = 'cubic')
        print('Weaving threads')
        sys.stdout.flush()
        #write outputs of threads
        self.in_dic_array = {}
        self.in_flag_array = {}
        for index_station in range(0, len(self.stations)):
            station_label = 'station'+str(index_station)
            self.in_dic_array[station_label] = {}
            self.in_flag_array[station_label] = {}
            for index_antenna, antenna_label in enumerate(self.lib.keys()):
                self.in_dic_array[station_label][antenna_label] , self.in_flag_array[station_label][antenna_label]  = thread_results[station_label][antenna_label].result()
            
        print('Finished griddata_Event in ', time.time() - griddata_initate_time, 's')
        sys.stdout.flush()
    
    def runEvent(self,energy_neutrino, phi_0, theta_0, x_0, y_0, z_0,eventid,inelasticity,**kwargs):
        '''
        This should contain the code used to run a single event in a thread
        '''
        if (self.n_events//1000 != 0):
            if(eventid%(self.n_events//1000) == 0):
                current_time  = time.time() - self.throw_start_time
                print ('Event (%i/%i) Time: %0.2f s ( %0.4f h)'%(eventid, self.n_events,current_time,current_time/3600.0)) #might want to comment out these print statements to run faster and spew less
        else:
            current_time  = time.time() - self.throw_start_time
            print ('Event (%i/%i) Time: %0.2f s ( %0.4f h)'%(eventid, self.n_events,current_time,current_time/3600.0))
        event_label = 'event%i'%eventid
        eventid, p_interact, p_earth, p_detect, electric_field_max, dic_max, observation_angle_max, \
        solution_max, index_station_max, index_antenna_max, info_i, \
        triggered, signals_out \
            = self.event(energy_neutrino, phi_0, theta_0, x_0, y_0, z_0, eventid, inelasticity, **kwargs)
        return eventid, p_interact, p_earth, p_detect, electric_field_max, dic_max, observation_angle_max, \
        solution_max, index_station_max, index_antenna_max, info_i, triggered, signals_out
        
    def throw(self, energy_neutrino=1.e9, 
              theta_0=None, phi_0=None, x_0=None, y_0=None, z_0=None, phi_vertex = None, r_vertex = None,
              anti=False, n_events=10000, detector_volume_radius=6000., detector_volume_depth=3000., 
              outfile=None,seed = None,method = 'cubic',electricFieldDomain = 'freq',include_noise = False,summed_signals = False,
              plot_geometry = False, plot_signals = False, trigger_threshold = 0,trigger_threshold_units = 'V',plot_filetype_extension = 'svg',image_path = './',
              use_interp_threading = False,use_event_threading = False, do_beamforming = False, n_beams = 15, n_baselines = 2, output_all_solutions = False,save_signals = False,
              pre_trigger_angle = None):
        '''
        electricFieldDomain should be either freq or time.  The freq domain uses
        the older electric field calculation, while the 'time' uses the new one.
        '''
        self.throw_start_time = time.time()
        if trigger_threshold_units == 'fpga':
            if do_beamforming == False:
                print('WARNING!!!')
                print('THRESHOLD GIVEN IN UNITS FOR FPGA, BUT BEAMFORMING NOT ENABLED')
                print('ENABLEING BEAMFORMING')
                sys.stdout.flush()
                do_beamforming = True
            if electricFieldDomain == 'freq':
                print('WARNING!!!')
                print('THRESHOLD GIVEN IN UNITS FOR FPGA, FREQ DOMAIN DOES NOT SUPPORT THIS')
                print('BREAKING')
                sys.stdout.flush()
                return 0
            
        self.outfile = outfile
        self.save_signals = save_signals
        self.n_events = n_events
        self.n_cores = cpu_count()
        self.lock = threading.RLock() #Hopefully fixes multithreading plotting
        #seed for testing purposes (if want replicated data)
        if (seed != None):
            numpy.random.seed(seed)
        else:
            numpy.random.seed()
        #These are used to set the seed for each event to force reproducible randomness when a seed is used.
        #This issue arises say for when using a pre_trigger at different levels.  If different events are
        #run then the position in the seed list is altered, because noise is generated a different number of
        #times.  This way the seed is set within each event (only if the pre_trigger is met, to save time)
        #and each event operates in the same way whether previous events were run or not. 
        event_seeds = numpy.random.randint(numpy.iinfo(numpy.uint32).max,size=self.n_events)
        energy_neutrinos = energy_neutrino * numpy.ones(self.n_events)
    
        #Direction neutrino came from (characterized by phi_0, theta_0)
        if theta_0 == None:
            theta_0 = numpy.degrees(numpy.arccos(numpy.random.uniform(1., -1., size=self.n_events))) # deg
        else:
            print('Using input theta_0')
            if numpy.logical_or(isinstance(theta_0,list) == True,isinstance(theta_0,tuple) == True):
                theta_0 = numpy.array(theta_0,dtype=float)
            else:
                theta_0 = theta_0.astype(float)
            
        if phi_0 == None:
            phi_0 = numpy.random.uniform(0., 360., size=self.n_events) # deg
        else:
            print('Using input phi_0')
            if numpy.logical_or(isinstance(phi_0,list) == True,isinstance(phi_0,tuple) == True):
                phi_0 = numpy.array(phi_0,dtype=float)
            else:
                phi_0 = phi_0.astype(float)
        
        
        #Location of neutrino interaction (characterized by [x_0, y_0, z_0] or [phi_vertex, theta_vertex, z_0] )
        if z_0 == None:
            z_0 = numpy.random.uniform(-1. * detector_volume_depth, 0., size=self.n_events) # m #maybe something to double check later, make sure doesn't give solutions outside of earth
        else:
            print('Using input z_0')
            if numpy.logical_or(isinstance(z_0,list) == True,isinstance(z_0,tuple) == True):
                z_0 = numpy.array(z_0,dtype=float)
            else:
                z_0 = z_0.astype(float)
                
        if numpy.logical_and(phi_vertex  != None, r_vertex != None):
            if numpy.logical_or(x_0 != None, y_0 != None):
                print('phi_vertex and r_vertex are given and will be used')
                print('Ignoring provide x_0, y_0 coordinates, if you want to use x_0, y_0 please ensure phi_vertex = None and r_vertex = None')
            
            if numpy.logical_or(isinstance(phi_vertex,list) == True,isinstance(phi_vertex,tuple) == True):
                phi_vertex = numpy.array(phi_vertex,dtype=float)
            else:
                phi_vertex = phi_vertex.astype(float)
                
            if numpy.logical_or(isinstance(r_vertex,list) == True,isinstance(r_vertex,tuple) == True):
                r_vertex = numpy.array(r_vertex,dtype=float)
            else:
                r_vertex = r_vertex.astype(float)
                
            x_0 = r_vertex * numpy.cos(numpy.radians(phi_vertex))
            y_0 = r_vertex * numpy.sin(numpy.radians(phi_vertex))
            
        elif numpy.logical_and(x_0  != None, y_0 != None):
            if numpy.logical_or(phi_vertex  != None, r_vertex != None):
                print('x_0 and y_0 are given and will be used')
                print('Ignoring provide phi_vertex, r_vertex coordinates, if you want to use phi_vertex, r_vertex please ensure x_0 = None and y_0 = None')
            
            if numpy.logical_or(isinstance(x_0,list) == True,isinstance(x_0,tuple) == True):
                x_0 = numpy.array(x_0,dtype=float)
            else:
                x_0 = x_0.astype(float)
                
            if numpy.logical_or(isinstance(y_0,list) == True,isinstance(y_0,tuple) == True):
                y_0 = numpy.array(y_0,dtype=float)
            else:
                y_0 = y_0.astype(float)
        
        else:
            print('Using randomized phi_vertex and r_vertex to calculate x_0, y_0')
            phi_vertex = numpy.random.uniform(0., 360., size=self.n_events) # deg
            alpha_max_radians = detector_volume_radius / gnosim.utils.constants.radius_earth # radians
            alpha = numpy.arccos(numpy.random.uniform(1., numpy.cos(alpha_max_radians), size=self.n_events)) # radians
            r_vertex = gnosim.utils.constants.radius_earth * alpha
            x_0 = r_vertex * numpy.cos(numpy.radians(phi_vertex))
            y_0 = r_vertex * numpy.sin(numpy.radians(phi_vertex))
        
        len_array = numpy.array([len(x_0),len(y_0),len(z_0),len(phi_0),len(theta_0)])
        
        if numpy.size(numpy.unique(len_array)) != 1:
            print('Breaking early, something went wrong dyring definitions of neutrino coordinate assigment.  The below numbers should all be the same:')
            print('numpy.array([len(x_0),len(y_0),len(z_0),len(phi_0),len(theta_0)]) = ',len_array)
            print('Check that self.n_events given matches length of all given coordinates')
            return 0
        
        #Response function preparations
        response_mode = 'v7'
        if electricFieldDomain == 'time':
            print('Loading Response Functions')
            if int(response_mode.split('v')[-1]) >= 6:
                upsample_factor = 40
            else:
                upsample_factor = 20
            
            self.signal_times, self.h_fft,self.sys_fft,self.freqs_response = gnosim.interaction.askaryan.calculateTimes(up_sample_factor = upsample_factor,mode=response_mode)
            apply_hard_cut = False
            hard_low = 130e6
            hard_high = 750e6
            if apply_hard_cut == True:
                freqs_cut = numpy.logical_and(self.freqs_response > hard_low, self.freqs_response < hard_high)
                self.sys_fft = numpy.multiply(self.sys_fft,freqs_cut) #This forces a strong bandwidth cut which makes signals that are dominant in low frequency below the band width not be dominated by the noise in that region over the response.  
        
            print('Preparing digitization')
            sys.stdout.flush()
            #The following is for digitization.  
        
            random_time_offsets = numpy.random.uniform(-1, 1, size=self.n_events)
            dc_offsets = numpy.zeros(self.n_events) #Perhaps something to change later.  Added now for ease. 
            
            #theta_obs_rad,R,Energy_GeV,n,t_offset,attenuation,beam_pattern_factor,u, 
            #h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential = False,
            #include_noise = False, resistance = 50, temperature = 320
            #print(self.signal_times)
            #print(len(self.signal_times))
            
            noise_signal  = numpy.array([])
            for i in range(100):
                noise_signal_i = gnosim.interaction.askaryan.quickSignalSingle( 0,1,energy_neutrino,1.8,\
                          0,0,0,self.signal_times,self.h_fft,self.sys_fft,self.freqs_response,\
                          plot_signals=False,plot_spectrum=False,plot_potential = False,\
                          include_noise = True, resistance = 50, temperature = 320)[3]
                noise_signal = numpy.append(noise_signal,noise_signal_i)
            self.noise_rms = numpy.std(noise_signal)
            self.scale_noise_to = 3
            
            #The following is for the beamfoforming
            if do_beamforming == True:
                power_calculation_sum_length = 16 #How long each power sum window is
                power_calculation_interval = 8 #How frequent each power sum window begins
                 
                z_array = []
                for sz in self.config['stations']['positions']:
                    for az in self.config['antennas']['positions']:
                        z_array.append(sz[2] + az[2])
                z_array = numpy.array(z_array)
                index_refraction_array = gnosim.earth.antarctic.indexOfRefraction(z_array, ice_model=self.config['detector_volume']['ice_model']) 
                mean_index = numpy.mean(index_refraction_array)
                self.beam_dict = gnosim.sim.fpga.getBeams(self.config, n_beams, n_baselines , mean_index , self.digital_sampling_period ,power_calculation_sum_length = power_calculation_sum_length, power_calculation_interval = power_calculation_interval, verbose = False)
                print(self.beam_dict)
                colormap = pylab.cm.gist_ncar #nipy_spectral, Set1,Paired   
                self.beam_colors = [colormap(i) for i in numpy.linspace(0, 1,n_beams+1)] #I put the +1 backs it was making the last beam white, hopefully if I put this then the last is still white but is never called
        
        #Preparing output arrays
        
        if output_all_solutions == False:
            self.len_info_per_event = self.n_antenna
        else:
            self.len_info_per_event = len(self.solutions)*self.n_antenna
        #For more info about the below type of dtype check out HDF5's discussion of them in 
        #O'Reilly, Python and HDF5: Chapter 7. More About Types - Compound Types
        self.info_dtype = numpy.dtype([('eventid','i'),('station',numpy.uint16),('antenna',numpy.uint16),('has_solution',numpy.bool_),('pre_triggered',numpy.bool_),('triggered',numpy.bool_),('solution','S10'),('time','f'),('distance','f'),('theta_ant','f'),('observation_angle','f'),('electric_field','f'),('electric_field_digitized','f'),('dominant_freq','f'),('a_h','f'),('a_v','f'),('SNR','f'),('beam_pattern_factor','f'),('fpga_max','i'),('seed',numpy.uint32)])
        p_interact = numpy.zeros(self.n_events)
        p_earth = numpy.zeros(self.n_events)
        p_detect = numpy.zeros(self.n_events)
        #inelasticity = numpy.zeros(self.n_events)
        inelasticity = gnosim.interaction.inelasticity.inelasticityArray(energy_neutrinos, mode='cc') ## GENERALIZE THIS LATER for anti neutrino, etc. 
        electric_field_max = numpy.zeros(self.n_events)
        observation_angle_max = numpy.zeros(self.n_events)
        solution_max = numpy.zeros(self.n_events)
        index_station_max = numpy.zeros(self.n_events)
        index_antenna_max = numpy.zeros(self.n_events)
        t_max = numpy.zeros(self.n_events)
        d_max = numpy.zeros(self.n_events)
        theta_ray_max = numpy.zeros(self.n_events)
        theta_ant_max = numpy.zeros(self.n_events)
        a_v_max = numpy.zeros(self.n_events)
        a_h_max = numpy.zeros(self.n_events)
        info = numpy.empty(self.n_events * self.len_info_per_event , dtype = self.info_dtype)
        
        if self.outfile:
            self.file = h5py.File(self.outfile, 'w')
            # ORIGINAL 28 MAY 2014
            #self.file.attrs['geometric_factor'] = (4. * numpy.pi) * (numpy.pi * detector_volume_radius**2 * detector_volume_depth) # m^3 sr
            # ORIGINAL 28 MAY 2014
            # NEW CURVATURE
            if (seed != None):
                self.file.attrs['seed'] = seed
            else:
                self.file.attrs['seed'] = 'None'
            
            self.file.attrs['geometric_factor'] = (4. * numpy.pi) \
                                             * (2. * numpy.pi * gnosim.utils.constants.radius_earth**2 \
                                                * (1. - numpy.cos(detector_volume_radius / gnosim.utils.constants.radius_earth))\
                                                * detector_volume_depth) # m^3 sr not that the 2 pi (1 - cos theta) is the area of a cap, that is then multiplied by a differential depth (depth of ice) to get volume, then 4 pi to get m^3 sr
                                                
            # NEW CURVATURE

            self.file.attrs['config'] = self.config_file
            #self.file.attrs['ice_model'] = gnosim.earth.antarctic.ice_model_default
            self.file.attrs['ice_model'] = self.config['detector_volume']['ice_model']
            self.file.attrs['trigger_mode'] = trigger_threshold_units
            self.file.attrs['trigger_threshold'] = trigger_threshold
            if pre_trigger_angle == None:
                self.file.attrs['pre_trigger_angle'] = 'None'
            else:
                self.file.attrs['pre_trigger_angle'] = pre_trigger_angle

            self.file.create_dataset('energy_neutrino', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('inelasticity', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('x_0', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('y_0', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('z_0', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('theta_0', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('phi_0', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)

            self.file.create_dataset('p_interact', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('p_earth', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('p_detect', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            
            self.file.create_dataset('index_station', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('index_antenna', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)

            self.file.create_dataset('electric_field', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('observation_angle', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('solution', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('t', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('d', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('theta_ray', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('theta_ant', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('a_v', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('a_h', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('info', ( self.n_events * self.len_info_per_event , ) , dtype=self.info_dtype, compression='gzip', compression_opts=9, shuffle=True)

            self.file.create_group('signals')
        
        
        general_prep_time = time.time() - self.throw_start_time
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
        
        hull_time = time.time() - self.throw_start_time
        print('About to run griddata_Event:')
        sys.stdout.flush()
        ############################################################################
        #Interpolating values from using griddata:
        ############################################################################
        
        #If memory becomes an issue this might need to be adapted to run for chunks of events 
        if use_interp_threading == True:
            self.multiThreadGridDataEvent(x_0, y_0, z_0, method = method,n_cores = self.n_cores)
        else:
            self.griddata_Event(x_0, y_0, z_0, method = method)   
        print('Succesfully ran griddata_Event:')
        griddata_time = time.time() - self.throw_start_time
        ############################################################################
        #Using interpolated values for further calculations on an event/event basis:
        ############################################################################
        if use_event_threading == True:
            futures = []
            #initiate threads
            print('Submitting Event Threads')
            sys.stdout.flush()
            with concurrent.futures.ThreadPoolExecutor(max_workers = self.n_cores) as executor:
                for ii in range(0, self.n_events):
                    futures.append(executor.submit(self.event, energy_neutrinos[ii], phi_0[ii], theta_0[ii], x_0[ii], y_0[ii], z_0[ii], \
                                    ii,inelasticity[ii], anti=anti,electricFieldDomain = electricFieldDomain, \
                                    include_noise = include_noise,plot_signals=plot_signals,plot_geometry=plot_geometry,\
                                    summed_signals = summed_signals,trigger_threshold = trigger_threshold, trigger_threshold_units = trigger_threshold_units, \
                                    plot_filetype_extension=plot_filetype_extension, image_path = image_path,
                                    random_time_offset = random_time_offsets[ii],\
                                    dc_offset = dc_offsets[ii], do_beamforming = do_beamforming, output_all_solutions = output_all_solutions,
                                    pre_trigger_angle = pre_trigger_angle, event_seed = event_seeds[ii]))

            for future in concurrent.futures.as_completed(futures):
                #Note eventid must be first output for other outputs to use it properly
                eventid, p_interact[eventid], p_earth[eventid], p_detect[eventid], electric_field_max[eventid], dic_max, observation_angle_max[eventid], \
                solution_max[eventid], index_station_max[eventid], index_antenna_max[eventid], info[(eventid * self.len_info_per_event ):((eventid+1) * self.len_info_per_event )], \
                triggered, signals_out = future.result()
                event_label = 'event%i'%eventid
                
                if eventid%1000 == 0:
                    sys.stdout.flush()
                
                if p_detect[ii] == 1.:
                    t_max[ii] = dic_max['t']
                    d_max[ii] = dic_max['d']
                    theta_ray_max[ii] = dic_max['theta']
                    theta_ant_max[ii] = dic_max['theta_ant']
                    a_v_max[ii] = dic_max['a_v']
                    a_h_max[ii] = dic_max['a_h']
                    
                if self.outfile: 
                    if numpy.logical_and(self.save_signals == True,triggered == True):
                        #This region I will need to be careful adjustig when/if I add multithreading per event. 
                        #Note to future self, there is a section in 'Python and HDF5' about multithreading with HDF5
                        self.file['signals'].create_group(event_label)
                        for index_station in range(self.config['stations']['n']):
                            station_label = 'station%i'%index_station
                            self.file['signals'][event_label].create_dataset(station_label, numpy.shape(signals_out[station_label]), dtype='f', compression='gzip', compression_opts=9, shuffle=True)  
                            self.file['signals'][event_label][station_label][...] = signals_out[station_label]
                    
        else:
            print('Running Events')
            sys.stdout.flush()
            for ii in range(0, self.n_events):
                event_label = 'event%i'%ii
                current_time  = time.time() - self.throw_start_time
                if (self.n_events//1000 != 0):
                    if(ii%(self.n_events//1000) == 0):
                        print ('Event (%i/%i) Time: %0.2f s ( %0.4f h)'%(ii, self.n_events,current_time,current_time/3600.0)) #might want to comment out these print statements to run faster and spew less
                        sys.stdout.flush()
                else:
                    print ('Event (%i/%i) Time: %0.2f s ( %0.4f h)'%(ii, self.n_events,current_time,current_time/3600.0))
                    sys.stdout.flush()
                eventid, p_interact[ii], p_earth[ii], p_detect[ii], electric_field_max[ii], dic_max, observation_angle_max[ii], \
                    solution_max[ii], index_station_max[ii], index_antenna_max[ii], info[(ii * self.len_info_per_event ):((ii+1) * self.len_info_per_event )], \
                    triggered, signals_out \
                    = self.event(energy_neutrinos[ii], phi_0[ii], theta_0[ii], x_0[ii], y_0[ii], z_0[ii], \
                                ii,inelasticity[ii], anti=anti,electricFieldDomain = electricFieldDomain, \
                                include_noise = include_noise,plot_signals=plot_signals,plot_geometry=plot_geometry,\
                                summed_signals = summed_signals,trigger_threshold = trigger_threshold, trigger_threshold_units = trigger_threshold_units, \
                                plot_filetype_extension=plot_filetype_extension, image_path = image_path,
                                random_time_offset = random_time_offsets[ii],\
                                dc_offset = dc_offsets[ii], do_beamforming = do_beamforming, output_all_solutions = output_all_solutions,
                                pre_trigger_angle = pre_trigger_angle, event_seed = event_seeds[ii])
                #print(info[(ii * self.len_info_per_event ):((ii+1) * self.len_info_per_event )])
                if numpy.logical_and(self.save_signals == True,triggered == True):
                    #This region I will need to be careful adjustig when/if I add multithreading per event. 
                    #Note to future self, there is a section in 'Python and HDF5' about multithreading with HDF5
                    self.file['signals'].create_group(event_label)
                    for index_station in range(self.config['stations']['n']):
                        station_label = 'station%i'%index_station
                        self.file['signals'][event_label].create_dataset(station_label, numpy.shape(signals_out[station_label]), dtype='f', compression='gzip', compression_opts=9, shuffle=True)  
                        self.file['signals'][event_label][station_label][...] = signals_out[station_label]
                if p_detect[ii] == 1.:
                    t_max[ii] = dic_max['t']
                    d_max[ii] = dic_max['d']
                    theta_ray_max[ii] = dic_max['theta']
                    theta_ant_max[ii] = dic_max['theta_ant']
                    a_v_max[ii] = dic_max['a_v']
                    a_h_max[ii] = dic_max['a_h']
        sys.stdout.flush()
                
        if self.outfile:
            print('Writing data after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            
            print('Writing energy_neutrino after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['energy_neutrino'][...] = energy_neutrinos
            
            print('Writing inelasticity after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['inelasticity'][...] = inelasticity
            
            print('Writing x_0 after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['x_0'][...] = x_0
            
            print('Writing y_0 after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['y_0'][...] = y_0
            
            print('Writing z_0 after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['z_0'][...] = z_0
            
            print('Writing theta_0 after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['theta_0'][...] = theta_0
            
            print('Writing phi_0 after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['phi_0'][...] = phi_0
            

            print('Writing p_interact after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['p_interact'][...] = p_interact
            
            print('Writing p_earth after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['p_earth'][...] = p_earth
            
            print('Writing p_detect after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['p_detect'][...] = p_detect

            print('Writing index_station after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['index_station'][...] = index_station_max
            
            print('Writing index_antenna after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['index_antenna'][...] = index_antenna_max

            print('Writing electric_field after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['electric_field'][...] = electric_field_max
            
            print('Writing observation_angle after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['observation_angle'][...] = observation_angle_max
            
            print('Writing solution types after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['solution'][...] = solution_max
            
            print('Writing t after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['t'][...] = t_max
            
            print('Writing d after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['d'][...] = d_max
            
            print('Writing theta_ray after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['theta_ray'][...] = theta_ray_max
            
            print('Writing theta_ant after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['theta_ant'][...] = theta_ant_max
            
            print('Writing a_v after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['a_v'][...] = a_v_max
            
            print('Writing a_h after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['a_h'][...] = a_h_max
            
            print('Writing info after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['info'][...] = info
            
            self.file.close()
    
            
        current_time  = time.time() - self.throw_start_time
        print('Throw finished after %0.3f s'%current_time)
        print('Time Breakdown:')
        print('Time performing general prep: %0.3f s'%(general_prep_time))
        print('Time loading hulls: %0.3f s'%(hull_time - general_prep_time))
        print('Time interpolating with griddata: %0.3f s'%(griddata_time- hull_time))
        print('Time in event calculations:  %0.3f s'%(current_time - griddata_time))
        
def makeIndexHTML(path = './',filetype = 'svg'):
    '''
    Makes a crude html image browser of the created images.
    filytpe should not have the .
    Path should have / at the end
    '''
    header = os.path.realpath(path).split('/')[-1]
    infiles = glob.glob('%s*%s'%(path,filetype))
    
    infiles_num = []
    
    for infile in infiles:
        if len(infile.split('-event')) > 1:
            infiles_num.append(int(infile.split('-event')[-1].replace('.' + filetype,'')))
        else:
            infiles_num.append(-1) #will put all non-conforming files at front before sorted event files.
    infiles = numpy.array(infiles)[numpy.argsort(infiles_num)] #sorts files in index by event number
        
    #I want to sort by event number here!
    image_list = ''
    for infile in infiles:
        image_list = image_list + '\t<img class="mySlides" src="' + infile.split('/')[-1] + '" style="width:100%">\n'
    
    #print(image_list)
    
    
    template = """<!DOCTYPE html>
    <html>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <style>
    .mySlides {display:none;}
    </style>
    <body>
    
    <head>
    <title> RCC KICP | Dan Southall </title>
    </head>
    <p><strong> Dan Southall </strong> | <a href="https://kicp.uchicago.edu/people/profile/daniel_southall.html"> KICP Profile </a> | <a href="../../index.html"> Home </a></p>
    
    <h2 class="w3-center"><strong> """ + header + """</strong></h2>
    
    <input id="slide_index" size="4" value="1" onchange="showDivs(parseInt(document.getElementById('slide_index').value))">
    
    <div class="w3-content w3-display-container"> 
    """ + image_list + """
    </div>
    
    <button class="w3-button w3-black w3-display-left" onclick="plusDivs(-1)">&#10094;</button>
    <button class="w3-button w3-black w3-display-right" onclick="plusDivs(1)">&#10095;</button>
    
    </div>
    <script>
    var slideIndex = 1;
    showDivs(slideIndex);

    function plusDivs(n) {
      showDivs(slideIndex += n);
    }

    function showDivs(n) {
      var i;
      var x = document.getElementsByClassName("mySlides");
      slideIndex =n;
      if (n > x.length) {slideIndex = 1}    
      if (n < 1) {slideIndex = x.length}
      for (i = 0; i < x.length; i++) {
         x[i].style.display = "none";  
      }
      x[slideIndex-1].style.display = "block"; 
      document.getElementById("slide_index").value = slideIndex;
      location.hash = "#" + slideIndex;
      document.getElementById("filename").innerHTML = x[slideIndex-1].getAttribute("src");
    }
    
    function load() 
    {
      var maybe = parseInt(location.hash.slice(1));
      if (!isNaN(maybe)) 
      {
        showDivs(maybe); 
      }
      else showDivs(1); 
    }
    </script>

    </body>
    </html>
    """
    print(template)
    outfile_name = path + 'index'
    if os.path.isfile(outfile_name +'.html'):
        print('Outfile Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(outfile_name))
        outfile_name = outfile_name + '_new'
        while os.path.isfile(outfile_name+'.html'):
            outfile_name = outfile_name + '_new'
    outfile = open(outfile_name + '.html','w')
    outfile.write(template)
    outfile.close()


############################################################

if __name__ == "__main__":

    config_file = sys.argv[1]
    energy_neutrino = float(sys.argv[2]) # GeV
    n_events = int(sys.argv[3])
    index = int(sys.argv[4])
    #solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2'])
    solutions = numpy.array(['direct', 'cross', 'reflect'])
    #solutions = numpy.array(['cross'])
    #detector_volume_radius = float(sys.argv[5]) # m, 1200 for Ross surface, 51000 for Minna bluff, >6000 for subterranean
    #detector_volume_depth = float(sys.argv[6]) # m, 500 for Ross and Minna, 3000 for subterranean
    if len(sys.argv) == 6:
        seed = int(sys.argv[5])
        print('Using seed from sys.argv[5]: ', seed)
    else:
        seed = None
        if seed == None:
            print('Using no seed')
        else:
            print('Using internally (antarcticsim.py) defined seed: ', seed)
    #SEED FOR TESTNG:
    #seed = 1#None
    config_file_fix = config_file.replace('/home/dsouthall/Projects/GNOSim/','')
    config_file_fix = config_file_fix.replace('gnosim/sim/ConfigFiles/Config_dsouthall/','')
    config_file_fix = config_file_fix.replace('./','')
    if (seed != None):
        outfile = '/home/dsouthall/scratch-midway2/results_2019_Feb_%s_%.2e_GeV_%i_events_%i_seed_%i.h5'%(config_file_fix.replace('.py', ''),
                                                                    energy_neutrino,
                                                                    n_events,
                                                                    seed,
                                                                    index)
        print('\n\n!!!Using Seed!!! \n\n Seed: ', seed, '\nOutfile Name: \n', outfile)
    else:
        outfile = '/home/dsouthall/scratch-midway2/results_2019_Feb_%s_%.2e_GeV_%i_events_%i.h5'%(config_file_fix.replace('.py', ''),
                                                                energy_neutrino,
                                                                n_events,
                                                                index)
        print('Outfile Name: \n', outfile)
    if os.path.isfile(outfile):
        print('Outfile Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(outfile))
        outfile = './' + outfile.split('/')[-1]
        while os.path.isfile(outfile):
            outfile = outfile.replace('.h5','_new.h5')
    
    #making image directory
    image_extension = 'svg'
    image_path = '/home/dsouthall/public_html/images/' + outfile.replace('.h5','').split('/')[-1] #should end with a / before inputting into throw
    if os.path.exists(image_path):
        print('Image Directory Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(image_path))
        image_path = image_path + '_new'
        while os.path.exists(image_path):
            image_path = image_path + '_new'
    
    os.makedirs(image_path) 
    image_path = image_path + '/'
    print('Images will be saved to ', image_path)
    
    
    #Creating Sim and throwing events
    my_sim = Sim(config_file, solutions=solutions,pre_split = True)
    #trigger_threshold_units should be one of 'adu', 'V', or 'fpga'.  If neither is assumes 'V'
    #adu is in digitized units where noise rms is set to 3 (hardcoded, search noise_rms).  If noise is turned off this still scales as if noise were present to 3.
    #V is volts of signal
    #fpga uses the magnitude of a beamformed-powersummed signal.  The magnitude of this is not as intuitive as adu or V but is more like what is done at Pole.
    #all of these assume time domain, as the freq domain portion of the code is not maintained. 
    
    #Used for testing: 10 adu, 11500 fpga, 11342 fpga for 10Hz noise triggering
    sys.stdout.flush()
    my_sim.throw(energy_neutrino, n_events=n_events,
                 detector_volume_radius=my_sim.config['detector_volume']['radius'],
                 detector_volume_depth=my_sim.config['detector_volume']['depth'],
                 outfile=outfile,seed=seed,electricFieldDomain = 'time',include_noise = True,summed_signals = True, 
                 plot_geometry = False, plot_signals = False, trigger_threshold = 11500, trigger_threshold_units = 'fpga',
                 plot_filetype_extension = image_extension,image_path = image_path,use_interp_threading = True,use_event_threading = True,
                 do_beamforming = True, n_beams = 15, n_baselines = 2,output_all_solutions = True,save_signals = False,
                 pre_trigger_angle = 10.0)
    sys.stdout.flush()
    #For pulser location that Kaeli is looking at:
    #r_vertex = numpy.array([5214.0]), phi_vertex = numpy.array([0.0]), z_0 = numpy.array([-1450.0])
    
    print('Trying to print station geometry and antenna orientations')
    try:
        fig = gnosim.sim.detector.plotArrayFromConfig(my_sim.config,only_station = 'all',verbose = False)
        fig.savefig('%s%s_array_geometry.%s'%(image_path,outfile.split('/')[-1].replace('.h5',''),image_extension),bbox_inches='tight')
        pylab.close(fig)
    except Exception as e:
        print('Failed to save image %s%s_array_geometry.%s'%(image_path,outfile.split('/')[-1].replace('.h5',''),image_extension))
        print(e)
    
    print('Trying to create index.html file for new images')
    try:
        makeIndexHTML(path = image_path ,filetype = image_extension)
    except Exception as e:
        print('Something went wrong in making index.html')
        print(e)
    sys.stdout.flush()
    #python /home/dsouthall/Projects/GNOSim/sim/antarcticsim.py config energy n_events index 
    #python /home/dsouthall/Projects/GNOSim/gnosim/sim/antarcticsim.py /home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_octo_-200_polar_120_rays.py 1.0e8 50000 1 
    #f.close()

############################################################


