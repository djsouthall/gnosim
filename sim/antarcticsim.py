#!/usr/bin/env python

'''
Simulation
'''

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

import gnosim.utils.linalg
import gnosim.earth.earth
import gnosim.earth.ice
import gnosim.trace.refraction_library
#from gnosim.trace.refraction_library import *
import gnosim.interaction.askaryan
import gnosim.detector.detector
import gnosim.detector.fpga
import gnosim.utils.misc
pylab.ion() #this turns interactive mode on.  I should test with this off

############################################################

import cProfile, pstats, io

def profile(fnc):
    '''
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
    '''
    
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

def getAcceptedTriggerUnits():
    '''
    Gives a list of the accepted units (labels of) for triggering in the simulation.
    
    'adu'   Will trigger in digitized units.  In the simulation noise will be scaled to the value set in the configuration file.  
            If noise is turned off this still scales as if noise were present and scaled appropriately.  This works on a per signal 
            basis (i.e. beamforming does not effect it).
    'V'     Will trigger on voltage in units of V.  This works on a per signal basis (i.e. beamforming does not effect it).
    'fpga'  Will trigger in units of adu^2 using the beamforming power sum.  This requires do_beamforming == True.  An appropriate
            trigger level can be determined using gnosim.analysis.threshold_scan_tool.py

    Returns
    -------
    accepted_units : numpy.ndarray of str
        The accepted labels for unit types to trigger on.
    '''
    accepted_units = numpy.array(['fpga','adu','V'])
    return accepted_units

class Sim:
    '''
    The object containing all information and functions specific to running the simuation.

    Parameters
    ----------
    config_file : str
        A string containing the address of the configuration file.
    solutions : numpy.ndarray of str, optional
        A list of the solution types to load.  Often either all accepted solution types (the Default), or the same list omitting
        the _2 strings, which represent libraries of solution that reflect of the bottom of the ice.  Too see acceptable solution
        types use gnosim.trace.refraction_library.getAcceptedSolutions().  (Default is numpy.array([]) which will result in
        all solution types being used).
    electricFieldDomain : str, optional
        Selects the type of Askaryan radiation calculation to use.  
        Options:    'time'  for the newer time domain calculation that allows many more options in the simulation and is more maintained, 
                    'freq'  which is the old Askaryan calculation method which does the calculation solely in the frequency domain and 
                            is no longer maintined. 
        (Default is 'time').
    do_beamforming : bool, optional
        Enables beamforming for the trigger.  If this is True then the units selected when using self.throw should correspondingly be 'fpga'.
        Otherwise a simple threshold trigger is applied with the specified units per signal, without beamforming.  (Default is True).

    Attributes
    ----------
    config_file : str
        A string containing the address of the configuration file.
    config : dict
        The configuration dictionary loaded from a config file.
    ice : gnosim.earth.ice.Ice
        The ice object containing the appropriate ice model.
    solutions : numpy.ndarray of str, optional
        A list of the solution types to load.
    electricFieldDomain : str
        The selected type of Askaryan radiation calculation to use.
    stations : list of gnosim.sim.detctor.Station objects
        Station objects corresponding to the stations (and antennas) specified in the configuration file.
    n_antenna : int
        The total number of antennas across all stations.  Used interally.
    do_beamforming : bool
        Determines whether or not to perform beamforming.
    n_cores : int
        The number of cores available for multithreading.
    lock : threading.Lock object
        A lock used in multithreading to avoid multiple threads writing simultaneously.
    info_dtype : numpy.dtype
        The data type that will be used to store the meta data about each calculated solution type for each antenna for each station for each event.
        Convenient for organization and performing cuts on many types of data simulatenously.  This is also the data type that the majority of the
        meta data will be stored in on output as well (except for some attributes and signal waveforms).
    in_flag_array : dict, optional
        A dictionary that contains bools for if particular event has a solution for a particular solution type/antenna/station.  Only present if self.throw or
        self.makeFlagDicArrayFromInfo are run.
    in_dic_array, optional
        A dictionary that contains the interpolated values for each events different solution types/antennas/stations.  Only present if self.throw or
        self.makeFlagDicArrayFromInfo are run.
    throw_start_time : float, optional
        The internal clock time of when the simulation first ran self.throw.  Only present self.throw is run.
    pre_split : bool, optional
        Determines whether to attempt to load from pre split libraries.  If true (and the pre split libraries are calculated and 
        saved appropriately) this avoids lengthy calculations which seperate the rays ito the different solution types.  (Default is False).  
        Only present self.throw is run.
    outfile : str, optional
        The address/name of the output file.  Only present self.throw is run.
    file : h5py.File(), optional
        The h5py.File object for the output file.  Only present self.throw is run.
    save_signals : bool, optional
        Enables saving the waveforms for triggered events.  Waveforms are saved in the output file as a dictionary under the 'signals' header.
        This dictionary is organized at the top level by event, then branching to station label. 
        The signals from a station are in the form of a numpy.ndarray which contains the signals for all antennas in the station.
        Each antenna in each station is stored as row in the numpy.ndarray corresponding to that station.  The final row of
        this ndarray is the times corresponding to the signals in every other row.  Signals are given in units of adu and times
        in units of ns.
        # TODO: Add a script that converts these to be in the same format as real data taken from an ARA station.
    n_events : int, optional
        The number of events (to be) thrown.  Only present self.throw is run.
    len_info_per_event : int, optional
        The number of elements to expect for the length of the info object per event.  Used internally for indexing purposes.  Only present self.throw is run.

    See Also
    For more info about how to work with info_dtype check out HDF5's discussion of custome dtypes in O'Reilly, Python and HDF5: Chapter 7. More About Types - Compound Types
    '''

    def __init__(self, config_file,solutions = numpy.array([]),electricFieldDomain = 'time',do_beamforming = True):
        #pre_split False unless using a library already sorted into different
        #directories by solution type.
        self.config_file = config_file
        #self.config = eval(''.join(open(config_file).readlines()))
        self.config = yaml.load(open(config_file))
        self.ice = gnosim.earth.ice.Ice(self.config['detector_volume']['ice_model'])
        
        #Filtering solutions to acceptable values.
        accepted_solutions = gnosim.trace.refraction_library.getAcceptedSolutions()
        if numpy.size(solutions) == 0:
            solutions = accepted_solutions
        solutions = accepted_solutions[numpy.isin(accepted_solutions,solutions)]
        if len(solutions) == 0:
            print('Selection of solution types did not match predefined values.  Using default types.')
            self.solutions = accepted_solutions
            print(self.solutions)
        else:
            self.solutions = solutions

        accepted_domains = numpy.array(['time','freq'])
        electricFieldDomain = accepted_domains[numpy.isin(accepted_domains,electricFieldDomain)]
        if len(electricFieldDomain) == 0:
            print('Selection of domain type did not match predefined values.  Using default type \'%s\''%accepted_domains[0])
            self.electricFieldDomain = accepted_domains[0]
        else:
            self.electricFieldDomain = electricFieldDomain[0]

        self.stations = []
        for ii in range(0, self.config['stations']['n']):
            station_label = 'station'+str(ii)
            x_station, y_station, z_station = self.config['stations']['positions'][ii]
            station = gnosim.detector.detector.Station(x_station, y_station, z_station, self.config, station_label,solutions = self.solutions)
            self.stations.append(station)
            
        self.n_antenna = sum([len(self.stations[s].antennas) for s in range(len(self.stations))])
        print('Total Number of Antennas = ', self.n_antenna)

        if self.electricFieldDomain == 'time':

            print('Preparing digitization')
            sys.stdout.flush()
            #The following is for digitization.  
        
            for station in self.stations:
                station.calculateNoiseRMS()
                        
            #The following is for the beamforming
            self.do_beamforming = do_beamforming
            if do_beamforming == True:
                print('Preparing for beamforming')
                for station in self.stations:
                    z_array = []
                    for antenna in station.antennas:
                        z_array.append(antenna.z)
                    z_array = numpy.array(z_array)
                    index_refraction_array = self.ice.indexOfRefraction(z_array) 
                    mean_index = numpy.mean(index_refraction_array) #used to estimate angles for each beam.
                    station.getBeams(mean_index)
        
        self.n_cores = cpu_count()
        #self.lock = threading.RLock() #Hopefully fixes multithreading plotting (it didn't)
        self.lock = threading.Lock()

        #For more info about the below type of dtype check out HDF5's discussion of them in 
        #O'Reilly, Python and HDF5: Chapter 7. More About Types - Compound Types
        self.info_dtype = numpy.dtype([('eventid','i'),('station',numpy.uint16),('antenna',numpy.uint16),('has_solution',numpy.bool_),('pre_triggered',numpy.bool_),('triggered',numpy.bool_),('solution','S10'),('time','f'),('distance','f'),('theta_ant','f'),('theta_ray','f'),('observation_angle','f'),('electric_field','f'),('electric_field_digitized','f'),('dominant_freq','f'),('a_s','f'),('a_p','f'),('SNR','f'),('signal_reduction_factor','f'),('fpga_max','i'),('seed',numpy.uint32)])

        
    #@profile 
    def event(self, energy_neutrino, phi_0, theta_0, x_0, y_0, z_0, eventid, inelasticity, anti=False, include_noise = True, 
            plot_signals=False, plot_geometry=False, summed_signals=True, trigger_threshold = 0.0, trigger_threshold_units = 'fpga', 
            plot_filetype_extension = 'svg',image_path = './',  random_time_offset = 0.0, dc_offset = 0.0, do_beamforming = False, 
            output_all_solutions = True, pre_trigger_angle = None, event_seed = None, return_fig_array = False):
        '''
        Note that the freq domain option is outdated and does not just do the same thing differently.  It does 
        what older version of the code attempted did.  Does not have a lot of the newer additions such as noise.  

        Parameters
        ----------
        energy_neutrino : float
            The energy of the neutrino.  This should be the raw energy of the neutrino (i.e. before inelasticity is applied).  Given in GeV.
        phi_0 : float
            The azimuthal spherical coordinate for the direction neutrino came from.  Given in degrees.
        theta_0 : float
            The polar spherical coordinate for the direction neutrino came from.  Given in degrees.
        x_0 : float
            The x cartesian ice frame coordinate of the neutrino interaction location.  Given in m.
        y_0 : float
            The y cartesian ice frame coordinate of the neutrino interaction location.  Given in m.
        z_0 : float
            The z cartesian ice frame coordinate of the neutrino interaction location.  Given in m.
        eventid : int
            The index associated with this event.
        inelasticity : float
            The inelasticty factor of the interaction of the neutrino in ice.  Represents the portion of energy that is actually transferred 
            to the energy of the shower.
        anti : bool, optional
            Selects either a neutrino (anti == False) or anti neutrino (anti == True).  (Default is False).
        include_noise : bool, optional
            Enables the addition of noise to the time domain signals.  Note supported for electricFieldDomain = 'freq'.  (Default is True).
        plot_signals : bool, optional
            Enables plotting of the waveforms, as well as of the beam forming plots and some meta data.  Only plots for trigger events. (Default is False).
        plot_geometry : bool, optional
            Enables plotting of the neutrino location, rays, and antennas.  Only plots for trigger events. (Default is False).
        summed_signals : bool, optional
            If true, then signals resulting from different solution types are combined into a single waveform per antenna.  Otherwise
            only the waveform of the solution type with the maximum signal per antenna will be used.  (Default is True).
        trigger_threshold : float, optional
            The trigger threshold to be applied on the signal or set of signals.  This should correspond to trigger_threshold_units.
            (Default is 0.0).
        trigger_threshold_units : str, optional
            This selects to units used for triggering.  To see the options try getAcceptedTriggerUnits().  If this is
            'fpga' then do_beamforming must be True.  (Default is 'fpga').
        plot_filetype_extension : str, optional
            Sets the file extension for any saved images.  (Default is 'svg').
        image_path : str, optional
            Sets the path of where any created images will be saved if enabled. (Default is './').
        random_time_offset : float, optional
            A small random jitter in timing to ensure that no systematic error is introduced from perfect timing in the MC simulation.  Given in ns.  (Default is 0.0).
        dc_offset : float, optional
            An offset to be given to any signals.  Given in V. (Default is 0.0).
        do_beamforming : bool, optional
            Enables beamforming.  (Default is True).
        output_all_solutions : bool, optional
            Enables all solution types to be output, otherwise only the solution type with the maximum signal per antenna is output.  (Default is True).
        pre_trigger_angle : float, optional
            If given, then a pre trigger will be applied to the event such that calculations of Askaryan radiation/electric field will only be conducted 
            if ANY of the possible solution types (for all antennas and stations) have an observation angle within pre_trigger_angle number of degrees to
            the Cherenkov angle.  Essentially the Cherenkov cone must be observable within the pre trigger angular window from at least one antenna in the
            array in order for the calculations to proceed.

            i.e. if pre_trigger_angle is 10.0 degrees then signals will only be calculated if one of the solutions was emitted (observed on cone) at an
            angle: theta_c - 10.0 deg < theta_obs < theta_c + 10.0 deg.

            If ANY of the solution types of ANY of the antennas in the entire array satisfies the pre trigger, then all calculations for that event proceed, 
            not just the solution types that independently satisfied the pre trigger threshold.  (Default is None).
        event_seed : int, optional
            This sets the state of the random object to be used internally for the event, allowing it to be reproducable regardless of the number of times random
            calls were made externally.  In general this should be calculated using numpy.random.randint(numpy.iinfo(numpy.uint32).max,size=self.n_events).
            (Default is None).
        return_fig_array : bool, optional
            Enables an additional ouput array containing the figure objects.  (Default is False).

        Returns
        -------
        eventid : int
            The index associated with this event.
        p_interact : float
            The probability that the neutrino interacts in a sphere containing a cubic meter of ice. 
        p_earth : float
            The probability of survival for the neutrino passing through the earth.
        p_detect : bool
            True if the event is obserable at all (i.e. has at least one solution type visible across the array).
        event_electric_field_max : float
            The maximum electric field measured for the event across all antennas.
        dic_max : dict
            Contains info about 'd', 'r', 't', 'theta', 'theta_ant', 'a_s', 'a_p', and 'z', for the solution type that resulted in event_electric_field_max. 
        event_solution_max : str
            The solution type that resulted in event_electric_field_max.
        event_index_station_max : int
            The index of the station observed event_electric_field_max.
        event_index_antenna_max : int
            The index of the antenna in the station that observed event_electric_field_max.
        info : numpy.ndarray of self.info_dtype
            Contains all meta data about the event for each antenna and solution type (if output_all_solutions == True).
        triggered : bool
            True if array triggers on the event.
        signals_out : dict
            A dictionary containing the waveforms for the event.  Each stations signals are seperated as entries in the dict,
            but each antenna in each station is stored as row in the numpy.ndarray corresponding to that station.  The final row of
            this ndarray is the times corresponding to the signals in every other row.  Signals are given in units of adu and times
            in units of ns.
        fig_array : list of figures, optional
            A list of the figures produced during the running of this function.
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
        has_solution_array = numpy.zeros( len(self.solutions)*self.n_antenna )
        station_wide_solution_index = -1 #add one to it for each antenna, used for indexing temporary array
        for index_station, station in enumerate(self.stations):
            # Loop over station antennas
            for index_antenna, antenna in enumerate(self.stations[index_station].antennas):
                for solution in antenna.solutions:
                    station_wide_solution_index += 1
                    has_solution_array[station_wide_solution_index] = self.in_flag_array[station.label][antenna.label][solution][eventid]
                    x_antennas[station_wide_solution_index] = antenna.x
                    y_antennas[station_wide_solution_index] = antenna.y
                    z_antennas[station_wide_solution_index] = antenna.z
                    temporary_info[station_wide_solution_index] = numpy.array([(eventid,index_station,index_antenna,has_solution_array[station_wide_solution_index],0,0,solution,self.in_dic_array[station.label][antenna.label][solution]['t'][eventid],self.in_dic_array[station.label][antenna.label][solution]['d'][eventid],self.in_dic_array[station.label][antenna.label][solution]['theta_ant'][eventid],self.in_dic_array[station.label][antenna.label][solution]['theta'][eventid],-999.0,-999.0,-999.0,-999.0,self.in_dic_array[station.label][antenna.label][solution]['a_s'][eventid],self.in_dic_array[station.label][antenna.label][solution]['a_p'][eventid],-999.0,-999.0,-999,event_seed)],dtype = self.info_dtype)
            
        index_of_refraction_at_neutrino = self.ice.indexOfRefraction(z_0)
        cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./index_of_refraction_at_neutrino))
        rs = numpy.sqrt((x_0 - x_antennas)**2 + (y_0 - y_antennas)**2)
        phi_rays = numpy.degrees(numpy.arctan2(y_0 - y_antennas, x_0 - x_antennas)) % 360. # deg #points along thrown ray towards neutrino from antenna. 

        multi_vector_rays_to_neutrino_at_neutrino = gnosim.utils.linalg.angToVec(phi_rays, temporary_info['theta_ray']) #at neutrino event, I think theta_ray points along ray towards neutrino.  So for an upward going wave vector ray towards the antenna, theta_ray > 90.0, directly downward ray towards antenna is 0.0 deg  
        multi_emission_wave_vector = - multi_vector_rays_to_neutrino_at_neutrino #Unit vectors along ray towards antenna as emitted by neutrino.  In the ice frame.

        vector_to_neutrino_source = gnosim.utils.linalg.angToVec(phi_0, theta_0) # Direction neutrino came from
        vector_neutrino_travel_dir = -vector_to_neutrino_source # Direction the shower is propogating (he direction the neutrino is heading)
        
        #multi_vector_to_neutrino_source = numpy.tile(vector_to_neutrino_source,(len(phi_rays),1)) 
        multi_vector_neutrino_travel_dir = numpy.tile(vector_neutrino_travel_dir,(len(phi_rays),1))
        
        #observation_angles = gnosim.utils.linalg.angTwoVec(multi_vector_to_neutrino_source, multi_vector_rays_to_neutrino_at_neutrino) # deg
        observation_angles = gnosim.utils.linalg.angTwoVec(multi_vector_neutrino_travel_dir, multi_vector_rays_to_neutrino_at_neutrino) # deg
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
        
        if trigger_threshold_units == 'fpga':
            if do_beamforming == False:
                print('WARNING!!!')
                print('THRESHOLD GIVEN IN UNITS FOR FPGA, BUT BEAMFORMING NOT ENABLED')
                print('ENABLEING BEAMFORMING')
                sys.stdout.flush()
                do_beamforming = True
            if self.electricFieldDomain == 'freq':
                print('WARNING!!!')
                print('THRESHOLD GIVEN IN UNITS FOR FPGA, FREQ DOMAIN DOES NOT SUPPORT THIS')
                print('BREAKING')
                sys.stdout.flush()
                return 0
        
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
        p_interact = gnosim.earth.earth.probInteract(energy_neutrino, self.ice.density(z_0), anti=anti) #Should not use effective_energy_neutrino, as that only becomes effective AFTER interaction, these test if there is an interaction at all
        
        # Probability for neutrino to make it through the Earth
        #note p_earth has no random components
        p_earth = gnosim.earth.earth.probSurvival(energy_neutrino, theta_0, self.ice, elevation=z_0, anti=anti) #Should not use effective_energy_neutrino, as that only becomes effective AFTER interaction, these test if there is an interaction at all

        
        #Only do triggering if any pass pre_trigger
        if numpy.any(temporary_info['pre_triggered'] == True):

            multi_vector_rays_to_neutrino_at_antenna = gnosim.utils.linalg.angToVec(phi_rays, temporary_info['theta_ant']) #at antenna event, I think theta_ray points along ray towards neutrino.  So for an upward going wave vector ray towards the antenna, theta_ray > 90.0, directly downward ray towards antenna is 0.0 deg  
            multi_detection_wave_vector = - multi_vector_rays_to_neutrino_at_antenna #Unit vectors along ray towards antenna as observed at antenna.  In the ice frame.

            #Set event seed:
            #Seperate RandomState object used for each event to futur proof for multithreading
            #to avoid issues with reproducability with a global RandomState
            random_local = numpy.random.RandomState(seed = event_seed)
            #random_local.normal() etc.,  would need to pass random_local to any function that uses random
            
            if self.electricFieldDomain == 'time':
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
                askaryan_fp_fft = {}
                for station in self.stations:
                    for antenna in station.antennas:
                        if ~numpy.isin(antenna.antenna_response_dir,askaryan_fp_fft.keys()):
                            askaryan_fp = gnosim.interaction.askaryan.F_p(effective_energy_neutrino,antenna.signal_times,index_of_refraction_at_neutrino,1.0)
                            askaryan_fp_fft[antenna.antenna_response_dir] = numpy.fft.rfft(askaryan_fp)
                    
            #Calculating the times to digitize the signal in advance, should be done after a pretrigger
            minimum_time = 1e20
            maximum_time = -1e20

            phi_ray_to_neutrino = {}
            station_wide_solution_index = -1 #add one to it for each antenna, used for indexing temporary array
            for station in self.stations:
                phi_ray_to_neutrino[station.label] = {}
                for antenna in station.antennas:
                    phi_ray_to_neutrino[station.label][antenna.label] = {}
                    for solution in antenna.solutions:
                        station_wide_solution_index += 1
                        phi_ray_to_neutrino[station.label][antenna.label][solution] = phi_rays[station_wide_solution_index]

                    event_times = numpy.sort(temporary_info[temporary_info['has_solution'] == True]['time'])
                    if minimum_time > antenna.signal_times[0] + event_times[0]:
                        minimum_time = antenna.signal_times[0] + event_times[0]
                    if maximum_time < antenna.signal_times[-1] + event_times[-1]:
                        maximum_time = antenna.signal_times[-1] + event_times[-1]
                    if minimum_time == 1e20:
                        minimum_time = antenna.signal_times[0]
                    if maximum_time == -1e20:
                        maximum_time = antenna.signal_times[-1]
            if return_fig_array == True:
                fig_array = []
            
            # Loop over stations
            station_wide_antenna_index = -1 #add one to it for each antenna, used for indexing temporary array
            for index_station, station in enumerate(self.stations):
                digital_sample_times = numpy.arange(minimum_time,maximum_time,station.digital_sampling_period) + random_time_offset #these + random_time_offset #these
                # Loop over station antennas
                if self.electricFieldDomain == 'time':
                    time_analog[station.label] = {}
                    V_analog[station.label] = {}
                    time_digital[station.label] = {}
                    V_digital[station.label] = {}
                    if numpy.logical_and(include_noise == True,summed_signals == True):
                        V_just_noise[station.label] = {}
                for index_antenna, antenna in enumerate(station.antennas):
                    station_wide_antenna_index += 1
                    first_solution_index = station_wide_antenna_index * len (antenna.solutions) #useful for things that are antenna specific but calculated for each solution
                    
                    if self.electricFieldDomain == 'time':
                        time_analog[station.label][antenna.label] = {}
                        V_analog[station.label][antenna.label] = {}
                        
                        if numpy.logical_and(include_noise == True,summed_signals == True):
                            V_just_noise[station.label][antenna.label] = {}

                        for solution in antenna.solutions:
                            time_analog[station.label][antenna.label][solution] = []
                            V_analog[station.label][antenna.label][solution] = []
                            if numpy.logical_and(include_noise == True,summed_signals == True):
                                V_just_noise[station.label][antenna.label][solution] = []

                    electric_field_max = 0. 
                    r = rs[first_solution_index]

                    flag_array = temporary_info[ numpy.logical_and(temporary_info['station'] == index_station, temporary_info['antenna'] == index_antenna)]['has_solution']
                    if numpy.any(flag_array):
                        has_solution = 1

                        for ii, solution in enumerate(antenna.solutions):
                            total_solution_index = station_wide_antenna_index * len (antenna.solutions) + ii
                            if temporary_info[total_solution_index]['has_solution'] == True:
                                #So now it will always get to the above trigger only if ANY pass pre trigger, bit will still only calculate the ones that have solutions regardless if they were the solutions that pretriggered
                                # Direction of outgoing ray from antenna to interaction vertex
                                
                                solution_cut = numpy.logical_and(temporary_info['solution'] == solution.encode(),numpy.logical_and(temporary_info['station'] == index_station, temporary_info['antenna'] == index_antenna))
                                observation_angle = temporary_info[ solution_cut ]['observation_angle']
                                
                                d = temporary_info[ solution_cut ]['distance'] #m
                                theta_ant_deg = temporary_info[ solution_cut ]['theta_ant'] #deg
                                
                                '''
                                theta_ray_from_ant_at_neutrino  = temporary_info[ solution_cut ]['theta_ray'][0]
                                phi_ray_from_ant_at_neutrino    = phi_ray_to_neutrino[station.label][antenna.label][solution]
                                theta_ray_from_ant_at_antenna   = temporary_info[ solution_cut ]['theta_ant'][0]
                                phi_ray_from_ant_at_antenna     = phi_ray_from_ant_at_neutrino
                                theta_neutrino_source_dir       = theta_0
                                phi_neutrino_source_dir         = phi_0
                                '''
                                vec_neutrino_travel_dir         = vector_neutrino_travel_dir
                                emission_wave_vector            = multi_emission_wave_vector[station_wide_solution_index]
                                detection_wave_vector           = multi_detection_wave_vector[station_wide_solution_index]

                                signal_reduction_factor = antenna.getAntennaResponseFactor(vec_neutrino_travel_dir , emission_wave_vector , detection_wave_vector , self.in_dic_array[station.label][antenna.label][solution]['a_s'][eventid] , self.in_dic_array[station.label][antenna.label][solution]['a_p'][eventid])
                                #vec_neutrino_travel_dir , emission_wave_vector , detection_wave_vector , a_s , a_p )
                                if self.electricFieldDomain == 'time':                                                                        
                                    if include_noise == True:
                                        V_noiseless, u , dominant_freq, V_noise, SNR = gnosim.interaction.askaryan.quickSignalSingle( numpy.deg2rad(observation_angle),\
                                          temporary_info[ solution_cut ]['distance'],effective_energy_neutrino,index_of_refraction_at_neutrino,\
                                          self.in_dic_array[station.label][antenna.label][solution]['t'][eventid],signal_reduction_factor,\
                                          antenna.signal_times,antenna.h_fft,antenna.sys_fft,antenna.freqs_response,fp_fft = askaryan_fp_fft[antenna.antenna_response_dir],plot_signals=False,plot_spectrum=False,plot_potential = False,\
                                          include_noise = True, resistance = antenna.resistance, noise_temperature = antenna.noise_temperature, random_local = random_local)  #expects ovbservation_angle to be in radians (hence the deg2rad on input)
                                        
                                        if summed_signals == True:
                                            V_just_noise[station.label][antenna.label][solution] = numpy.add(V_noise,-V_noiseless) #subtracting away raw signal from noisy signal to get just the noise
                                        electric_array = V_noise
                                    else:
                                        V_noiseless, u , dominant_freq = gnosim.interaction.askaryan.quickSignalSingle( numpy.deg2rad(observation_angle),\
                                          temporary_info[ solution_cut ]['distance'],effective_energy_neutrino,index_of_refraction_at_neutrino,\
                                          self.in_dic_array[station.label][antenna.label][solution]['t'][eventid],signal_reduction_factor,\
                                          antenna.signal_times,antenna.h_fft,antenna.sys_fft,antenna.freqs_response,fp_fft = askaryan_fp_fft[antenna.antenna_response_dir],plot_signals=False,plot_spectrum=False,plot_potential = False,\
                                          include_noise = False, resistance = antenna.resistance, noise_temperature = antenna.noise_temperature, random_local = random_local)  #expects ovbservation_angle to be in radians (hence the deg2rad on input)
                                        
                                        SNR = -999.
                                        electric_array = V_noiseless
                                    
                                    electric_field = numpy.max(numpy.abs(electric_array))#maybe shouldn't be abs.  Need to change some lines that comapre to this if made not abs here.  
                                    electric_field_digitized = gnosim.detector.fpga.digitizeSignal([0,1],numpy.array([electric_field,electric_field]),[0.5],station.sampling_bits,antenna.noise_rms,station.scale_noise_to, dc_offset = 0, plot = False)[0] #I don't like this workaround.... :(
                                    
                                    V_analog[station.label][antenna.label][solution] = electric_array
                                    time_analog[station.label][antenna.label][solution] = u
                                else:
                                    if self.electricFieldDomain != 'freq':
                                        print('Electric field domain selection did not fit one of the\ntwo expected values.  Continuing with old method: freq.')
                                        sys.stdout.flush()
                                    frequency = numpy.linspace(antenna.frequency_low, antenna.frequency_high, 100) # GHz
                                    #Note below doesn't use effective_energy_neutrino because it has inelasticity as an input parameter and does that calculation internally.  This is the problem with trying to carry along old code 
                                    electric_field \
                                        = gnosim.interaction.askaryan.electricFieldFrequencyDomainRaw(frequency, temporary_info[ solution_cut ]['distance'], observation_angle,
                                                                                    energy_neutrino, inelasticity, index_of_refraction_at_neutrino) # V m^-1 GHz^-1, dimensionless, expects observation_angle to be in degrees
                                    electric_field *= signal_reduction_factor
                                    electric_array, electric_field, dominant_freq = antenna.totalElectricField(frequency, electric_field, theta_ant_deg) # V m^-1 #THIS WAS CHANGED THETA WAS ADDED
                                    SNR = -999.
                                    electric_field_digitized = -999.
                                
                                temporary_info[total_solution_index]['electric_field']           = electric_field
                                temporary_info[total_solution_index]['electric_field_digitized'] = electric_field_digitized
                                temporary_info[total_solution_index]['dominant_freq']            = dominant_freq
                                temporary_info[total_solution_index]['SNR']                      = SNR
                                temporary_info[total_solution_index]['signal_reduction_factor']  = signal_reduction_factor
                                
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
                        
                    else:
                        #This event has no solution for this antenna
                        #has_solution = 0
                        solution_type_max = antenna.solutions[0]
                    
                    #Temporary_info but only the max solution type.  Should defult to the first solution type, with values already filled out previously to be filler values
                    info[ sum([len(self.stations[s].antennas) for s in range(0,index_station)]) + index_antenna] = temporary_info[numpy.logical_and(numpy.logical_and(temporary_info['station'] == index_station,temporary_info['antenna'] == index_antenna),temporary_info['solution'] == solution_type_max.encode() )] #numpy.array(temporary_info['solution']).astype(str) == solution_type_max
                    max_solution_cut = numpy.logical_and(temporary_info['solution'] == solution_type_max.encode(),numpy.logical_and(temporary_info['station'] == index_station, temporary_info['antenna'] == index_antenna))
                    dic_max['d'] = temporary_info[ max_solution_cut ]['distance']
                    dic_max['r'] = rs[max_solution_cut]
                    dic_max['t'] = temporary_info[ max_solution_cut ]['time']
                    dic_max['theta'] = temporary_info[ max_solution_cut ]['theta_ray']
                    dic_max['theta_ant'] = temporary_info[ max_solution_cut ]['theta_ant']
                    dic_max['a_s'] = temporary_info[ max_solution_cut ]['a_s']
                    dic_max['a_p'] = temporary_info[ max_solution_cut ]['a_p']
                    dic_max['z'] = z_0#self.in_dic_array[station.label][antenna.label][solution_type_max]['z'][eventid]
                
                
                #Only do triggering if any pass pre_trigger
                if numpy.any(temporary_info[temporary_info['station'] == index_station]['pre_triggered'] == True):
                    # Triggering Code below:
                    signals_out[station.label] = numpy.array([])
                    if numpy.any(info['has_solution']) == True:
                        if self.electricFieldDomain == 'time':
                            for index_antenna, antenna in enumerate(station.antennas):
                                u_in = []
                                V_in = []
                                if numpy.logical_and(include_noise == True,summed_signals == True):
                                    V_just_noise_in = []
                                if summed_signals == False:
                                    max_V_in_val = 0
                                    max_E_val_solution_type = ''
                                for solution in antenna.solutions:
                                    if self.in_flag_array[station.label][antenna.label][solution][eventid]:
                                        u_in.append(time_analog[station.label][antenna.label][solution])
                                        V_in.append(V_analog[station.label][antenna.label][solution])
                                        
                                        if numpy.logical_and(include_noise == True,summed_signals == True):
                                            V_just_noise_in.append(V_just_noise[station.label][antenna.label][solution])
                                        if summed_signals == False:
                                            current_max = max(numpy.fabs(V_analog[station.label][antenna.label][solution]))
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
                                        u_out = numpy.array(time_analog[station.label][antenna.label][max_E_val_solution_type])
                                        V_out = numpy.array(V_analog[station.label][antenna.label][max_E_val_solution_type])
                                    Vd_out, ud_out = gnosim.detector.fpga.digitizeSignal(u_out,V_out,digital_sample_times,station.sampling_bits,antenna.noise_rms,station.scale_noise_to, dc_offset = dc_offset, plot = False)
                                    
                                else:
                                    V_out = numpy.array([])
                                    u_out = numpy.array([])
                                    Vd_out = numpy.array([])
                                    ud_out = numpy.array([])
                            
                                time_analog[station.label][antenna.label] = u_out
                                V_analog[station.label][antenna.label] = V_out
                                time_digital[station.label][antenna.label] = ud_out
                                V_digital[station.label][antenna.label] = Vd_out
                            
                            
                            min_time = digital_sample_times[0]
                            max_time = digital_sample_times[-1]
                            dt = digital_sample_times[1] - digital_sample_times[0]
                        
                        #Triggering
                        if do_beamforming == True:
                            #Here is where I perform the beamforming algorithms. 
                            
                            Vd_out_sync, ud_out_sync  = gnosim.detector.fpga.syncSignals(V_digital[station.label],time_digital[station.label], min_time, max_time, dt)
                            formed_beam_powers, beam_powersums = gnosim.detector.fpga.fpgaBeamForming(Vd_out_sync,ud_out_sync, station.beam_dict , plot1 = False, plot2 = False, save_figs = False, cap_bits = station.beamforming_power_sum_bit_cap)
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
                            top_val_theta_ant = numpy.array([station.beam_dict['theta_ant'][beam_label] for beam_label in top_val_beams])
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
                            Vd_out_sync, ud_out_sync  = gnosim.detector.fpga.syncSignals(V_digital[station.label],time_digital[station.label], min_time, max_time, dt)
                    else:
                        triggered = False    
                            
                    if triggered == True:
                        info['triggered'] = True
                        temporary_info['triggered'] = True
                        signals_out[station.label] = numpy.vstack((Vd_out_sync, ud_out_sync[0,:]))
                        
                        try:
                            print('Triggered on event %i at Time: %0.3f'%(eventid,time.time() - self.throw_start_time))
                        except:
                            print('Triggered on event %i'%(eventid))
                        sys.stdout.flush()
                        if plot_geometry == True:
                            neutrino_loc = [x_0, y_0, z_0]
                            if len(info[info['has_solution'] == True]) > 0:
                                self.lock.acquire()
                                try:
                                    print('Plotting Signals Acquired a lock')
                                    fig = gnosim.trace.refraction_library.plotGeometry(self.stations,neutrino_loc,phi_0,temporary_info[numpy.logical_and(temporary_info['has_solution'] == True,temporary_info['station'] == index_station)],self.ice)
                                    if return_fig_array == True:
                                        fig_array.append(fig)

                                    try:
                                        fig.savefig('%s%s_all_antennas-event%i.%s'%(image_path,self.outfile.split('/')[-1].replace('.h5',''),eventid,plot_filetype_extension),bbox_inches='tight')
                                        pylab.close(fig)
                                    except Exception as e:
                                        print('Failed to save image for plotGeometry on event %i'%(eventid))
                                        print(e)
                                        sys.stdout.flush()
                                finally:
                                    print('Plotting Geometry Released a lock')
                                    self.lock.release()
                        
                        
                        if numpy.logical_and(self.electricFieldDomain == 'time',plot_signals == True):
                            self.lock.acquire()
                            try:
                                print('Plotting Signals Acquired a lock')

                                fig = pylab.figure(figsize=(16.,11.2)) #my screensize
                                
                                n_rows = len(station.antennas)
                                ntables = 4 #With below lines is 5 for beamforming == True
                                height_ratios = [2,2,n_rows+1,n_rows+1]
                                if do_beamforming == True:
                                    ntables += 1
                                    height_ratios.append(0.5*sum(height_ratios))
                                    
                                gs_left = gridspec.GridSpec(n_rows, 2, width_ratios=[3, 2]) #should only call left plots.  pylab.subplot(gs_left[0]),pylab.subplot(gs_left[2]),...
                                gs_right = gridspec.GridSpec(ntables, 2, width_ratios=[3, 2], height_ratios=height_ratios) #should only call odd tables pylab.subplot(gs_right[1])
                                #if do_beamforming == True:
                                #    gs_beam_forming = gridspec.GridSpec(ntables, 3, width_ratios=[3, 1,5], height_ratios=height_ratios)
                                    
                                #ax = pylab.subplot(gs_left[0])
                                
                                first_in_loop = True
                                axis2 = []
                                max_ax1_range = numpy.array([1e20,-1e20])
                                for index_antenna, antenna in enumerate(station.antennas):
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
                                    ax1.plot(time_analog[station.label][antenna.label],V_analog[station.label][antenna.label],label='s%ia%i'%(index_station,index_antenna),linewidth=0.6,c = c1)
                                    ax2.plot(time_digital[station.label][antenna.label],V_digital[station.label][antenna.label],label='s%ia%i'%(index_station,index_antenna),linewidth=0.4,c = c2)
                                    
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
                                    ax2.set_ylim(max_ax1_range * station.scale_noise_to / antenna.noise_rms)
                                    
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
                                reduction_factor =       ['%0.3g'%i for i in info['signal_reduction_factor']]
                                df = pandas.DataFrame({'antenna':antenna , '$\\theta_\mathrm{ant}$ (deg)':theta_ant , '$\\theta_\mathrm{emit}$ (deg)':observation_angle,'d$_\mathrm{path}$ (m)':distance, 'Reduction Factor':reduction_factor})
                                table = pylab.table(cellText = df.values, colLabels = df.columns, loc = 'center')
                                table.auto_set_font_size(False)
                                table.set_fontsize(10)
                                
                                
                                #TABLE 4: Max Voltage and SNR per Antenna
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
                                    
                                    gs_beam_forming = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[9], wspace=0.2, hspace=0.1, width_ratios=[1,12,6])
                                    #table_fig = pylab.subplot(gs_beam_forming[13])
                                    table_fig = pylab.subplot(gs_beam_forming[1])
                                    #table_fig = pylab.subplot(gs_right[9])
                                    table_ax = pylab.gca()
                                    table_fig.patch.set_visible(True)
                                    
                                    for beam_index, beam_label in enumerate(station.beam_dict['beams'].keys()):
                                        table_ax.plot(beam_powersums[beam_label],label = '%s, $\\theta_{ant} = $ %0.2f'%(beam_label,station.beam_dict['theta_ant'][beam_label]),color = station.beam_colors[beam_index])

                                    pylab.yticks(rotation=45)
                                    table_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                    #pylab.legend(loc='upper right', bbox_to_anchor=(1.05, 0.5))
                                    #table_ax.axis('tight')
                                pylab.subplots_adjust(left = 0.03, bottom = 0.05, right = 0.97, top = 0.97, wspace = 0.15, hspace = 0.28)
                                if return_fig_array == True:
                                    fig_array.append(fig)   

                                try:
                                    pylab.savefig('%s%s-event%i.%s'%(image_path,self.outfile.split('/')[-1].replace('.h5',''),eventid,plot_filetype_extension),bbox_inches='tight')
                                    pylab.close(fig)
                                    #print('Saved image %s%s-event%i.%s'%(image_path,self.outfile,eventid,plot_filetype_extension))
                                except Exception as e:
                                    print('Failed to save image for plotSignals on event %i'%(eventid))
                                    print(e)
                                    sys.stdout.flush()
                            finally:
                                print('Plotting Signals Released a lock')
                                self.lock.release()
                    else:
                        info['triggered'] = False
                        temporary_info['triggered'] = False
        else:
            dic_max['d'] = temporary_info[0]['distance']
            dic_max['r'] = rs[0]
            dic_max['t'] = temporary_info[0]['time']
            dic_max['theta'] = temporary_info[0]['theta_ray']
            dic_max['theta_ant'] = temporary_info[0]['theta_ant']
            dic_max['a_s'] = temporary_info[0]['a_s']
            dic_max['a_p'] = temporary_info[0]['a_p']
            dic_max['z'] = z_0#self.in_dic_array[station.label][antenna.label][solution_type_max]['z'][eventid]
        if output_all_solutions == True:
            info = temporary_info
        p_detect = numpy.any(info['has_solution'])
        if return_fig_array == True:
            return eventid, p_interact, p_earth, p_detect, event_electric_field_max, dic_max, event_observation_angle_max, event_solution_max, event_index_station_max, event_index_antenna_max, info, triggered, signals_out, fig_array
        else:
            return eventid, p_interact, p_earth, p_detect, event_electric_field_max, dic_max, event_observation_angle_max, event_solution_max, event_index_station_max, event_index_antenna_max, info, triggered, signals_out
    
    def makeFlagArray(self, x_query, y_query , z_query):
        '''
        This function takes a set of x,y,z coordinates and determines whether each set
        is within the set of solutions. It checks the points against the
        self.concave_hull bounding functions.  Defines self.in_flag_array.

        Parameters
        ----------
        x_query : numpy.ndarray of floats
            The list of x cartesian coordinates in the ice frame of the neutrino interactions.
        y_query : numpy.ndarray of floats
            The list of y cartesian coordinates in the ice frame of the neutrino interactions.
        z_query : numpy.ndarray of floats
            The list of z cartesian coordinates in the ice frame of the neutrino interactions.
        '''
        in_flag_array = {}
        for index_station, station in enumerate(self.stations):
            in_flag_array[station.label] = {}
            for antenna in self.stations[index_station].antennas:
                print('Creating in_flag_array for:', antenna.label)
                r_query = numpy.sqrt((x_query - antenna.x)**2 + (y_query - antenna.y)**2)
                
                if ((type(r_query) != numpy.ndarray) or (type(z_query) != numpy.ndarray)):
                    if ((type(r_query) != list) or (type(z_query) != list)):
                        r_query = numpy.array([r_query])
                        z_query = numpy.array([z_query])
                    else:
                        r_query = numpy.array(r_query)
                        z_query = numpy.array(z_query)
                
                in_flag_array[station.label][antenna.label] = {}

                for solution in antenna.solutions:
                    print('\tSolution Type:', solution)

                    in_bound = numpy.logical_and((z_query >= antenna.concave_hull[solution]['z_min']),z_query <= antenna.concave_hull[solution]['z_max'])
                    r_in_hull = numpy.logical_and((r_query >= antenna.concave_hull[solution]['f_inner_r_bound'](z_query)),(r_query <= antenna.concave_hull[solution]['f_outer_r_bound'](z_query)))
                    has_solution = numpy.logical_and(in_bound,r_in_hull)
                    
                    if numpy.all( has_solution == False ):
                        print('No solutions found for', antenna.label, solution)
                        in_flag_array[station.label][antenna.label][solution] = numpy.array([False]*len(r_query))

                    in_flag_array[station.label][antenna.label][solution] = has_solution
        self.in_flag_array = in_flag_array

    def makeFlagDicArrayFromInfo(self,info):
        '''
        This is intended for use in offline mode.  Given the info = reader['info'][...] array
        from a previously computed simulation, this can recreate the in_dic_array and in_flag_array
        so they can be used to reproduce events without running a full simulation.

        Parameter
        ---------
        info : numpy.ndarray of self.info_dtype
            Contains all meta data about the event for each antenna and solution type (if output_all_solutions was True when this was generated).
        '''
        key_dict = {
                'theta':'theta_ray',
                'theta_ant':'theta_ant',
                'd':'distance',
                't':'time'}
        if numpy.isin('a_h',list(info.dtype.fields.keys())):
            key_dict['a_s'] = 'a_h' #Backwards compatibility for info created before relabelling.
        elif numpy.isin('a_s',list(info.dtype.fields.keys())):
            key_dict['a_s'] = 'a_s'
        if numpy.isin('a_v',list(info.dtype.fields.keys())):
            key_dict['a_p'] = 'a_v' #Backwards compatibility for info created before relabelling.
        elif numpy.isin('a_p',list(info.dtype.fields.keys())):
            key_dict['a_p'] = 'a_p'

        in_dic_array = {}
        in_flag_array = {}   
        for index_station, station in enumerate(self.stations):
            station_cut = info['station'] == index_station
            in_dic_array[station.label] = {}
            in_flag_array[station.label] = {}
            for index_antenna, antenna in enumerate(station.antennas):
                in_dic_array[station.label][antenna.label] = {}
                in_flag_array[station.label][antenna.label] = {}
                antenna_cut = info['antenna'] == index_antenna
                for solution in antenna.solutions:
                    in_dic_array[station.label][antenna.label][solution] = {}
                    solution_cut = info['solution'] == solution.encode()
                    cut = numpy.logical_and(solution_cut,numpy.logical_and(antenna_cut,station_cut))

                    if sum(cut) != 0:
                        in_flag_array[station.label][antenna.label][solution] = info[cut]['has_solution']
                        
                    else:
                        in_flag_array[station.label][antenna.label][solution] = False
                    for key in list(key_dict.keys()):
                        in_dic_array[station.label][antenna.label][solution][key] = info[cut][key_dict[key]]
        self.in_dic_array = in_dic_array
        self.in_flag_array = in_flag_array

    def griddata_Event(self, x_query, y_query , z_query, method = 'cubic',events_per_calc = 10000000):
        '''
        This function takes a set of x,y,z coordinates and determines whether neutrino location is observable (has a solution).  
        It then obtains information about the neutrino event by interpolating the queried position against the ray tracing library
        for each event, using interpolation and the scipy.interpolate.griddata function.  Solution types for each event that
        are not observed are masked with values of -999.0.  self.in_dic_array is then populated used these interpolated values, (sorted
        by station label -> antenna label -> solution type -> key - > eventid).

        Parameters
        ----------
        x_query : numpy.ndarray of floats
            The list of x cartesian coordinates in the ice frame of the neutrino interactions.
        y_query : numpy.ndarray of floats
            The list of y cartesian coordinates in the ice frame of the neutrino interactions.
        z_query : numpy.ndarray of floats
            The list of z cartesian coordinates in the ice frame of the neutrino interactions.
        method : str, optional
            The selected method of interpolation.  (Default is 'cubic').
        events_per_calc : int, optional
            Limits the number of events sent through griddata at a time.  Is present as a feature to potentially reduce memory usage.
            (Default is 10000000).
        '''
        griddata_initate_time = time.time()
        self.makeFlagArray(x_query, y_query , z_query)
        self.in_dic_array = {}
        for index_station,station in enumerate(self.stations):
            self.in_dic_array[station.label] = {}
            for antenna in self.stations[index_station].antennas:
                print('Running Events Through Griddata Interpolation for:', antenna.label)
                r_query = numpy.sqrt((x_query - antenna.x)**2 + (y_query - antenna.y)**2)
                
                if ((type(r_query) != numpy.ndarray) or (type(z_query) != numpy.ndarray)):
                    if ((type(r_query) != list) or (type(z_query) != list)):
                        r_query = numpy.array([r_query])
                        z_query = numpy.array([z_query])
                    else:
                        r_query = numpy.array(r_query)
                        z_query = numpy.array(z_query)
                
                self.in_dic_array[station.label][antenna.label] = {}

                for solution in antenna.solutions:
                    print('\tSolution Type:', solution)
                    self.in_dic_array[station.label][antenna.label][solution] = {}
                    has_solution = self.in_flag_array[station.label][antenna.label][solution]
                    
                    if numpy.all( has_solution == False ):
                        for key in antenna.lib.data[solution].keys():
                            self.in_dic_array[station.label][antenna.label][solution][key] = []
                        continue

                    zm_query = numpy.ma.masked_array( z_query, mask = ~has_solution)
                    rm_query = numpy.ma.masked_array( r_query, mask = ~has_solution)

                    left_event = 0
                    
                    for key in antenna.lib.data[solution].keys():
                        self.in_dic_array[station.label][antenna.label][solution][key] = numpy.zeros_like(z_query)
                    
                    while left_event < len(z_query):
                        cut = numpy.arange(left_event,min(left_event+events_per_calc,len(z_query)))
                        for key in antenna.lib.data[solution].keys():
                            self.in_dic_array[station.label][antenna.label][solution][key][cut] = numpy.ma.filled(numpy.ma.masked_array(scipy.interpolate.griddata((antenna.lib.data[solution]['r'],antenna.lib.data[solution]['z']),antenna.lib.data[solution][key],(rm_query[cut], zm_query[cut]),method=method,fill_value=-999.0),mask = ~has_solution[cut],fill_value = -999.0))
                        print('\t\t%s : %i/%i'%(solution,min(left_event+events_per_calc,len(z_query)),len(z_query)))
                        left_event += events_per_calc
        print('Finished griddata_Event in ', time.time() - griddata_initate_time, 's')
    
    def singleAntennaGridDataEvent(self, r_query , z_query, antenna, in_flag_array, method = 'cubic',events_per_calc = 10000000):
            '''
            This will do what griddata_Event does but for a single antenna at time. This is intended to be used during multithreading. 
            in_flag_array should be the dict for only that antenna.

            Parameters
            ----------
            r_query : numpy.ndarray of floats
                The list of polar r coordinates in cylindrical coordinates in the ice frame (centered at this particular antenna).
            z_query : numpy.ndarray of floats
                The list of z cartesian coordinates in the ice frame of the neutrino interactions.
            antenna : gnosim.detector.detector.Antenna
                The antenna object containing information about this particular antenna.
            in_flag_array : dict
                The in_flag_array corresponding to the particular antenna passed to this function. 
            method : str, optional
                The selected method of interpolation.  (Default is 'cubic').
            events_per_calc : int, optional
                Limits the number of events sent through griddata at a time.  Is present as a feature to potentially reduce memory usage.
                (Default is 10000000).

            Returns
            -------
            out_dic_array : dict
                A dictionary containing the results of interpolation.   (Sorted by solution type -> key - > eventid).
            '''

            print('Running Events Through Griddata Interpolation for:', antenna.label)
            if ((type(r_query) != numpy.ndarray) or (type(z_query) != numpy.ndarray)):
                if ((type(r_query) != list) or (type(z_query) != list)):
                    r_query = numpy.array([r_query])
                    z_query = numpy.array([z_query])
                else:
                    r_query = numpy.array(r_query)
                    z_query = numpy.array(z_query)

            out_dic_array = {}
            
            for solution in antenna.solutions:
                out_dic_array[solution] = {}

                in_bound = numpy.logical_and((z_query >= antenna.concave_hull[solution]['z_min']),z_query <= antenna.concave_hull[solution]['z_max'])
                r_in_hull = numpy.logical_and((r_query >= antenna.concave_hull[solution]['f_inner_r_bound'](z_query)),(r_query <= antenna.concave_hull[solution]['f_outer_r_bound'](z_query)))
                has_solution = in_flag_array[solution]
                
                if numpy.all( has_solution == False ):
                    print('No solutions found for', antenna.label, solution)
                    for key in antenna.lib.data[solution].keys():
                        out_dic_array[solution][key] = []
                    continue

                zm_query = numpy.ma.masked_array( z_query, mask = ~has_solution)
                rm_query = numpy.ma.masked_array( r_query, mask = ~has_solution)
                
                left_event = 0
                
                for key in antenna.lib.data[solution].keys():
                    out_dic_array[solution][key] = numpy.zeros_like(z_query)
                
                while left_event < len(z_query):
                    cut = numpy.arange(left_event,min(left_event+events_per_calc,len(z_query)))
                    for key in antenna.lib.data[solution].keys():
                        out_dic_array[solution][key][cut] = numpy.ma.filled(numpy.ma.masked_array(scipy.interpolate.griddata((antenna.lib.data[solution]['r'],antenna.lib.data[solution]['z']),antenna.lib.data[solution][key],(rm_query[cut], zm_query[cut]),method=method,fill_value=-999.0),mask = ~has_solution[cut],fill_value = -999.0))
                    print('\t\t%s : %i/%i'%(solution,min(left_event+events_per_calc,len(z_query)),len(z_query)))
                    left_event += events_per_calc
            print('Done Interpolation for:', antenna.label)
            sys.stdout.flush()
            return out_dic_array
    
    def multiThreadGridDataEvent(self, x_query, y_query , z_query, method = 'cubic',n_cores = 4,events_per_calc = 10000000):
        '''
        This function manages the multithreading of calls to singleAntennaGridDataEvent.

        Parameters
        ----------
        x_query : numpy.ndarray of floats
            The list of x cartesian coordinates in the ice frame of the neutrino interactions.
        y_query : numpy.ndarray of floats
            The list of y cartesian coordinates in the ice frame of the neutrino interactions.
        z_query : numpy.ndarray of floats
            The list of z cartesian coordinates in the ice frame of the neutrino interactions.
        method : str, optional
            The selected method of interpolation.  (Default is 'cubic').
        n_cores : int, optional
            The number of cores available for multithreading.  (Default is 4).
        events_per_calc : int, optional
            Limits the number of events sent through griddata at a time (in a single thread).  Is present as a feature to potentially reduce memory usage.
            (Default is 10000000).
        '''
        griddata_initate_time = time.time()
        self.makeFlagArray(x_query, y_query, z_query)
        #initiate threads
        print('Submitting threads')
        thread_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers = n_cores) as executor:
            for index_station,station in enumerate(self.stations):
                thread_results[station.label] = {}
                for antenna in self.stations[index_station].antennas:
                    r_query = numpy.sqrt((x_query - antenna.x)**2 + (y_query - antenna.y)**2)
                    thread_results[station.label][antenna.label] = executor.submit(self.singleAntennaGridDataEvent, r_query , z_query, antenna,self.in_flag_array[station.label][antenna.label], method = 'cubic',events_per_calc = events_per_calc)
        
        print('Weaving threads')
        sys.stdout.flush()
        #write outputs of threads
        self.in_dic_array = {}
        for index_station, station in enumerate(self.stations):
            self.in_dic_array[station.label] = {}
            for antenna in self.stations[index_station].antennas:
                self.in_dic_array[station.label][antenna.label] = thread_results[station.label][antenna.label].result()
            
        print('Finished griddata_Event in ', time.time() - griddata_initate_time, 's')
        sys.stdout.flush()
    
    def throw(self, energy_neutrino=1.e9, 
              theta_0=None, phi_0=None, x_0=None, y_0=None, z_0=None, phi_vertex = None, r_vertex = None,
              anti=False, n_events=100000, detector_volume_radius=6000., detector_volume_depth=3000., 
              outfile=None,seed = None,pre_split = True, method = 'cubic', include_noise = True,summed_signals = True,
              plot_geometry = False, plot_signals = False, trigger_threshold = 0.0,trigger_threshold_units = 'fpga',plot_filetype_extension = 'svg',image_path = './',
              use_interp_threading = True,use_event_threading = True, output_all_solutions = True,save_signals = True,
              pre_trigger_angle = None):
        '''
        The work horse of the simulation.  This function 'throws' (generates) neutrinos in the ice volume for the given energy, distributing them throughout the
        ice and orienting them randomly.  This then performs calculations to determine if these neutrinos are observed by the stations defined in the config file.
        It saves meta data about each event and can also save signal wave forms, etc. 
        
        energy_neutrino : numpy.ndarray of floats, optional
            The energy that each neutrino will have.  Given in GeV.  (Default is 1.0e9).
        theta_0 : numpy.ndarray of floats, optional
            The polar spherical coordinates for the directions each neutrino came from.  Given in degrees.  (Default is None).
        phi_0 : numpy.ndarray of floats, optional
            The azimuthal spherical coordinates for the directions each neutrino came from.  Given in degrees.  (Default is None).
        x_0 : numpy.ndarray of floats
            The x cartesian ice frame coordinates of each neutrinos interaction location.  Given in m.  (Default is None).
            x_0 and y_0 are only used if both are specified, otherwise it will default to randomized values.  These inputs are
            mutually exclusive with phi_vertex and r_vertex.  If both sets are given then the coordinates given by phi_vertex, r_vertex are used.
        y_0 : numpy.ndarray of floats
            The y cartesian ice frame coordinates of each neutrinos interaction location.  Given in m.  (Default is None).
            x_0 and y_0 are only used if both are specified, otherwise it will default to randomized values.  These inputs are
            mutually exclusive with phi_vertex and r_vertex.  If both sets are given then the coordinates given by phi_vertex, r_vertex are used.
        z_0 : numpy.ndarray of floats
            The z cartesian ice frame coordinates of each neutrinos interaction location.  Given in m.  (Default is None).
        phi_vertex : numpy.ndarray of floats, optional
            The angular coordinate of each neutrino location in cylindrical coordinates in the ice frame.  Given in deg.  (Default is None).
            phi_vertex and r_vertex are only used if both are specified, otherwise it will default to randomized values.  These inputs are
            mutually exclusive with x_0 and y_0.  If both sets are given then the coordinates given by phi_vertex, r_vertex are used.
        r_vertex : numpy.ndarray of floats, optional
            The radial coordinate of each neutrino location in cylindrical coordinates in the ice frame.  Given in m.  (Default is None).
            phi_vertex and r_vertex are only used if both are specified, otherwise it will default to randomized values.  These inputs are
            mutually exclusive with x_0 and y_0.  If both sets are given then the coordinates given by phi_vertex, r_vertex are used.
        anti : bool, optional
            Selects either a neutrino (anti == False) or anti neutrino (anti == True).  (Default is False).
        n_events : int, optional
            The number of events to be thrown. 
        detector_volume_radius : , optional
            The radius of the 'detector' (ice to be used as detector/populated with neutrinos).  Given in m.  This should not exceed that
            used when making the ray tracing libraries defined in the configuration file.  Making it slightly lower than that used in the 
            ray tracing library construction can help avoid edge effects from the interpolation.
        detector_volume_depth : , optional
            The depth of the 'detector' (ice to be used as detector/populated with neutrinos).  Given in m.  This should not exceed that
            used when making the ray tracing libraries defined in the configuration file.  Making it slightly lower than that used in the 
            ray tracing library construction can help avoid edge effects from the interpolation.
        outfile : str, optional
            The address/name of the output file.  Only present self.throw is run.
        seed : int, optional
            The seed used for setting the random state of the simulation.  This will allow for testing with reproducable randomly generated neutrinos,
            noise, etc.
        pre_split : bool, optional
            Determines whether to attempt to load from pre split libraries.  If true (and the pre split libraries are calculated and 
            saved appropriately) this avoids lengthy calculations which seperate the rays ito the different solution types.  (Default is False).
        method : str, optional
            The selected method of interpolation.  (Default is 'cubic').
        include_noise : bool, optional
            Enables the addition of noise to the time domain signals.  Note supported for electricFieldDomain = 'freq'.  (Default is True).
        summed_signals : bool, optional
            If True, then signals resulting from different solution types are combined into a single waveform per antenna.  Otherwise
            only the waveform of the solution type with the maximum signal per antenna will be used.  (Default is True).
        plot_geometry : bool, optional
            Enables plotting of the neutrino location, rays, and antennas.  Only plots for trigger events. (Default is False).
        plot_signals : bool, optional
            Enables plotting of the waveforms, as well as of the beam forming plots and some meta data.  Only plots for trigger events. (Default is False).
        trigger_threshold : float, optional
            The trigger threshold to be applied on the signal or set of signals.  This should correspond to trigger_threshold_units.
            (Default is 0.0).
        trigger_threshold_units : str, optional
            This selects to units used for triggering.  To see the options try getAcceptedTriggerUnits().  If this is
            'fpga' then do_beamforming must be True.  (Default is 'fpga').
        plot_filetype_extension : str, optional
            Sets the file extension for any saved images.  (Default is 'svg').
        image_path : str, optional
            Sets the path of where any created images will be saved if enabled. (Default is './').
        use_interp_threading : bool, optional
            Enables multithreading during the interpolation portion of the code.  Significantly speeds up users perception of start up time (computationally
            this obviously still takes the same amount of time).  (Default is True).  Note that if this is enabled then live plotting is disabled, due to
            matplotlib currently not being thread safe.  Plots can be generated after the fact. (See gnosim.analysis.testing_single_event.py).
        use_event_threading : bool, optional
            Enables multithreading during the interpolation portion of the code.  Significantly speeds up users perception of total run time (computationally
            this obviously still takes the same amount of time).  (Default is True).  Note that if this is enabled then live plotting is disabled, due to
            matplotlib currently not being thread safe.  Plots can be generated after the fact. (See gnosim.analysis.testing_single_event.py).
        output_all_solutions : bool, optional
            Enables all solution types to be output, otherwise only the solution type with the maximum signal per antenna is output.  (Default is True).
        save_signals : bool, optional
            Enables saving the waveforms for triggered events.  Waveforms are saved in the output file as a dictionary under the 'signals' header.
            This dictionary is organized at the top level by event, then branching to station label. 
            The signals from a station are in the form of a numpy.ndarray which contains the signals for all antennas in the station.
            Each antenna in each station is stored as row in the numpy.ndarray corresponding to that station.  The final row of
            this ndarray is the times corresponding to the signals in every other row.  Signals are given in units of adu and times
            in units of ns.
            # TODO: Add a script that converts these to be in the same format as real data taken from an ARA station.
        pre_trigger_angle : float, optional
            If given, then a pre trigger will be applied to the event such that calculations of Askaryan radiation/electric field will only be conducted 
            if ANY of the possible solution types (for all antennas and stations) have an observation angle within pre_trigger_angle number of degrees to
            the Cherenkov angle.  Essentially the Cherenkov cone must be observable within the pre trigger angular window from at least one antenna in the
            array in order for the calculations to proceed.

            i.e. if pre_trigger_angle is 10.0 degrees then signals will only be calculated if one of the solutions was emitted (observed on cone) at an
            angle: theta_c - 10.0 deg < theta_obs < theta_c + 10.0 deg.

            If ANY of the solution types of ANY of the antennas in the entire array satisfies the pre trigger, then all calculations for that event proceed, 
            not just the solution types that independently satisfied the pre trigger threshold.  

            NOTE:  Setting a pre trigger can speed up the code significantly, but it must be done with care or else information may be lost.
            It is highly recommended that you first run a simulation with you desired settings and pre_trigger_angle set to None.  Then with this
            preliminary simulation analyse the losses you expect by not computing these angles (using gnosim/analysis/pre_trigger_set_tool.py for example).
            Then with this knowledge set the pre trigger for future simulations as you see fit.

            (Default is None).
        '''

        if numpy.logical_and(numpy.logical_or(plot_geometry,plot_signals),numpy.logical_or(use_event_threading,use_interp_threading)):
            print('Note: Pylab/Matplotlib are not threadig safe, so plotting will be disabled while threading is enabled.')
            plot_geometry = False
            plot_signals = False

        self.throw_start_time = time.time()
        
        self.pre_split = pre_split
        self.outfile = outfile
        self.save_signals = save_signals
        self.n_events = n_events
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

        #In the morning I want to turn this to be part of loading the antenna/station.  Will need to add it to the config file. 
        if self.electricFieldDomain == 'time':
            random_time_offsets = numpy.random.uniform(-1, 1, size=self.n_events)
            dc_offsets = numpy.zeros(self.n_events) #Perhaps something to change later.  Added now for ease. 

        #Preparing output arrays
        
        if output_all_solutions == False:
            self.len_info_per_event = self.n_antenna
        else:
            self.len_info_per_event = len(self.solutions)*self.n_antenna

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
        a_p_max = numpy.zeros(self.n_events)
        a_s_max = numpy.zeros(self.n_events)
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
            self.file.attrs['ice_model'] = self.ice.ice_model
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
            
            self.file.create_dataset('index_station', (self.n_events,), dtype='i', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('index_antenna', (self.n_events,), dtype='i', compression='gzip', compression_opts=9, shuffle=True)

            self.file.create_dataset('electric_field', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('observation_angle', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('solution', (self.n_events,), dtype='i', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('t', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('d', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('theta_ray', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('theta_ant', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('a_p', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('a_s', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)

            self.file.create_dataset('random_time_offsets', (self.n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            self.file.create_dataset('info', ( self.n_events * self.len_info_per_event , ) , dtype=self.info_dtype, compression='gzip', compression_opts=9, shuffle=True)

            self.file.create_group('signals')
        
        
        general_prep_time = time.time() - self.throw_start_time
        #Loading Hulls (or creating if hulls have not been previously determined in the necessary folder)
        
        print('About to run griddata_Event:')
        sys.stdout.flush()
        ############################################################################
        #Interpolating values from using griddata:
        ############################################################################
        print('Loading ray tracing libraries for interpolation.')
        for station in self.stations:
            station.loadLib(pre_split = self.pre_split)
            station.loadConcaveHull()

        #If memory becomes an issue this might need to be adapted to run for chunks of events 
        if use_interp_threading == True:
            self.multiThreadGridDataEvent(x_0, y_0, z_0, method = method,n_cores = self.n_cores)
        else:
            self.griddata_Event(x_0, y_0, z_0, method = method)   
        print('Succesfully ran griddata_Event:')
        griddata_time = time.time() - self.throw_start_time

        print('Use of ray tracing libraries complete, unloading.')
        for station in self.stations:
            station.deleteLib(verbose=False) #to save memory
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
                                    ii,inelasticity[ii], anti=anti,include_noise = include_noise,plot_signals=plot_signals,plot_geometry=plot_geometry,\
                                    summed_signals = summed_signals,trigger_threshold = trigger_threshold, trigger_threshold_units = trigger_threshold_units, \
                                    plot_filetype_extension=plot_filetype_extension, image_path = image_path,
                                    random_time_offset = random_time_offsets[ii],\
                                    dc_offset = dc_offsets[ii], do_beamforming = self.do_beamforming, output_all_solutions = output_all_solutions,
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
                    a_p_max[ii] = dic_max['a_p']
                    a_s_max[ii] = dic_max['a_s']
                    
                if self.outfile: 
                    if numpy.logical_and(self.save_signals == True,triggered == True):
                        #This region I will need to be careful adjustig when/if I add multithreading per event. 
                        #Note to future self, there is a section in 'Python and HDF5' about multithreading with HDF5
                        self.file['signals'].create_group(event_label)
                        for index_station, station in enumerate(self.stations):
                            self.file['signals'][event_label].create_dataset(station.label, numpy.shape(signals_out[station.label]), dtype='f', compression='gzip', compression_opts=9, shuffle=True)  
                            self.file['signals'][event_label][station.label][...] = signals_out[station.label]
                    
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
                                ii,inelasticity[ii], anti=anti, include_noise = include_noise,plot_signals=plot_signals,plot_geometry=plot_geometry,\
                                summed_signals = summed_signals,trigger_threshold = trigger_threshold, trigger_threshold_units = trigger_threshold_units, \
                                plot_filetype_extension=plot_filetype_extension, image_path = image_path,
                                random_time_offset = random_time_offsets[ii],\
                                dc_offset = dc_offsets[ii], do_beamforming = self.do_beamforming, output_all_solutions = output_all_solutions,
                                pre_trigger_angle = pre_trigger_angle, event_seed = event_seeds[ii])
                #print(info[(ii * self.len_info_per_event ):((ii+1) * self.len_info_per_event )])
                if numpy.logical_and(self.save_signals == True,triggered == True):
                    #This region I will need to be careful adjustig when/if I add multithreading per event. 
                    #Note to future self, there is a section in 'Python and HDF5' about multithreading with HDF5
                    self.file['signals'].create_group(event_label)
                    for index_station, station in enumerate(self.stations):
                        self.file['signals'][event_label].create_dataset(station.label, numpy.shape(signals_out[station.label]), dtype='f', compression='gzip', compression_opts=9, shuffle=True)  
                        self.file['signals'][event_label][station.label][...] = signals_out[station.label]
                if p_detect[ii] == 1.:
                    t_max[ii] = dic_max['t']
                    d_max[ii] = dic_max['d']
                    theta_ray_max[ii] = dic_max['theta']
                    theta_ant_max[ii] = dic_max['theta_ant']
                    a_p_max[ii] = dic_max['a_p']
                    a_s_max[ii] = dic_max['a_s']
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
            
            print('Writing a_p after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['a_p'][...] = a_p_max
            
            print('Writing a_s after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['a_s'][...] = a_s_max

            print('Writing random_time_offsets after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['random_time_offsets'][...] = random_time_offsets
            
            print('Writing info after %0.3f s'%(time.time() - self.throw_start_time))
            sys.stdout.flush()
            self.file['info'][...] = info
            
            self.file.close()
    
            
        current_time  = time.time() - self.throw_start_time
        print('Throw finished after %0.3f s'%current_time)
        print('Time Breakdown:')
        print('Time performing general prep: %0.3f s'%(general_prep_time))
        print('Time interpolating with griddata: %0.3f s'%(griddata_time- general_prep_time))
        print('Time in event calculations:  %0.3f s'%(current_time - griddata_time))


############################################################

if __name__ == '__main__':

    config_file = sys.argv[1]
    energy_neutrino = float(sys.argv[2]) # GeV
    n_events = int(sys.argv[3])
    index = int(sys.argv[4])
    if len(sys.argv) == 6:
        seed = int(sys.argv[5])
        print('Using seed from sys.argv[5]: ', seed)
    else:
        seed = None
        if seed == None:
            print('Using no seed')
        else:
            print('Using internally (antarcticsim.py) defined seed: ', seed)

    sim_config = yaml.load(open(config_file))
    station_config_file = sim_config['station_config_file']

    station_config_file_fix = station_config_file.split('/')[-1].replace('.py','')

    if (seed != None):
        outfile = sim_config['outfile_dir'] + '%s_%s_%.2e_GeV_%i_events_%i_seed_%i.h5'%(    sim_config['outfile_name_root'],
                                                                                            station_config_file_fix,
                                                                                            energy_neutrino,
                                                                                            n_events,
                                                                                            seed,
                                                                                            index)
        print('\n\n!!!Using Seed!!! \n\n Seed: ', seed, '\nOutfile Name: \n', outfile)
    else:
        outfile = sim_config['outfile_dir'] + '%s_%s_%.2e_GeV_%i_events_%i.h5'%(    sim_config['outfile_name_root'],
                                                                                    station_config_file_fix,
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
    image_path = sim_config['image_path_root'] + outfile.replace('.h5','').split('/')[-1] #should end with a / before inputting into throw
    if os.path.exists(image_path):
        print('Image Directory Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(image_path))
        image_path = image_path + '_new'
        while os.path.exists(image_path):
            image_path = image_path + '_new'
    
    os.makedirs(image_path) 
    image_path = image_path + '/'
    print('Images will be saved to ', image_path)
    
    
    
    #Creating Sim and throwing events
    my_sim = Sim(station_config_file, solutions=numpy.array(sim_config['solutions']),electricFieldDomain = sim_config['electricFieldDomain'],do_beamforming = sim_config['do_beamforming'])
    
    sys.stdout.flush()
    my_sim.throw(   energy_neutrino,
                    n_events                = n_events,
                    detector_volume_radius  = my_sim.config['detector_volume']['radius'],
                    detector_volume_depth   = my_sim.config['detector_volume']['depth'],
                    outfile                 = outfile,
                    seed                    = seed, 
                    include_noise           = sim_config['include_noise'],
                    summed_signals          = sim_config['summed_signals'],
                    pre_split               = sim_config['pre_split'],
                    plot_geometry           = sim_config['plot_geometry'],
                    plot_signals            = sim_config['plot_signals'],
                    trigger_threshold       = sim_config['trigger_threshold'],
                    trigger_threshold_units = sim_config['trigger_threshold_units'],
                    plot_filetype_extension = sim_config['image_extension'],
                    image_path              = image_path,
                    use_interp_threading    = sim_config['use_event_threading'],
                    use_event_threading     = sim_config['use_event_threading'],
                    output_all_solutions    = sim_config['output_all_solutions'],
                    save_signals            = sim_config['save_signals'],
                    pre_trigger_angle       = sim_config['pre_trigger_angle'] )
    sys.stdout.flush()
    
    print('Trying to print station geometry and antenna orientations')
    try:
        fig = gnosim.detector.detector.plotArrayFromConfig(my_sim.config,my_sim.solutions,only_station = 'all',verbose = False)
        fig.savefig('%s%s_array_geometry.%s'%(image_path,outfile.split('/')[-1].replace('.h5',''),sim_config['image_extension']),bbox_inches='tight')
        pylab.close(fig)
    except Exception as e:
        print('Failed to save image %s%s_array_geometry.%s'%(image_path,outfile.split('/')[-1].replace('.h5',''),sim_config['image_extension']))
        print(e)
    
    print('Trying to create index.html file for new images')
    try:
        gnosim.utils.misc.makeIndexHTML(path = image_path ,filetype = sim_config['image_extension'])
    except Exception as e:
        print('Something went wrong in making index.html')
        print(e)
    sys.stdout.flush()

############################################################


