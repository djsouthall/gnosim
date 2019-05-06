#!/usr/bin/env python3

'''
This is meant to rerun calculations from the simulaton for a single event.
I.e. it will use an output file to get the necessary info to mostly reproduce
the event (minus the interpolation, which it uses values from the info file
for).  
'''

import sys
import numpy
import h5py
import matplotlib
import pylab
import yaml
import os
import os.path
import glob
import gnosim.sim.antarcticsim
import ast
pylab.ion()
  
if __name__ == "__main__":
    pylab.close('all')
    #Parameters
    #results_2019_testingMay1_real_config_antarctica_180_rays_signed_fresnel_3.00e+09_GeV_1000000_events_1_seed_1.h5
    infile = '/home/dsouthall/scratch-midway2/results_2019_testingApril18_real_config_antarctica_180_rays_signed_fresnel_3.00e+09_GeV_2_events_4.h5'#results_2019_April15_real_config_full_station_3.00e+09_GeV_10000_events_1_seed_1.h5'#'/home/dsouthall/scratch-midway2/April8/results_2019_April_real_config_antarctica_180_rays_signed_fresnel_1.00e+08_GeV_1000000_events_0_seed_1.h5'#'/home/dsouthall/scratch-midway2/results_2019_April_real_config_full_station_3.00e+09_GeV_10000_events_5_seed_1.h5'#'/home/dsouthall/scratch-midway2/results_2019_April_real_config_full_station_3.00e+09_GeV_10000_events_1_seed_1.h5'#results_2019_Mar_config_dipole_octo_-200_antarctica_180_rays_3.00e+09_GeV_10000_events_1.h5
    plot_geometry = True
    plot_signals = True
    choose_n = 2 #How many of the triggered events to run
    #Loading (set up to only be done first time file is executed.)
    try:
        print('Pre_loaded = ', pre_loaded, 'skipping loading') #TODO: This works for exec(open(filename).read()) but not for %run in ipython...  Fix it somehow?
    except Exception as e:
        print(e)
        pre_loaded = True
        reader = h5py.File(infile , 'r')
        
        info = reader['info'][...]
        solutions = numpy.array(['direct'])#gnosim.trace.refraction_library.getAcceptedSolutions()[0:3]
        trigger = 0#reader.attrs['trigger_threshold']

        #Loading sim config dictionary (How this is stored has changed, hence if/trys below)
        if numpy.isin('sim_config',list(reader.attrs.keys())):
            sim_config = ast.literal_eval(reader.attrs['sim_config'])
        else:
            try:
                sim_config = ast.literal_eval(reader.attrs['config'])
            except:
                sim_config = yaml.load(open(reader.attrs['config']))

        #Loading station config dictionary (How this is stored has changed, hence if/trys below)
        if numpy.isin('sim_config',list(reader.attrs.keys())):
            station_config = ast.literal_eval(reader.attrs['station_config'])
        else:
            station_config = yaml.load(open(sim_config['station_config_file']))

        testSim = gnosim.sim.antarcticsim.Sim(station_config, solutions = solutions, electric_field_domain = 'time',do_beamforming = True)
        for station in testSim.stations:
            station.loadLib(pre_split = True, build_lib = False)
            station.loadConcaveHull()
        testSim.makeFlagDicArrayFromInfo(info) #Necessary for running events without interpolation (calling throw)
        

    #'''
    eventids = numpy.array([0])#numpy.sort(numpy.unique(info[info['triggered']]['eventid'])) #numpy.array([932807])#
    try:
        do_events = numpy.random.choice(eventids,choose_n,replace=False)
    except:
        try:
            do_events = numpy.unique(numpy.random.choice(eventids,choose_n,replace=True))
        except:
            print('eventids = ',eventids)
            print('This drew an error')
            print('defaulting to eventids = numpy.array([])')
            do_events = numpy.array([])

    for eventid in do_events:
        print('Attempting event ',eventid)
        event_info = info[info['eventid'] == eventid]
        if numpy.isin('random_time_offsets',list(reader.keys())):
            random_time_offset = reader['random_time_offsets'][eventid]
        else:
            print('Loading a simulation creaed before time offsets saved, creating random one (may differ slightly from original calculation)')
            random_time_offset = random_time_offsets = numpy.random.uniform(-1, 1, size=1) #This isn't perfect, future sim outputs have this saved for reproducability

        if numpy.logical_or(reader.attrs['pre_trigger_angle'] == 'None',reader.attrs['pre_trigger_angle'] == b'None'):
            pre_trigger_angle = None
        else:
            pre_trigger_angle = reader.attrs['pre_trigger_angle']

        if numpy.isin('event_seeds',list(reader.keys())):
            event_seed = reader['event_seeds'][eventid] 
        else:
            event_seed = numpy.unique(event_info['seed'])[0] #Backwards compatability while building the sim.
        dc_offset = 0 #This feature is not yet implemented really.  It would just be storing zeros if it were.


        eventid, p_interact, p_earth, p_detect, info_out, triggered, signals_out, fig_array = \
            testSim.event(reader['energy_neutrino'][eventid], reader['phi_0'][eventid], reader['theta_0'][eventid], reader['x_0'][eventid], reader['y_0'][eventid], reader['z_0'][eventid], \
                        eventid,reader['inelasticity'][eventid], anti=False, include_noise = True,plot_signals=plot_signals,plot_geometry=plot_geometry,\
                        summed_signals = True,trigger_threshold = trigger , trigger_threshold_units = reader.attrs['trigger_mode'].decode(), \
                        plot_filetype_extension='svg', image_path = './', random_time_offset = random_time_offset,\
                        dc_offset = dc_offset, do_beamforming = testSim.do_beamforming, output_all_solutions = True,
                        pre_trigger_angle = pre_trigger_angle, event_seed = event_seed,return_fig_array = True)
        for fig in fig_array:
            fig.show()
