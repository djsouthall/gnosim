'''
This is meant to rerun calculations from the simulaton for a single event.
I.e. it will use an output file to get the necessary info to mostly reproduce
the event (except fo rinterpolation, whih it uses values fromm the info file
for.  
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

sys.path.append("/home/dsouthall/Projects/GNOSim/")
import gnosim.sim.antarcticsim
pylab.ion()
  
if __name__ == "__main__":
    pylab.close('all')
    #Parameters
    infile = '/home/dsouthall/scratch-midway2/results_2019_Mar_config_dipole_octo_-200_antarctica_180_rays_3.00e+09_GeV_10000_events_1.h5'
    plot_geometry = False
    plot_signals = True
    choose_n = 10 #How many of the triggered events to run
    #Loading (set up to only be done first time file is executed.)
    try:
        print('Pre_loaded = %i, skipping loading',pre_loaded)
    except:
        reader = h5py.File(infile , 'r')
        
        info = reader['info'][...]
        solutions = numpy.array(['direct', 'cross', 'reflect'])
        config_file = reader.attrs['config']
        trigger = 0#reader.attrs['trigger_threshold']
        testSim = gnosim.sim.antarcticsim.Sim(config_file, solutions = solutions, electricFieldDomain = 'time',do_beamforming = True)
        for station in testSim.stations:
            station.loadLib(pre_split = True, build_lib = False)
            station.loadConcaveHull()
        testSim.makeFlagDicArrayFromInfo(info) #Necessary for running events without interpolation (calling throw)
        pre_loaded = True

    #'''
    eventids = numpy.unique(info[info['triggered']]['eventid'])
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
        event_info = info[info['eventid'] == eventid]
        if numpy.isin('random_time_offsets',list(reader.keys())):
            random_time_offset = reader['random_time_offsets'][eventid]
        else:
            print('Loading a simulation creaed before time offsets saved, creating random one (may differ slightly from original calculation)')
            random_time_offset = random_time_offsets = numpy.random.uniform(-1, 1, size=1) #This isn't perfect, future sim outputs have this saved for reproducability

        if reader.attrs['pre_trigger_angle'] == 'None':
            pre_trigger_angle = None
        else:
            pre_trigger_angle = reader.attrs['pre_trigger_angle']
        dc_offset = 0 #This feature is not yet implemented really.  It would just be storing zeros if it were.
        eventid, p_interact, p_earth, p_detect, event_electric_field_max, dic_max, event_observation_angle_max, event_solution_max, event_index_station_max, event_index_antenna_max, info_out, triggered, signals_out, fig_array = \
            testSim.event(reader['energy_neutrino'][eventid], reader['phi_0'][eventid], reader['theta_0'][eventid], reader['x_0'][eventid], reader['y_0'][eventid], reader['z_0'][eventid], \
                        eventid,reader['inelasticity'][eventid], anti=False, include_noise = True,plot_signals=plot_signals,plot_geometry=plot_geometry,\
                        summed_signals = True,trigger_threshold = trigger , trigger_threshold_units = reader.attrs['trigger_mode'], \
                        plot_filetype_extension='svg', image_path = './', random_time_offset = random_time_offset,\
                        dc_offset = dc_offset, do_beamforming = testSim.do_beamforming, output_all_solutions = True,
                        pre_trigger_angle = pre_trigger_angle, event_seed = numpy.unique(event_info['seed'])[0],return_fig_array = True)
        for fig in fig_array:
            fig.show()
