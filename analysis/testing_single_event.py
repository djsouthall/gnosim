#!/usr/bin/env python3
'''
Purpose and Description
-----------------------
This script allows you to examine single events after a full simulation has been run.  It uses the meta data in the
output file to recreate the necessary infrastructue to run the event again using the antarcticsim code.  The only portion
of a normal simulation that does not occur is the interpolation against the grid, as this is very time consuming.  Instead
it uses the previously interpolated values stored in the output file.  The output of this will contain all of the details
of a particular event.  This has been left open and not encapsualted as a function because I think it is the sort of script
that will change a lot based on the needs of the particular user, and it is nice for them to not have to work around a strict
structure.  

Running
-------
It is recommended that you run this in a command line interface like ipython with the %run command or using the 
command exec(open('./script_path/script.py').read()) .

Debugging
---------
If you wish to examine the code at a particular point for an event that is not what you expected then
I recommend adding a pdb.set_trace() line into the antarcticsim.event() function where desirable, and then run this script
with the selected event id.  Select the infile you wish to examine.

Main Parameters
---------------
infile : str
    The path of the simulation output data you wish to work with.
plot_geometry : bool
    Enabling plot_geometry will plot the rays paths through the ice for that event.
plot_signals : bool
    Enabling plot_signals will plot the phased array signals and beam forming sums.
choose_n : int
    The maximum number of events to loop over.  This code chooses a list of eligble events to plot (eventids below) by
    applying the cuts specified in the Cut section.  Then it chooses from event_ids using the draw method specified
    by draw_order to obtaina subset of events to create do_events, which is looped over.  The lengh of do_events is capped 
    by choose_n.  If less than choose_n events pass the cut then only those events will be run.
draw_order : str
    Selects how do_events is chosen from eventids.  Either 'random' (selects randomly) or 'ordered' (selects in the order
    that eventids has them indexed).  
solutions : numpy.ndarray of floats
    This is the list of solution types you wish to recreate when running the calculations again.  You can provide less solutions
    than were originally used for the creation of the data if you wish to isolate a particular solution type, but running more
    (i.e. enabling bottom-reflecting events if they weren't before) will cause issues because these solution types aren't
    interpolated.
trigger_threshold : int or float
    This is the trigger level for running the event again.  If you want to guarentee plotting then you should set this to 0.
    The units of this should match the units set by trigger_threshold_units.  If this is set to None the the original threshold
    is used.
trigger_threshold_units : str
    This selects the units you wish to use for the current recalculation.  If this is set to None the the original threshold
    is used.
'''
import os
import sys
sys.path.append(os.environ['GNOSIM_DIR'])
import os.path
import numpy
import h5py
import matplotlib
import pylab
import yaml
import glob
import gnosim.sim.antarcticsim
import ast
pylab.ion()

if __name__ == "__main__":
    pylab.close('all')

    ###------------###
    ### Parameters ###
    ###------------###

    infile = '/home/dsouthall/scratch-midway2/results_2019_testing_input_event_locations_real_config_antarctica_180_rays_signed_fresnel_1.00e+08_GeV_0_events_1.h5'
    plot_geometry = True
    plot_signals = True
    choose_n = 1 #How many of the triggered events to run
    trigger_threshold = 0
    trigger_threshold_units = None
    draw_order = 'random' #'ordered'

    ###--------------###
    ### Loading Data ###
    ###--------------###

    reader = h5py.File(infile , 'r')
    info = reader['info'][...]
    solutions = numpy.array([s.decode() for s in numpy.unique(info['solution'])])#gnosim.trace.refraction_library.getAcceptedSolutions()[0:3]#numpy.array(['direct'])#gnosim.trace.refraction_library.getAcceptedSolutions()[0:3]

    ###-----------------###
    ### Performing Cuts ###
    ###-----------------###

    '''
    #This is an example where I was looking at a particular effect in the polarization, and shows out one might use this section.
    polarization_emission_theta = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))

    cut_A = numpy.logical_and(info['theta_ant'] < 30.0, info['theta_ant'] > 25.0)
    cut_B = numpy.logical_or(polarization_emission_theta <= 85.0, polarization_emission_theta > 95.0)
    cut_C = info['has_solution']

    cut = numpy.logical_and(numpy.logical_and(cut_A,cut_B),cut_C)

    eventids = numpy.array(info[cut][numpy.argsort(numpy.fabs(info[cut]['a_p']))]['eventid'])
    '''

    #Standard application
    cut_A = info['has_solution']
    cut_B = info['triggered']#numpy.ones_like(cut_A)#info['triggered']

    cut = numpy.logical_and(cut_A,cut_B)
    eventids = numpy.unique(info[cut]['eventid'])
    #eventids = numpy.array([32])


    if type(eventids) != numpy.ndarray:
        if numpy.size(eventids) == 1:
            eventids = numpy.array([eventids])
        else:
            eventids = numpy.array(eventids)

    ###----------------###
    ### Picking Events ###
    ###----------------###

    if draw_order == 'random':
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
    elif draw_order == 'ordered':
        try:
            do_events = eventids[0:numpy.min([choose_n,numpy.size(eventids)])]
        except:
            print('eventids = ',eventids)
            print('This drew an error')
            print('defaulting to eventids = numpy.array([])')
            do_events = numpy.array([])

    ###--------------###
    ### Preparations ###
    ###--------------###

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

    testSim = gnosim.sim.antarcticsim.Sim(station_config, solutions = solutions, electric_field_domain = 'time',do_beamforming = True, pre_split = True, load_lib = False)

    testSim.makeFlagDicArrayFromInfo(info,eventids = do_events) #Necessary for running events without interpolation (calling throw)

    if trigger_threshold == None:
        trigger_threshold = reader.attrs['trigger_threshold']
    if trigger_threshold_units == None:
        trigger_threshold_units = reader.attrs['trigger_mode'].decode()

    ###----------------###
    ### Running Events ###
    ###----------------###
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
                        summed_signals = True,trigger_threshold = trigger_threshold , trigger_threshold_units = trigger_threshold_units, \
                        plot_filetype_extension='svg', image_path = './', random_time_offset = random_time_offset,\
                        dc_offset = dc_offset, do_beamforming = testSim.do_beamforming, output_all_solutions = True,
                        pre_trigger_angle = pre_trigger_angle, event_seed = event_seed,return_fig_array = True)
        
        for fig in fig_array:
            fig.show()


    ###-------------------###
    ### Misc Calculations ###
    ###-------------------###

    if plot_geometry:
        brewsters_angle = numpy.rad2deg(numpy.arctan(testSim.ice.indexOfRefraction(0.1)/testSim.ice.indexOfRefraction(-0.1))) 
        print('Brewsters Angle = %0.3f'%brewsters_angle)
