'''
This is meant to run a simulation of the siulation for a single event.
I.e. it will use an output file to get the necessary info to mostly reproduce
the event (except fo rinterpolation, whih it uses values fromm the info file
for.  
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
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

sys.path.append("/home/dsouthall/Projects/GNOSim/")
import gnosim.utils.quat
import gnosim.earth.earth
import gnosim.earth.antarctic
import gnosim.trace.refraction_library_beta
#from gnosim.trace.refraction_library_beta import *
import gnosim.interaction.askaryan
import gnosim.interaction.askaryan_testing
import gnosim.sim.detector
import gnosim.sim.fpga
pylab.ion()

def redoEventFromInfo(reader,eventid,energy_neutrino,index_of_refraction,signal_times,h_fft,sys_fft,freqs_response,sampling_bits,noise_rms,scale_noise_to,digital_sampling_period,random_time_offset,dc_offset,beam_dict,trigger_threshold_units, trigger_threshold,include_noise = True,summed_signals = True,do_beamforming = True, plot_geometry = True, plot_signals = True,plot_first = False,single_plot_signals   = False,single_plot_spectrum  = False,single_plot_angles = False, single_plot_potential = False,plot_each_beam = False):
    '''
    '''
    event_label = 'event%i'%eventid
    config = yaml.load(open(reader.attrs['config']))
    info = reader['info'][...]
    solutions = numpy.unique(info['solution'])
    info = info[info['eventid'] == eventid]
    inelasticity = reader['inelasticity'][eventid]
    x_0 = reader['x_0'][eventid]
    y_0 = reader['y_0'][eventid]
    z_0 = reader['z_0'][eventid]
    phi_0 = reader['phi_0'][eventid]
    p_interact = reader['p_interact'][eventid]
    p_earth = reader['p_earth'][eventid]
    signals_out = {}
    
    if numpy.isin('seed',list(info.dtype.fields.keys())):
        random_local = numpy.random.RandomState(seed = numpy.unique(info[info['eventid'] == eventid]['seed'])[0])
    else:
        random_local = numpy.random.RandomState()   
    if numpy.size(info) != 0:
    
        time_analog = {}
        V_analog = {}
        time_digital = {}
        V_digital = {}
        if numpy.logical_and(include_noise == True,summed_signals == True):
            V_just_noise = {}
        
        minimum_time = 1e20
        maximum_time = -1e20
        for index_station in numpy.unique(info['station']):
            # Loop over station antennas
            station_label = 'station%i'%index_station
            station_cut = info['station'] == index_station
            for index_antenna in range(0, config['antennas']['n']):
                antenna_label = config['antennas']['types'][index_antenna]
                antenna_cut = numpy.logical_and(info['antenna'] == index_antenna, station_cut)
                for solution in solutions:
                    solution_cut = numpy.logical_and(info['solution'] == solution, antenna_cut)
                    if info[solution_cut]['has_solution'] == 1:
                        minimum_time = numpy.min([minimum_time,signal_times[0] + info[solution_cut]['time']])
                        maximum_time = numpy.max([maximum_time,signal_times[-1] + info[solution_cut]['time']])
        if minimum_time == 1e20:
            minimum_time = signal_times[0]
        if maximum_time == -1e20:
            maximum_time = signal_times[-1]
        digital_sample_times = numpy.arange(minimum_time,maximum_time,digital_sampling_period) + random_time_offset #these
        
        for index_station in numpy.unique(info['station']):
            station_label = 'station%i'%index_station
            station_cut = info['station'] == index_station
            time_analog[station_label] = {}
            V_analog[station_label] = {}
            time_digital[station_label] = {}
            V_digital[station_label] = {}
            if numpy.logical_and(include_noise == True,summed_signals == True):
                V_just_noise[station_label] = {}
            for index_antenna in range(0, config['antennas']['n']):
                antenna_label = config['antennas']['types'][index_antenna]
                antenna_cut = numpy.logical_and(info['antenna'] == index_antenna, station_cut)
                time_analog[station_label][antenna_label] = {}
                V_analog[station_label][antenna_label] = {}
                if numpy.logical_and(include_noise == True,summed_signals == True):
                    V_just_noise[station_label][antenna_label] = {}
                for solution in numpy.unique(info['solution']):
                    solution_cut = numpy.logical_and(info['solution'] == solution, antenna_cut)
                    sub_info = info[solution_cut]
                    #print(sub_info)
                    if sub_info['has_solution'] == 1:
                        if include_noise == True:
                            
                            V_noiseless, u , dominant_freq, V_noise, SNR = gnosim.interaction.askaryan.quickSignalSingle( numpy.deg2rad(sub_info['observation_angle']),\
                              sub_info['distance'],energy_neutrino*inelasticity,index_of_refraction,\
                              sub_info['time'],sub_info['a_v'],sub_info['beam_pattern_factor'],\
                              signal_times,h_fft,sys_fft,freqs_response,plot_signals=single_plot_signals,plot_spectrum=single_plot_spectrum,plot_angles = single_plot_angles,plot_potential = single_plot_potential,\
                              include_noise = True, resistance = 50, temperature = 320,random_local = random_local)  #expects ovbservation_angle to be in radians (hence the deg2rad on input)
                            
                            if summed_signals == True:
                                V_just_noise[station_label][antenna_label][solution] = numpy.add(V_noise,-V_noiseless) #subtracting away raw signal from noisy signal to get just the noise
                            electric_array = V_noise
                            #electric_array_digitized, u_digitized = gnosim.sim.fpga.digitizeSignal(u,V_noise,digital_sample_times,stations[index_station].antennas[index_antenna].sampling_bits,noise_rms,scale_noise_to, dc_offset = dc_offset, plot = False)
                        else:
                            V_noiseless, u , dominant_freq = gnosim.interaction.askaryan.quickSignalSingle(numpy.deg2rad(sub_info['observation_angle']),\
                              sub_info['distance'],energy_neutrino*inelasticity,index_of_refraction,\
                              sub_info['time'],sub_info['a_v'],sub_info['beam_pattern_factor'],\
                              signal_times,h_fft,sys_fft,freqs_response,plot_signals=single_plot_signals,plot_spectrum=single_plot_spectrum,plot_angles = single_plot_angles,plot_potential = single_plot_potential,\
                              include_noise = False, resistance = 50, temperature = 320,random_local = random_local)  #expects ovbservation_angle to be in radians (hence the deg2rad on input)
                            
                            SNR = -999.
                            electric_array = V_noiseless
                        if plot_first == True:
                            plot_first = False
                            plots = True
                            single_plot_signals   = False
                            single_plot_spectrum  = False
                            single_plot_potential = False
                            single_plot_angles    = False
                        V_analog[station_label][antenna_label][solution] = electric_array
                        time_analog[station_label][antenna_label][solution] = u
                    else:
                        V_analog[station_label][antenna_label][solution] = []
                        time_analog[station_label][antenna_label][solution] = []
                
            
            # Triggering Preparation below:
            signals_out[station_label] = numpy.array([])
            station_cut = info['has_solution'] == 1
            if numpy.any(info['has_solution']) == True:
                for index_antenna in range(0, config['antennas']['n']):
                    antenna_label = config['antennas']['types'][index_antenna]
                    antenna_cut = numpy.logical_and(info['antenna'] == index_antenna, station_cut)
                    u_in = []
                    V_in = []
                    if numpy.logical_and(include_noise == True,summed_signals == True):
                        V_just_noise_in = []
                    if summed_signals == False:
                        max_V_in_val = 0
                        max_E_val_solution_type = ''
                    for solution in solutions:
                        solution_cut = numpy.logical_and(info['solution'] == solution, antenna_cut)
                        sub_info = info[solution_cut]
                        if sub_info['has_solution'] == 1:
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
                        Vd_out, ud_out = gnosim.sim.fpga.digitizeSignal(u_out,V_out,digital_sample_times,sampling_bits,noise_rms,scale_noise_to, dc_offset = dc_offset, plot = False)
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
            triggered = False
            if do_beamforming == True:
                #Here is where I perform the beamforming algorithms. 
                
                Vd_out_sync, ud_out_sync  = gnosim.sim.fpga.syncSignals(time_digital[station_label],V_digital[station_label], min_time, max_time, dt)
                formed_beam_powers, beam_powersums = gnosim.sim.fpga.fpgaBeamForming(ud_out_sync, Vd_out_sync, beam_dict , config, plot1 = plot_each_beam, plot2 = False, save_figs = False)
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
                top_val_theta_ant = numpy.array([beam_dict['theta_ant'][beam_label] for beam_label in top_val_beams])
                #Currently don't know what to do with these values.  They will be written out as I progress but
                #right now I am just testing that they can be calculate without breaking the simulation.
                #Right now I am only storing the 3 highest values.  It is likely that I want to store every beam
                #that satisfies the trigger condiditon?
            
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
            
            #Plotting
            if triggered == True:
                print('Triggered on event %i'%eventid)
                signals_out[station_label] = numpy.vstack((Vd_out_sync, ud_out_sync[0,:]))
                if plot_geometry == True:
                    origin = []
                    for index_antenna in info[info['has_solution'] == 1]['antenna']:
                        station_loc = numpy.array(config['stations']['positions'][index_station],dtype=float)
                        antenna_loc = numpy.add(numpy.array(config['antennas']['positions'][index_antenna],dtype=float),station_loc)
                        origin.append(list(antenna_loc))
                    
                    neutrino_loc = numpy.array([x_0, y_0, z_0],dtype=float)
                    if len(info[info['has_solution'] == 1]) > 0:
                        fig = gnosim.trace.refraction_library_beta.plotGeometry(origin,neutrino_loc,phi_0,info[numpy.logical_and(info['has_solution'] == 1,info['station'] == index_station)])
                        '''
                        try:
                            fig.savefig('%s%s_all_antennas-event%i.%s'%(image_path,outfile.split('/')[-1].replace('.h5',''),eventid,plot_filetype_extension),bbox_inches='tight')
                            pylab.close(fig)
                        except:
                            print('Failed to save image %s%s_all_antennas-event%i.%s'%(image_path,outfile.split('/')[-1].replace('.h5',''),eventid,plot_filetype_extension))
                        '''
                
                
                if plot_signals == True:
                    #might need to account for when signals are not present in certain detectors
                    #print('Attempting to plot', eventid)
                    temporary_info = numpy.zeros(config['antennas']['n'],info.dtype)      
                    for index_antenna in range(0, config['antennas']['n']):
                        antenna_label = config['antennas']['types'][index_antenna]
                        antenna_cut = numpy.logical_and(info['antenna'] == index_antenna, station_cut)
                        temporary_info[index_antenna] = info[antenna_cut][numpy.argmax(info[antenna_cut]['electric_field'])]
                        
                        for solution in numpy.unique(info['solution']):
                            solution_cut = numpy.logical_and(info['solution'] == solution, antenna_cut)
                            sub_info = info[solution_cut]
                            temporary_info
                    
                    fig = pylab.figure(figsize=(16.,11.2)) #my screensize
                    
                    n_rows = config['antennas']['n']
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
                        antenna_label = config['antennas']['types'][index_antenna]
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
                        ax2.set_ylim(max_ax1_range * scale_noise_to / noise_rms)
                        
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
                    
                    df = pandas.DataFrame({'E$_\\nu$ (GeV)':'%0.4g'%(energy_neutrino) , 'Inelasticity':'%0.4g'%inelasticity , 'p_interact':'%0.4g'%p_interact, 'p_earth':'%0.4g'%p_earth},index=[0])
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
                    antenna =           ['%i'%i for i in temporary_info['antenna'].astype(int)]
                    observation_angle = ['%0.5g'%i for i in temporary_info['observation_angle'].astype(float)]
                    theta_ant =         ['%0.5g'%i for i in temporary_info['theta_ant'].astype(float)]
                    distance =          ['%0.3g'%i for i in temporary_info['distance'].astype(float)]
                    beam_factor =       ['%0.3g'%i for i in temporary_info['beam_pattern_factor']]
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
                    antenna =           ['%i'%i for i in temporary_info['antenna'].astype(int)]
                    electric_field =    ['%0.3g'%i for i in temporary_info['electric_field'].astype(float)]
                    dom_freqs =         ['%0.3g'%i for i in (temporary_info['dominant_freq']/1e6).astype(float)]
                    SNRs =              ['%0.3g'%i for i in temporary_info['SNR'].astype(float)]
                    df = pandas.DataFrame({'antenna':antenna , '$V_\mathrm{max}$ (V)':electric_field , 'SNR':SNRs, '$f_\mathrm{max}$ (MHz)':dom_freqs})
                    table = pylab.table(cellText = df.values , colLabels = df.columns, loc = 'center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    
                    #TABLE 5: THE TABLE THAT'S ACTUALLY A PLOT AND ONLY SOMETIMES SHOWS UP DEPENDING ON SETTINGS :D
                    
                    if do_beamforming == True:
                        
                        colormap = pylab.cm.gist_ncar #nipy_spectral, Set1,Paired   
                        beam_colors = [colormap(i) for i in numpy.linspace(0, 1,len(beam_dict['beams'].keys())+1)] #I put the +1 backs it was making the last beam white, hopefully if I put this then the last is still white but is never called
        
                        
                        gs_beam_forming = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[9], wspace=0.1, hspace=0.1, width_ratios=[0.1,6,3])
                        #table_fig = pylab.subplot(gs_beam_forming[13])
                        table_fig = pylab.subplot(gs_beam_forming[1])
                        #table_fig = pylab.subplot(gs_right[9])
                        table_ax = pylab.gca()
                        table_fig.patch.set_visible(True)
                        
                        for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
                            table_ax.plot(beam_powersums[beam_label],label = '%s, $\\theta_{ant} = $ %0.2f'%(beam_label,beam_dict['theta_ant'][beam_label]),color = beam_colors[beam_index])
                            #print(beam_powersums[beam_label])
                        #for line_index,line in enumerate(table_ax.lines):
                        #    line.set_color(beam_colors[line_index])
                        #xlim = table_ax.get_xlim()
                        #table_ax.set_xlim( ( xlim[0] , xlim[1]*1.5 ) )
                        
                        #box = table_ax.get_position()
                        #table_ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                        pylab.yticks(rotation=45)
                        table_ax.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
                        #pylab.legend(loc='upper right', bbox_to_anchor=(1.05, 0.5))
                        #table_ax.axis('tight')
                        
                    pylab.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.99, top = 0.97, wspace = 0.15, hspace = 0.28)
                    #pylab.show(block=True)
                    '''
                    try:
                        pylab.savefig('%s%s-event%i.%s'%(image_path,outfile.split('/')[-1].replace('.h5',''),eventid,plot_filetype_extension),bbox_inches='tight')
                        pylab.close(fig)
                        #print('Saved image %s%s-event%i.%s'%(image_path,outfile,eventid,plot_filetype_extension))
                    except:
                        print('Failed to save image %s%s-event%i.%s'%(image_path,outfile,eventid,plot_filetype_extension))
                    '''
            return triggered,signals_out
                
def plotFromReader(reader,eventid,trigger_threshold_units = 'fpga', trigger_threshold = 7000, do_beamforming = True,beam_dict = {}, plot_signals = True, plot_geometry = False):
    '''
    Unlike redoEventFromInfo, this does not redo the calculation of the signals, 
    but rather assumes reader['signals'] exists with saved waveforms.  As such this
    only works for events that triggered, and only has access to the digitized 
    waveforms.  Plots will differ slightly from those in redoEventFromInfo because
    of this.
    '''
    if numpy.logical_and(trigger_threshold_units == 'fpga', numpy.logical_or(beam_dict == {}, do_beamforming == False)): 
        print('fpga triggering requested but either beam_dict is not given, or do_beamforming is set to False')
        if beam_dict != {}:
            print('Setting do_beamorming = True')
            do_beamforming = True
        else:
            print('Breaking')
            return
    event_label = 'event%i'%eventid
    config = yaml.load(open(reader.attrs['config']))
    info = reader['info'][...]
    solutions = numpy.unique(info['solution'])
    info = info[info['eventid'] == eventid]
    inelasticity = reader['inelasticity'][eventid]
    x_0 = reader['x_0'][eventid]
    y_0 = reader['y_0'][eventid]
    z_0 = reader['z_0'][eventid]
    phi_0 = reader['phi_0'][eventid]
    p_interact = reader['p_interact'][eventid]
    p_earth = reader['p_earth'][eventid]
    if numpy.isin('signals',numpy.array(list(reader.keys()))):
        for index_station in range(config['stations']['n']):
            station_label = 'station%i'%index_station
            station_cut = info['station'] == index_station
            if numpy.isin(event_label,numpy.array(list(reader['signals']))):
                signals = reader['signals'][event_label][station_label][...]
                Vd_out_sync = signals[0:(numpy.shape(signals)[0]-1),:]
                ud_out_sync = numpy.tile(signals[-1,:],(numpy.shape(Vd_out_sync)[0],1))
                
                if do_beamforming == True:
                    #Here is where I perform the beamforming algorithms. 
                    formed_beam_powers, beam_powersums = gnosim.sim.fpga.fpgaBeamForming(ud_out_sync, Vd_out_sync, beam_dict , config, plot1 = plot_each_beam, plot2 = False, save_figs = False)
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
                    top_val_theta_ant = numpy.array([beam_dict['theta_ant'][beam_label] for beam_label in top_val_beams])
                    #Currently don't know what to do with these values.  They will be written out as I progress but
                    #right now I am just testing that they can be calculate without breaking the simulation.
                    #Right now I am only storing the 3 highest values.  It is likely that I want to store every beam
                    #that satisfies the trigger condiditon?
                
                if trigger_threshold_units == 'adu':
                    if numpy.size(V_out) > 0:
                        if numpy.any(Vd_out_sync > trigger_threshold):
                            triggered = True
                elif trigger_threshold_units == 'fpga':
                    if top_vals[0] > trigger_threshold:
                        triggered = True
                else:
                    print('Must use either adu or fpga trigger')
                    return
                
                #Plotting
                if triggered == True:
                    print('Triggered on event %i'%eventid)
                    if plot_geometry == True:
                        origin = []
                        for index_antenna in info[info['has_solution'] == 1]['antenna']:
                            station_loc = numpy.array(config['stations']['positions'][index_station])
                            antenna_loc = numpy.add(numpy.array(config['antennas']['positions'][index_antenna]),station_loc)
                            
                            origin.append(list(antenna_loc))
                        
                        neutrino_loc = [x_0, y_0, z_0]
                        if len(info[info['has_solution'] == 1]) > 0:
                            fig = gnosim.trace.refraction_library_beta.plotGeometry(origin,neutrino_loc,phi_0,info[numpy.logical_and(info['has_solution'] == 1,info['station'] == index_station)])
                    
                    if plot_signals == True:
                        #might need to account for when signals are not present in certain detectors
                        #print('Attempting to plot', eventid)
                        temporary_info = numpy.zeros(config['antennas']['n'],info.dtype)      
                        for index_antenna in range(0, config['antennas']['n']):
                            antenna_label = config['antennas']['types'][index_antenna]
                            antenna_cut = numpy.logical_and(info['antenna'] == index_antenna, station_cut)
                            temporary_info[index_antenna] = info[antenna_cut][numpy.argmax(info[antenna_cut]['electric_field'])]
                            '''
                            for solution in numpy.unique(info['solution']):
                                solution_cut = numpy.logical_and(info['solution'] == solution, antenna_cut)
                                sub_info = info[solution_cut]
                            '''
                        fig = pylab.figure(figsize=(16.,11.2)) #my screensize
                        
                        n_rows = config['antennas']['n']
                        ntables = 4 #With below lines is 5 for beamforming == True
                        height_ratios = [2,2,n_rows+1,n_rows+1]
                        if do_beamforming == True:
                            ntables += 1
                            height_ratios.append(0.9*sum(height_ratios))
                            
                        gs_left = gridspec.GridSpec(n_rows, 2, width_ratios=[3, 2]) #should only call left plots.  pylab.subplot(gs_left[0]),pylab.subplot(gs_left[2]),...
                        gs_right = gridspec.GridSpec(ntables, 2, width_ratios=[3, 2], height_ratios=height_ratios) #should only call odd tables pylab.subplot(gs_right[1])
                        if do_beamforming == True:
                            gs_beam_forming = gridspec.GridSpec(ntables, 3, width_ratios=[3, 1,5], height_ratios=height_ratios)
                            
                        
                        first_in_loop = True
                        axis2 = []
                        max_ax2_range = numpy.array([1e20,-1e20])
                        for index_antenna in range(0, n_rows):
                            antenna_label = config['antennas']['types'][index_antenna]
                            if first_in_loop == True:
                                first_in_loop = False
                                pylab.title('Event %i'%eventid)
                                ax = pylab.subplot(gs_left[2*index_antenna])
                            
                            ax2 = pylab.subplot(gs_left[2*index_antenna],sharex = ax,sharey = ax) #this is not perfect and can be janky with zooming.   
                            axis2.append(ax2)   
                            c1 = 'b'
                            c2 = 'r'

                            ax2.plot(ud_out_sync[index_antenna,:],Vd_out_sync[index_antenna,:],label='s%ia%i'%(index_station,index_antenna),linewidth=0.4,c = c2)
                            
                            if ( n_rows // 2 == index_antenna):
                                ax2.set_ylabel('adu',fontsize=12, color=c2)
                                
                            ax2.legend(fontsize=8,framealpha=0.0,loc='upper right')
                            ax2.tick_params('y', colors=c2)
                            
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
                        
                        df = pandas.DataFrame({'E$_\\nu$ (GeV)':'%0.4g'%(energy_neutrino) , 'Inelasticity':'%0.4g'%inelasticity , 'p_interact':'%0.4g'%p_interact, 'p_earth':'%0.4g'%p_earth},index=[0])
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
                        antenna =           ['%i'%i for i in temporary_info['antenna'].astype(int)]
                        observation_angle = ['%0.5g'%i for i in temporary_info['observation_angle'].astype(float)]
                        theta_ant =         ['%0.5g'%i for i in temporary_info['theta_ant'].astype(float)]
                        distance =          ['%0.3g'%i for i in temporary_info['distance'].astype(float)]
                        beam_factor =       ['%0.3g'%i for i in temporary_info['beam_pattern_factor']]
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
                        antenna =           ['%i'%i for i in temporary_info['antenna'].astype(int)]
                        electric_field =    ['%0.3g'%i for i in temporary_info['electric_field'].astype(float)]
                        dom_freqs =         ['%0.3g'%i for i in (temporary_info['dominant_freq']/1e6).astype(float)]
                        SNRs =              ['%0.3g'%i for i in temporary_info['SNR'].astype(float)]
                        df = pandas.DataFrame({'antenna':antenna , '$V_\mathrm{max}$ (V)':electric_field , 'SNR':SNRs, '$f_\mathrm{max}$ (MHz)':dom_freqs})
                        table = pylab.table(cellText = df.values , colLabels = df.columns, loc = 'center')
                        table.auto_set_font_size(False)
                        table.set_fontsize(10)
                        
                        #TABLE 5: THE TABLE THAT'S ACTUALLY A PLOT AND ONLY SOMETIMES SHOWS UP DEPENDING ON SETTINGS :D
                        
                        if do_beamforming == True:
                            
                            colormap = pylab.cm.gist_ncar #nipy_spectral, Set1,Paired   
                            beam_colors = [colormap(i) for i in numpy.linspace(0, 1,len(beam_dict['beams'].keys())+1)] #I put the +1 backs it was making the last beam white, hopefully if I put this then the last is still white but is never called
            
                            
                            gs_beam_forming = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[9], wspace=0.1, hspace=0.1, width_ratios=[0.1,6,3])
                            #table_fig = pylab.subplot(gs_beam_forming[13])
                            table_fig = pylab.subplot(gs_beam_forming[1])
                            #table_fig = pylab.subplot(gs_right[9])
                            table_ax = pylab.gca()
                            table_fig.patch.set_visible(True)
                            
                            for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
                                table_ax.plot(beam_powersums[beam_label],label = '%s, $\\theta_{ant} = $ %0.2f'%(beam_label,beam_dict['theta_ant'][beam_label]),color = beam_colors[beam_index])
                                #print(beam_powersums[beam_label])
                            #for line_index,line in enumerate(table_ax.lines):
                            #    line.set_color(beam_colors[line_index])
                            #xlim = table_ax.get_xlim()
                            #table_ax.set_xlim( ( xlim[0] , xlim[1]*1.5 ) )
                            
                            #box = table_ax.get_position()
                            #table_ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                            pylab.yticks(rotation=45)
                            table_ax.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
                            #pylab.legend(loc='upper right', bbox_to_anchor=(1.05, 0.5))
                            #table_ax.axis('tight')
                            
                        pylab.subplots_adjust(left = 0.05, bottom = 0.05, right = 0.99, top = 0.97, wspace = 0.15, hspace = 0.28)
                        #pylab.show(block=True)
                        '''
                        try:
                            pylab.savefig('%s%s-event%i.%s'%(image_path,outfile.split('/')[-1].replace('.h5',''),eventid,plot_filetype_extension),bbox_inches='tight')
                            pylab.close(fig)
                            #print('Saved image %s%s-event%i.%s'%(image_path,outfile,eventid,plot_filetype_extension))
                        except:
                            print('Failed to save image %s%s-event%i.%s'%(image_path,outfile,eventid,plot_filetype_extension))
                        '''
                
                
            
            
            else:
                print('Signals for %s station %s not found in reader'%(event_label,station_label))        
    else:
        print('Signals for %s not found in reader'%(event_label))
            
    
    
    
    
    
    
    
    




  
if __name__ == "__main__":
    pylab.close('all')
    
    #reader = h5py.File('./Output/results_2019_Jan_config_dipole_octo_-200_polar_120_rays_3.00e+09_GeV_100_events_1_seed_10.h5' , 'r')
    
    #reader = h5py.File('./Output/results_2019_Jan_config_dipole_octo_-200_polar_120_rays_3.00e+09_GeV_1000_events_1_seed_2.h5' , 'r')
    #reader = h5py.File('' , 'r')
    reader = h5py.File('/home/dsouthall/scratch-midway2/results_2019_Feb_config_dipole_octo_-200_polar_120_rays_3.00e+09_GeV_50000_events_1_seed_3.h5' , 'r')
    reader2 = h5py.File('./results_2019_Feb_config_dipole_octo_-200_polar_120_rays_3.00e+09_GeV_1002_events_1_seed_4.h5' , 'r')
    
    #reader_kaeli = h5py.File('./results_2019_Jan_real_config_3.00e+09_GeV_1_events_1_seed_1.h5' , 'r')
    
    config = yaml.load(open(reader.attrs['config']))
    info = reader['info'][...]
    info2 = reader2['info'][...]
    #options
    energy_neutrino = 3.e9 # GeV
    choose_n = 10 #number of events to run code on (randomly selected from events with solutions
    sampling_bits = 7
    scale_noise_to = 3
    digital_sampling_freq = 1.5 #GHz
    digital_sampling_period = 1/digital_sampling_freq
    n_beams = 15
    n_baselines = 2
    power_calculation_sum_length = 16 #How long each power sum window is
    power_calculation_interval = 8 #How frequent each power sum window begins
    trigger_threshold_units = 'fpga'
    trigger_threshold = 0#7000
    
    
    use_fromReader = False
    use_redo = True
    
    multi_plot_signals = True
    plot_geometry = False
    
    #The three single_ plots correspond to plots in gnosim.interaction.askaryan.quickSignalSingle
    single_plot_signals   = False
    single_plot_spectrum  = False
    single_plot_angles  = False
    single_plot_potential = False
    
    #add_signals_plot = True 
    remove_noise_overlap = True
    signal_response_version = 'v7'
    if int(signal_response_version.split('v')[-1]) >= 6:
        up_sample_factor = 40
    else:
        up_sample_factor = 20
    
    include_noise = True
    summed_signals = True
    do_beamforming = True
    plot_first = True
    plot_each_beam = False
    #preparations
    #'''
    #'''
    #Signal Calculations
    
    z_0 = numpy.array(list(reader['z_0']))
    n_array = gnosim.earth.antarctic.indexOfRefraction(z_0, ice_model=config['detector_volume']['ice_model']) 
    mean_index = numpy.mean(n_array)
    
    eventids = numpy.unique(info[info['triggered']==1]['eventid'])
    #eventids = numpy.unique(info[info['pre_triggered'] == 1]['eventid'])
    #eventids = info2[numpy.where(info2['triggered'])[0][numpy.isin(numpy.where(info2['triggered'])[0],numpy.where(info['triggered'])[0],invert = True)]]['eventid']
    beam_dict = gnosim.sim.fpga.getBeams(config, n_beams, n_baselines , mean_index ,digital_sampling_period ,power_calculation_sum_length = power_calculation_sum_length, power_calculation_interval = power_calculation_interval, verbose = False)
    
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
    
    random_time_offsets = numpy.random.uniform(-1, 1, size=len(n_array))
    ###########################
    input_u, h_fft, sys_fft, freqs = gnosim.interaction.askaryan.calculateTimes(up_sample_factor=up_sample_factor,mode = signal_response_version)
    print('LENGTH OF FREQS:',len(freqs))
    noise_signal  = numpy.array([])
    
    for i in range(100):
        V_noiseless, u , dominant_freq, V_noise, SNR = gnosim.interaction.askaryan.quickSignalSingle( 0,1,energy_neutrino,1.8,\
                      0,0,0,input_u,h_fft,sys_fft,freqs,\
                      plot_signals=False,plot_spectrum=False,plot_potential = False,\
                      include_noise = True, resistance = 50, temperature = 320)
        noise_signal = numpy.append(noise_signal,V_noise)
    noise_rms = numpy.std(noise_signal)
    
    print('External noise_rms: %f',noise_rms)
    
    
    
    if use_redo == True:
        create_dataset = False
        if create_dataset == True:
            outfile = 'test.h5'
            file = h5py.File(outfile, 'w')
        for eventid in do_events:
            event_label = 'event%i'%eventid
            print('On %s'%event_label)
            print(info[numpy.logical_and(info['eventid'] == eventid,info['has_solution']==1)])
            triggered,out_data = redoEventFromInfo(reader,eventid,energy_neutrino,n_array[eventid],input_u,h_fft,sys_fft,freqs,sampling_bits,noise_rms,scale_noise_to,digital_sampling_period,random_time_offsets[eventid],0,beam_dict,trigger_threshold_units,trigger_threshold,include_noise = include_noise,summed_signals = summed_signals,do_beamforming = do_beamforming, plot_geometry = plot_geometry, plot_signals = multi_plot_signals,plot_first = plot_first,single_plot_signals   = single_plot_signals,single_plot_spectrum  = single_plot_spectrum, single_plot_angles = single_plot_angles,single_plot_potential = single_plot_potential, plot_each_beam = plot_each_beam)
            if triggered == True:
                if create_dataset == True:
                    file.create_group(event_label)
                    for index_station in range(config['stations']['n']):
                        station_label = 'station%i'%index_station
                        if create_dataset == True:
                            file[event_label].create_dataset(station_label, numpy.shape(out_data[station_label]), dtype='f', compression='gzip', compression_opts=9, shuffle=True)  
                            file[event_label][station_label][...] = out_data[station_label]
        if create_dataset == True:
            file.close()
            out_reader = h5py.File(outfile , 'r')
    if use_fromReader == True:    
        for eventid in do_events:
            event_label = 'event%i'%eventid
            print('On %s'%event_label)
            plotFromReader(reader,eventid,trigger_threshold_units = 'fpga', trigger_threshold = trigger_threshold, do_beamforming = True,beam_dict =  beam_dict, plot_signals = multi_plot_signals, plot_geometry = plot_geometry)
            print(info[numpy.logical_and(info['eventid'] == eventid,info['has_solution']==1)])
    
    '''
    info2 = info[numpy.isin(info['eventid'],eventids)]
    info2 = info2[info2['has_solution'] == 1]
    pylab.figure()
    pylab.hist(info2[info2['has_solution'] == 1]['observation_angle'])
    pylab.figure()
    info2 = reader2['info'][...]
    pylab.hist(info2[info2['has_solution'] == 1]['electric_field_digitized'])
    '''
    
