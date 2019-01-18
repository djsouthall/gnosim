"""
Practical and Accurate Calculations of Askaryan Radiation
Source: Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283
"""

import numpy
import pylab
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
import scipy.misc
import time
import math
import copy
sys.path.append("/home/dsouthall/Projects/GNOSim/")
from matplotlib import gridspec
import pandas

import gnosim.utils.constants
import gnosim.interaction.inelasticity
import gnosim.utils.quat
import gnosim.earth.earth
import gnosim.earth.antarctic
import gnosim.trace.refraction_library_beta
from gnosim.trace.refraction_library_beta import *
import gnosim.interaction.askaryan
import gnosim.sim.detector
import gnosim.interaction.askaryan_testing
pylab.ion()
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

def calculateDigitalTimes(u_min,u_max,sampling_period,random_time_offset = 0):
    '''
    This calculates the times to sample a signal given some u_min and u_max. 
    
    This should be the same for every antenna (i.e. it is on the array clock).
    
    So I shouldbe able to create it in advance for a wide window, and then sample
    signals later for the subset of time that they occupy. 
    
    This probably doesn't need to be a function.
    '''
    sample_times = numpy.arange(u_min,u_max,sampling_period) + random_time_offset
    #sample_times = sample_times[numpy.logical_and(sample_times <= u[-1],sample_times >= u[1])] 
    return sample_times
    
def digitizeSignal(u,V,sample_times,bytes,scale_noise_from,scale_noise_to, dc_offset = 0, plot = False):
    '''
    This function is meant to act as the ADC for the simulation.  It will sample
    the signal using the provided sampling rate.  The value of
    bytes sets the number of voltage bins, those will be distributed from 
    -2**(bytes-1)+1 to 2**(bytes-1).  The input V will be sampled and scaled linearly
    using a linear scaling with V_new = (scale_noise_to/scale_noise_from) * V_sampled
    And then V_sampled is scaled to be one of the byte values using a floor.  Everything
    outside of the max will be rounded.  
    
    Sampleing rate should be in GHz
    '''
    V = V + dc_offset
    #sampling_period = 1.0 / sampling_rate #ns
    #sample_times = numpy.arange(u[1],u[-1],sampling_period) + random_time_offset
    #sample_times = sample_times[numpy.logical_and(sample_times <= u[-1],sample_times >= u[1])] #otherwise interpolation error for out of bounds. 
    V_sampled = scipy.interpolate.interp1d(u,V,bounds_error = False,fill_value = 0.0)(sample_times) #sampletimes will now extend beyond the interpolated range, but here it returns 0 voltage
    
    #byte_vals = numpy.linspace(-2**(bytes-1)+1,2**(bytes-1),2**bytes,dtype=int)
    byte_vals = numpy.array([-2**(bytes-1)+1,2**(bytes-1)],dtype=int) #only really need endpoints
    slope = scale_noise_to/scale_noise_from
    f = scipy.interpolate.interp1d(byte_vals/slope,byte_vals,bounds_error = False,fill_value = (byte_vals[0],byte_vals[-1]) )
    V_bit = numpy.floor(f(V_sampled)) #not sure if round or floor should be used to best approximate the actual process.
    
    if plot == True:
        pylab.figure()
        ax = pylab.subplot(2,1,1)
        pylab.ylabel('V (V)')
        pylab.xlabel('t (ns)')
        pylab.scatter(u,V,label='Signal')
        pylab.stem(sample_times,V_sampled,bottom = dc_offset, linefmt='r-', markerfmt='rs', basefmt='r-',label='Interp Sampled at %0.2f GSPS'%sampling_rate)
        
        ax = pylab.subplot(2,1,2,sharex=ax)
        pylab.ylabel('Voltage (Scaled so VRMS = 3)')
        pylab.xlabel('t (ns)')
        pylab.plot(u,V*slope)
        pylab.stem(sample_times,V_bit,bottom = dc_offset*slope, linefmt='r-', markerfmt='rs', basefmt='r-',label='Interp Sampled at %0.2f GSPS'%sampling_rate)
    return V_bit, sample_times

def getBeams( config, n_beams, n_baselines , n , dt ,power_calculation_sum_length = 16, power_calculation_interval = 8, verbose = False):
    '''
    The goal of this function is to determine the beam and subbeam time delays 
    semiautomatically given a config file.
    
    Currently the minimum time shift is assigned to the smallest baseline.  Thus
    every other timeshift resulting from larger baselines must be a multiple of the
    minimum baseline. i.e. all subbeam baselines must be in integer multiples of 
    the minimum baseline.  
    Currently requires all other baselines to be an integer multiple of the minimum baseline
    
    n should be average index of refraction of array (used for angle estimates)
    dt should be in nanoseconds (used for angle estimates)
    '''
    n_antennas = config['antennas']['n']
    min_antennas_per_subbeam =  2#numpy.round(n_antennas/3) #if the number of antennas satisfying a baseline is less than this that beam won't be counted
    relative_antenna_depths = numpy.array(config['antennas']['positions'])[:,2]
    #relative_antenna_depths = relative_antenna_depths - relative_antenna_depths[len(relative_antenna_depths)//2] #shifts the timings to be relative to the centerish antenna
    
    baselines = []
    for ii in relative_antenna_depths:
        for jj in relative_antenna_depths:
            baselines.append(numpy.abs(ii - jj))
    baselines = numpy.sort(numpy.unique(baselines))
    baselines = baselines[baselines!= 0][range(n_baselines)]
    
    antenna_list = numpy.arange(n_antennas)
    beam_dict = {'attrs' :  {'power_calculation_sum_length' : power_calculation_sum_length,
                             'power_calculation_interval'   : power_calculation_interval},
                 'beams':{}}
    #Both power_calculation_sum_length and power_calculation_interval can probably be made into input parameters if needed
    subbeam_list = [] 
    min_baseline = min(baselines)
    shifted_beam_index = {}             
    for baseline_index, baseline in enumerate(baselines):
        use_in_subbeam = numpy.zeros_like(relative_antenna_depths)
        for antenna_index, initial_depth in enumerate(relative_antenna_depths):
            if numpy.all(use_in_subbeam) == True:
                break
            subbeam_antenna_cut = numpy.arange(antenna_index,len(relative_antenna_depths))
            subbeam_antenna_list = antenna_list[subbeam_antenna_cut]
            subbeam_depth_list = relative_antenna_depths[subbeam_antenna_cut]
            subbeam_cut = (((subbeam_depth_list - initial_depth) % baseline) == 0 )
            use_in_subbeam[subbeam_antenna_cut] = numpy.logical_or(use_in_subbeam[subbeam_antenna_cut],subbeam_cut)
            if sum(subbeam_cut) >= min_antennas_per_subbeam:
                subbeam_list.append(numpy.array(subbeam_antenna_list[subbeam_cut]))
    
    if verbose == True:
        print(subbeam_list) 
    
    all_time_delays = numpy.array([])
    beam_dict['theta_ant'] = {}
    for beam_index in range(n_beams):
        beam_label = 'beam%i'%beam_index
        beam_dict['beams'][beam_label] = {}
        theta_ant_beam = 0
        total_ant = 0
        for subbeam_index, subbeam in enumerate(subbeam_list):
            subbeam_label = 'subbeam%i'%subbeam_index
            baseline = min(numpy.unique(numpy.abs(numpy.diff(relative_antenna_depths[subbeam]))))
            ms = numpy.arange(-n_beams/(2/baseline),n_beams/(2/baseline),baseline,dtype=int) #it is sort of sloppy to calculate this each time (only needs ot be done once per baseline) but this function is only done once so whatever.
            if baseline % min_baseline != 0:
                continue
                
            #print('gnosim.utils.constants.speed_light * ms[beam_index] * dt  / ( n * baseline)',gnosim.utils.constants.speed_light * ms[beam_index] * dt  / ( n * baseline))
            #theta_elevation = 0
            theta_elevation = numpy.rad2deg(numpy.arcsin(gnosim.utils.constants.speed_light * ms[beam_index] * dt  / ( n * baseline) ))
            theta_ant = 90-theta_elevation    
            beam_dict['beams'][beam_label][subbeam_label] = {'baseline'       : baseline,
                                                    'antennas'       : subbeam,
                                                    'depths'         : relative_antenna_depths[subbeam],
                                                    'time_delays'    : numpy.array( ms[beam_index]  * relative_antenna_depths[subbeam],dtype=int),
                                                    'theta_elevation': theta_elevation,
                                                    'theta_ant': theta_ant,
                                                    'adjusted_m' : ms[beam_index]
                                                    }
            theta_ant_beam += len(subbeam)*theta_ant
            total_ant += len(subbeam)
            all_time_delays = numpy.append(all_time_delays,beam_dict['beams'][beam_label][subbeam_label]['time_delays'])
        beam_dict['theta_ant'][beam_label] = theta_ant_beam/total_ant
    beam_dict['attrs']['unique_delays'] = numpy.array(numpy.sort(numpy.unique(all_time_delays)),dtype=int)
    if verbose == True:
        for k in beam_dict['beams'].keys():
            print('\n',k)
            if 'beam' in k:
                for key in beam_dict['beams'][k].keys():
                    print(key)
                    print(beam_dict['beams'][k][key])

    return beam_dict

def syncSignals( u_in, V_in, min_time, max_time, u_step ):
    '''
    Given a dictionary with an array (or empty array) for each antenna, This 
    will extend the temporal range of each signal to be the same, and 
    produce an ndarray with each row representing a signal and
    appropriately place the V_in along this extended timeline.  This should be 
    used before beam summing. 
    '''
    u_out = numpy.tile(numpy.arange(min_time,max_time+u_step,u_step), (len(list(V_in.keys())),1))
    V_out = numpy.zeros_like(u_out)
    for antenna_index,antenna_label in enumerate(list(V_in.keys())):
        V = V_in[antenna_label]
        u = u_in[antenna_label]
        if numpy.size(u) != 0:
            left_index = numpy.argmin(abs(u_out[antenna_index] - u[0]))
            right_index = left_index + len(V)
            V_out[antenna_index,left_index:right_index] +=  V
    return V_out, u_out

def fpgaBeamForming(u_in, V_in, beam_dict , config, plot1 = False, plot2 = False, save_figs = False):
    '''
    This is one function which uses the code from what were the sumBeams and 
    doFPGAPowerCalcAllBeams functions, but puts them in one to avoid the extra 
    time from calling multiple functions. 
    
    Expects u_in and V_in to be the same dimensions, with the same number
    of rows as there are antennas. The shallowest detector should be the
    first row of the input matrix. 
    
    beam_dict should come from the getBeams function, and is not included here
    because it only needs to be called once, whereas this should be called for
    each signal.
    '''
    #####
    #Doing the beam summing portion below:
    #####
    n_antennas = config['antennas']['n']
    signal_length = numpy.shape(V_in)[1]
    #below is the fastest way I could figure out of doing these sums.  Originally
    #a roll function was used, however this took too long.  I tried to reduce
    #the number of times things are redundently done by calculating them in advance
    #as well as avoid things like appending and such.  It isn't as readable as I would
    #like it to be but it is faster and reproduces the same results as the old
    #algorithm.
    zeros_delay = numpy.arange(signal_length) #any zero delays are the same, no need to do this arange multiple times
    delay_indices = numpy.zeros((len(beam_dict['attrs']['unique_delays']),signal_length)) #will hold the indices for each delay that replicate the roll required.  Ordered in the same way as beam_dict['attrs']['unique_delays'], so call the correct row by from indexing that
    for index, shift in enumerate(beam_dict['attrs']['unique_delays']):
        if shift < 0:
            delay_indices[index,0:signal_length + shift] = numpy.arange(-shift,signal_length)
            delay_indices[index,signal_length + shift:signal_length] = numpy.arange(0,-shift)
        elif shift > 0:
            delay_indices[index,0:shift] = numpy.arange(signal_length - shift, signal_length)
            delay_indices[index,shift:signal_length] = numpy.arange(0,signal_length - shift)
        else:
            delay_indices[index,:] = zeros_delay
    delay_indices = numpy.array(delay_indices,dtype=int)
    
    formed_beam_powers = {}
    for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
        formed_beam_powers[beam_label] = {}
        if plot1 == True:
            fig = pylab.figure(figsize=(16.,11.2))
        first_in_loop  = True
        for subbeam_index, subbeam_label in enumerate(beam_dict['beams'][beam_label].keys()):
            first_in_sub_loop = True
            for i in range(len(beam_dict['beams'][beam_label][subbeam_label]['antennas'])):
                if first_in_sub_loop:
                    first_in_sub_loop = False
                    V_subbeam = V_in[beam_dict['beams'][beam_label][subbeam_label]['antennas'][i],:][delay_indices[numpy.where(beam_dict['attrs']['unique_delays']==beam_dict['beams'][beam_label][subbeam_label]['time_delays'][i])[0][0],:]]
                else:
                    V_subbeam = numpy.add(V_subbeam, V_in[beam_dict['beams'][beam_label][subbeam_label]['antennas'][i],:][delay_indices[numpy.where(beam_dict['attrs']['unique_delays']==beam_dict['beams'][beam_label][subbeam_label]['time_delays'][i])[0][0],:]])
            #beam_dict['beams'][beam_label][subbeam_label]['beam_power_signal'] = V_subbeam**2
            formed_beam_powers[beam_label][subbeam_label] = V_subbeam**2
            if plot1 == True:
                if first_in_loop == True:
                    first_in_loop = False
                    ax = pylab.subplot(len(beam_dict['beams'][beam_label].keys()),1,subbeam_index+1)
                    pylab.title('%s'%(beam_label))
                    pylab.plot(V_subbeam,label = '$\\theta_\mathrm{ant} = $%0.2f'%(beam_dict['beams'][beam_label][subbeam_label]['theta_ant']))
                    pylab.xlabel('time steps')
                    pylab.ylabel('%s (abu)'%subbeam_label)
                    pylab.legend(loc='upper right')
                else:
                    pylab.subplot(len(beam_dict['beams'][beam_label].keys()),1,subbeam_index+1,sharex = ax)
                    pylab.plot(V_subbeam,label = '$\\theta_\mathrm{ant} = $%0.2f'%(beam_dict['beams'][beam_label][subbeam_label]['theta_ant']))
                    pylab.xlabel('time steps')
                    pylab.ylabel('%s (abu)'%subbeam_label)
                    pylab.legend(loc='upper right')
        if save_figs == True:
            try:
                pylab.savefig('%s%s.%s'%(image_path,beam_label,plot_filetype_extension),bbox_inches='tight')
                pylab.close(fig)
            except:
                print('Failed to save image %s%s.%s'%(image_path,beam_label,plot_filetype_extension))
    #####
    #Doing the power window calculation portion
    #####
    beam_powersums = {}
    if plot2 == True:
        fig = pylab.figure(figsize=(16.,11.2))
        
    test_interval = 8
    test_length = 16
    left = numpy.arange(0,signal_length - beam_dict['attrs']['power_calculation_sum_length'] + 1,beam_dict['attrs']['power_calculation_interval']) #probably need to cap 
    span = numpy.arange( beam_dict['attrs']['power_calculation_sum_length'] )
    spans = numpy.tile(span,(len(left),1))
    lefts = numpy.tile(left,(len(span),1)).T
    indices = numpy.add(spans,lefts) #This ends up doing one more frame on the end than Eric's original code, but I can't tell why that frame isn't included so I am keeping it. 
    
    for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
        first_subbeam = True
        
        for subbeam_label in beam_dict['beams'][beam_label].keys():
            subbeam_powersum = numpy.sum(formed_beam_powers[beam_label][subbeam_label][indices],axis=1)
            if first_subbeam == True:
                first_subbeam = False
                beam_powersums[beam_label] = subbeam_powersum
            else:
                beam_powersums[beam_label] = numpy.add(beam_powersums[beam_label],subbeam_powersum)
        if plot2 == True:
            #getting weighted angle in crude way, angle is not a real angle anyways
            total_n = 0
            weighted_theta_ant = 0
            for subbeam_label in beam_dict['beams'][beam_label].keys():
                weighted_theta_ant += beam_dict['beams'][beam_label][subbeam_label]['theta_ant'] * len(beam_dict['beams'][beam_label][subbeam_label]['antennas'])
                total_n += len(beam_dict['beams'][beam_label][subbeam_label]['antennas'])
            weighted_theta_ant = weighted_theta_ant / total_n
            pylab.plot(beam_powersums[beam_label],label = '%s, $\\theta_{ant} = $ %0.2f'%(beam_label,weighted_theta_ant))
    if plot2 == True:
        ax = pylab.gca()
        colormap = pylab.cm.gist_ncar #nipy_spectral, Set1,Paired   
        colors = [colormap(i) for i in numpy.linspace(0, 1,len(ax.lines))]
        for line_index,line in enumerate(ax.lines):
            line.set_color(colors[line_index])
        pylab.legend()
        
        if save_figs == True:
            try:
                pylab.savefig('%spowersum_allbeams.%s'%(image_path,plot_filetype_extension),bbox_inches='tight')
                pylab.close(fig)
            except:
                print('Failed to save image %spowersum_allbeams.%s'%(image_path,plot_filetype_extension))
            
    return formed_beam_powers, beam_powersums

############################################################

#
if __name__ == "__main__":
    pylab.close('all')
    energy_neutrino = 3.e9 # GeV
    n = 1.78
    c = gnosim.utils.constants.speed_light #m/ns
    
    R = 1000. #m
    cherenkov_angle = numpy.arccos(1./n)
    cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./n))
    h_fft,sys_fft,freqs = gnosim.interaction.askaryan.loadSignalResponse()
    input_u, h_fft, sys_fft, freqs = gnosim.interaction.askaryan.calculateTimes(up_sample_factor=20)
    inelasticity = 0.2
    noise_rms = numpy.std(gnosim.interaction.askaryan.quickSignalSingle(0,R,inelasticity*energy_neutrino,n,R,0,0,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)[3])
    V_noiseless, u, dominant_freq, V_noise,  SNR = gnosim.interaction.askaryan.quickSignalSingle(numpy.deg2rad(50),R,inelasticity*energy_neutrino,n,2500,0.7,0.7,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)
    sampling_rate = 1.5 #GHz
    sampling_period = 1/sampling_rate
    bytes = 7
    scale_noise_from = noise_rms
    scale_noise_to = 3
    
    random_time_offset = numpy.random.uniform(-5.0,5.0) #ns
    dc_offset = 0.0 #V
    sample_times=calculateDigitalTimes(u[0],u[-1],sampling_period,  random_time_offset = random_time_offset)
    V_bit, sampled_times = digitizeSignal(u,V_noise,sample_times,bytes,scale_noise_from,scale_noise_to, dc_offset = dc_offset, plot = False)
    dt = sampled_times[1] - sampled_times[0]
    #################################################################
    config_file = '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_dipole_octo_-200_polar_120_rays.py'
    config = yaml.load(open(config_file))
    config_file2 = '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/real_config.py'
    config2 = yaml.load(open(config_file2))
    
    n_beams = 15
    n_baselines = 2
    
    beam_dict = getBeams( config, n_beams, n_baselines , n , dt, verbose = False )
    #beam_dict = getBeams( config2, n_beams, n_baselines , n , dt, verbose = False )
    
    from gnosim.trace.refraction_library_beta import *
    reader = h5py.File('./results_2019_Jan_config_dipole_octo_-200_polar_120_rays_3.00e+09_GeV_100_events_1_seed_4_new_new_new_new_new_new.h5' , 'r')
    #reader = h5py.File('./Output/results_2019_Jan_config_dipole_octo_-200_polar_120_rays_3.10e+09_GeV_100_events_1_seed_3.h5' , 'r')
    
    
    info = reader['info'][...]
    #info_cut = info[numpy.logical_and(info['SNR'] > 1 , info['SNR'] < 10) ]
    info_cut = info[numpy.logical_and(info['SNR'] > 1 , info['SNR'] < 100) ]
    #events 15, 92
    '''
    eventids = numpy.unique(info_cut[info_cut['has_solution']==1]['eventid'])
    choose_n = 1
    try:
        do_events = numpy.random.choice(eventids,choose_n,replace=False)
    except:
        do_events = numpy.unique(numpy.random.choice(eventids,choose_n,replace=True))
    plot_beams = False
    plot_sums = False
    for eventid in do_events:
        print('On event %i'%eventid)
        V, u, Vd, ud = gnosim.interaction.askaryan_testing.signalsFromInfo(eventid,reader,input_u,n,h_fft,sys_fft,freqs,include_noise = True,resistance = 50, temperature = 320)
        sub_info = info[info['eventid'] == eventid]
        Vd2, ud2 = syncSignals(ud,Vd)
        print('Mean angle for event %i is %0.2f'%(eventid, numpy.mean(info[info['eventid'] == eventid]['theta_ant'])))
        print(beam_dict['beams']['beam0']['subbeam0'])
        beam_powersums = testingBeamCalculations(ud, Vd, beam_dict , config, plot1 = plot_beams, plot2 = plot_sums)
        print(beam_dict['beams']['beam0']['subbeam0'])
    
    beam_label_list = numpy.array(list(beam_powersums.keys()))
    stacked_beams = numpy.zeros((len(beam_label_list),len(beam_powersums[beam_label_list[0]])))
    for beam_index, beam_label in enumerate(beam_label_list):
        stacked_beams[beam_index,:] = beam_powersums[beam_label]
    max_vals = numpy.max(stacked_beams,axis=1)
    
    keep_top = 3
    top_val_indices = numpy.argsort(max_vals)[-numpy.arange(1,keep_top+1)]
    top_vals = max_vals[top_val_indices]
    top_val_beams = beam_label_list[top_val_indices]
    top_val_theta_ant = numpy.array([beam_dict['theta_ant'][beam_label] for beam_label in top_val_beams]) 
    print(max_vals)
    print(top_vals)
    print(top_val_beams)
    print(top_val_theta_ant)
    '''
    V_in = {}
    u_in = {}
    min_u = 1e20
    max_u = -1e20
    solutions = ['direct','cross','reflect']
    
    do_solutions = False
    for station_index in numpy.arange(1):
        station_label = 'station%i'%station_index
        V_in[station_label] = {}
        u_in[station_label] = {}
        for antenna_index in numpy.arange(8):
            antenna_label = 'antenna%i'%antenna_index
            V_in[station_label][antenna_label] = {}
            u_in[station_label][antenna_label] = {}
            if do_solutions == True:
                for solution in solutions:
                    V_in[station_label][antenna_label][solution] = numpy.random.rand(100)
                    u_in[station_label][antenna_label][solution] = numpy.arange(100)+antenna_index+numpy.random.randint(low=-10,high=10)
                    min_u = numpy.min([min_u,u_in[station_label][antenna_label][solution][0]])
                    max_u = numpy.max([max_u,u_in[station_label][antenna_label][solution][-1]])
            else:
                V_in[station_label][antenna_label] = numpy.random.rand(100)
                u_in[station_label][antenna_label] = numpy.arange(100)+antenna_index+numpy.random.randint(low=-10,high=10)
                min_u = numpy.min([min_u,u_in[station_label][antenna_label][0]])
                max_u = numpy.max([max_u,u_in[station_label][antenna_label][-1]])
    dt = 1
    V_out,u_out = syncSignals( u_in['station0'], V_in['station0'], min_u, max_u, dt )
    
    
    #use the above to test a trigger algorithm
    
############################################################
