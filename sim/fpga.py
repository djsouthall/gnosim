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


def digitizeSignal(u,V,sampling_rate,bytes,scale_noise_from,scale_noise_to, dc_offset = 0, random_time_offset = 0, plot = False):
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
    sampling_period = 1.0 / sampling_rate #ns
    sample_times = numpy.arange(u[1],u[-1],sampling_period) + random_time_offset
    sample_times = sample_times[numpy.logical_and(sample_times <= u[-1],sample_times >= u[1])] #otherwise interpolation error for out of bounds. 
    V_sampled = scipy.interpolate.interp1d(u,V)(sample_times)
    
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
    
    for beam_index in range(n_beams):
        beam_label = 'beam%i'%beam_index
        beam_dict['beams'][beam_label] = {}
        for subbeam_index, subbeam in enumerate(subbeam_list):
            subbeam_label = 'subbeam%i'%subbeam_index
            baseline = min(numpy.unique(numpy.abs(numpy.diff(relative_antenna_depths[subbeam]))))
            ms = numpy.arange(-n_beams/(2/baseline),n_beams/(2/baseline),baseline,dtype=int) #it is sort of sloppy to calculate this each time (only needs ot be done once per baseline) but this function is only done once so whatever.
            if baseline % min_baseline != 0:
                continue
                
            theta_elevation = numpy.rad2deg(numpy.arcsin(c * ms[beam_index] * dt  / ( n * baseline) ))
            theta_ant = 90-theta_elevation    
            beam_dict['beams'][beam_label][subbeam_label] = {'baseline'       : baseline,
                                                    'antennas'       : subbeam,
                                                    'depths'         : relative_antenna_depths[subbeam],
                                                    'time_delays'    : numpy.array( ms[beam_index]  * relative_antenna_depths[subbeam],dtype=int),
                                                    'theta_elevation': theta_elevation,
                                                    'theta_ant': theta_ant,
                                                    'adjusted_m' : ms[beam_index]
                                                    }
    if verbose == True:
        for k in beam_dict['beams'].keys():
            print('\n',k)
            if 'beam' in k:
                for key in beam_dict['beams'][k].keys():
                    print(key)
                    print(beam_dict['beams'][k][key])

    return beam_dict
    
def syncSignals( u_in, V_in ):
    '''
    Given an array of u_in times and V_in signals with the same number
    of rows as there are antennas. This will extend the temporal range of
    each signal to be the same, and appropriately place the V_in along this
    extended timeline.  This should be used before beam summing. 
    '''
    if numpy.shape(u_in)[0] <= 1:
        return V_in.flatten(),u_in.flatten()
    else:
        #print(u_in)
        #print(numpy.shape(u_in))
        u_step = u_in[0,1]-u_in[0,0]
        u_out_min = min(u_in[:,0])
        u_out_max = max(u_in[:,-1])
        u_out = numpy.tile(numpy.arange(u_out_min,u_out_max+u_step,u_step), (numpy.shape(V_in)[0],1))
        V_out = numpy.zeros_like(u_out)
        for i in range(numpy.shape(V_in)[0]):
            V = V_in[i]
            u = u_in[i]
            if len(u) == 0:
                u = u_out
                V = numpy.zeros_like(u_out)   
            left_index = numpy.argmin(abs(u_out - u[0]))
            right_index = left_index + len(V)
            cut = numpy.arange(left_index,right_index)
            V_out[i][cut] += V
        return V_out, u_out
            
def sumBeams( u_in, V_in, beam_dict , config , plot = False,save_figs = False, image_path = './', plot_filetype_extension = 'png'):
    '''
    Expects u_in and V_in to be the same dimensions, with the same number
    of rows as there are antennas. The shallowest detector should be the
    first row of the input matrix. 
    '''
    n_antennas = config['antennas']['n']
    out_beams = beam_dict
    beam_dict['attrs']['signal_length'] = numpy.shape(V_in)[1]
    for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
        beam = beam_dict['beams'][beam_label]
        if plot == True:
            fig = pylab.figure(figsize=(16.,11.2))
        first_in_loop  = True
        for subbeam_index, subbeam_label in enumerate(beam.keys()):
            subbeam = beam[subbeam_label]
            V_subbeam = numpy.zeros_like(V_in[0,:]) #might need a check conditional in case only one row is entered. 
            for i in range(len(subbeam['antennas'])):
                V_subbeam = numpy.add(V_subbeam, numpy.roll(V_in[subbeam['antennas'][i],:],subbeam['time_delays'][i]))
            out_beams['beams'][beam_label][subbeam_label]['beam_power_signal'] = numpy.array(V_subbeam**2)
            if plot == True:
                if first_in_loop == True:
                    first_in_loop = False
                    ax = pylab.subplot(len(beam.keys()),1,subbeam_index+1)
                    pylab.title('%s'%(beam_label))
                    pylab.plot(V_subbeam,label = '$\\theta_\mathrm{ant} = $%0.2f'%(subbeam['theta_ant']))
                    pylab.xlabel('time steps')
                    pylab.ylabel('%s (abu)'%subbeam_label)
                    pylab.legend(loc='upper right')
                else:
                    pylab.subplot(len(beam.keys()),1,subbeam_index+1,sharex = ax)
                    pylab.plot(V_subbeam,label = '$\\theta_\mathrm{ant} = $%0.2f'%(subbeam['theta_ant']))
                    pylab.xlabel('time steps')
                    pylab.ylabel('%s (abu)'%subbeam_label)
                    pylab.legend(loc='upper right')
        if save_figs == True:
            try:
                pylab.savefig('%s%s.%s'%(image_path,beam_label,plot_filetype_extension),bbox_inches='tight')
                pylab.close(fig)
                #print('Saved image %s%s-event%i.%s'%(image_path,self.outfile,eventid,plot_filetype_extension))
            except:
                print('Failed to save image %s%s.%s'%(image_path,beam_label,plot_filetype_extension))
    return out_beams

def doFPGAPowerCalcAllBeams(summed_beam_dict,plot = False):
    '''
    This replicates the results of Eric's code but does so using more numpy operations
    to be much faster adn useable in the simulation.
    '''
    beam_powers = {}
    power_beam_dict = summed_beam_dict
    if plot == True:
        fig = pylab.figure(figsize=(16.,11.2))
        #ax = pylab.subplot(len( summed_beam_dict['beams'].keys() ), 1, 1)
        
    test_interval = 8
    test_length = 16
    left = numpy.arange(0,beam_dict['attrs']['signal_length'] - beam_dict['attrs']['power_calculation_sum_length'] + 1,beam_dict['attrs']['power_calculation_interval']) #probably need to cap 
    span = numpy.arange( beam_dict['attrs']['power_calculation_sum_length'] )
    spans = numpy.tile(span,(len(left),1))
    lefts = numpy.tile(left,(len(span),1)).T
    indices = numpy.add(spans,lefts) #This ends up doing one more frame on the end than Eric's original code, but I can't tell why that frame isn't included so I am keeping it. 
    
    for beam_index, beam_label in enumerate(summed_beam_dict['beams'].keys()):
        first_subbeam = True
        
        for subbeam_label in summed_beam_dict['beams'][beam_label].keys():
            power_beam_dict['beams'][beam_label][subbeam_label]['power_sum'] = numpy.sum(summed_beam_dict['beams'][beam_label][subbeam_label]['beam_power_signal'][indices],axis=1)
            #power_beam_dict['beams'][beam_label][subbeam_label]['power_sum'] = doFPGAPowerCalcSingleBeamOld(summed_beam_dict['beams'][beam_label][subbeam_label]['beam_power_signal'], beam_dict['attrs']['power_calculation_sum_length'], beam_dict['attrs']['power_calculation_interval'])
            if first_subbeam == True:
                first_subbeam = False
                beam_powers[beam_label] = power_beam_dict['beams'][beam_label][subbeam_label]['power_sum']
            else:
                beam_powers[beam_label] = numpy.add(beam_powers[beam_label],power_beam_dict['beams'][beam_label][subbeam_label]['power_sum'])
        if plot == True:
            #getting weighted angle in crude way, angle is not a real angle anyways
            total_n = 0
            weighted_theta_ant = 0
            for subbeam_label in summed_beam_dict['beams'][beam_label].keys():
                weighted_theta_ant += summed_beam_dict['beams'][beam_label][subbeam_label]['theta_ant'] * len(summed_beam_dict['beams'][beam_label][subbeam_label]['antennas'])
                total_n += len(summed_beam_dict['beams'][beam_label][subbeam_label]['antennas'])
            weighted_theta_ant = weighted_theta_ant / total_n
            pylab.plot(beam_powers[beam_label],label = '%s, $\\theta_{ant} = $ %0.2f'%(beam_label,weighted_theta_ant))
    if plot == True:
        ax = pylab.gca()
        colormap = pylab.cm.gist_ncar #nipy_spectral, Set1,Paired   
        colors = [colormap(i) for i in numpy.linspace(0, 1,len(ax.lines))]
        for line_index,line in enumerate(ax.lines):
            line.set_color(colors[line_index])
        pylab.legend()
    return power_beam_dict, beam_powers

def testingBeamCalculations(u_in, V_in, beam_dict , config, plot1 = False, plot2 = False):
    '''
    This function should do everything that would be calculated per event in the simulation
    such that I can profile the individual functios to ensure it all runs quickly.
    '''
    V, u = syncSignals(ud,Vd)
    summed_beam_dict = sumBeams( u, V, beam_dict , config, plot = plot1,save_figs=False)
    power_beam_dict, beam_powers = doFPGAPowerCalcAllBeams(summed_beam_dict,plot = plot2)
    
############################################################

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
    bytes = 7
    scale_noise_from = noise_rms
    scale_noise_to = 3
    
    random_time_offset = numpy.random.uniform(-5.0,5.0) #ns
    dc_offset = 0.0 #V
    V_bit, sampled_times = digitizeSignal(u,V_noise,sampling_rate,bytes,scale_noise_from,scale_noise_to, random_time_offset = random_time_offset, dc_offset = dc_offset, plot = False)
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
    reader = h5py.File('./Output/results_2018_Dec_config_dipole_octo_-200_polar_120_rays_3.00e+09_GeV_100_events_1_seed_6.h5' , 'r')
    info = reader['info'][...]
    #info_cut = info[numpy.logical_and(info['SNR'] > 1 , info['SNR'] < 10) ]
    info_cut = info[numpy.logical_and(info['SNR'] > 1 , info['SNR'] < 100) ]
    #events 15, 92
    eventids = numpy.unique(info_cut[info_cut['has_solution']==1]['eventid'])
    try:
        do_events = numpy.random.choice(eventids,5,replace=False)
    except:
        do_events = numpy.unique(numpy.random.choice(eventids,5,replace=True))
    plot_beams = False
    plot_sums = True
    for eventid in do_events:
        print('On event %i'%eventid)
        V, u, Vd, ud = gnosim.interaction.askaryan_testing.signalsFromInfo(eventid,reader,input_u,n,h_fft,sys_fft,freqs,include_noise = True,resistance = 50, temperature = 320)
        sub_info = info[info['eventid'] == eventid]
        Vd2, ud2 = syncSignals(ud,Vd)
        print('Mean angle for event %i is %0.2f'%(eventid, numpy.mean(info[info['eventid'] == eventid]['theta_ant'])))
        testingBeamCalculations(ud, Vd, beam_dict , config, plot1 = plot_beams, plot2 = plot_sums)
    
    
    
############################################################
