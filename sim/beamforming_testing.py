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
pylab.ion()
import gnosim.interaction.askaryan_testing
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
                
            theta_elevation = numpy.rad2deg(numpy.arcsin(gnosim.utils.constants.speed_light #m/ns * ms[beam_index] * dt  / ( n * baseline) ))
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

@profile
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

def sumBeamsFast( u_in, V_in, beam_dict , config , plot = False,save_figs = False, image_path = './', plot_filetype_extension = 'png'):
    '''
    Expects u_in and V_in to be the same dimensions, with the same number
    of rows as there are antennas. The shallowest detector should be the
    first row of the input matrix. 
    '''
    n_antennas = config['antennas']['n']
    out_beams = beam_dict
    beam_dict['attrs']['signal_length'] = numpy.shape(V_in)[1]
    delay_indices = numpy.zeros((len(beam_dict['attrs']['unique_delays']),beam_dict['attrs']['signal_length']))
    signal_indices = numpy.arange(beam_dict['attrs']['signal_length'])
    for index, shift in enumerate(beam_dict['attrs']['unique_delays']):
        delay_indices[index,:] = numpy.roll(signal_indices,shift)
    delay_indices = numpy.array(delay_indices,dtype=int)
    for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
        #beam = beam_dict['beams'][beam_label]
        if plot == True:
            fig = pylab.figure(figsize=(16.,11.2))
        first_in_loop  = True
        for subbeam_index, subbeam_label in enumerate(beam_dict['beams'][beam_label].keys()):
            #subbeam = beam[subbeam_label]
            #V_subbeam = numpy.zeros_like(V_in[0,:]) #might need a check conditional in case only one row is entered. 
            first_in_sub_loop = True
            for i in range(len(beam_dict['beams'][beam_label][subbeam_label]['antennas'])):
                #signal = V_in[beam_dict['beams'][beam_label][subbeam_label]['antennas'][i],:]
                
                #V_subbeam = numpy.add(V_subbeam, signal[delay_indices[beam_dict['attrs']['unique_delays'] == beam_dict['beams'][beam_label][subbeam_label]['time_delays'][i]][0]])
                #V_subbeam = numpy.add(V_subbeam, V_in[beam_dict['beams'][beam_label][subbeam_label]['antennas'][i],:][delay_indices[beam_dict['attrs']['unique_delays'] == beam_dict['beams'][beam_label][subbeam_label]['time_delays'][i]][0]])
                if first_in_sub_loop:
                    first_in_sub_loop = False
                    V_subbeam = V_in[beam_dict['beams'][beam_label][subbeam_label]['antennas'][i],:][delay_indices[numpy.where(beam_dict['attrs']['unique_delays']==beam_dict['beams'][beam_label][subbeam_label]['time_delays'][i])[0][0],:]]
                else:
                    V_subbeam = numpy.add(V_subbeam, V_in[beam_dict['beams'][beam_label][subbeam_label]['antennas'][i],:][delay_indices[numpy.where(beam_dict['attrs']['unique_delays']==beam_dict['beams'][beam_label][subbeam_label]['time_delays'][i])[0][0],:]])
            out_beams['beams'][beam_label][subbeam_label]['beam_power_signal'] = V_subbeam**2
            if plot == True:
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
                #print('Saved image %s%s-event%i.%s'%(image_path,self.outfile,eventid,plot_filetype_extension))
            except:
                print('Failed to save image %s%s.%s'%(image_path,beam_label,plot_filetype_extension))
    return out_beams

def sumBeamsFastest( u_in, V_in, beam_dict , config , plot = False,save_figs = False, image_path = './', plot_filetype_extension = 'png'):
    '''
    Expects u_in and V_in to be the same dimensions, with the same number
    of rows as there are antennas. The shallowest detector should be the
    first row of the input matrix. 
    '''
    n_antennas = config['antennas']['n']
    out_beams = beam_dict
    beam_dict['attrs']['signal_length'] = numpy.shape(V_in)[1]
    
    #below is the fastest way I could figure out of doing these sums.  Originally
    #a roll function was used, however this took too long.  I tried to reduce
    #the number of times things are redundently done by calculating them in advance
    #as well as avoid things like appending and such.  It isn't as readable as I would
    #like it to be but it is faster and reproduces the same results as the old
    #algorithm.
    zeros_delay = numpy.arange(beam_dict['attrs']['signal_length']) #any zero delays are the same, no need to do this arange multiple times
    delay_indices = numpy.zeros((len(beam_dict['attrs']['unique_delays']),beam_dict['attrs']['signal_length'])) #will hold the indices for each delay that replicate the roll required.  Ordered in the same way as beam_dict['attrs']['unique_delays'], so call the correct row by from indexing that
    for index, shift in enumerate(beam_dict['attrs']['unique_delays']):
        if shift < 0:
            delay_indices[index,0:beam_dict['attrs']['signal_length'] + shift] = numpy.arange(-shift,beam_dict['attrs']['signal_length'])
            delay_indices[index,beam_dict['attrs']['signal_length'] + shift:beam_dict['attrs']['signal_length']] = numpy.arange(0,-shift)
        elif shift > 0:
            delay_indices[index,0:shift] = numpy.arange(beam_dict['attrs']['signal_length'] - shift, beam_dict['attrs']['signal_length'])
            delay_indices[index,shift:beam_dict['attrs']['signal_length']] = numpy.arange(0,beam_dict['attrs']['signal_length'] - shift)
        else:
            delay_indices[index,:] = zeros_delay
    delay_indices = numpy.array(delay_indices,dtype=int)
    
    for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
        if plot == True:
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
            out_beams['beams'][beam_label][subbeam_label]['beam_power_signal'] = V_subbeam**2
            if plot == True:
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
    return out_beams
 
def doFPGAPowerCalcSingleBeamOld(beam, sum_length=16, interval=8):
    '''
    This is straight from Eric's code
    '''
    num_frames = int(math.floor((len(beam)-sum_length) / interval))
        
    power = numpy.zeros(num_frames)
    for frame in range(num_frames):
        for i in range(frame*interval, frame*interval + sum_length):
            power[frame] += beam[i]
    return numpy.array(power)

@profile
def doFPGAPowerCalcAllBeamsOld(summed_beam_dict,plot = False):
    '''
    This is an adapted version of Eric's code to account for my organizational
    structure for the various beams and subbeams. 
    '''
    beam_powers = {}
    power_beam_dict = summed_beam_dict
    if plot == True:
        fig = pylab.figure(figsize=(16.,11.2))
        #ax = pylab.subplot(len( summed_beam_dict['beams'].keys() ), 1, 1)
    for beam_index, beam_label in enumerate(summed_beam_dict['beams'].keys()):
        first_subbeam = True
        for subbeam_label in summed_beam_dict['beams'][beam_label].keys():
            power_beam_dict['beams'][beam_label][subbeam_label]['power_sum'] = doFPGAPowerCalcSingleBeamOld(summed_beam_dict['beams'][beam_label][subbeam_label]['beam_power_signal'], beam_dict['attrs']['power_calculation_sum_length'], beam_dict['attrs']['power_calculation_interval'])
            if first_subbeam == True:
                first_subbeam = False
                beam_powers[beam_label] = power_beam_dict['beams'][beam_label][subbeam_label]['power_sum']
            else:
                beam_powers[beam_label] = numpy.add(beam_powers[beam_label],power_beam_dict['beams'][beam_label][subbeam_label]['power_sum'])
        
        if plot == True:
            #pylab.subplot(len( summed_beam_dict['beams'].keys() ), 1,beam_index + 1,sharey = ax)
            #getting weighted angle in crude way
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

@profile
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
            beam_dict['beams'][beam_label][subbeam_label]['power_sum'] = numpy.sum(formed_beam_powers[beam_label][subbeam_label][indices],axis=1)
            if first_subbeam == True:
                first_subbeam = False
                beam_powersums[beam_label] = beam_dict['beams'][beam_label][subbeam_label]['power_sum']
            else:
                beam_powersums[beam_label] = numpy.add(beam_powersums[beam_label],beam_dict['beams'][beam_label][subbeam_label]['power_sum'])
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

def beamCalculations(u_in, V_in, beam_dict , config, plot1 = False, plot2 = False):
    '''
    This function should do everything that would be calculated per event in the simulation
    such that I can profile the individual functios to ensure it all runs quickly.
    '''
    V, u = syncSignals(ud,Vd)
    
    summed_beam_dict = sumBeams( u, V, beam_dict , config, plot = plot1,save_figs=False)
    summed_beam_dict2 = sumBeamsFastest( u, V, beam_dict , config, plot = plot1,save_figs=False)
    
    power_beam_dict, beam_powers = doFPGAPowerCalcAllBeamsOld(summed_beam_dict,plot = plot2)
    power_beam_dict2, beam_powers2 = doFPGAPowerCalcAllBeams(summed_beam_dict2,plot = plot2)
    formed_beam_powers, beam_powersums = fpgaBeamForming(u, V, beam_dict , config, plot1 = plot1, plot2 = plot2, save_figs = False)
    print('check0',summed_beam_dict == summed_beam_dict2)
    print('check1',power_beam_dict == power_beam_dict2)
    for beami in numpy.arange(14):
        beam = 'beam%i'%beami
        print('beam_powers %s same?:\n'%beam,numpy.all(beam_powers[beam] == numpy.delete(beam_powersums[beam],-1))) #popping last element because the old alg didn't include it (still not sure why..)
        #print(beam_powers[beam])
        #print(numpy.delete(beam_powers3[beam],-1))
    
############################################################

if __name__ == "__main__":
    old_testing = True
    new_testing = False
    
    pylab.close('all')
    energy_neutrino = 3.e9 # GeV
    n = 1.78
    c = gnosim.utils.constants.speed_light #m/ns
    
    R = 1000. #m
    cherenkov_angle = numpy.arccos(1./n)
    cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./n))
    h_fft,sys_fft,freqs = gnosim.interaction.askaryan_testing.loadSignalResponse()
    input_u, h_fft, sys_fft, freqs = gnosim.interaction.askaryan_testing.calculateTimes(up_sample_factor=20)
    inelasticity = 0.2
    noise_rms = numpy.std(gnosim.interaction.askaryan_testing.quickSignalSingle(0,R,inelasticity*energy_neutrino,n,R,0,0,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)[3])
    V_noiseless, u, dominant_freq, V_noise,  SNR = gnosim.interaction.askaryan_testing.quickSignalSingle(numpy.deg2rad(50),R,inelasticity*energy_neutrino,n,2500,0.7,0.7,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)
    sampling_rate = 1.5 #GHz
    bytes = 7
    scale_noise_from = noise_rms
    scale_noise_to = 3
    
    random_time_offset = numpy.random.uniform(-5.0,5.0) #ns
    dc_offset = 0.0 #V
    V_bit, sampled_times = gnosim.sim.fpga.digitizeSignal(u,V_noise,sampling_rate,bytes,scale_noise_from,scale_noise_to, random_time_offset = random_time_offset, dc_offset = dc_offset, plot = False)
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
    choose_n = 1
    try:
        do_events = numpy.random.choice(eventids,choose_n,replace=False)
    except:
        do_events = numpy.unique(numpy.random.choice(eventids,choose_n,replace=True))
    #do_events = eventids[numpy.random.randint(0,len(eventids),size = 5)]
    plot_beams = False
    plot_sums = False
    for eventid in do_events:
        #Note noise is kind of jank and will always be the same
        print('On event %i'%eventid)
        V, u, Vd, ud = gnosim.interaction.askaryan_testing.signalsFromInfo(eventid,reader,input_u,n,h_fft,sys_fft,freqs,include_noise = True,resistance = 50, temperature = 320)
        sub_info = info[info['eventid'] == eventid]
        Vd2, ud2 = syncSignals(ud,Vd)
        print('Mean angle for event %i is %0.2f'%(eventid, numpy.mean(info[info['eventid'] == eventid]['theta_ant'])))
        beamCalculations(ud, Vd, beam_dict , config, plot1 = plot_beams, plot2 = plot_sums)
        #dips in signals near the ends is likely due to zeros created when synching signals.  These are regions with no overlapping noise, so leass overall power from noise. 
    #sumBeams( ud, Vd, beam_dict , config, plot = True)
    
    i = 2
    beam_label = 'beam0'
    subbeam_label = 'subbeam0'
    signal = Vd2[beam_dict['beams'][beam_label][subbeam_label]['antennas'][i],:]
    signal_length = len(signal)
    delay_indices = numpy.zeros((len(beam_dict['attrs']['unique_delays']),signal_length))
    signal_indices = numpy.arange(signal_length)
    for index, shift in enumerate(beam_dict['attrs']['unique_delays']):
        delay_indices[index,:] = numpy.roll(signal_indices,shift)
    delay_indices_old = numpy.array(delay_indices,dtype=int)
    desired_indexing = delay_indices[numpy.where(beam_dict['attrs']['unique_delays']==beam_dict['beams'][beam_label][subbeam_label]['time_delays'][i])[0][0],:]
    attempted_indexing = delay_indices[beam_dict['attrs']['unique_delays'] == beam_dict['beams'][beam_label][subbeam_label]['time_delays'][i]][0]
    
    delay_indices = numpy.zeros((len(beam_dict['attrs']['unique_delays']),signal_length))
    for index, shift in enumerate(beam_dict['attrs']['unique_delays']):
        if shift < 0:
            delay_indices[index,0:signal_length + shift] = numpy.arange(-shift,signal_length)
            delay_indices[index,signal_length + shift:signal_length] = numpy.arange(0,-shift)
        elif shift > 0:
            delay_indices[index,0:shift] = numpy.arange(signal_length - shift, signal_length)
            delay_indices[index,shift:signal_length] = numpy.arange(0,signal_length - shift)
        else:
            delay_indices[index,:] = numpy.arange(signal_length)
    delay_indices = numpy.array(delay_indices,dtype=int)
    
    
############################################################
