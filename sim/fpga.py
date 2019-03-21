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
import gnosim.utils.misc
import gnosim.sim.antarcticsim
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
    
def digitizeSignal(u,V,sample_times,digitizer_bytes,scale_noise_from,scale_noise_to, dc_offset = 0, plot = False):
    '''
    This function is meant to act as the ADC for the simulation.  It will sample
    the signal using the provided sampling rate.  The value of
    digitizer_bytes sets the number of voltage bins, those will be distributed from 
    -2**(digitizer_bytes-1)+1 to 2**(digitizer_bytes-1).  The input V will be sampled and scaled linearly
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
    
    #byte_vals = numpy.linspace(-2**(digitizer_bytes-1)+1,2**(digitizer_bytes-1),2**digitizer_bytes,dtype=int)
    byte_vals = numpy.array([-2**(digitizer_bytes-1)+1,2**(digitizer_bytes-1)],dtype=int) #only really need endpoints
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

def fpgaBeamForming(u_in, V_in, beam_dict , plot1 = False, plot2 = False, save_figs = False, trim_sums = False,trim_amount = 50, cap_bytes = 5):
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
    if len(numpy.shape(V_in)) == 1:
        signal_length = len(V_in)
    elif len(numpy.shape(V_in)) == 2:
        signal_length = numpy.shape(V_in)[1]
    else:
        signal_length = 0
        
    #FPGA does beamforming with only 5 bits despite signals being at 7, so they are capped here:
    byte_vals = numpy.array([-2**(cap_bytes-1)+1,2**(cap_bytes-1)],dtype=int) #only really need endpoints
    V_in[V_in > byte_vals[-1]] = byte_vals[-1]#upper cap
    V_in[V_in < byte_vals[0]] = byte_vals[0]  #lower cap
    
    
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
                if trim_sums == True:
                    zero_cut = subbeam_powersum == 0
            else:
                beam_powersums[beam_label] = numpy.add(beam_powersums[beam_label],subbeam_powersum)
                if trim_sums == True:
                    zero_cut = numpy.logical_or(zero_cut,subbeam_powersum == 0) 
        if trim_sums == True:
            #print('sum(zero_cut) == ', sum(zero_cut))
            #print(zero_cut)
            #print(1.0*(~zero_cut))
            #print(beam_powersums[beam_label])
            beam_powersums[beam_label] = numpy.multiply(beam_powersums[beam_label], 1.0*(~zero_cut))
            #print(beam_powersums[beam_label])
            beam_powersums[beam_label] = beam_powersums[beam_label][numpy.arange(trim_amount,len(beam_powersums[beam_label]) - trim_amount)]
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
        colors = gnosim.utils.misc.getColorMap(len(ax.lines))
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

def getScaleSystemResponseScale(station, desired_noise_rms = 20.4E-3, save_new_response = False):
    '''
    The absolute scale of the system response has been hard to obtain, so we
    have decided to just scale it such that the noise level (which is independent
    of antenna response scale) matches experiment.  The value of noise given
    by Eric was 1adu  6.8mV, so for a 3adu noise rms the desired_noise_rms = 20.4mV.
    The mode selects which version of the responses you are calculating this factor
    for.  
    '''

    for index_antenna, antenna in enumerate(station.antennas):
        signal_times = antenna.signal_times
        h_fft = antenna.h_fft
        sys_fft = antenna.sys_fft
        freqs_response = antenna.freqs_response

        slope = desired_noise_rms/antenna.noise_rms
        print('Antenna %i Simple method of scaling: '%(index_antenna), slope)
        sys_fft = sys_fft*slope
        
        if save_new_response == True:
            try:
                antenna_outfile = antenna.antenna_response_dir.replace('.npy','_new.npy')
                system_outfile = antenna.system_response_dir.replace('.npy','_new.npy')
                if os.path.isfile(antenna_outfile):
                    print('Outfile Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(antenna_outfile))
                    antenna_outfile = antenna_outfile.replace('.npy','_new.npy')
                    while os.path.isfile(antenna_outfile):
                        antenna_outfile = antenna_outfile.replace('.npy','_new.npy')
                if os.path.isfile(system_outfile):
                    print('Outfile Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(system_outfile))
                    system_outfile = system_outfile.replace('.npy','_new.npy')
                    while os.path.isfile(system_outfile):
                        system_outfile = system_outfile.replace('.npy','_new.npy')
                
                print('Saving:\n %s \n %s'%(antenna_outfile,system_outfile))
                numpy.save(antenna_outfile, numpy.array(list(zip(freqs_response,h_fft))))
                numpy.save(system_outfile, numpy.array(list(zip(freqs_response,sys_fft))))
            except Exception as e:
                print('Something went wrong in saving the responses')
                print(e)
    return slope, sys_fft

############################################################

#
if __name__ == "__main__":
    pylab.close('all')

    config_file = '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_dipole_octo_-200_polar_120_rays.py'
    testSim = gnosim.sim.antarcticsim.Sim(config_file,electricFieldDomain = 'time',do_beamforming = True)

    slope,sys_fft = getScaleSystemResponseScale(testSim.stations[0],desired_noise_rms = 20.4E-3,save_new_response = True)
    print('Simple method of scaling: ', slope)
    
############################################################
