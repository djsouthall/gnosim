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
pylab.ion()
############################################################

if __name__ == "__main__":
    pylab.close('all')
    
    #Paremeters
    signal_response_version = 'v7'
    #config = yaml.load(open('/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/real_config.py'))
    #config = yaml.load(open('/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/real_config_reduced2.py'))
    config = yaml.load(open('/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_dipole_octo_-200_polar_120_rays.py'))
    create_signals = True
    load_signals = False
    random_roll = True
    signal_file = '/home/dsouthall/Projects/GNOSim/gnosim/analysis/samples/1771_1.csv'
    if numpy.logical_and(create_signals == True,load_signals == True):
        print('Currently only supports one signal option per run.  Setting load_signals to False')
        load_signals = False
    if load_signals == True:
        print('Using real configuration config file to match with loaded sample')
        config = yaml.load(open('/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/real_config.py'))
    trim_sums = False #This only returns the regions of powersumming where zeros are not being added, as well as additional windowing
    trim_amount = 100
    cut_zeros = False
    
    if int(signal_response_version.split('v')[-1]) >= 6:
        up_sample_factor = 40
    else:
        up_sample_factor = 20
        
    noise_length_multiplier = 1000 #for the noise used in the triggering
    noise_rms_length_multiplier = 100 #for the noise used to get an understanding of the noise_rms which is used to digitized the noise used for triggering
    threshold_array = numpy.arange(4000,11000,100, dtype=int)
    temperature = 320#K
    resistance = 50 #Ohm
    n_beams = 15
    n_baselines = 2
    power_calculation_sum_length = 16 #How long each power sum window is
    power_calculation_interval = 8 #How frequent each power sum window begins
    
    dc_offset = 0
    scale_noise_to = 3 #noise rms set to this in adu
    sampling_bits = 7
    digital_sampling_period = 1.0/1.5 #ns where 1.5 is in GHz
    random_time_offset = numpy.random.uniform(low=-10.0,high=10.0)
    
    plot_analog_signals     = False # Plots the analog signals used in the calculations.  Only works if create_signals == True, otherwise no analog signals will be present 
    plot_digital_signals    = False  # Plots the digitized signals used in the calculations.
    plot_summed_beams       = False # Plots the power sum beams
    plot_spectrum           = False # Plots the sprectrum of one noise signal to see responses etc.
    plot_rate_per_beam      = True  # Plots the trigger rate per beam vs function of threshold
    plot_fit                = True  # Plots the fit curve for the trigger rate as a function of threshold
    
    desired_trigger_rate = 10.0 #Hz
    #######################
    
    #######################
    
    #Get index of refraction at array
    z_array = []
    for sz in config['stations']['positions']:
        for az in config['antennas']['positions']:
            z_array.append(sz[2] + az[2])
    z_array = numpy.array(z_array)
    index_refraction_array = gnosim.earth.antarctic.indexOfRefraction(z_array, ice_model=config['detector_volume']['ice_model']) 
    mean_index = numpy.mean(index_refraction_array)
    #Create the beams
    beam_dict = gnosim.sim.fpga.getBeams( config, n_beams, n_baselines , mean_index , digital_sampling_period, verbose = False )
    
    
    if create_signals == True:
        #Load Response Curves:
        signal_times, h_fft, sys_fft, freqs_response = gnosim.interaction.askaryan.calculateTimes(up_sample_factor=up_sample_factor,mode=signal_response_version)
        dt = signal_times[1] - signal_times[0]
        
        if plot_spectrum == True:
            #plotting a noise spectrum        
            noise_signal_i = gnosim.interaction.askaryan.quickSignalSingle( 0,1,0,mean_index,\
                              0,0,0,signal_times,h_fft,sys_fft,freqs_response,\
                              plot_signals=False,plot_spectrum=True,plot_potential = False,\
                              include_noise = True, resistance = resistance, temperature = temperature)[3]
        #calculating some noise just for noise_rms
        noise_signal  = numpy.array([])
        for i in range(int(numpy.max([1,noise_rms_length_multiplier]))):
            #Artificially extending noise to be longer
            noise_signal_i = gnosim.interaction.askaryan.quickSignalSingle( 0,1,0,mean_index,\
                          0,0,0,signal_times,h_fft,sys_fft,freqs_response,\
                          plot_signals=False,plot_spectrum=False,plot_potential = False,\
                          include_noise = True, resistance = resistance, temperature = temperature)[3]  #This is just producing a 0eV energy pulse, so only thing returned is noise
            noise_signal = numpy.append(noise_signal,noise_signal_i)
        noise_rms = numpy.std(noise_signal)
        
        
        #Create the noise signal for triggering
        noise_times  = []
        noise_digital_times = []
        noise_signals  = []
        noise_digital_signals = []
        signal_ticks_x = []
        signal_ticks_y = []
        
        for index_antenna in range(config['antennas']['n']):
            noise_signal  = numpy.array([])
            signal_tick_x = []
            print('index_antenna',index_antenna)
            for i in range(int(numpy.max([1,noise_length_multiplier]))):
                #Artificially extending noise to be longer
                noise_signal_i = gnosim.interaction.askaryan.quickSignalSingle( 0.0,1.0,0.0,mean_index,\
                              0.0,0.0,0.0,signal_times,h_fft,sys_fft,freqs_response,\
                              plot_signals=False,plot_spectrum=False,plot_potential = False,\
                              include_noise = True, resistance = resistance, temperature = temperature)[3]  #This is just producing a 0eV energy pulse, so only thing returned is noise
                signal_tick_x.append(len(noise_signal))
                noise_signal = numpy.append(noise_signal,noise_signal_i)
                signal_tick_x.append(len(noise_signal)-1)
                #if add_gaps == True:
                #    noise_signal = numpy.append(noise_signal,numpy.zeros_like(noise_signal_i))
                #    this messes with the rates
            if random_roll == True:
                roll_amount = numpy.random.randint(low = 0, high = len(noise_signal))
                print('rolling by ', roll_amount)
                noise_signal = numpy.roll(noise_signal,roll_amount)
                
            noise_signals.append(noise_signal)
            noise_time = numpy.arange(len(noise_signal)) * dt
            noise_times.append(noise_time)
            digital_sample_times = numpy.arange(noise_time[0],noise_time[-1],digital_sampling_period) + random_time_offset #these + random_time_offset #these
            noise_digital, time_digital = gnosim.sim.fpga.digitizeSignal(noise_time,noise_signal,digital_sample_times,sampling_bits,noise_rms,scale_noise_to,dc_offset = dc_offset,plot = False)
            noise_digital_signals.append(noise_digital)
            noise_digital_times.append(time_digital)
            signal_ticks_x.append(numpy.array(noise_time)[signal_tick_x])
            signal_ticks_y.append(numpy.array(noise_signal)[signal_tick_x])
          
        noise_signals = numpy.array(noise_signals) 
        noise_times = numpy.array(noise_times) 
        noise_digital_signals = numpy.array(noise_digital_signals)
        noise_digital_times = numpy.array(noise_digital_times)
        noise_rms = numpy.std(noise_signal)
        signal_ticks_x = numpy.array(signal_ticks_x)
        signal_ticks_y = numpy.array(signal_ticks_y)
        if plot_analog_signals == True:
            pylab.figure(figsize=(16.,11.2))
            first_in_loop = True
            for index_antenna in range(config['antennas']['n']):
                if first_in_loop == True:
                    first_in_loop = False
                    ax = pylab.subplot(config['antennas']['n'],1,index_antenna+1)
                else:
                    pylab.subplot(config['antennas']['n'],1,index_antenna+1,sharex = ax, sharey = ax)
                pylab.plot(noise_times[index_antenna],noise_signals[index_antenna])
                pylab.plot(signal_ticks_x[index_antenna],signal_ticks_y[index_antenna],'r*',label = 'Sub-Signal Endpoints')
                pylab.ylabel('Noise Signals (V)')
            pylab.xlabel('Times (ns)')
            pylab.legend()
        
    elif load_signals == True:
        signal_file_data = numpy.genfromtxt(signal_file,delimiter = ',')
        relevant_rows = numpy.array([0,1,2,3,4,6,7])
        noise_digital_signals = signal_file_data[relevant_rows]
        noise_digital_times = numpy.tile(numpy.arange(numpy.shape(signal_file_data)[1])*digital_sampling_period, (config['antennas']['n'],1))
    else:
        print('Neither mode selected.  Breaking')
        
    if plot_digital_signals == True:
        fig = pylab.figure(figsize=(16.,11.2))  
        if load_signals == True:
            fig.suptitle('Loaded Real Signals')
        elif create_signals == True:
            fig.suptitle('Generated Noise Signals')
        first_in_loop = True
        for index_antenna in range(config['antennas']['n']):
            if first_in_loop == True:
                first_in_loop = False
                ax = pylab.subplot(config['antennas']['n'],1,index_antenna+1)
            else:
                pylab.subplot(config['antennas']['n'],1,index_antenna+1,sharex = ax, sharey = ax)
            pylab.plot(noise_digital_times[index_antenna],noise_digital_signals[index_antenna])
            pylab.ylabel('Sig (adu)')
            pylab.xlabel('Times (ns)')
    
    #Calculate times to sample
    
    #digitize signal and noise:
    '''
    pylab.figure(figsize=(16.,11.2)) #my screensize
    pylab.plot(time_digital[time_digital < 500],noise_digital[time_digital < 500],label='noise')
    pylab.ylabel('Noise (adu)')
    pylab.xlabel('t (ns)')
    pylab.show()
    '''
    
    formed_beam_powers, beam_powersums = gnosim.sim.fpga.fpgaBeamForming(noise_digital_times, noise_digital_signals, beam_dict , config, plot1 = False, plot2 = False, save_figs = False,trim_sums = trim_sums,trim_amount = trim_amount)
    if cut_zeros == True:
        #'''###
        beam_cut = numpy.zeros((len(beam_powersums.keys()),len(beam_powersums[list(beam_powersums.keys())[0]])))
        for beam_label, beam in beam_powersums.items():
            print(beam == 0)
            print(len(beam))
            print(sum(beam == 0))
            beam_cut[0,:] = beam ==0
        beam_cut2 = ~numpy.any(beam_cut,axis=0)

        for beam_label, beam in beam_powersums.items():
            beam_powersums[beam_label] = beam[beam_cut2]
        #'''###
        
    beam_labels = list(beam_powersums.keys())
    
    
    hits = numpy.zeros((len(list(beam_dict['beams'].keys())),len(threshold_array)),dtype = int)
    for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
        beam = beam_powersums[beam_label]
        hits[beam_index,:] = numpy.sum(numpy.greater_equal(numpy.tile(beam,(len(threshold_array),1)),numpy.tile(threshold_array,(len(beam),1)).T),axis = 1)
    all_hits = numpy.sum(hits, axis=0)

    '''
    hits = []
    for beam_label in beam_labels:
        beam = beam_powersums[beam_label]
        hits_in_single_beam=[]
        for threshold in threshold_array:
            hits_in_single_beam.append(sum(beam >= threshold))
        hits.append(hits_in_single_beam)
    all_hits = numpy.sum(numpy.array(hits), axis=0)
    
    hits = numpy.zeros((len(list(beam_dict['beams'].keys())),len(threshold_array)),dtype = int)
    for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
        beam = beam_powersums[beam_label]
        for threshold_index, threshold in enumerate(threshold_array):
            hits[beam_index,threshold_index] = sum(beam >= threshold)
    all_hits = numpy.sum(hits, axis=0)
    '''
    colormap = pylab.cm.gist_ncar #nipy_spectral, Set1,Paired   
    beam_colors = [colormap(i) for i in numpy.linspace(0, 1,len(beam_dict['beams'].keys())+1)] #I put the +1 backs it was making the last beam white, hopefully if I put this then the last is still white but is never called

    if plot_summed_beams == True:
        fig = pylab.figure(figsize=(16.,11.2)) #my screensize
        if load_signals == True:
            fig.suptitle('Loaded Real Signals')
        elif create_signals == True:
            fig.suptitle('Generated Noise Signals')
        ax = pylab.gca()

        for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
            ax.plot(beam_powersums[beam_label],label = '%s, $\\theta_{ant} = $ %0.2f'%(beam_label,beam_dict['theta_ant'][beam_label]),color = beam_colors[beam_index], alpha = 0.8)

        pylab.ylabel('Summed Power (adu$^2$)')
        pylab.xlabel('Sum Index')
        pylab.legend()
        #pylab.yticks(rotation=45)
        #ax.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
    
    if plot_rate_per_beam == True:
        fig = pylab.figure(figsize=(16.,11.2)) #my screensize
        if load_signals == True:
            fig.suptitle('Loaded Real Signals, Per Beam')
        elif create_signals == True:
            fig.suptitle('Generated Noise Signals, Per Beam')
        for beam_index, beam_label in enumerate(beam_dict['beams'].keys()):
            only_plot_cut = numpy.array(hits[beam_index]) > 0
            pylab.plot(threshold_array[only_plot_cut], numpy.array(hits[beam_index])[only_plot_cut]/(numpy.shape(noise_digital_times)[1]*digital_sampling_period*1.e-9), 'o',color = beam_colors[beam_index],label = '%s, $\\theta_{ant} = $ %0.2f'%(beam_label,beam_dict['theta_ant'][beam_label]))
            #pylab.plot(threshold_array[only_plot_cut], numpy.array(hits[beam_index])[only_plot_cut], 'o',label = beam_labels[beam_index])#/(numpy.shape(noise_digital_times)[1]*digital_sampling_period*1.e-9), 'o')
        only_plot_cut = numpy.array(all_hits) > 0
        pylab.plot(threshold_array[only_plot_cut], all_hits[only_plot_cut]/(numpy.shape(noise_digital_times)[1]*digital_sampling_period*1.e-9), '-',c='black',label = 'All')
        #pylab.plot(threshold_array[only_plot_cut], all_hits[only_plot_cut], '-',c='black',label = 'All hits')#/(numpy.shape(noise_digital_times)[1]*digital_sampling_period*1.e-9), '-',c='black')
        pylab.yscale('log')
        pylab.grid(True)
        pylab.legend()
        pylab.xlabel('Threshold (adu$^2$)')
        pylab.ylabel('Trigger Rate (Hz)')
    
    '''
    try:
    '''
    ####
    indx = numpy.where(all_hits > 0)[0] #why > 1?  >0? #it was 1
    
    #print(indx)
    indx_fit = numpy.where((all_hits > 0.00002*max(all_hits) ) & (all_hits < 0.01*max(all_hits) ))[0]
    if numpy.size(indx_fit) < 3:
        indx_fit = indx
    print(indx_fit)
    ##fit using polyfit in semi-log space:
    logy   = numpy.log(all_hits[indx_fit]/(numpy.shape(noise_digital_times)[1]*digital_sampling_period*1.e-9))  #Linearly fitting the log of both sides expecting a relationship like y = a * exp(-b * x) -->  ln(y) = ln(a) - b*x
    logy_w = numpy.log(1./(numpy.sqrt(all_hits[indx_fit])*(numpy.shape(noise_digital_times)[1]*digital_sampling_period*1.e-9)))  #weights on fit
    #logy_w = numpy.ones_like(logy)
    coeff  = numpy.polyfit(threshold_array[indx_fit], logy, w=logy_w, deg=1)
    poly   = numpy.poly1d(coeff)
    #yfit  = lambda x : numpy.power(10, poly(threshold_array[indx_fit]))
    yfit   = lambda x : numpy.exp(poly(threshold_array[indx]))
    yfit_extended = lambda x1 : numpy.exp(poly(threshold_array))

    print(coeff)
    if plot_fit == True:
        fig = pylab.figure(figsize=(16.,11.2)) #my screensize
        if load_signals == True:
            fig.suptitle('Loaded Real Signals, All Beams')
        elif create_signals == True:
            fig.suptitle('Generated Noise Signals, All Beams')
        pylab.plot(threshold_array[indx], yfit(threshold_array[indx]), '--', color='red', lw=1,label = 'Initial Fit')
        #pylab.plot(threshold_array, yfit_extended(threshold_array), '--', color='red', lw=1)
        pylab.errorbar(threshold_array[indx], all_hits[indx]/(numpy.shape(noise_digital_times)[1]*digital_sampling_period*1.e-9), #xerr=numpy.zeros(len(indx)),
                     yerr=1./(numpy.sqrt(all_hits[indx])*(numpy.shape(noise_digital_times)[1]*digital_sampling_period*1.e-9)),
                     ecolor='black', elinewidth=2, fmt='o', ms=4,
                     #elinewidth=2, fmt='o', ms=2,
                     #label='{:.2f}, fitPow={:0.3f}'.format(labels[beam_index], coeff[0]))
                     #label='{:.2f}'.format(labels[beam_index]))
                     )
        func = lambda t,a,b: a*numpy.exp(b*t)
        x = threshold_array[indx] 
        y = all_hits[indx]/(numpy.shape(noise_digital_times)[1]*digital_sampling_period*1.e-9)
        p0=(numpy.exp(coeff[1]),coeff[0])
        popt, pcov = scipy.optimize.curve_fit( func,  x,  y,  p0=p0)#(numpy.exp(coeff[0]),-coeff[1])
        pylab.plot(x, func(x, *popt), 'g--',label='Corrected Fit:\n $y = a\cdot e^{b \cdot x}$\n a=%0.4g, b=%0.4g' % tuple(popt))
        pylab.legend()
        pylab.xlabel('Threshold (adu$^2$)')
        pylab.ylabel('Trigger Rate (Hz)')
    '''
    except Exception as e:
        print('Plot failed:')
        print(e)
    '''
    for beam_label, beam in beam_powersums.items():
        beam_index = int(beam_label.split('beam')[-1])
        if beam_index < 10:
            beam_index = str('0%i'%beam_index)
        else:
            beam_index = str(beam_index)
        print('%s - Mean: %0.3g    RMS: %0.3g'%('beam' + beam_index, numpy.mean(beam), numpy.std(beam)))
        
    trigger_level = numpy.log(desired_trigger_rate/popt[0])/popt[1]
    print('To trigger on noise at %0.2f Hz set trigger level to %i adu^2'%(desired_trigger_rate,trigger_level))
