
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
from matplotlib import gridspec
import pandas
import gnosim.sim.antarcticsim
import gnosim.utils.misc
pylab.ion()
############################################################

if __name__ == "__main__":
    pylab.close('all')
    #Paremeters
    config_file = '/home/dsouthall/Projects/GNOSim/gnosim/detector/station_config/config_dipole_octo_-200_polar_120_rays.py'
    #config_file = '/home/dsouthall/Projects/GNOSim/gnosim/detector/station_config/test.py'
    config = yaml.load(open(config_file))
    solutions = numpy.array(['direct', 'cross', 'reflect'])
    
    random_roll = True

    cut_zeros = False
    
    noise_length_multiplier = 100 #for the noise used in the triggering
    noise_rms_length_multiplier = 100 #for the noise used to get an understanding of the noise_rms which is used to digitized the noise used for triggering
    threshold_array = numpy.arange(4000,11000,100, dtype=int)
 
    dc_offset = 0

    random_time_offset = numpy.random.uniform(low=-1.0,high=10.0)
    
    plot_analog_signals     = True # Plots the analog signals used in the calculations. 
    plot_digital_signals    = True  # Plots the digitized signals used in the calculations.
    plot_summed_beams       = False # Plots the power sum beams
    plot_spectrum           = False # Plots the sprectrum of one noise signal to see responses etc.
    plot_rate_per_beam      = True  # Plots the trigger rate per beam vs function of threshold
    plot_fit                = True  # Plots the fit curve for the trigger rate as a function of threshold
    
    desired_trigger_rate = 10.0 #Hz
    #######################
    
    #######################
    testSim = gnosim.sim.antarcticsim.Sim(config_file,electricFieldDomain = 'time',do_beamforming = True,solutions = solutions)
    #come back and fix once antennas are properly set up
    for index_station, station in enumerate(testSim.stations):
        z_array = []
        noise_temp_array = []
        resistance_array = []
        noise_rms_array = []
        station.calculateNoiseRMS()
        for antenna in station.antennas:
            z_array.append(antenna.z)
            noise_temp_array.append(antenna.noise_temperature)
            resistance_array.append(antenna.resistance)
            noise_rms_array.append(antenna.noise_rms)
        z_array = numpy.array(z_array)
        noise_temp_array = numpy.array(noise_temp_array)
        resistance = numpy.array(resistance_array)
        noise_rms_array = numpy.array(noise_rms_array)
        index_refraction_array = testSim.ice.indexOfRefraction(z_array) 
        mean_index = numpy.mean(index_refraction_array)

        if plot_spectrum == True:
            #plotting a noise spectrum

            noise_signal_i = gnosim.interaction.askaryan.quickSignalSingle( 0.0,1.0,0.0,mean_index,\
                              0.0,0.0,testSim.stations[0].antennas[0].signal_times,testSim.stations[0].antennas[0].h_fft,testSim.stations[0].antennas[0].sys_fft,testSim.stations[0].antennas[0].freqs_response,\
                              plot_signals=False,plot_spectrum=True,plot_potential = False,\
                              include_noise = True, resistance = numpy.mean(resistance), noise_temperature = numpy.mean(noise_temp_array))[3]

        noise_rms = numpy.mean(noise_rms_array)

        #Create the noise signal for triggering
        noise_times  = []
        noise_digital_times = []
        noise_signals  = []
        noise_digital_signals = []
        signal_ticks_x = []
        signal_ticks_y = []

        for index_antenna,antenna in enumerate(station.antennas):
            noise_signal  = numpy.array([])
            signal_tick_x = []
            print('index_antenna',index_antenna)
            for i in range(int(numpy.max([1,noise_length_multiplier]))):
                #Artificially extending noise to be longer
                noise_signal_i = gnosim.interaction.askaryan.quickSignalSingle( 0.0,1.0,0.0,mean_index,\
                              0.0,0.0,antenna.signal_times,antenna.h_fft,antenna.sys_fft,antenna.freqs_response,\
                              plot_signals=False,plot_spectrum=False,plot_potential = False,\
                              include_noise = True, resistance = antenna.resistance, noise_temperature = antenna.noise_temperature)[3]  #This is just producing a 0eV energy pulse, so only thing returned is noise
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
            dt = antenna.signal_times[1] - antenna.signal_times[0]
            noise_time = numpy.arange(len(noise_signal)) * dt
            noise_times.append(noise_time)
        noise_times = numpy.array(noise_times)
        min_time = numpy.min(noise_times)
        #Here I should calculate igital samping times from all antennas.  
        digital_sample_times = numpy.arange(numpy.min(noise_times),numpy.max(noise_times),station.digital_sampling_period) + random_time_offset #these + random_time_offset #these
        
        for index_antenna,antenna in enumerate(station.antennas):
            noise_digital, time_digital = gnosim.detector.fpga.digitizeSignal(noise_times[index_antenna],noise_signals[index_antenna],digital_sample_times,station.sampling_bits,antenna.noise_rms,station.scale_noise_to,dc_offset = dc_offset,plot = False)
            noise_digital_signals.append(noise_digital)
            noise_digital_times.append(time_digital)
            signal_ticks_x.append(numpy.array(noise_times[index_antenna])[signal_tick_x])
            signal_ticks_y.append(numpy.array(noise_signals[index_antenna])[signal_tick_x])
          
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
            for index_antenna,antenna in enumerate(station.antennas):
                if first_in_loop == True:
                    first_in_loop = False
                    ax = pylab.subplot(len(station.antennas),1,index_antenna+1)
                else:
                    pylab.subplot(len(station.antennas),1,index_antenna+1,sharex = ax, sharey = ax)
                pylab.plot(noise_times[index_antenna],noise_signals[index_antenna])
                pylab.plot(signal_ticks_x[index_antenna],signal_ticks_y[index_antenna],'r*',label = 'Sub-Signal Endpoints')
                pylab.ylabel('Noise Signals (V)')
            pylab.xlabel('Times (ns)')
            pylab.legend()
        
        if plot_digital_signals == True:
            fig = pylab.figure(figsize=(16.,11.2))  
            fig.suptitle('Generated Noise Signals')
            first_in_loop = True
            for index_antenna, antenna in enumerate(station.antennas):
                if first_in_loop == True:
                    first_in_loop = False
                    ax = pylab.subplot(len(station.antennas),1,index_antenna+1)
                else:
                    pylab.subplot(len(station.antennas),1,index_antenna+1,sharex = ax, sharey = ax)
                pylab.plot(noise_digital_times[index_antenna],noise_digital_signals[index_antenna])
                pylab.ylabel('Sig (adu)')
                pylab.xlabel('Times (ns)')
    
        formed_beam_powers, beam_powersums = gnosim.detector.fpga.fpgaBeamForming(noise_digital_signals,noise_digital_times, station.beam_dict , plot1 = False, plot2 = False, save_figs = False, cap_bits = station.beamforming_power_sum_bit_cap)
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
        
        
        hits = numpy.zeros((len(list(station.beam_dict['beams'].keys())),len(threshold_array)),dtype = int)
        for beam_index, beam_label in enumerate(station.beam_dict['beams'].keys()):
            beam = beam_powersums[beam_label]
            hits[beam_index,:] = numpy.sum(numpy.greater_equal(numpy.tile(beam,(len(threshold_array),1)),numpy.tile(threshold_array,(len(beam),1)).T),axis = 1)
        all_hits = numpy.sum(hits, axis=0)

        beam_colors = gnosim.utils.misc.getColorMap(len(station.beam_dict['beams'].keys()))

        if plot_summed_beams == True:
            fig = pylab.figure(figsize=(16.,11.2)) #my screensize
            fig.suptitle('Generated Noise Signals')
            ax = pylab.gca()

            for beam_index, beam_label in enumerate(station.beam_dict['beams'].keys()):
                ax.plot(beam_powersums[beam_label],label = '%s, $\\theta_{ant} = $ %0.2f'%(beam_label,station.beam_dict['theta_ant'][beam_label]),color = beam_colors[beam_index], alpha = 0.8)

            pylab.ylabel('Summed Power (adu$^2$)')
            pylab.xlabel('Sum Index')
            pylab.legend()
            #pylab.yticks(rotation=45)
            #ax.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
        
        if plot_rate_per_beam == True:
            fig = pylab.figure(figsize=(16.,11.2)) #my screensize
            fig.suptitle('Generated Noise Signals, Per Beam')
            for beam_index, beam_label in enumerate(station.beam_dict['beams'].keys()):
                only_plot_cut = numpy.array(hits[beam_index]) > 0
                pylab.plot(threshold_array[only_plot_cut], numpy.array(hits[beam_index])[only_plot_cut]/(numpy.shape(noise_digital_times)[1]*station.digital_sampling_period*1.e-9), 'o',color = beam_colors[beam_index],label = '%s, $\\theta_{ant} = $ %0.2f'%(beam_label,station.beam_dict['theta_ant'][beam_label]))
                #pylab.plot(threshold_array[only_plot_cut], numpy.array(hits[beam_index])[only_plot_cut], 'o',label = beam_labels[beam_index])#/(numpy.shape(noise_digital_times)[1]*station.digital_sampling_period*1.e-9), 'o')
            only_plot_cut = numpy.array(all_hits) > 0
            pylab.plot(threshold_array[only_plot_cut], all_hits[only_plot_cut]/(numpy.shape(noise_digital_times)[1]*station.digital_sampling_period*1.e-9), '-',c='black',label = 'All')
            #pylab.plot(threshold_array[only_plot_cut], all_hits[only_plot_cut], '-',c='black',label = 'All hits')#/(numpy.shape(noise_digital_times)[1]*station.digital_sampling_period*1.e-9), '-',c='black')
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
        logy   = numpy.log(all_hits[indx_fit]/(numpy.shape(noise_digital_times)[1]*station.digital_sampling_period*1.e-9))  #Linearly fitting the log of both sides expecting a relationship like y = a * exp(-b * x) -->  ln(y) = ln(a) - b*x
        logy_w = numpy.log(1./(numpy.sqrt(all_hits[indx_fit])*(numpy.shape(noise_digital_times)[1]*station.digital_sampling_period*1.e-9)))  #weights on fit
        #logy_w = numpy.ones_like(logy)
        coeff  = numpy.polyfit(threshold_array[indx_fit], logy, w=logy_w, deg=1)
        poly   = numpy.poly1d(coeff)
        #yfit  = lambda x : numpy.power(10, poly(threshold_array[indx_fit]))
        yfit   = lambda x : numpy.exp(poly(threshold_array[indx]))
        yfit_extended = lambda x1 : numpy.exp(poly(threshold_array))

        print(coeff)
        if plot_fit == True:
            fig = pylab.figure(figsize=(16.,11.2)) #my screensize
            fig.suptitle('Generated Noise Signals, All Beams')
            pylab.plot(threshold_array[indx], yfit(threshold_array[indx]), '--', color='red', lw=1,label = 'Initial Fit')
            #pylab.plot(threshold_array, yfit_extended(threshold_array), '--', color='red', lw=1)
            pylab.errorbar(threshold_array[indx], all_hits[indx]/(numpy.shape(noise_digital_times)[1]*station.digital_sampling_period*1.e-9), #xerr=numpy.zeros(len(indx)),
                         yerr=1./(numpy.sqrt(all_hits[indx])*(numpy.shape(noise_digital_times)[1]*station.digital_sampling_period*1.e-9)),
                         ecolor='black', elinewidth=2, fmt='o', ms=4,
                         #elinewidth=2, fmt='o', ms=2,
                         #label='{:.2f}, fitPow={:0.3f}'.format(labels[beam_index], coeff[0]))
                         #label='{:.2f}'.format(labels[beam_index]))
                         )
            func = lambda t,a,b: a*numpy.exp(b*t)
            x = threshold_array[indx] 
            y = all_hits[indx]/(numpy.shape(noise_digital_times)[1]*station.digital_sampling_period*1.e-9)
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
