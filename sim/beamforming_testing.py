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

#### BELOW IS ERICS CODE COPIED ####

import detector.geometry as geometry
import config as cfg

import math
import numpy as np

def doFPGABeamForming(wfms, subbeam0=True, subbeam1=False, subbeam2=False):
    '''
    input 'wfms' should already have sampling rate set to appropriate rate
    '''
    coherent_sums_0=[]
    coherent_sums_1=[]
    coherent_sums_2=[]

    for i in range(cfg.NUM_FPGA_BEAMS):
        _coherent_sums_0 = numpy.zeros(len(wfms[0]))
        _coherent_sums_1 = numpy.zeros(len(wfms[0]))
        _coherent_sums_2 = numpy.zeros(len(wfms[0]))
        for j in range(geometry.nantenna):
            if subbeam0 and cfg.subbeam_0_codes[j] != 99:
                _coherent_sums_0 += numpy.roll(wfms[j], cfg.subbeam_0_delays[i]*cfg.subbeam_0_codes[j])
            if subbeam1 and cfg.subbeam_1_codes[j] != 99:
                _coherent_sums_1 += numpy.roll(wfms[j], cfg.subbeam_1_delays[i]*cfg.subbeam_1_codes[j])
            if subbeam2 and cfg.subbeam_2_codes[j] != 99:
                _coherent_sums_2 += numpy.roll(wfms[j], cfg.subbeam_2_delays[i]*cfg.subbeam_2_codes[j])
                
        coherent_sums_0.append(_coherent_sums_0)
        coherent_sums_1.append(_coherent_sums_1)
        coherent_sums_2.append(_coherent_sums_2)

    return coherent_sums_0, coherent_sums_1, coherent_sums_2

def doFPGAPowerCalcSingleBeam(beam, sum_length=16, interval=8):
    num_frames = int(math.floor((len(beam)-sum_length) / interval))
        
    power = numpy.zeros(num_frames)
    for frame in range(num_frames):
        for i in range(frame*interval, frame*interval + sum_length):
            power[frame] += (beam[i] * beam[i]) #/ sum_length
    return power

def doFPGAPowerCalcAllBeams(beam, sum_length, interval):
    beam_powers = []
    for i in range(cfg.NUM_FPGA_BEAMS):
        beam_powers.append(doFPGAPowerCalcSingleBeam(beam[i], sum_length, interval)  )
     
        #total_power.append(numpy.sum([power_beam8, power_beam4a, power_beam4b],axis=0,dtype=numpy.int))

    return numpy.array(beam_powers,dtype=int)

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

############################################################

if __name__ == "__main__":
    old_testing = True
    new_testing = False
    
    pylab.close('all')
    energy_neutrino = 3.e9 # GeV
    n = 1.78
    R = 1000. #m
    cherenkov_angle = numpy.arccos(1./n)
    cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./n))
    h_fft,sys_fft,freqs = gnosim.interaction.askaryan_testing.loadSignalResponse()
    input_u, h_fft, sys_fft, freqs = gnosim.interaction.askaryan_testing.calculateTimes(up_sample_factor=20)
    print(n)
    if new_testing == True:
        inelasticity = 0.2#gnosim.interaction.inelasticity.inelasticity(energy_neutrino, mode='cc')
        #Testing the digitizations of the signal.  
        V_noiseless, u, dominant_freq, V_noise,  SNR = gnosim.interaction.askaryan_testing.quickSignalSingle(numpy.deg2rad(50),R,inelasticity*energy_neutrino,n,2500,0.7,0.7,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)
        noise_rms = numpy.std(gnosim.interaction.askaryan_testing.quickSignalSingle(0,R,inelasticity*energy_neutrino,1.8,2500,0,0,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)[3])
        sampling_rate = 1.5 #GHz
        bytes = 7
        scale_noise_from = noise_rms
        scale_noise_to = 3
        
        random_time_offset = numpy.random.uniform(-5.0,5.0) #ns
        dc_offset = 0.0 #V
        V_bit, sampled_times = gnosim.interaction.askaryan_testing.digitizeSignal(u,V_noise,sampling_rate,bytes,scale_noise_from,scale_noise_to, random_time_offset = random_time_offset, dc_offset = dc_offset, plot = False)
        

    if old_testing == True:
        #Testing making a table for an event
        from gnosim.trace.refraction_library_beta import *
        reader = h5py.File('./Output/results_2018_Dec_config_dipole_octo_-200_polar_120_rays_3.00e+09_GeV_100_events_1_seed_1.h5' , 'r')
        
        info = reader['info'][...]
        
        for eventid in [15]:
            #Note noise is kind of jank and will always be the same
            V, u, Vd, ud = gnosim.interaction.askaryan_testing.signalsFromInfo(eventid,reader,input_u,n,h_fft,sys_fft,freqs,include_noise = True,resistance = 50, temperature = 320)
            sub_info = info[info['eventid'] == eventid]
            origin = [[0,0,-200],[0,0,-201],[0,0,-202],[0,0,-203],[0,0,-204],[0,0,-205],[0,0,-206],[0,0,-207]]
            neutrino_loc = [reader['x_0'][eventid],reader['y_0'][eventid],reader['z_0'][eventid]]
            phi_0 = reader['phi_0'][eventid]

############################################################
