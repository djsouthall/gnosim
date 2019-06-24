'''
This is a tool to help you determine the appropriate pre-trigger for your situation.
'''

import os
import sys
sys.path.append(os.environ['GNOSIM_DIR'])
import os.path
import numpy
import pylab
import numpy
import h5py
import pylab
import yaml
import glob
import scipy
import time
import math
import copy
from matplotlib import gridspec
from matplotlib.colors import LogNorm

pylab.ion()
############################################################

if __name__ == "__main__":
    load_reader = False
    load_values = False
    
    weight_by_p_earth = True
    
    plot_1 = True
    plot_2 = False
    calculate_cuts = False
    start_time = time.time()
    if load_reader == True:
        print('(%0.2f s)'%(time.time() - start_time) + '	Loading Reader')    
        reader = h5py.File(os.environ['GNOSIM_DATA'] + '/feb_testing_real_config_108/results_2019_Feb_real_config_1.00e+08_GeV_1000000_events_merged.h5' , 'r')
        config = yaml.load(open(reader.attrs['config_0']))
    
    if load_values == True:
        print('(%0.2f s)'%(time.time() - start_time) + '	Loading Necessary Portions of Info')
        info = reader['info']['eventid','has_solution','distance','observation_angle','fpga_max'][reader['info']['has_solution'][...]]
        
        print('(%0.2f s)'%(time.time() - start_time) + '	Loading Energies')
        energies = (reader['energy_neutrino'][...]*reader['inelasticity'][...])[info['eventid']]   
        
        print('(%0.2f s)'%(time.time() - start_time) + '	Calculating ed_ratio') 
        ed_ratio = energies / info['distance']
        if weight_by_p_earth == True:
            p_earth = reader['p_earth'][...][info['eventid']]
    for trigger_level in numpy.array([11500]):
        print('(%0.2f s)'%(time.time() - start_time) + '	On trigger level:', trigger_level)
        bins = [1000,360]
        
        if weight_by_p_earth == True:
            w1 = p_earth
            w2 = numpy.multiply(info['fpga_max'] > trigger_level,p_earth)
            w1 = numpy.multiply(w1,w1 > 1e-2)
            w2 = numpy.multiply(w2,w2 > 1e-2)
        else:
            w1 = numpy.ones_like(info['fpga_max'])
            w2 = info['fpga_max'] > trigger_level
        
        
        if plot_1 == True:
            hist_fig = pylab.figure()
            x = energies
            if weight_by_p_earth == True:
                hist_fig.suptitle('Trigger = %0.1f (Weighted by $p_\mathrm{earth}$'%trigger_level)
            else:
                hist_fig.suptitle('Trigger = %0.1f'%trigger_level)
            
            ax1 = pylab.subplot(2,3,1)
            
            pylab.hist2d(x,info['observation_angle'],bins=bins,weights = w1,norm=LogNorm())
            pylab.ylabel('Observation Angle (deg)')
            pylab.xlabel('E (GeV)')
            pylab.colorbar()
            
            
            pylab.subplot(2,3,4,sharex = ax1, sharey = ax1)
            
            pylab.hist2d(x,info['observation_angle'],bins=bins,weights = w2 ,norm=LogNorm())
            pylab.ylabel('Observation Angle (deg)')
            pylab.xlabel('E (GeV) (Only showing those with fpga max above %0.2f)'%trigger_level)
            pylab.colorbar()
            
            ############
            
            hist_fig.suptitle('Trigger = %0.1f'%trigger_level)
            x = 1/info['distance']
            ax2 = pylab.subplot(2,3,2,sharey = ax1)
            pylab.hist2d(x,info['observation_angle'],bins=bins,weights = w1,norm=LogNorm())
            pylab.ylabel('Observation Angle (deg)')
            pylab.xlabel('1/d (1/m)')
            pylab.colorbar()
            
            
            pylab.subplot(2,3,5,sharex = ax2, sharey = ax1)
            pylab.hist2d(x,info['observation_angle'],bins=bins,weights = w2 ,norm=LogNorm())
            pylab.ylabel('Observation Angle (deg)')
            pylab.xlabel('1/d (1/m) (Only showing those with fpga max above %0.2f)'%trigger_level)
            pylab.colorbar()
            
            ###############
            
            hist_fig.suptitle('Trigger = %0.1f'%trigger_level)
            x = ed_ratio
            ax3 = pylab.subplot(2,3,3, sharey = ax1)
            pylab.hist2d(x,info['observation_angle'],bins=bins,weights = w1,norm=LogNorm())
            pylab.ylabel('Observation Angle (deg)')
            pylab.xlabel('E/d (GeV/m)')
            pylab.colorbar()
            
            
            pylab.subplot(2,3,6,sharex = ax3, sharey = ax1)
            pylab.hist2d(x,info['observation_angle'],bins=bins,weights = w2 ,norm=LogNorm())
            pylab.ylabel('Observation Angle (deg)')
            pylab.xlabel('E/d (GeV/m) (Only showing those with fpga max above %0.2f)'%trigger_level)
            pylab.colorbar()
    
        #########
        if plot_2 == True:
            hist_fig2 = pylab.figure()
            bins = [1000,1000]
            x = energies
            y = 1/info['distance']
            hist_fig2.suptitle('Trigger = %0.1f'%trigger_level)
            ax1 = pylab.subplot(2,1,1)

            pylab.hist2d(x,y,bins=bins,weights = w1,norm=LogNorm())
            pylab.ylabel('1/d (1/m)')
            pylab.xlabel('E (GeV)')
            pylab.colorbar()

            ax1 = pylab.subplot(2,1,2)

            pylab.hist2d(x,y,bins=bins,weights = w2,norm=LogNorm())
            pylab.ylabel('1/d (1/m)')
            pylab.xlabel('E (GeV)')
            pylab.colorbar()
        
        if calculate_cuts == True:
            angle_range_secondary = [30,80] #Some broader range of angles that might trigger if close enough or high enough energy.
            ed_ratio_range_secondary = [0,2e5] #only solutions matching both of these (or the below cut) are calculated
            
            angle_range_cone = [45,65]  #All angles in this range calculated, essentially selecting on cone events
            
            angle_secondary_cut = numpy.logical_and(info['observation_angle'] > angle_range_secondary[0], info['observation_angle'] < angle_range_secondary[-1])
            ed_ratio_secondary_cut = numpy.logical_and(ed_ratio > ed_ratio_range_secondary[0], ed_ratio < ed_ratio_range_secondary[-1])
            
            secondary_cut = numpy.logical_and(angle_secondary_cut,ed_ratio_secondary_cut)
            secondary_cut = numpy.isin(info['eventid'],info['eventid'][secondary_cut]) #Any event with ANYTHING passing pretrigger passes
            
            cone_cut = numpy.logical_and(info['observation_angle'] > angle_range_cone[0], info['observation_angle'] < angle_range_cone[-1])
            cone_cut = numpy.isin(info['eventid'],info['eventid'][cone_cut]) #Any event with ANYTHING passing pretrigger passes
            
            total_cut = numpy.logical_or(secondary_cut,cone_cut)
            
            if weight_by_p_earth == True:
                p_earth_sum = numpy.sum(p_earth)
                print('Calculations below made using p_earth as a weight')
                print('With a pre trigger cut on angles between %0.2f and %0.2f degrees (cone cut) you would see:'%(angle_range_cone[0],angle_range_cone[-1]))
                print('(cone cut) Percentage of events with solutions that would be calculated: %0.3f'%( 100 * numpy.sum(p_earth[cone_cut]) / p_earth_sum))
                print('(cone cut) Weighted Percentage of events that are calculated that would have triggered at %i: %0.3f '%(trigger_level,100 * numpy.sum(p_earth[cone_cut][info['fpga_max'][cone_cut] > trigger_level]) / numpy.sum(p_earth[info['fpga_max'] > trigger_level])))
                
                print('With a pre trigger cut on angles between %0.2f and %0.2f degrees, as well as \na cut on the ed_ratio range from %0.3g to %0.3g (energy/distance cut) you would see:'%(angle_range_secondary[0],angle_range_secondary[-1], ed_ratio_range_secondary[0], ed_ratio_range_secondary[-1]))
                print('(energy/distance cut) Percentage of events with solutions that would be calculated: %0.3f'%( 100 * numpy.sum(p_earth[secondary_cut]) / p_earth_sum))
                print('(energy/distance cut) Weighted Percentage of events that are calculated that would have triggered at %i: %0.3f '%(trigger_level,100 * numpy.sum(p_earth[secondary_cut][info['fpga_max'][secondary_cut] > trigger_level]) / numpy.sum(p_earth[info['fpga_max'] > trigger_level])))
                
                print('With both pre triggers applied (passes if either passes) you would see:')
                print('Percentage of events with solutions that would be calculated: %0.3f'%( 100 * numpy.sum(p_earth[total_cut]) / p_earth_sum))
                print('Weighted Percentage of events that are calculated that would have triggered at %i: %0.3f '%(trigger_level,100 * numpy.sum(p_earth[total_cut][info['fpga_max'][total_cut] > trigger_level]) / numpy.sum(p_earth[info['fpga_max'] > trigger_level])))
            else:
                print('With a pre trigger cut on angles between %0.2f and %0.2f degrees (cone cut) you would see:'%(angle_range_cone[0],angle_range_cone[-1]))
                print('(cone cut) Percentage of events with solutions that would be calculated: %0.3f'%( 100 * numpy.sum(cone_cut) / len(info)))
                print('(cone cut) Percentage of events that are calculated that would have triggered at %i: %0.3f '%(trigger_level,100 * numpy.sum(info['fpga_max'][cone_cut] > trigger_level) / numpy.sum(info['fpga_max'] > trigger_level)))
                
                print('With a pre trigger cut on angles between %0.2f and %0.2f degrees, as well as \na cut on the ed_ratio range from %0.3g to %0.3g (energy/distance cut) you would see:'%(angle_range_secondary[0],angle_range_secondary[-1], ed_ratio_range_secondary[0], ed_ratio_range_secondary[-1]))
                print('(energy/distance cut) Percentage of events with solutions that would be calculated: %0.3f'%( 100 * numpy.sum(secondary_cut) / len(info)))
                print('(energy/distance cut) Percentage of events that are calculated that would have triggered at %i: %0.3f '%(trigger_level,100 * numpy.sum(info['fpga_max'][secondary_cut] > trigger_level) / numpy.sum(info['fpga_max'] > trigger_level)))
                
                print('With both pre triggers applied (passes if either passes) you would see:')
                print('Percentage of events with solutions that would be calculated: %0.3f'%( 100 * numpy.sum(total_cut) / len(info)))
                print('Percentage of events that are calculated that would have triggered at %i: %0.3f '%(trigger_level,100 * numpy.sum(info['fpga_max'][total_cut] > trigger_level) / numpy.sum(info['fpga_max'] > trigger_level)))
                         
            
    
    
    
    
    
