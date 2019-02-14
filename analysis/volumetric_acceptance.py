'''
This file will eventually load in the various configuration types and other information
to calculate and plot volumetric acceptance of various configuration types as a
function of neutrino energy.

Run in python command line:
exec(open('./gnosim/sim/volumetric_acceptance_dsouthall.py').read())


'''
import glob
import sys
import os
import numpy
import h5py
import pylab
import yaml
import pickle
import types
from matplotlib.colors import LogNorm
sys.path.append('/home/dsouthall/Projects/GNOSim/')
import gnosim.utils.constants
import gnosim.utils.bayesian_efficiency
import gnosim.earth.antarctic
import gnosim.trace.refraction_library
from matplotlib.colors import LogNorm
pylab.ion()


def volumetricAcceptance(reader,verbose = True):
    '''
    Calculates the volumetric acceptance of a run in # km^3 sr
    '''
    config = yaml.load(open(reader.attrs['config_0']))
    n_events = len(info)
    
    if verbose == True:
        print('Loading relevant parts of info')
    info = numpy.unique(reader['info']['eventid','triggered'])
    
    if verbose == True:
        print('Loading z_0')
    z_0 = reader['z_0'][...]
    
    if verbose == True:
        print('Loading p_earth')
    p_earth = reader['p_earth'][...]
    
    if verbose == True:
        print('Loading p_interact')
    p_interact = reader['p_interact'][...]
    
    if verbose == True:
        print('Calculating VA_sum')
    
    VA_sum = numpy.sum(p_earth * (gnosim.earth.antarctic.density(z_0) / gnosim.utils.constants.density_water) * (info['triggered']) )
    VA  = ((reader.attrs['geometric_factor'] / gnosim.utils.constants.km_to_m**3)/n_events) *  VA_sum # km^3 sr
    error = VA/numpy.sqrt(VA_sum)
    
    if verbose == True:
        print( 'VA = %0.4g km^3 sr +/- %0.4g km^3 sr'%(VA, error))
        print( 'VA = %0.4g km^3 sr +/- %0.4g percent'%(VA, 100.0 * error/VA))
    return VA, error

############################
if __name__ == "__main__":
    print('Loading reader')
    reader = h5py.File('/home/dsouthall/scratch-midway2/feb_testing_real_config_108/results_2019_Feb_real_config_1.00e+08_GeV_1000000_events_merged.h5' , 'r')
    config = yaml.load(open(reader.attrs['config_0']))
    print('Loading relevant parts of info')
    info = numpy.unique(reader['info']['eventid','triggered'])
    print('Loading z_0')
    z_0 = reader['z_0'][...]
    print('Loading p_earth')
    p_earth = reader['p_earth'][...]
    print('Loading p_interact')
    p_interact = reader['p_interact'][...]
    print('Calculating VA_sum')
    n_events = len(info)
    VA_sum = numpy.sum(p_earth * (gnosim.earth.antarctic.density(z_0) / gnosim.utils.constants.density_water) * (info['triggered']) )
    VA  = ((reader.attrs['geometric_factor'] / gnosim.utils.constants.km_to_m**3)/n_events) *  VA_sum # km^3 sr
    error = VA/numpy.sqrt(VA_sum)
    print( 'VA = %0.4g km^3 sr +/- %0.4g km^3 sr'%(VA, error))
    print( 'VA = %0.4g km^3 sr +/- %0.4g percent'%(VA, 100.0 * error/VA))
    


    '''
    if plot_ratios == True:
        norm_data = h5py.File(norm_name, 'r')
        sorted_cut = numpy.argsort(norm_data['energy_neutrino'][...])
        norm_VA = norm_data['volumetric_acceptance'][...][sorted_cut]
        norm_error = norm_data['error'][...][sorted_cut]
        norm_threshold = norm_data['mean_electric_threshold'][...][sorted_cut]
        norm_energy_neutrino = norm_data['energy_neutrino'][...][sorted_cut]

    for bandwidth_label in list(bandwidths.keys()):
        for SNR in SNRs:
            print('Analyzing data for SNR:',SNR)
            if plot_bool == True:
                fig = pylab.figure()
                
                if plot_ratios:
                    pylab.subplot(211)
                    pylab.title('Volumetric Acceptance with SNR = %0.1f (%s)'%(SNR,bandwidth_label),fontsize = 30)
                    ax = pylab.gca()
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    pylab.tick_params(labelsize=16)
                    #pylab.xlabel('Neutrino Energy (GeV)',fontsize = 20)
                    pylab.ylabel('V$\Omega$ (km$^3$ sr)',fontsize = 20)
                    pylab.grid(b=True, which='major', color='k', linestyle='-')
                    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    
                    pylab.subplot(212)
                    pylab.title('Normalized to %.1f dBi (%s) at %.1f m with %.1f $\sigma$)'%(norm_gain,norm_bandwidth_label,norm_depth,norm_SNR),fontsize = 16)
                    ax = pylab.gca()
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    pylab.tick_params(labelsize=16)
                    pylab.xlabel('Neutrino Energy (GeV)',fontsize = 20)
                    pylab.ylabel('Normalized V$\Omega$',fontsize = 20)
                    pylab.grid(b=True, which='major', color='k', linestyle='-')
                    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                else:
                    pylab.title('Volumetric Acceptance with SNR = %0.1f (%s)'%(SNR,bandwidth_label),fontsize = 30)
                    ax = pylab.gca()
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    pylab.tick_params(labelsize=16)
                    pylab.xlabel('Neutrino Energy (GeV)',fontsize = 20)
                    pylab.ylabel('V$\Omega$ (km$^3$ sr)',fontsize = 20)
                    pylab.grid(b=True, which='major', color='k', linestyle='-')
                    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            for depth in depths:
                print('Analyzing data for Depth:',depth)
                
                if numpy.logical_and(do_calculations,~skip_old):
                    infiles = glob.glob(bandwidths[bandwidth_label]['in_dir'] +'*_%i*.h5'%(depth))
                else:
                    infiles = glob.glob(bandwidths[bandwidth_label]['out_dir'] +'*_%i*.h5'%(depth))
                    
                for gain in gains[str(depth)]:
                    print('Analyzing data for gain:',gain)
                    outname = bandwidths[bandwidth_label]['out_dir'] +'volumetric_acceptance_data_z0_%i_gain_%s_SNR_%s.h5'%(depth, str(gain).replace('.','p'), str(SNR).replace('.','p'))
                    should_skip = numpy.logical_and(skip_old, os.path.isfile(outname) == True)
                    if numpy.logical_and(do_calculations, ~should_skip):
                        VA = numpy.array([])
                        error = numpy.array([])
                        threshold = numpy.array([])
                        energy_neutrino = numpy.array([])
                        for infile in infiles:
                            print('On file:',infile)
                            reader = h5py.File(infile, 'r')
                            VA_out , error_out, threshold_out, energy_neutrino_out = getInfo(reader,SNR,gain,freq,verbose=verbose)
                            VA = numpy.append(VA,VA_out)
                            error = numpy.append(error,error_out)
                            threshold = numpy.append(threshold,threshold_out)
                            energy_neutrino = numpy.append(energy_neutrino,energy_neutrino_out)
                            reader.close()
                    else:
                        if should_skip:
                            print('Skipping calculations for:\n',outname)
                        print('Loading file:',outname)
                        reader = h5py.File(outname, 'r')
                        VA = reader['volumetric_acceptance'][...]
                        error = reader['error'][...]
                        threshold = reader['mean_electric_threshold'][...]
                        energy_neutrino = reader['energy_neutrino'][...]
                        reader.close()
                    
                    cut = VA>0#numpy.logical_and( (VA > 0 ) , (error < 1.0*VA) )
                    if numpy.logical_and(numpy.logical_and(do_calculations, ~should_skip),save_data):
                        print('Creating outfile:',outname)
                        outfile  = h5py.File(outname, 'w')
                        outfile.attrs['gain'] = gain
                        outfile.attrs['z_0'] = depth
                        
                        outfile.create_dataset('volumetric_acceptance', (sum(cut),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                        outfile.create_dataset('error', (sum(cut),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                        outfile.create_dataset('mean_electric_threshold', (sum(cut),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                        outfile.create_dataset('energy_neutrino', (sum(cut),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                        outfile['volumetric_acceptance'][...] = VA[cut]
                        outfile['error'][...] = error[cut]
                        outfile['mean_electric_threshold'][...] = threshold[cut]
                        outfile['energy_neutrino'][...] = energy_neutrino[cut]
                        outfile.close()
                    
                    if plot_bool:
                        #pylab.plot(energy_neutrino[cut],VA[cut])
                        label = 'Depth: %i, Gain: %0.1f'%(depth, gain)
                        sorted_cut = numpy.argsort(energy_neutrino)[numpy.isin(numpy.argsort(energy_neutrino),numpy.where(cut)[0])]
                        if plot_ratios:
                            pylab.subplot(211)
                            pylab.errorbar(energy_neutrino[sorted_cut], VA[sorted_cut], yerr=error[sorted_cut],fmt='o-',label=label)
                            pylab.subplot(212)
                            ratio_cut_on_data = numpy.isin(energy_neutrino[sorted_cut],norm_energy_neutrino)
                            ratio_cut_on_norm = numpy.isin(norm_energy_neutrino,energy_neutrino[sorted_cut])
                            ratios = numpy.divide((VA[sorted_cut])[ratio_cut_on_data],norm_VA[ratio_cut_on_norm])
                            rel_err = numpy.sqrt(numpy.divide((error[sorted_cut])[ratio_cut_on_data],(VA[sorted_cut])[ratio_cut_on_data])**2 + numpy.divide(norm_error[ratio_cut_on_norm],norm_VA[ratio_cut_on_norm])**2)
                            pylab.errorbar((energy_neutrino[sorted_cut])[ratio_cut_on_data], ratios , yerr=numpy.multiply(rel_err,ratios),fmt='o-',label=label)
                        else:
                            pylab.errorbar(energy_neutrino[sorted_cut], VA[sorted_cut], yerr=error[sorted_cut],fmt='o-',label=label)
                        
            if plot_bool == True:
                if plot_ratios:
                    pylab.subplot(211)
                    pylab.legend(loc = 'upper left',fontsize = 16)
                    pylab.ylim(8e-7,1.5e2)
                    pylab.xlim(8e5,2e10)
                    ax = pylab.gca()
                    ax.minorticks_on()
                    
                    pylab.subplot(212)
                    #pylab.legend(loc = 'lower left',fontsize = 16)
                    pylab.xlim(8e5,2e10)
                    pylab.ylim(5e-3,5e1)
                    #ax = pylab.gca()
                    #ax.minorticks_on()
                    pylab.subplots_adjust(top = 0.96, bottom = 0.07, right = 0.99, left = 0.07, hspace = 0.2, wspace = 0.2)
                else:
                    pylab.legend(loc = 'upper left',fontsize = 16)
                    pylab.ylim(8e-7,1.5e2)
                    pylab.xlim(8e5,2e10)

    if plot_ratios == True:
        norm_data.close()
    '''
