'''
This file will eventually load in the various configuration types and other information
to calculate and plot volumetric acceptance of various configuration types as a
function of neutrino energy.

Run in python command line:
exec(open('./gnosim/sim/volumetric_acceptance_dsouthall.py').read())

#NOTE IN THE CURRENT STATE THIS WAS DESIGNED FOR A MODIFIED BUILD OF THE CODE
FOR COMMIT cc81fcdb81da4e848ad649f32d30611d02f7cd5a  THIS WILL EVENTUALLY BE 
MODIFIED TO BE USED WITH THE NEW CODE, BUT THERE MIGHT CURRENTLY BE SOME
MISMATCHED CODE.  THIS IS STILL BEING ADDED/TRACKED BECAUSE IT IS A HELPFUL
FRAMEWORK

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
sys.path.append('/home/dsouthall/Projects/StationConfigurationTesting/')
import gnosim.utils.constants
import gnosim.utils.bayesian_efficiency
import gnosim.earth.greenland
import gnosim.trace.refraction_library
from matplotlib.colors import LogNorm
pylab.ion()


def volumetricAcceptance(reader,electric_field_threshold, verbose = True):
    '''
    Calculates the volumetric acceptance of a run in # km^3 sr
    '''
    VA_sum = numpy.sum(reader['p_earth'][...] \
                                  * (gnosim.earth.greenland.density(reader['z_0'][...]) / gnosim.utils.constants.density_water)
                                  * (numpy.greater(reader['electric_field'][...] , electric_field_threshold)) )
    VA  = VA_sum * reader.attrs['geometric_factor'] / float(reader['p_interact'].shape[0]) * gnosim.utils.constants.km_to_m**-3 # km^3 sr
    error = VA/numpy.sqrt(VA_sum)
    if verbose == True:
        print( sum((reader['electric_field'][...] > electric_field_threshold) ),  '/' , float(reader['p_interact'].shape[0]) , 'events detected')
        print( 'error:', error)
        print( '% error:', 100.0 * error/VA)
    return VA, error
                                  
def electricFieldThreshold( snr , gain, temperature, bandwidth, frequency, verbose = True):
    '''
    Resistance assumed to be 50 Ohms
    Temperature (K)
    Bandwidth (GHz)
    Gain (dBi)
    frequency (GHz)
    Calculates the electric field threshold using the temperature, resistance, and BW
    to determine a V_RMS, and the signal to noise, gain, and frequency to calculate
    the electric field using the antenna factor formula.

    Currently uncertain about if the gain is being correctly utilized in this formula.
    If an array of frequencies is given this should be able to output an array of 
    correspondng thresholds.  This would be helpful if the threshold should be event
    by event depending on the dominant E field frequency. 
    '''
    V_rms = numpy.sqrt(gnosim.utils.constants.boltzmann * temperature * 50.0 * bandwidth * gnosim.utils.constants.GHz_to_Hz)
    thresh = snr * V_rms * 9.73 * frequency / (gnosim.utils.constants.speed_light * gnosim.utils.rf.amplitude(gain))  #f/c gives 1/m because f (GHz) and c (m/ns)
    if verbose == True:
        print('V_rms = ', V_rms*1000000,'uV')
        print('Bandwidth = %.2f MHz'%(bandwidth*1000))
        print('Thresh = %0.3e V/m'%(thresh))
    return thresh
def getInfo(reader,snr,gain,frequency, verbose = True):
    #reader = h5py.File(setup['reader_file'], 'r')
    config = yaml.load(open(reader.attrs['config']))
    energy_neutrino = reader['energy_neutrino'][0] #GeV
    if frequency == 0:
        frequency = reader['weighted_freq'][...]
    if verbose == True:
        print('\nDepth:', config['stations']['positions'][0][2], 'm')
        print('Gain:',gain,'dBi')
        print('SNR:', snr)
        if (numpy.size(frequency) == 1):
            print('Trigger frequency:',frequency*1000,'MHz')
        else:
            print('Mean Trigger frequency:',numpy.mean(frequency)*1000,'MHz')
        print('Neutrio Energy:',energy_neutrino,'GeV')
        print('Number of events:',len(reader['t'][...]))
    
    thresh = electricFieldThreshold( snr , gain, config['antenna_definitions']['simple']['temp'], config['antenna_definitions']['simple']['frequency_high'] - config['antenna_definitions']['simple']['frequency_low'], frequency,verbose = verbose)
    VA, error = volumetricAcceptance(reader,thresh,verbose = verbose)
    if verbose == True:
        if numpy.size(thresh == 1):
            print('Electric field threshold =', thresh*1000,'mV/m')
        else:
            print('Minimum Electric field threshold applied =', min(thresh)*1000,'mV/m')
            print('Maximum Electric field threshold applied =', max(thresh)*1000,'mV/m')
        print('Volumetric acceptance =', VA,'km^3 sr')
    return VA, error, numpy.mean(thresh), energy_neutrino

############################
bandwidths = {'50-200MHz':{'in_dir':'/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/VA_50_200MHz/Merged/','out_dir':'/home/dsouthall/Projects/StationConfigurationTesting/Output/50_200MHz/'},\
              '200-800MHz':{'in_dir':'/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/VA_200_800MHz/Merged/','out_dir':'/home/dsouthall/Projects/StationConfigurationTesting/Output/200_800MHz/'}}
#bandwidths = {'50-200MHz':{'in_dir':'/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/VA_50_200MHz/Merged/','out_dir':'/home/dsouthall/Projects/StationConfigurationTesting/Output/50_200MHz/'}}
#bandwidths = {'200-800MHz':{'in_dir':'/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/VA_200_800MHz/Merged/','out_dir':'/home/dsouthall/Projects/StationConfigurationTesting/Output/200_800MHz/'}}

#PUT FREQ = 0 TO USE ARRAY OF FREQUENCIES FROM SIM FILE
freq = 0#0.2 #GHz
gains = {'-2':[2.0,10.0],'-5':[2.0,10.0],'-60':[2.0],'-100':[2.0]}#{'-2':[2.0,6.0,10.0],'-30':[2.0],'-100':[2.0]}
depths = [-2,-5,-60,-100]#[-2,-30,-100]
SNRs = [1.5,2.0,2.3,3.0,4.0]
plot_bool = True
save_data = True
verbose = True
do_calculations = True # if false, uses the out directory to load in previously generated out files of the same naming scheme
skip_old = True #Skips any set of calculations for which there is already an existing output file
plot_ratios = True
#THe three below select which curve to normalize to if plot_ratios = True

norm_gain = 2.0
norm_SNR = 2.0
norm_depth = -100
norm_bandwidth_label = '200-800MHz'

########################################################################
#sys.stdout = open('/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/output_VA_50_200MHz.txt', 'w')
#sys.stdout = open('/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/output_VA_200-800MHz.txt', 'w')

norm_name = bandwidths[norm_bandwidth_label]['out_dir'] +'volumetric_acceptance_data_z0_%i_gain_%s_SNR_%s.h5'%(norm_depth, str(norm_gain).replace('.','p'), str(norm_SNR).replace('.','p'))

if numpy.logical_and(plot_ratios == True, do_calculations == True):
    if os.path.isfile(norm_name) == False:
        print('Trying to do calculations while also plotting ratios.  It is recommended\nto only perform ratio caculation afterward.  Setting plot_ratios = False')
        print('File you wish to normalize to is not yet created:')
        print(norm_name)
        plot_ratios = False
    
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