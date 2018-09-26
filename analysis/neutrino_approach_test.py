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
                                  
def electricFieldThreshold( snr , gain, temperature, bandwidth, frequency, verbose = True):
    '''
    Resistance assumed to be 50 Ohms
    Temperature (K)
    Bandwidth (GHz)
    Gain (dBi)
    frequency (GHz)
    Calculates the electric field threshold using the temperature, resistacne, and BW
    to determine a V_RMS, and the signal to noise, gain, and frequency to calculate
    the electric field using the antenna factor formula.

    Currently uncertain about if the gain is being correctly utilized in this formula.
    If an array of frequencies is given this should be able to output an array of 
    correspondng thresholds.  This would be helpful if the threshold should be event
    by evet depending on the dominant E field frequency. 
    '''
    V_rms = numpy.sqrt(gnosim.utils.constants.boltzmann * temperature * 50.0 * bandwidth * gnosim.utils.constants.GHz_to_Hz)
    if verbose == True:
        print('V_rms = ', V_rms*1000000,'uV')
        print('Bandwidth = %.2f MHz'%(bandwidth*1000))
    return snr * V_rms * 9.73 * frequency / (gnosim.utils.constants.speed_light * gnosim.utils.rf.amplitude(gain)) #f/c gives m because f (GHz) and c (m/ns)

def getInfo(reader,snr,gain,frequency, verbose = True,hist_bins = 180,hist_range = (0.0,180.0)):
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
            print('Mean Trigger frequency:',numpy.mean(frequency[frequency > 0])*1000,'MHz')
        print('Neutrio Energy:',energy_neutrino,'GeV')
        print('Number of events:',len(reader['t'][...]))
    
    thresh = electricFieldThreshold( snr , gain, config['antenna_definitions']['simple']['temp'], config['antenna_definitions']['simple']['frequency_high'] - config['antenna_definitions']['simple']['frequency_low'], frequency,verbose = verbose)
    arrived = reader['p_earth'][...]
    density_factor = (gnosim.earth.greenland.density(reader['z_0'][...]) / gnosim.utils.constants.density_water)
    detected = reader['electric_field'][...] > thresh
    theta_weight = numpy.multiply(numpy.multiply(arrived,detected),density_factor)
    theta_ant = reader['theta_ant'][...]
    theta_0 = reader['theta_0'][...]
    theta_0_hist,edges = numpy.histogram(theta_0,bins = hist_bins,range = hist_range,weights = theta_weight)
    theta_ant_hist = numpy.histogram(theta_ant,bins = hist_bins,range = hist_range,weights = theta_weight)[0]
    theta_theta_hist, theta_hist_xedges, theta_hist_yedges = numpy.histogram2d(theta_ant,theta_0,bins = (hist_bins,hist_bins),range = (hist_range,hist_range),weights = theta_weight)
    theta = (edges[1:] + edges[:-1]) / 2.0
    if verbose == True:
        if numpy.size(thresh == 1):
            print('Electric field threshold =', thresh*1000,'mV/m')
        else:
            print('Minimum Electric field threshold applied =', min(thresh)*1000,'mV/m')
            print('Maximum Electric field threshold applied =', max(thresh)*1000,'mV/m')
    return theta_0_hist,theta_ant_hist, theta_theta_hist, theta_hist_xedges ,  theta_hist_yedges , theta, numpy.mean(frequency[frequency > 0])*1000, energy_neutrino

############################
#'''
bandwidths = {'50-200MHz':{'in_dir':'/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/VA_50_200MHz/Merged/','out_dir':'/home/dsouthall/Projects/StationConfigurationTesting/Output/50_200MHz/Neutrino_Approach/'},\
              '200-800MHz':{'in_dir':'/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/VA_200_800MHz/Merged/','out_dir':'/home/dsouthall/Projects/StationConfigurationTesting/Output/200_800MHz/Neutrino_Approach/'}}
#'''
#bandwidths = {'50-200MHz':{'in_dir':'/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/VA_50_200MHz/Merged/','out_dir':'/home/dsouthall/Projects/StationConfigurationTesting/Output/50_200MHz/Neutrino_Approach/'}}
#bandwidths = {'200-800MHz':{'in_dir':'/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/VA_200_800MHz/Merged/','out_dir':'/home/dsouthall/Projects/StationConfigurationTesting/Output/200_800MHz/Neutrino_Approach/'}}

#PUT FREQ = 0 TO USE ARRAY OF FREQUENCIES FROM SIM FILE
freq = 0#0.2 #GHz
gains = [2.0,10.0]
depths = {'2.0':[-2.0,-5.0,-60,-100.0],'10.0':[-2.0,-5.0]}
color_key = {'2.0':{'-2.0':'C0','-5.0':'C2','-60.0':'C4','-100.0':'C5'},'10.0':{'-2.0':'C1','-5.0':'C3'}} #color_key[gain][depth]
SNRs = [2.0]#[1.5,2.0,2.3,3.0,4.0]
plot_bool = True
save_data = False

verbose = True
do_calculations = False # if false, uses the out directory to load in previously generated out files of the same naming scheme

#THe three below select which curve to normalize to if plot_ratios = True
hist_bins = 360
hist_range = (0,180.0)

########################################################################
#sys.stdout = open('/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/output_VA_50_200MHz.txt', 'w')
#sys.stdout = open('/project2/avieregg/dsouthall/gnosim_output/volumetric_acceptance/output_VA_200-800MHz.txt', 'w')



for SNR in SNRs:
    print('Analyzing data for SNR:',SNR)
    for bandwidth_label in list(bandwidths.keys()):
        outname = bandwidths[bandwidth_label]['out_dir'] +'neutrino_approach_data_SNR_%s_BW_%s.h5'%( str(SNR).replace('.','p'),bandwidth_label)
        if do_calculations:
            threshold = numpy.array([])
            energy_neutrino = numpy.array([])
            theta_neutrino_hist = numpy.array([])
            theta_ant_hist = numpy.array([])
            theta = numpy.array([])
            z_0 = numpy.array([])
            gain_array = numpy.array([])
            for gain in gains:
                print('Analyzing data for gain:',gain)
                for depth in depths[str(gain)]:
                    print('Analyzing data for Depth:',depth)
                    infiles = glob.glob(bandwidths[bandwidth_label]['in_dir'] +'*_%i*.h5'%(depth))
                    theta_theta_hist = numpy.zeros((hist_bins,hist_bins))
                    for infile in infiles:#[infiles[0]]:
                        print('On file:',infile)
                        reader = h5py.File(infile, 'r')
                        theta_neutrino_hist_out , theta_ant_hist_out, theta_theta_hist_out, theta_theta_hist_out_xedges, theta_theta_hist_out_yedges, theta_out, threshold_out, energy_neutrino_out = getInfo(reader,SNR,gain,freq,hist_bins = hist_bins,verbose=verbose)
                        theta_theta_hist = numpy.add(theta_theta_hist,theta_theta_hist_out)
                        threshold = numpy.append(threshold,[threshold_out]*len(theta_neutrino_hist_out))
                        z_0 = numpy.append(z_0,[depth]*len(theta_neutrino_hist_out))
                        energy_neutrino = numpy.append(energy_neutrino,[energy_neutrino_out]*len(theta_neutrino_hist_out))
                        theta_neutrino_hist = numpy.append(theta_neutrino_hist,theta_neutrino_hist_out)
                        theta_ant_hist = numpy.append(theta_ant_hist,theta_ant_hist_out)
                        theta = numpy.append(theta,theta_out)
                        gain_array = numpy.append(gain_array,[gain]*len(theta_neutrino_hist_out))
                        reader.close()
                    if plot_bool:
                        fig = pylab.figure()
                        pylab.title('Source Angle v.s. Signal Arrival Angles\nwith Depth = %0.2fm Gain = %0.2f SNR = %0.1f (%s)'%(depth,gain,SNR,bandwidth_label),fontsize = 30)
                        
                        pylab.xlabel('Signal Arrival Angles\n$\\theta_{antenna}$ (deg)',fontsize = 20)
                        pylab.ylabel('Direction of Neutrino Source\n$\\theta_0$ (deg)',fontsize = 20)
                        
                        X,Y = numpy.meshgrid(theta_theta_hist_out_xedges,theta_theta_hist_out_yedges)
                        pylab.pcolormesh(Y,X,theta_theta_hist.T,norm=LogNorm())
                        pylab.colorbar()
                if numpy.logical_and(do_calculations,save_data):
                    print('Creating outfile:',outname)
                    outfile  = h5py.File(outname, 'w')
                    outfile.attrs['SNR'] = SNR
                    
                    outfile.create_dataset('theta_neutrino_hist', (len(theta_neutrino_hist),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                    outfile.create_dataset('theta_antenna_hist', (len(theta_neutrino_hist),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                    outfile.create_dataset('mean_electric_threshold', (len(theta_neutrino_hist),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                    outfile.create_dataset('energy_neutrino', (len(theta_neutrino_hist),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                    outfile.create_dataset('theta_0', (len(theta_neutrino_hist),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                    outfile.create_dataset('z_0', (len(theta_neutrino_hist),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                    outfile.create_dataset('gain', (len(theta_neutrino_hist),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                    outfile['theta_neutrino_hist'][...] = theta_neutrino_hist
                    outfile['theta_antenna_hist'][...] = theta_ant_hist
                    outfile['mean_electric_threshold'][...] = threshold
                    outfile['energy_neutrino'][...] = energy_neutrino
                    outfile['theta_0'][...] = theta
                    outfile['z_0'][...] = z_0
                    outfile['gain'][...] = gain_array
                    outfile.close()
        else:
            #Things not doing calculations
            reader = h5py.File(outname, 'r')
            theta_neutrino_hist = reader['theta_neutrino_hist'][...]
            theta_ant_hist = reader['theta_antenna_hist'][...]
            threshold = reader['mean_electric_threshold'][...]
            energy_neutrino = reader['energy_neutrino'][...]
            theta = reader['theta_0'][...]
            z_0 = reader['z_0'][...]
            gain_array = reader['gain'][...]
            reader.close()
                    
                     
        if plot_bool:
            
            #Neutrino Source Hist Plot
            fig = pylab.figure()
            pylab.title('Sensitivity to Neutrino Source Angles\nwith SNR = %0.1f (%s)'%(SNR,bandwidth_label),fontsize = 30)
            ax = pylab.gca()
            pylab.tick_params(labelsize=16)
            pylab.xlabel('Neutrino Energy (GeV)',fontsize = 20)
            pylab.ylabel('Direction of Neutrino Source\n$\\theta_0$ (deg)',fontsize = 20)
            #ax.set_xscale('log')
            #ax.set_yscale('log')
            
            offset = -1.0 
            label_once = {'2.0':{'-2.0':False,'-5.0':False,'-60.0':False,'-100.0':False},'10.0':{'-2.0':False,'-5.0':False}}
            depths = {'2.0':[-2.0,-5.0,-60.0,-100.0],'10.0':[-2.0,-5.0]}
            pylab.xticks(numpy.arange(len(numpy.unique(energy_neutrino))), ['%.2e'%(energy) for energy in numpy.sort(numpy.unique(energy_neutrino))] ,rotation=30)
            pylab.yticks(numpy.arange(0.0,181.0,20.0), numpy.arange(0.0,181.0,20.0))
            for energy in numpy.sort(numpy.unique(energy_neutrino)):
                offset = offset + 1.0
                energy_cut = (energy_neutrino == energy)
                if max(theta_neutrino_hist[energy_cut]) == 0.0:
                    continue
                for depth in numpy.sort(numpy.unique(z_0))[::-1]:
                    depth_cut = (z_0 == depth)
                    for gain in numpy.sort(numpy.unique(gain_array)):
                        if numpy.isin(depth,depths[str(gain)]):
                            gain_cut = (gain_array == gain)
                            cut = numpy.logical_and(gain_cut ,numpy.logical_and(depth_cut,energy_cut))
                            if label_once[str(gain)][str(depth)] == False:
                                label = 'Depth: %0.1fm Gain: %0.1fdBi'%(depth,gain)
                                pylab.plot( numpy.array([offset]*sum(cut)) + 0.9*(theta_neutrino_hist[cut]/max(theta_neutrino_hist[energy_cut])) , theta[cut],label = label,c = color_key[str(gain)][str(depth)])
                                label_once[str(gain)][str(depth)] = True
                            else:
                                pylab.plot( numpy.array([offset]*sum(cut)) + 0.9*(theta_neutrino_hist[cut]/max(theta_neutrino_hist[energy_cut])) , theta[cut],c = color_key[str(gain)][str(depth)])
                                        
                            pylab.legend(loc = 'upper left',fontsize = 16)
                            pylab.ylim(0.0,220.0)
                            pylab.grid(b=True, which='major', color='k', linestyle='-')
                            pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                            
            #Signal Angle Hist Plot
            fig = pylab.figure()
            pylab.title('Sensitivity for Signal Arrival Angles\nwith SNR = %0.1f (%s)'%(SNR,bandwidth_label),fontsize = 30)
            ax = pylab.gca()
            pylab.tick_params(labelsize=16)
            pylab.xlabel('Neutrino Energy (GeV)',fontsize = 20)
            pylab.ylabel('Signal Arrival Angles\n$\\theta_{antenna}$ (deg)',fontsize = 20)
            #ax.set_xscale('log')
            #ax.set_yscale('log')
            
            offset = -1.0 
            label_once = {'2.0':{'-2.0':False,'-5.0':False,'-60.0':False,'-100.0':False},'10.0':{'-2.0':False,'-5.0':False}}
            pylab.xticks(numpy.arange(len(numpy.unique(energy_neutrino))), ['%.2e'%(energy) for energy in numpy.sort(numpy.unique(energy_neutrino))] ,rotation=30)
            pylab.yticks(numpy.arange(0.0,181.0,20.0), numpy.arange(0.0,181.0,20.0))
            for energy in numpy.sort(numpy.unique(energy_neutrino)):
                offset = offset + 1.0
                energy_cut = (energy_neutrino == energy)
                if max(theta_ant_hist[energy_cut]) == 0.0:
                    continue
                for depth in numpy.sort(numpy.unique(z_0))[::-1]:
                    depth_cut = (z_0 == depth)
                    for gain in numpy.sort(numpy.unique(gain_array)):
                        if numpy.isin(depth,depths[str(gain)]):
                            gain_cut = (gain_array == gain)
                            cut = numpy.logical_and(gain_cut ,numpy.logical_and(depth_cut,energy_cut))
                            if label_once[str(gain)][str(depth)] == False:
                                label = 'Depth: %0.1fm Gain: %0.1fdBi'%(depth,gain)
                                pylab.plot( numpy.array([offset]*sum(cut)) + 0.9*(theta_ant_hist[cut]/max(theta_ant_hist[energy_cut])) , theta[cut],label = label,c = color_key[str(gain)][str(depth)])
                                label_once[str(gain)][str(depth)] = True
                            else:
                                pylab.plot( numpy.array([offset]*sum(cut)) + 0.9*(theta_ant_hist[cut]/max(theta_ant_hist[energy_cut])) , theta[cut],c = color_key[str(gain)][str(depth)])
                                        
                            pylab.legend(loc = 'upper left',fontsize = 16)
                            pylab.ylim(0.0,220.0)
                            pylab.grid(b=True, which='major', color='k', linestyle='-')
                            pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            ''' 
            #2d Hist Plot
            for energy in numpy.sort(numpy.unique(energy_neutrino)):
                energy_cut = (energy_neutrino == energy)
                all_depths = numpy.histogram2d(theta_ant_hist[energy_cut],theta_neutrino_hist[energy_cut],bins=180,range=(0.0,180.0))
                
                fig, axes = plt.subplots(nrows=len(numpy.unique(z_0)), ncols=1)
                pylab.title('Source Angle v.s. Signal Arrival Angles\nwith SNR = %0.1f and E = %0.2e GeV (%s)'%(SNR,energy,bandwidth_label),fontsize = 30)
                
                pylab.xlabel('Direction of Neutrino Source\n$\\theta_0$ (deg)',fontsize = 20)
                pylab.ylabel('Signal Arrival Angles\n$\\theta_{antenna}$ (deg)',fontsize = 20)
                #ax.set_xscale('log')
                #ax.set_yscale('log')
                
                if max(theta_ant_hist[energy_cut]) == 0.0:
                    continue
                for depth_index,depth in numpy.sort(numpy.unique(z_0))[::-1]:
                    ax = axes.flat[depth_index]
                    depth_cut = (z_0 == depth)
                    for gain in numpy.sort(numpy.unique(gain_array)):
                        if numpy.isin(depth,depths[str(gain)]):
                            gain_cut = (gain_array == gain)
                            cut = numpy.logical_and(gain_cut ,numpy.logical_and(depth_cut,energy_cut))
                            if label_once[str(gain)][str(depth)] == False:
                                label = 'Depth: %0.1fm Gain: %0.1fdBi'%(depth,gain)
                                pylab.plot( numpy.array([offset]*sum(cut)) + 0.9*(theta_ant_hist[cut]/max(theta_ant_hist[energy_cut])) , theta[cut],label = label,c = color_key[str(gain)][str(depth)])
                                label_once[str(gain)][str(depth)] = True
                            else:
                                pylab.plot( numpy.array([offset]*sum(cut)) + 0.9*(theta_ant_hist[cut]/max(theta_ant_hist[energy_cut])) , theta[cut],c = color_key[str(gain)][str(depth)])
                                        
                            pylab.legend(loc = 'upper left',fontsize = 16)
                            pylab.ylim(0.0,220.0)
                            pylab.grid(b=True, which='major', color='k', linestyle='-')
                            pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            '''
