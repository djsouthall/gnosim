#!/usr/bin/env python3
'''
Purpose and Description
-----------------------
This analysis script computes the volumetric acceptance for a set of simulations.  The current version
will calculate a different volumetric acceptance for each file in the directory given, so if you wish 
to calculate VA for many simulations at a single energy simultaneously then you should use quick_merge
first to combine them into a single file.  This script also currently also supports the comparison with
other volumetric acceptance curves created say, between different configurations, or data from papers.
This feature is not particularly generic however and may require some tinkering.

Running
-------
It is recommended that you run this in a command line interface like ipython with the %run command or using the 
command exec(open('./script_path/script.py').read()) .

Main Parameters
---------------
calculate_data : bool
    Enables the actual calculations to be performed.  If you have already done them and saved them
    then this should be set to False and the data should be loaded in.
in_path : str
    Where the simulation output files are located that you want to perform calculations on.
name : str
    An identifier to add to the output volumetric acceptance data file name.
outdir : str
    The directory you wish to save the volumetric acceptance values in.
outname : str
    The name out the volumetric acceptance values file.
infiles : str
    The list of files for which to calculate volumetric acceptance.
plot_data : bool
    Enables the plotting of VA.  This should be set to True after the values are actually calculated and saved.
label : str
    The label in plots corresponding to the data that dataname points to.
dataname : str
    The location of a set of volumetric acceptance values.  Often this is the most recent set of VA calculations, which
    will be compared to older VA calculations.
plot_paper_comparison : bool
    This will also add the data from the Design and Performance Paper paper.
paper_comparison_file : str
    This is the path of the papers data.  This data is NOT formatted in the same way as the data that this script produces
    and is treated differently to account for this.
plot_self_comparison : bool
    Enables the plotting of some specified older version of volumetric acceptance values as calculated by this code.
self_comparison_file : str
    The list of other volumetric acceptance data to plot.
label_compare : str
    The label for this data in plots.
plot_ratios : bool
    Will attempt to plot the ratios of volumetric acceptance where energies line up.


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
import gnosim.utils.constants
import gnosim.utils.bayesian_efficiency
import gnosim.earth.ice
from matplotlib.colors import LogNorm
pylab.ion()


def volumetricAcceptance(reader,verbose = True):
    '''
    Calculates the volumetric acceptance of a run in # km^3 sr.

    Parameters
    ----------
    reader : HDF5 file
        This is the HDF5 file for the simulation (already opened).  To obtain this from an
        address of a file, run the command:
            reader = h5py.File( PATH_TO/FILENAME.h5 , 'r')
        This will open the file in reader mode.

    verbose : bool, optional
        This enables print statements.  (Default is True).

    Returns
    -------
    VA : float
        The calculated volumetric acceptance given in units of km^3 sr.
    error : float
        The calculated error for the volumetric acceptance given in units of km^3 sr.
    energy_neutrino : float
        The energy of the neutrino for which this volumetric acceptance was calcualted.  Given in GeV.
    '''
    if numpy.isin('config',list(reader.attrs)):
        config = yaml.load(open(reader.attrs['config']))
    elif numpy.isin('config_0',list(reader.attrs)):
        config = yaml.load(open(reader.attrs['config_0']))
    else:
        print('Config file not found...')
        pass
        
    if numpy.isin('geometric_factor_0',list(reader.attrs)):
        geometric_factor = reader.attrs['geometric_factor_0']
    elif numpy.isin('geometric_factor',list(reader.attrs)):
        geometric_factor = reader.attrs['geometric_factor']
    else:
        print('geometric_factor file not found...')
        pass

    if numpy.isin('ice_model_0',list(reader.attrs)):
        ice_model = reader.attrs['ice_model_0']
    elif numpy.isin('ice_model',list(reader.attrs)):
        ice_model = reader.attrs['ice_model']
    else:
        print('ice_model file not found...')
        pass
    
    if verbose == True:
        print('Loading relevant parts of info')
    info = numpy.unique(reader['info']['eventid','triggered']) #This will reduce info to just eventid and if a station triggered or didn't.  There could be multiples of events if one station triggers and another doesn't.
    unique_arr, indices, counts = numpy.unique(info['eventid'],return_index=True,return_counts=True) #Indices and counts can be used to determine which are multiples, so I can ensure they are set to Triggered, as I want the event to count as detected if a single station detects it.
    info['triggered'][indices[counts>1]] = True 
    #If any station triggered then event is triggered.  Counts would only be greater than 1 if two unique 
    #answers present (True and False) for triggered in the same event.  Which could occur for multiple 
    #stations, one triggering one not.  Events that have 1 and already triggered don't need to be altered.
    info = info[indices] #Now it is unique, but each triggered weights are appropriately set.
    #TODO: The above code is untested with multiple stations, should test.

    n_events = len(info)
    
    if verbose == True:
        print('Loading z_0')
    z_0 = reader['z_0'][...]
    
    if verbose == True:
        print('Loading p_earth')
    p_earth = reader['p_earth'][...]
    
    if verbose == True:
        print('Calculating VA_sum')
    ice = gnosim.earth.ice.Ice(ice_model,suppress_fun = True)
    VA_sum = numpy.sum(p_earth * (ice.density(z_0) / gnosim.utils.constants.density_water) * (info['triggered']) )
    VA  = ((geometric_factor / gnosim.utils.constants.km_to_m**3)/n_events) * VA_sum # km^3 sr
    error = VA/numpy.sqrt(VA_sum)
    energy_neutrino = reader['energy_neutrino'][0]
    
    if verbose == True:
        print( 'E = %0.3g GeV'%energy_neutrino)
        print( 'VA = %0.4g km^3 sr +/- %0.4g km^3 sr'%(VA, error))
        print( 'VA = %0.4g km^3 sr +/- %0.4g percent'%(VA, 100.0 * error/VA))

    return VA, error, energy_neutrino

############################
if __name__ == "__main__":
    ###------------------------###
    ### Calculation Parameters ###
    ###------------------------###

    calculate_data  = False
    in_path     = '/home/dsouthall/scratch-midway2/May3/'
    name        = 'new_polarization' 
    outdir      = in_path
    outname     = outdir +'volumetric_acceptance_data_%s.h5'%(name)

    #Below is the list of files for which you wish to calcualt VA.  
    #Either make an array over your own or choose the wildcard string appropriately.
    infiles     = glob.glob(in_path + 'results*.h5')


    ###---------------------###
    ### Plotting Parameters ###
    ###---------------------###

    plot_data = True
    label     = 'GNOSim - Current'
    dataname  = outname

    plot_paper_comparison = True
    paper_comparison_file = '/home/dsouthall/Projects/GNOSim/gnosim/analysis/DesignPerformancePaperData.py'

    plot_self_comparison  = True
    self_comparison_file  = '/home/dsouthall/scratch-midway2/April9/volumetric_acceptance_data_real_config_antarctica_180_rays_signed_fresnel.h5'
    label_compare         = 'GNOSim Old - No polarization'

    plot_ratios = True


    ###--------------###
    ### Calculations ###
    ###--------------###

    if calculate_data == True:
        print('Attempting to calculate volumetric acceptance for the following files:')
        for infile in infiles:
            print(infile)

        if os.path.exists(outname):
            print('Output file with name %s already exists, saving appending \'_new\' if necessary'%(outname))
            while os.path.exists(outname):
                outname = outname.replace('.h5','_new.h5')
        print('Saving as %s'%outname)
        writer = h5py.File(outname , 'w')

        writer.create_dataset('volumetric_acceptance', (len(infiles),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)  
        writer.create_dataset('error', (len(infiles),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)  
        writer.create_dataset('energy_neutrino', (len(infiles),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)  

        for index, infile in enumerate(infiles):
            print('Loading reader:\n%s'%infile)
            reader = h5py.File(infile , 'r')
            VA, error, energy_neutrino = volumetricAcceptance(reader,verbose = True)
            reader.close()
            writer['volumetric_acceptance'][index] = VA
            writer['error'][index] = error
            writer['energy_neutrino'][index] = energy_neutrino
         
        writer.close()


    ###----------###
    ### Plotting ###
    ###----------###
    
    if numpy.logical_or(plot_data,plot_ratios):
        markersize = 10
        linewidth = 5
        reader = h5py.File(dataname , 'r')
        energy_neutrino = reader['energy_neutrino'][...]
        
        VA = reader['volumetric_acceptance'][...]
        error = reader['error'][...]
        
        cut = VA != 0 
        sorted_cut = numpy.argsort(energy_neutrino)[numpy.isin(numpy.argsort(energy_neutrino),numpy.where(cut)[0])]

        if plot_paper_comparison == True:
            compare_data = yaml.load(open(paper_comparison_file))

        if plot_self_comparison == True:
            reader_compare = h5py.File(self_comparison_file , 'r')
            energy_neutrino_compare = reader_compare['energy_neutrino'][...]
        
            VA_compare = reader_compare['volumetric_acceptance'][...]
            error_compare = reader_compare['error'][...]
            
            cut_compare = VA_compare != 0 
            sorted_cut_compare = numpy.argsort(energy_neutrino_compare)[numpy.isin(numpy.argsort(energy_neutrino_compare),numpy.where(cut_compare)[0])]

    if plot_data:
        fig = pylab.figure(figsize=(16.,11.2))
        
        if plot_ratios == True:
            ax = pylab.subplot(2,1,1)
        else:
            ax = pylab.gca
        
        if plot_paper_comparison == True:
            compare_data = yaml.load(open(paper_comparison_file))
            for key in compare_data.keys():
                x = numpy.array(compare_data[key]['x'])
                y = numpy.array(compare_data[key]['y'])
                pylab.errorbar(x,y, yerr=None,fmt=compare_data[key]['style'],label=compare_data[key]['label'],capsize=5,markersize=markersize,linewidth=linewidth)
        
        if plot_self_comparison == True:
            pylab.errorbar(energy_neutrino_compare[sorted_cut_compare]/1e6, VA_compare[sorted_cut_compare], yerr=error_compare[sorted_cut_compare],color='orange',fmt='o-',label=label_compare,capsize=5,markersize=markersize,linewidth=linewidth)

        pylab.errorbar(energy_neutrino[sorted_cut]/1e6, VA[sorted_cut], yerr=error[sorted_cut],fmt='go-',label=label,capsize=5,markersize=markersize,linewidth=linewidth)
                    
            
        pylab.legend(loc = 'lower right',fontsize = 9)
        ax = pylab.gca()
        ax.minorticks_on()
        ax.set_xscale('log')
        ax.set_yscale('log')
        pylab.tick_params(labelsize=16)
        pylab.xlabel('Neutrino Energy (PeV)',fontsize = 20)
        pylab.ylabel('V$\Omega$ (km$^3$ sr)',fontsize = 20)
        pylab.grid(b=True, which='major', color='k', linestyle='-')
        pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            
    if plot_ratios:

        if plot_data:
            ax = pylab.subplot(2,1,2, sharex = ax)
        else:
            fig = pylab.figure(figsize=(16.,11.2))
            ax = fig.gca()

        
        if plot_paper_comparison == True:
            plot_paper_comparison = True
            compare_data = yaml.load(open(paper_comparison_file))
            rounded_energy_neutrinos_sorted = numpy.zeros_like(energy_neutrino[sorted_cut])
            for i,en in enumerate(energy_neutrino[sorted_cut]):
                rounded_energy_neutrinos_sorted[i] = float('%0.1g'%(en/1e6)) #rounds to 3 sig figs

            for key in compare_data.keys():
                x = numpy.array(compare_data[key]['x'])
                rounded_x = numpy.zeros_like(x)
                for i,en in enumerate(x):
                    rounded_x[i] = float('%0.1g'%(en)) #rounds to 3 sig figs
                y = numpy.array(compare_data[key]['y'])
                y = y[numpy.isin(rounded_x,rounded_energy_neutrinos_sorted)]
                x = x[numpy.isin(rounded_x,rounded_energy_neutrinos_sorted)]
                rounded_x = rounded_x[numpy.isin(rounded_x,rounded_energy_neutrinos_sorted)]
                y = numpy.divide(y,VA[sorted_cut][numpy.isin(rounded_energy_neutrinos_sorted,rounded_x)])
                pylab.errorbar(x,y, yerr=None,fmt=compare_data[key]['style'],label=compare_data[key]['label'],capsize=5,markersize=markersize,linewidth=linewidth)
        
        if plot_self_comparison == True:
            plot_self_comparison = True
            reader_compare = h5py.File(self_comparison_file , 'r')
            energy_neutrino_compare = reader_compare['energy_neutrino'][...]
            
            VA_compare = reader_compare['volumetric_acceptance'][numpy.isin(energy_neutrino_compare,energy_neutrino[sorted_cut])]
            error_compare = reader_compare['error'][numpy.isin(energy_neutrino_compare,energy_neutrino[sorted_cut])]
            energy_neutrino_compare = energy_neutrino_compare[numpy.isin(energy_neutrino_compare,energy_neutrino[sorted_cut])]
            cut_compare = VA_compare != 0 
            sorted_cut_compare = numpy.argsort(energy_neutrino_compare)[numpy.isin(numpy.argsort(energy_neutrino_compare),numpy.where(cut_compare)[0])]
            ratio_compare = numpy.divide(VA_compare[sorted_cut_compare],VA[sorted_cut])
            
            rel_err = numpy.sqrt(numpy.divide(error[sorted_cut],VA[sorted_cut])**2 + numpy.divide(error_compare[sorted_cut_compare],VA_compare[sorted_cut_compare])**2)

            pylab.errorbar(energy_neutrino_compare[sorted_cut_compare]/1e6, ratio_compare, yerr=numpy.multiply(ratio_compare,rel_err),color='orange',fmt='o-',label=label_compare,capsize=5,markersize=markersize,linewidth=linewidth)

        pylab.errorbar(energy_neutrino[sorted_cut]/1e6, numpy.ones_like(VA[sorted_cut]), yerr=None,fmt='go-',label=label,capsize=5,markersize=markersize,linewidth=linewidth)
                    
        if not plot_data:
            pylab.legend(loc = 'upper right',fontsize = 16)
        ax = pylab.gca()
        ax.minorticks_on()
        ax.set_xscale('log')
        ax.set_yscale('log')
        pylab.tick_params(labelsize=16)
        pylab.xlabel('Neutrino Energy (PeV)',fontsize = 20)
        pylab.ylabel('Ratios of V$\Omega$',fontsize = 20)
        pylab.grid(b=True, which='major', color='k', linestyle='-')
        pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            
            
            
            
            
            
            
            
