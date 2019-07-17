"""
Detector object
"""
import os
import sys
sys.path.append(os.environ['GNOSIM_DIR'])
import numpy
import scipy.interpolate
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import pylab
import types
import gnosim.trace.refraction_library
import gnosim.interaction.polarization
import gnosim.interaction.askaryan
import gnosim.utils.linalg
import gnosim.utils.misc
############################################################

class Station:
    '''
    Stores the attributes, information, and functions for a station of antennas.
    
    Parameters
    ----------
    x : float
        The x coordinate for the position of the station in the ice frame.  Given in m.
    y : float
        The y coordinate for the position of the station in the ice frame.  Given in m.
    z : float
        The z coordinate for the position of the station in the ice frame.  Given in m.
    config : dict
        The configuration dictionary loaded from a config file.
    station_label : str
        A label for the station, often used as a key in dictionaries during the simulation.  Should be unique for a
        particualar simulation (i.e. 'dipole0', 'dipole1', ... rather than just 'dipole').
    solutions : numpy.ndarray of str, optional
        A list of the solution types to load.  Often either all accepted solution types (the Default), or the same list omitting
        the _2 strings, which represent libraries of solution that reflect of the bottom of the ice.
    electric_field_domain : str, optional
        Selects the type of Askaryan radiation calculation to use.  
        Options:    'time'  for the newer time domain calculation that allows many more options in the simulation and is more maintained, 
                    'freq'  which is the old Askaryan calculation method which does the calculation solely in the frequency domain and 
                            is no longer maintined. 
        (Default is 'time').


    Attributes
    ----------
    x : float
        The x coordinate for the position of the station in the ice frame.  Given in m.
    y : float
        The y coordinate for the position of the station in the ice frame.  Given in m.
    z : float
        The z coordinate for the position of the station in the ice frame.  Given in m.
    config : dict
        The configuration dictionary loaded from a config file.
    label : str
        A label for the station, often used as a key in dictionaries during the simulation.  Should be unique for a
        particualar simulation (i.e. 'dipole0', 'dipole1', ... rather than just 'dipole').
    accepted_solutions : numpy.ndarray of str, optional
        A list of the possible allowable solution types to load.
    solutions : numpy.ndarray of str, optional
        A list of the solution types to load.  Often either all accepted solution types (the Default), or the same list omitting
        the _2 strings, which represent libraries of solution that reflect of the bottom of the ice.
    electric_field_domain : str, optional
        Selects the type of Askaryan radiation calculation to use.  
        Options:    'time'  for the newer time domain calculation that allows many more options in the simulation and is more maintained, 
                    'freq'  which is the old Askaryan calculation method which does the calculation solely in the frequency domain and 
                            is no longer maintined. 
        (Default is 'time').
    power_calculation_sum_length : int
        This sets the width (in bins of voltage data points) of a particular power sum window.  This is used in the beamforming calculation 
        during the power sum.  Specified in the configuration file.
    power_calculation_interval : int
        This sets the number of bins between successive sums in the power sum.  If this is less than power_calculation_sum_length then successive
        bins will overlap.  The common use case when this was written was to have the interval set to half of the sum length, so successive sums
        contain half of their values in common with the previous sum.  This is used in the beamforming calculation during the power sum.  
        Specified in the configuration file.
    beamforming_power_sum_bit_cap : int
        This sets number of bits to cap the power sum calculation (which will have units of adu^2).  This is used in gnosim.detector.fpga.fpgaBeamForming.
        Specified in the configuration file.
    n_beams : int
        The number of beams to be formed when creating a beam forming dictionary.  Specified in the configuration file.
    n_baselines : int
        This sets the number of baselines to be considered when creating the beam forming dictionary.  Currently this will automatically select the
        n_baselines that are smallest (in m).  I.e. if you had 8 antennas seperated evenly by 1m, then n_baselines = 2 would result in both the 1m
        and 2m baselines being used for subbeams.  If they 8 antennas were seperated evenly by 2m, then n_baselines = 2 would result in both the 2m
        and 4m baselines being used for subbeams.  A subbeam is created for each baseline for a given beam (assuming at least 2 antennas are seperated
        by the baseline).  Thus this parameter selects the number of subbeams to be used per beam.  Specified in the configuration file.
        Currently the minimum time shift is assigned to the smallest baseline.  Thus every other timeshift resulting from larger baselines must be a 
        multiple of the minimum baseline. i.e. all subbeam baselines must be in integer multiples of  the minimum baseline.  Currently requires all 
        other baselines to be an integer multiple of the minimum baseline.
    sampling_rate : float
        This is the sampling rate of the fpga digitizer.  It will sample from the electric fields at this rate, returning digitized signals.
        Given in GHz.  Specified in the configuration file.
    digital_sampling_period : float
        This is the sampling period of the fpga digitizer.  It will sample from the electric fields at this period, returning digitized signals.  
        Given in ns.  Calculated from sampling_rate.
    sampling_bits : int
        This sets the number of voltage bins for a digitized signal.  Signals will be digitized asymmetrically about 0 to this bit size with values
        ranging from -2**(sampling_bits-1)+1 to 2**(sampling_bits-1).
    scale_noise_to : int
        This scales the calculated 'analog' Askaryan calculations (in V) during digitization such that the noise_rms value is scale_noise_to adu.  
        The common use case is to set noise_rms to 3 adu.
    antennas : numpy.ndarray of Antenna objects
        An array containing all of the Antenna objects corresponding to this particular station.  Contains all antennas, with phased array antennas
        listed first and reconstruction antennas following. 
    antenna_keys : numpy.ndarray of str
        An array of the unique keys for each antenna in the array.
    phased_cut :  numpy.ndarray of bool
        An array of bools that can be used as a cut on antenna_keys or antennas to select only the antennas in the phased array.
    reconstruction_cut :  numpy.ndarray of bool
        An array of bools that can be used as a cut on antenna_keys or antennas to select only the antennas in the reconstruction array.
    noise_rms : numpy.ndarray float, optional
        The rms of the noise calculated for each antenna antenna.  Given in V.  Not present unless self.calculateNoiseRMS is run.
    beam_dict : dict, optional
        A dictionary containing all of the beam forming information for use by the gnosim.detector.fpga module.  Not present unless self.getBeams is run.
    beam_colors : dict, optional
        A dictionary containing colors corresponding to each beam for plotting purposes.  Not present unless self.getBeams is run.

    See Also
    --------
    Antenna
    gnosim.detector.fpga
    '''
    def __init__(self, station_key, config, solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2']), electric_field_domain = 'time'):        
        self.label = station_key
        self.config = config['stations'][self.label]
        self.x = self.config['position'][0]
        self.y = self.config['position'][1]
        self.z = self.config['position'][2]
        self.electric_field_domain = electric_field_domain
        self.accepted_solutions = gnosim.trace.refraction_library.getAcceptedSolutions()

        #TODO:  THe DAQ options below should be made into a digitizer class, along with the relevant functions.  Each antenna should then be associated with a specific digitizer.
        self.power_calculation_sum_length = self.config['DAQ']['power_calculation_sum_length']
        self.power_calculation_interval = self.config['DAQ']['power_calculation_interval']
        self.beamforming_power_sum_bit_cap = self.config['DAQ']['beamforming_power_sum_bit_cap']
        self.n_beams = self.config['DAQ']['n_beams']
        self.n_baselines = self.config['DAQ']['n_baselines']
        self.sampling_rate = self.config['DAQ']['sampling_rate_GHz']
        self.digital_sampling_period = 1.0  /  self.sampling_rate
        self.sampling_bits = self.config['DAQ']['sampling_bits']
        self.scale_noise_to = self.config['DAQ']['scale_noise_to']
        # List attributes of interest
        solutions = self.accepted_solutions[numpy.isin(self.accepted_solutions,solutions)]
        if len(solutions) == 0:
            print('Selection of solution types did not match predefined values.  Using default types.')
            self.solutions = self.accepted_solutions
            print(self.solutions)
        else:
            self.solutions = solutions

        self.buildStation() 

    def buildStation(self):
        '''
        Creates an Antenna object for each antenna specified in the configuration file.  Does not load the ray tracing libraries.  Call loadLib() for that.
        '''
        
        self.antennas = [] 
        self.antenna_keys = []
        self.phased_cut = []
        self.reconstruction_cut = []
        
        antenna_groups = numpy.append(numpy.tile('phased_antennas',(len(list(self.config['phased_antennas'].keys())),1)) , numpy.tile('reconstruction_antennas',(len(list(self.config['reconstruction_antennas'].keys())),1)))
        self.antenna_keys = numpy.append(list(self.config['phased_antennas'].keys()),list(self.config['reconstruction_antennas'].keys()))
        self.phased_cut = numpy.append(numpy.tile(True,(len(list(self.config['phased_antennas'].keys())),1)) , numpy.tile(False,(len(list(self.config['reconstruction_antennas'].keys())),1)))
        self.reconstruction_cut = numpy.logical_not(self.phased_cut)
        for index, key in enumerate(self.antenna_keys):
            x_antenna, y_antenna, z_antenna = self.config[antenna_groups[index]][key]['position']
            if numpy.isin('orientations',list(self.config[antenna_groups[index]][key].keys())) == True:                                   
                alpha_deg, beta_deg, gamma_deg =  self.config[antenna_groups[index]][key]['orientation']
            else:
                alpha_deg, beta_deg, gamma_deg = [0.0,0.0,0.0]

            antenna = Antenna(x_antenna + self.x, y_antenna + self.y, z_antenna + self.z, 
                                alpha_deg, beta_deg, gamma_deg,
                                self.config[antenna_groups[index]][key]['antenna_type'],
                                self.config[antenna_groups[index]][key]['noise_temperature'], 
                                self.config[antenna_groups[index]][key]['resistance'],
                                self.config[antenna_groups[index]][key]['lib'],
                                key,
                                solutions = self.solutions)
            
            if self.electric_field_domain == 'freq':
                antenna.setUpFrequencyDomain(self.config[antenna_groups[index]][key]['frequency_low'],self.config[antenna_groups[index]][key]['frequency_high'])
            elif self.electric_field_domain == 'time':
                antenna.addTimingInfo(os.path.expandvars(self.config[antenna_groups[index]][key]['system_response']),os.path.expandvars(self.config[antenna_groups[index]][key]['antenna_response']))
            self.antennas.append(antenna)
        self.antennas = numpy.array(self.antennas) 
        self.antenna_keys = numpy.array(self.antenna_keys)
        self.phased_cut = numpy.array(self.phased_cut)
        self.reconstruction_cut = numpy.array(self.reconstruction_cut)

    def loadLib(self,pre_split = False,build_lib = True):
        '''
        Calls the Antenna.loadLib function for each antenna which loads the gnosim.trace.refraction_library.RefractionLibrary 
        object corresponding to said antenna after cross checking the desired solutions are valid options.  If build_lib is True 
        then this will load the full ray tracing library, otherwise it just creates the RefractionLirary object which contains some 
        meta data about the library but not the actual ray tracing data.
        
        Parameters
        ----------
        pre_split : bool, optional
            Determines whether to attempt to load from pre split libraries.  If true (and the pre split libraries are calculated and 
            saved appropriately) this avoids lengthy calculations which seperate the rays ito the different solution types.  (Default is False).
        build_lib : bool, optional
            Must be called to actually populate most library information (such as the rays), however is left as an option such that 
            the rest of the library can be worked with as a less bulky object when necessary.  (Default is True).
        See Also
        --------
        gnosim.trace.refraction_library
        '''
        for antenna in self.antennas:
            antenna.loadLib(pre_split,build_lib = build_lib)
            self.solutions = self.solutions[numpy.isin(self.solutions,antenna.lib.solutions)] #will catch any inconsistencies between solutions present in library and intended list of solutions. 

    def deleteLib(self,verbose=False):
        '''
        Calls the Antenna.deleteLib function for each antenna which deletes the library object from memory 
        (to the best of its ability).  It is used after interpolation to free up RAM.
        
        Parameters
        ----------
        verbose : bool, optional
            Enables more print statements (Default is False).

        See Also
        --------
        gnosim.trace.refraction_library
        '''
        import gc
        
        if verbose == True:
            def memTools():
                '''
                Is a quickly included function to print some information about the freed up RAM.  This was straight from a stackExchange forum
                and is not exactly what I wanted, but was present for a quick test.
                '''
                mem=str(os.popen('free -t -m').readlines())
                print('%s'%mem[0])
                """
                Get a whole line of memory output, it will be something like below
                ['             total       used       free     shared    buffers     cached\n', 
                'Mem:           925        591        334         14         30        355\n', 
                '-/+ buffers/cache:        205        719\n', 
                'Swap:           99          0         99\n', 
                'Total:        1025        591        434\n']
                 So, we need total memory, usage and free memory.
                 We should find the index of capital T which is unique at this string
                """
                T_ind=mem.index('T')
                """
                Than, we can recreate the string with this information. After T we have,
                "Total:        " which has 14 characters, so we can start from index of T +14
                and last 4 characters are also not necessary.
                We can create a new sub-string using this information
                """
                mem_G=mem[T_ind+14:-4]
                """
                The result will be like
                1025        603        422
                we need to find first index of the first space, and we can start our substring
                from from 0 to this index number, this will give us the string of total memory
                """
                S1_ind=mem_G.index(' ')
                mem_T=mem_G[0:S1_ind]
                """
                Similarly we will create a new sub-string, which will start at the second value. 
                The resulting string will be like
                603        422
                Again, we should find the index of first space and than the 
                take the Used Memory and Free memory.
                """
                mem_G1=mem_G[S1_ind+8:]
                S2_ind=mem_G1.index(' ')
                mem_U=mem_G1[0:S2_ind]

                mem_F=mem_G1[S2_ind+8:]
                print( 'Summary = ' + mem_G)
                print( 'Total Memory = ' + mem_T +' MB')
                print( 'Used Memory = ' + mem_U +' MB')
                print( 'Free Memory = ' + mem_F +' MB')

                CPU_Pct=str(round(float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline()),2))

                #print( results)
                print(("CPU Usage = " + CPU_Pct))
        
        
        if verbose == True:
            print('Memory info before deleting:')
            memTools()  
        
        for antenna in self.antennas:
            antenna.deleteLib(verbose=verbose)    
        gc.collect()
        if verbose == True:
            print('Memory info after deleting:')
            memTools()  

    def loadConcaveHull(self):
        '''
        Calls the Antenna.loadConcaveHull() function for each antenna.
        
        See Also
        --------
        Antenna.loadConcaveHull
        '''
        for antenna in self.antennas:
            antenna.loadConcaveHull()

    def getBeams(self, n , verbose = False):
        '''
        This creates a dictionary containing all of the beam forming information for use by the gnosim.detector.fpga module. The goal of this 
        function is to determine the beam and subbeam time delays semiautomatically for the station. Currently the minimum time shift 
        is assigned to the smallest baseline.  Thus every other timeshift resulting from larger baselines must be a multiple of the 
        minimum baseline. i.e. all subbeam baselines must be in integer multiples of  the minimum baseline.  Currently requires all 
        other baselines to be an integer multiple of the minimum baseline.

        #TODO:  This currently will try to make the dictionary based on all antennas, and will likely break if anything but the
        anticipated linear geometry of station is given.  The antennnas should be separated into 2 categories, phased v.s. reconstruct.
        And the phased array ones must be in a line.  This should only work on those ones.  

        Parameters
        ----------
        n : float
            The index of refraction for the station (often an average of the various antenna location indices of refraction).
            This is used in calculations estimating the point direction of the beams.
        verbose : bool, optional
            Enables more print statements (Default is False).

        See Also
        --------
        gnosim.detector.fpga
        '''
        print('Using:\npower_calculation_sum_length = %i\npower_calculation_interval = %i\nn_baselines = %i'%(self.power_calculation_sum_length,self.power_calculation_interval,self.n_baselines))
        
        '''
        phased_antennas = []
        for antenna in self.antennas:
            if numpy.isin(antenna.label,self.antenna_keys[self.phased_cut]):
                phased_antennas.append(antenna)
        '''
        phased_antennas = self.antennas[self.phased_cut]
        n_antennas = len(phased_antennas)
        min_antennas_per_subbeam =  2#numpy.round(n_antennas/3) #if the number of antennas satisfying a baseline is less than this that beam won't be counted
        
        depths = numpy.array([antenna.z for antenna in phased_antennas])
        relative_antenna_depths = depths - max(depths)

        baselines = []
        for ii in relative_antenna_depths:
            for jj in relative_antenna_depths:
                baselines.append(numpy.abs(ii - jj))
        baselines = numpy.sort(numpy.unique(baselines))
        baselines = baselines[baselines!= 0][range(self.n_baselines)]
        
        antenna_list = numpy.arange(n_antennas)
        beam_dict = {'attrs' :  {'power_calculation_sum_length' : self.power_calculation_sum_length,
                                 'power_calculation_interval'   : self.power_calculation_interval},
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
        for beam_index in range(self.n_beams):
            beam_label = 'beam%i'%beam_index
            beam_dict['beams'][beam_label] = {}
            theta_ant_beam = 0
            total_ant = 0
            for subbeam_index, subbeam in enumerate(subbeam_list):
                subbeam_label = 'subbeam%i'%subbeam_index
                baseline = min(numpy.unique(numpy.abs(numpy.diff(relative_antenna_depths[subbeam]))))
                ms = numpy.arange(-self.n_beams/(2/baseline),self.n_beams/(2/baseline),baseline,dtype=int) #it is sort of sloppy to calculate this each time (only needs ot be done once per baseline) but this function is only done once so whatever.
                if baseline % min_baseline != 0:
                    continue
                    
                #print('gnosim.utils.constants.speed_light * ms[beam_index] * self.digital_sampling_period / ( n * baseline)',gnosim.utils.constants.speed_light * ms[beam_index] * self.digital_sampling_period / ( n * baseline))
                #theta_elevation = 0
                theta_elevation = numpy.rad2deg(numpy.arcsin(gnosim.utils.constants.speed_light * ms[beam_index] * self.digital_sampling_period / ( n * baseline) ))  #Double check this calculation!
                theta_ant = 90.0-theta_elevation
                #time_delays = numpy.array( ms[beam_index]  * relative_antenna_depths[subbeam],dtype=int) 
                time_delays = numpy.array( ms[beam_index]  * ((relative_antenna_depths[subbeam] - relative_antenna_depths[subbeam][0])//baseline),dtype=int) 
                beam_dict['beams'][beam_label][subbeam_label] = {'baseline'       : baseline,
                                                                 'antennas'       : subbeam,
                                                                 'depths'         : relative_antenna_depths[subbeam],
                                                                 'time_delays'    : time_delays,
                                                                 'theta_elevation': theta_elevation,
                                                                 'theta_ant'      : theta_ant,
                                                                 'adjusted_m'     : ms[beam_index]
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

        self.beam_dict = beam_dict
        self.beam_colors = gnosim.utils.misc.getColorMap(self.n_beams)

    def calculateNoiseRMS(self):
        '''
        Calls the Antenna.calculateNoiseRMS() function for each antenna.  Creates the self.noise_rms array.
        
        See Also
        --------
        Antenna.calculateNoiseRMS
        '''
        noise_rms = []
        for antenna in self.antennas:
            antenna.calculateNoiseRMS()
            noise_rms.append(antenna.noise_rms)
        self.noise_rms = numpy.array(noise_rms)

    def plotStation(self,verbose = False, fig = None , ax = None):
        '''
        This plots creates a 3d plot of the antenna locations in the station.

        Parameters
        ----------
        verbose : bool, optional
            Enables more print statements (Default is False).
        fig : matplotlib.pyplot.figure, optional
            The figure on which to plot on.  If None, then a figure will be created.  (Default is None).
        ax : matplotlib.pyplot.axis, optional
            The axis on which to plot on.  As this is a 3d plot, this must be a 3d projection axis.  These can be obtained with:
            fig.gca(projection='3d')
            If None, then a figure will be created.  (Default is None).
        '''
        from matplotlib import markers
        if fig == None:
            fig = pylab.figure()
        if ax == None:
            ax = fig.gca(projection='3d')
        
        xs_PA = []
        ys_PA = []
        zs_PA = []
        xs_RA = []
        ys_RA = []
        zs_RA = []
        first = True
        
        marker_exclusions = ['.']
        ms = numpy.array(list(markers.MarkerStyle.markers))[~numpy.isin(list(markers.MarkerStyle.markers),marker_exclusions)]

        first = True
        for antenna in self.antennas:
            if verbose == True:
                print('alpha = ', antenna.alpha_deg ,'deg')
                print('beta = ', antenna.beta_deg,'deg')
                print('gamma = ', antenna.gamma_deg,'deg')
                print('x = ', antenna.x, 'm')
                print('y = ', antenna.y, 'm')
                print('z = ', antenna.z, 'm')
            basis_X = antenna.R[:,0] #x basis vector of the antenna frame in the ice basis
            basis_Y = antenna.R[:,1] #y basis vector of the antenna frame in the ice basis
            basis_Z = antenna.R[:,2] #z basis vector of the antenna frame in the ice basis
            if first == True:
                first = False
                ax.quiver(antenna.x, antenna.y, antenna.z, basis_X[0], basis_X[1], basis_X[2],color='r',label = 'Antenna X',linestyle='--')
                ax.quiver(antenna.x, antenna.y, antenna.z, basis_Y[0], basis_Y[1], basis_Y[2],color='g',label = 'Antenna Y',linestyle='--')
                ax.quiver(antenna.x, antenna.y, antenna.z, basis_Z[0], basis_Z[1], basis_Z[2],color='b',label = 'Antenna Z',linestyle='--')
            else:
                ax.quiver(antenna.x, antenna.y, antenna.z, basis_X[0], basis_X[1], basis_X[2],color='r',linestyle='--')
                ax.quiver(antenna.x, antenna.y, antenna.z, basis_Y[0], basis_Y[1], basis_Y[2],color='g',linestyle='--')
                ax.quiver(antenna.x, antenna.y, antenna.z, basis_Z[0], basis_Z[1], basis_Z[2],color='b',linestyle='--')
            if numpy.all(numpy.isin(antenna.label,self.antenna_keys[self.phased_cut])):
                xs_PA.append([antenna.x])
                ys_PA.append([antenna.y])
                zs_PA.append([antenna.z])
            else:
                xs_RA.append([antenna.x])
                ys_RA.append([antenna.y])
                zs_RA.append([antenna.z])

        ax.scatter(xs_PA,ys_PA,zs_PA,marker=ms[0],s=50,label='Phased Array')
        ax.scatter(xs_RA,ys_RA,zs_RA,marker=ms[0],s=50,label='Reconstruction Array')
        #ax.set_xlim([min(xs) - 1, max(xs) + 1])
        #ax.set_ylim([min(ys) - 1, max(ys) + 1])
        #ax.set_zlim([min(zs) - 1, max(zs) + 1])
        ax.set_xlabel('Ice x',fontsize=16)
        ax.set_ylabel('Ice y',fontsize=16)
        ax.set_zlabel('Ice z',fontsize=16)
        ax.view_init(elev = 30.0, azim = 45.0)
        pylab.legend(fancybox=True, framealpha=0.5,fontsize=12)
        return fig

############################################################

def getAcceptedAntennaTypes():
    '''
    Returns a list of the currently supported antenna types.  If you add a new antenna type
    you should put the label in the list such that it is recognized.
    
    Returns
    -------
    accepted_types : numpy.ndarray of floats
        A list of the accepted antenna types (labels).
    '''
    accepted_types = numpy.array(['simple', 'dipole','old_dipole', 'angular_dependent'])
    return accepted_types

class Antenna:
    '''
    Stores the attributes, information, and functions for an Antenna.
    
    Parameters
    ----------
    x : float
        The x coordinate for the position of the antenna in the ice frame.  Given in m.
    y : float
        The y coordinate for the position of the antenna in the ice frame.  Given in m.
    z : float
        The z coordinate for the position of the antenna in the ice frame.  Given in m.
    alpha_deg : float
        The alpha euler angle coordinate for choosing the orientation of the antenna in the ice frame.
        Given in degrees.  These angles will be used with a rotation to orient the axis of the antenna.
        For additional information see the EulerAngleDefiniton.pdf file located in the sim folder.
    beta_deg : float
        The beta euler angle coordinate for choosing the orientation of the antenna in the ice frame.
        Given in degrees.  These angles will be used with a rotation to orient the axis of the antenna.
        For additional information see the EulerAngleDefiniton.pdf file located in the sim folder.
    gamma_deg : float
        The gamma euler angle coordinate for choosing the orientation of the antenna in the ice frame.
        Given in degrees.  These angles will be used with a rotation to orient the axis of the antenna.
        For additional information see the EulerAngleDefiniton.pdf file located in the sim folder.
    antenna_type : str
        This is the label of a particular antenna type defined in the code.  This will select how the antenna
        behaves, i.e. beam patterns, polarization sensitivity, etc.  To see current supported antenna types 
        try gnosim.detector.detector.getAcceptedAntennaTypes().
    noise_temperature : float
        The temperature to be used in the noise calculation.  Given in K.
        Note that the noise is also processed by the system response, which may be scaled to obtain a particular noise
        level for a certain temperature.
    resistance : float
        The resistance to be used in the noise calculation.  Given in Ohms. 
        Note that the noise is also processed by the system response, which may be scaled to obtain a particular noise
        level for a certain temperature.
    lib_dir : str
        The location of the ray tracing library corresponding to an antenna at this location.
    label : str
        A label for the antenna, often used as a key in dictionaries during the simulation.  Should be unique for a
        particualar simulation (i.e. 'dipole0', 'dipole1', ... rather than just 'dipole').
    solutions : numpy.ndarray of str, optional
        A list of the solution types to load.  Often either all accepted solution types (the Default), or the same list omitting
        the _2 strings, which represent libraries of solution that reflect of the bottom of the ice.


    Attributes
    ----------
    x : float
        The x coordinate for the position of the antenna in the ice frame.  Given in m.
    y : float
        The y coordinate for the position of the antenna in the ice frame.  Given in m.
    z : float
        The z coordinate for the position of the antenna in the ice frame.  Given in m.
    alpha_deg : float
        The alpha euler angle coordinate for choosing the orientation of the antenna in the ice frame.
        Given in degrees.  These angles will be used with a rotation to orient the axis of the antenna.
        For additional information see the EulerAngleDefiniton.pdf file located in the sim folder.
    beta_deg : float
        The beta euler angle coordinate for choosing the orientation of the antenna in the ice frame.
        Given in degrees.  These angles will be used with a rotation to orient the axis of the antenna.
        For additional information see the EulerAngleDefiniton.pdf file located in the sim folder.
    gamma_deg : float
        The gamma euler angle coordinate for choosing the orientation of the antenna in the ice frame.
        Given in degrees.  These angles will be used with a rotation to orient the axis of the antenna.
        For additional information see the EulerAngleDefiniton.pdf file located in the sim folder.
    antenna_type : str
        This is the label of a particular antenna type defined in the code.  This will select how the antenna
        behaves, i.e. beam patterns, polarization sensitivity, etc.  To see current supported antenna types 
        try gnosim.detector.detector.getAcceptedAntennaTypes().
    noise_temperature : float
        The temperature to be used in the noise calculation.  Given in K.
        Note that the noise is also processed by the system response, which may be scaled to obtain a particular noise
        level for a certain temperature.
    resistance : float
        The resistance to be used in the noise calculation.  Given in Ohms. 
        Note that the noise is also processed by the system response, which may be scaled to obtain a particular noise
        level for a certain temperature.
    lib_dir : str
        The location of the ray tracing library corresponding to an antenna at this location.
    label : str
        A label for the antenna, often used as a key in dictionaries during the simulation.  Should be unique for a
        particualar simulation (i.e. 'dipole0', 'dipole1', ... rather than just 'dipole').
    accepted_solutions : numpy.ndarray of str, optional
        A list of the possible allowable solution types to load.
    solutions : numpy.ndarray of str, optional
        A list of the solution types to load.  Often either all accepted solution types (the Default), or the same list omitting
        the _2 strings, which represent libraries of solution that reflect of the bottom of the ice.
    R : numpy.ndarray
        The Euler rotation matrix coresponding to the given alpha, beta, and gamma, euler angles.
    R_inv : numpy.ndarray
        The inverse of the Euler rotation matrix coresponding to the given alpha, beta, and gamma, euler angles.  Used when representing
        a vector in the ice frame in terms of the antenna's defined (by alpha,beta, and gamma) reference frame.
    frequency_low : float, optional
        The lower frequency bound for the old and unsupported frequency domain calculation of the Askaryan radiation.  Only present if the
        selected time domain for the calculation is 'freq'.
    frequency_high : float, optional
        The upper frequency bound for the old and unsupported frequency domain calculation of the Askaryan radiation.  Only present if the
        selected time domain for the calculation is 'freq'.
    lib : dict
        Contains the gnosim.trace.refraction_library.RefractionLibrary object corresponding to this antenna.
        This is only loaded if self.loadLib is run, as it takes a lot of memory.  Often only loaded while grid interpolation
        is occuring.
    concave_hull : dict, optional
        Contains the information about the concave hull (corresponding to lib), and relavent funtions/limits.  Only loaded if 
        self.loadConcaveHull() is run.
    antenna_response_dir : str, optional
        The directory/file containing the antenna response. Only present if self.addTimingInfo() is run.
    system_response_dir : str, optional
        The directory/file containing the system response. Only present if self.addTimingInfo() is run.
    h_fft : numpy.ndarray of cfloat, optional
        The values for the antenna response. (Should have units of m, i.e. effective height). Only present if self.addTimingInfo() is run.
    sys_fft : numpy.ndarray of cfloat, optional
        The values for the system response. (Should be unitless). Only present if self.addTimingInfo() is run.
    freqs_response : numpy.ndarray of float, optional
        The values for the frequencies corresponding to the above responses. Only present if self.addTimingInfo() is run.
    signal_times : numpy.ndarray of floats, optional
        The observer times for which to calculate the Askaryan radiation.  Should span both negative and positive times 
        to get the full details of the pulse.  Given in ns.  Only present if self.addTimingInfo() is run.
    noise_rms : float
        The rms of the noise calculated for this particular antenna.  Given in V.

    See Also
    --------
    Station
    gnosim.interaction.askaryan
    gnosim.sim:EulerAngleDefiniton.pdf
    '''
    def __init__(self, x, y, z, alpha_deg, beta_deg, gamma_deg, antenna_type, noise_temperature, resistance, lib_dir, label, solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2'])):
        self.x = x
        self.y = y
        self.z = z
        self.alpha_deg = alpha_deg
        self.beta_deg  = beta_deg
        self.gamma_deg = gamma_deg
        self.lib_dir = os.path.expandvars(lib_dir)
        self.label = label
        self.noise_temperature = noise_temperature
        self.resistance = resistance 

        self.R = gnosim.utils.linalg.eulerRotationMatrix(numpy.deg2rad(self.alpha_deg), numpy.deg2rad(self.beta_deg), numpy.deg2rad(self.gamma_deg))
        self.R_inv = numpy.linalg.inv(self.R)     

        accepted_types = getAcceptedAntennaTypes() #MAKE SURE TO UPDATE IF YOU ADD NEW ANTENNA TYPES.
        if numpy.isin(antenna_type,accepted_types):
            self.antenna_type = antenna_type
        else:
            print('Selection of antenna type did not match predefined values.  Using default type:')
            self.antenna_type = accepted_types[0]
            print(self.antenna_type)
    
        self.accepted_solutions = gnosim.trace.refraction_library.getAcceptedSolutions()

        # List attributes of interest
        solutions = self.accepted_solutions[numpy.isin(self.accepted_solutions,solutions)]
        if len(solutions) == 0:

            print('Selection of solution types did not match predefined values.  Using default types.')
            self.solutions = self.accepted_solutions
            print(self.solutions)
        else:
            self.solutions = solutions
  

    def setUpFrequencyDomain(self, frequency_low, frequency_high):
        '''
        This gives the antenna class the necessary features to be backwards compatable, but also confines
        all of the old code to a single function so that I can delete it when I decide to remove the
        frequency domain support. 
        
        Parameters
        ----------
        frequency_low : float, optional
            The lower frequency bound for the old and unsupported frequency domain calculation of the Askaryan radiation.  
        frequency_high : float, optional
            The upper frequency bound for the old and unsupported frequency domain calculation of the Askaryan radiation.  
        '''
        self.frequency_low = frequency_low
        self.frequency_high = frequency_high

        def totalElectricField(self, frequency, electric_field, theta_ant, n_steps=100):
            f = scipy.interpolate.interp1d(frequency, electric_field, bounds_error=False, fill_value=0.)
            delta_frequency = (self.frequency_high - self.frequency_low) / n_steps
            frequency_array = numpy.linspace(self.frequency_low, self.frequency_high, n_steps)
            electric_array = f(frequency_array) * numpy.sin(numpy.deg2rad(theta_ant))
            integrated_field = numpy.sum(electric_array) * delta_frequency # V m^-1
            if numpy.sum(electric_array) != 0:
                weighted_freq = numpy.sum(frequency_array * electric_array) / numpy.sum(electric_array)
            else:
                weighted_freq = min(frequency)
            return electric_array, integrated_field, weighted_freq # V m^-1

        self.totalElectricField = types.MethodType(totalElectricField,self)


    def loadLib(self,pre_split = False,build_lib = True):
        '''
        This loads the gnosim.trace.refraction_library.RefractionLibrary object corresponding to this antenna after 
        cross checking the desired solutions are valid options.  If build_lib is True then this will load the full ray
        tracing library, otherwise it just creates the RefractionLirary object which contains some meta data about the library
        but not the actual ray tracing data.
        
        Parameters
        ----------
        pre_split : bool, optional
            Determines whether to attempt to load from pre split libraries.  If true (and the pre split libraries are calculated and 
            saved appropriately) this avoids lengthy calculations which seperate the rays ito the different solution types.  (Default is False).
        build_lib : bool, optional
            Must be called to actually populate most library information (such as the rays), however is left as an option such that 
            the rest of the library can be worked with as a less bulky object when necessary.  (Default is True).
        See Also
        --------
        gnosim.trace.refraction_library
        '''
        if numpy.logical_and(pre_split == False,numpy.logical_not(len(self.solutions) == len(self.accepted_solutions))):
            print('Limiting Solution Types Currently only works for pre_split = True, using default solution types.')
            self.solutions = self.accepted_solutions
        self.lib = gnosim.trace.refraction_library.RefractionLibrary(os.path.expandvars(self.lib_dir),solutions=self.solutions,pre_split = pre_split,build_lib = build_lib)
        self.solutions = self.lib.solutions #Catches mistakes if the refraction library has a varying number of antennas.

    def deleteLib(self,verbose=False):
        '''
        This deletes the library object from memory (to the best of its ability).  It is used after interpolation to free up RAM.
        
        Parameters
        ----------
        verbose : bool, optional
            Enables more print statements (Default is False).

        See Also
        --------
        gnosim.trace.refraction_library
        '''
        import gc
        if verbose == True:
            def memTools():
                '''
                Is a quickly included function to print some information about the freed up RAM.  This was straight from a stackExchange forum
                and is not exactly what I wanted, but was present for a quick test.
                '''
                mem=str(os.popen('free -t -m').readlines())
                """
                Get a whole line of memory output, it will be something like below
                ['             total       used       free     shared    buffers     cached\n', 
                'Mem:           925        591        334         14         30        355\n', 
                '-/+ buffers/cache:        205        719\n', 
                'Swap:           99          0         99\n', 
                'Total:        1025        591        434\n']
                 So, we need total memory, usage and free memory.
                 We should find the index of capital T which is unique at this string
                """
                T_ind=mem.index('T')
                """
                Than, we can recreate the string with this information. After T we have,
                "Total:        " which has 14 characters, so we can start from index of T +14
                and last 4 characters are also not necessary.
                We can create a new sub-string using this information
                """
                mem_G=mem[T_ind+14:-4]
                """
                The result will be like
                1025        603        422
                we need to find first index of the first space, and we can start our substring
                from from 0 to this index number, this will give us the string of total memory
                """
                S1_ind=mem_G.index(' ')
                mem_T=mem_G[0:S1_ind]
                """
                Similarly we will create a new sub-string, which will start at the second value. 
                The resulting string will be like
                603        422
                Again, we should find the index of first space and than the 
                take the Used Memory and Free memory.
                """
                mem_G1=mem_G[S1_ind+8:]
                S2_ind=mem_G1.index(' ')
                mem_U=mem_G1[0:S2_ind]

                mem_F=mem_G1[S2_ind+8:]
                print( 'Summary = ' + mem_G)
                print( 'Total Memory = ' + mem_T +' MB')
                print( 'Used Memory = ' + mem_U +' MB')
                print( 'Free Memory = ' + mem_F +' MB')

                CPU_Pct=str(round(float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline()),2))

                #print( results)
                print(("CPU Usage = " + CPU_Pct))

        if ~numpy.isin('lib',list(self.__dict__.keys())):
            print('Antenna dict already deleted.')
        else:
            if verbose == True:
                print('Memory info before deleting:')
                memTools()
            del self.__dict__['lib']
            gc.collect()
            if verbose == True:
                print('Memory info after deleting:')
                memTools()  

    def loadConcaveHull(self):
        '''
        This loads the concave hull within the gnosim.trace.refraction_library.RefractionLibrary object corresponding to this antenna.
        
        See Also
        --------
        gnosim.trace.refraction_library
        '''
        print('Loading Hull For:',self.lib_dir)
        self.concave_hull = {}

        indir = self.lib_dir.replace('*.h5','')

        if os.path.expandvars(indir)[-1] == '/':
            indir = os.path.expandvars(indir) + 'concave_hull'
        else:
            indir = os.path.expandvars(indir) + '/concave_hull'

        generate = False
        for solution in self.solutions:
            if os.path.exists(indir + '/concave_hull_data_%s.h5'%solution):
                pass
            else:
                generate = True

        if generate == True:
            print('All of hull not previously generated, calculating now.')
            self.lib.saveEnvelope( indir.replace('/concave_hull','') )

        chull = self.lib.loadEnvelope( indir ,store_fit_data = False)
        for dkey in self.lib.data.keys():
            self.concave_hull[dkey] = {}
            self.concave_hull[dkey]['z_min'] = chull[dkey]['z_min']
            self.concave_hull[dkey]['z_max'] = chull[dkey]['z_max']
            self.concave_hull[dkey]['f_inner_r_bound'] = chull[dkey]['f_inner_r_bound']
            self.concave_hull[dkey]['f_outer_r_bound'] = chull[dkey]['f_outer_r_bound']


    def loadAngularAntennaResponse(self, angle=90.0, mode='vpol'):
        '''
        Loads antenna impulse response (related to gain) for a given angle. 

        Parameters
        ----------
        angle : float
            Incident observed angle in degrees
        
        mode : str, optional
            The label for a particular response function loading operation.  (Default is 'vpol').
            Currently unused parameter.
        
        '''    

        # Various Failure modes. 
        if(numpy.isnan(angle)):
            print(angle, "is nan. Using 90. ")
            angle = 90.0
        else:
            if(angle < 0.0):
                print(angle, "is less than 0. Taking Absolute value. ")
                angle = numpy.abs(angle)
            if(angle > 180.0):
                print(angle, "is greater than 180.")
                while(angle > 180.0): 
                    angle -= 180.0

        # Find index based on given angle. 
        entry_1 = int(numpy.floor(angle/10.0))
        entry_2 = int(numpy.ceil(angle/10.0))
        weight_1 = 1.0 - numpy.abs(angle - entry_1*10.0)/10.0
        weight_2 = 1.0 - weight_1

        # Various failure modes.
        if(entry_2 > 18): 
            print(entry_2, "is too high, out of range. ")
            entry_1 = 18
            weight_1 = 1.0
            entry_2 = 18
            weight_2 = 0.0
        
        # Return the linear interpolation
        self.h_fft = (self.response_vs_angle[entry_1]*weight_1 + self.response_vs_angle[entry_2]*weight_2) / 2.0
        self.freqs_response = self.response_freqs  # Terrible naming scheme

    def addTimingInfo(self, system_response_dir, antenna_response_dir):
        '''
        This loads the system response, antenna response, and corresponding frequencies, 
        and additionally calculates the times for which the Askaryan calculation will be calculated.  

        If a new response is added, you should ensure it can be loaded correctly with this funciton.

        Parameters
        ----------
        antenna_response_dir : str
            The directory/file containing the antenna response.
        system_response_dir : str
            The directory/file containing the system response.

        See Also
        --------
        gnosim.sim.response
        '''
        self.antenna_response_dir = os.path.expandvars(antenna_response_dir)

        self.system_response_dir = os.path.expandvars(system_response_dir)
        electronic_response = numpy.load(self.system_response_dir)
        freqs, sys_fft = numpy.hsplit(electronic_response, 2)
        self.sys_fft = numpy.ravel(sys_fft)
        self.freqs_response =  numpy.absolute(numpy.ravel(freqs).astype(float))
        t_step = 1/(2*max(self.freqs_response))*1e9 #ns
        self.signal_times = numpy.arange(-(len(self.freqs_response)-1),(len(self.freqs_response)-1))*t_step #To increase time duration of signal I should just need to upsample?

        # If antenna is angularly dependent, going to load G_realized(f, theta) into memory. 
        if(self.antenna_type == 'angular_dependent'):

            # Cheap way to check if the file type is correct
            if not("npz" in self.antenna_response_dir):
                print("Angularly dependent antenna type selected but no response file / wrong file specified. Using defult type:")
                self.antenna_type = accepted_types[0]
                print(self.antenna_type)
            else:
                response_file = numpy.load(self.antenna_response_dir)
                self.response_vs_angle = response_file['response_vs_angle']
                self.response_freqs = response_file['desired_freqs']
                self.response_angs = response_file['response_angs']

            self.loadAngularAntennaResponse()

        else:
            self.antenna_response_dir = os.path.expandvars(antenna_response_dir)
            antenna_response = numpy.load(self.antenna_response_dir)
            freqs, h_fft = numpy.hsplit(antenna_response, 2)
            self.h_fft = numpy.ravel(h_fft)

    def calculateNoiseRMS(self):
        '''
        The rms of the noise calculated for this particular antenna.  Done by producing signals for events with 0 neutrino energy such 
        that all that is returned is noise, but calculated identically to how it is generated for each event.  This is done for many events
        and the overall noise rms is taken.  This is usually used for scaling digitized signals.

        See Also
        --------
        gnosim.interaction.askaryan
        '''
        noise_signal  = numpy.array([])
        for i in range(100):
            #Values below are mostly just filler to get just the noise (As calculated for the Askaryan pulse).  I am doing this
            #by just calculating signals with 0 energy.  The noise is overlaying on them, but because there is no signal we just
            #get the noise.  Not pretty but....
            noise_signal_i = gnosim.interaction.askaryan.quickSignalSingle( 0,1,0,1.8,\
                      0,0,self.signal_times,self.h_fft,self.sys_fft,self.freqs_response,\
                      plot_signals=False,plot_spectrum=False,plot_potential = False,\
                      include_noise = True, resistance = self.resistance, noise_temperature = self.noise_temperature)[3]
            noise_signal = numpy.append(noise_signal,noise_signal_i)
        self.noise_rms = numpy.std(noise_signal)

    def antennaFrameCoefficients(self,in_vector):
        '''
        Takes a vector in ice frame coordinates and converts returns the coefficients for the antenna frame.

        R should be calculated in advance using R = gnosim.utils.linalg.eulerRotationMatrix(alpha_rad, beta_rad, gamma_rad), and stored
        with its inverse as self.R and self.R_inv.  
        
        This is intended to perform the extrinsic rotation of a vector using the Euler angles alpha, beta, gamma.  
        The output vector is intended to be in the frame in the basis frame defined by the given Euler angles. This returns the coefficients 
        of the vector in the antenna frame.   

        I.e. if the vector u, given in the ice basis (x,y,z) as u = a x + b y + c z  is represented in the ice frame, 
        this returns the coefficients A,B,C of the antenna frame basis (X,Y,Z), such that u = A X + B Y + C Z = a x + b y + c z  
        
        Parameters
        ----------
        in_vector : numpy.ndarray of floats
            The 3 dimensional vector defined in the ice frame to be represented in the antenna frame.  
            From the above example this would be: 
            in_vector  = numpy.array([a,b,c]) 
            such that:
            u = a*x + b*y + c*z 
            where:
            x,y,z are the units vectors in the ice frame (defined to be x = numpy.array([1,0,0]), y = numpy.array([0,1,0]), z = numpy.array([0,0,1]))

        Returns
        -------
        out_vector : numpy.ndarray of floats
            The 3 dimensional vector defined in the antenna frame.  
            From the above example this would be: 
            out_vector  = numpy.array([A,B,C])
            such that:
            u = A*X + B*Y + C*Z
            where:
            X, Y, and Z are the unit vectors for the antenna frame.  The vector u points in the same direction, but is just represented in the new frame.
        '''
        out_vector = numpy.dot(self.R_inv,in_vector)
        return out_vector 

    def getAntennaResponseFactor( self , vec_neutrino_travel_dir , emission_wave_vector , detection_wave_vector , a_s , a_p , return_polarizations = False):
        '''
        This calculates the net reduction in signal seen by the antenna (before system response).  Includes effects from 
        beam pattern, polarization sensitivity, etc. depending on the antenna type chosen.

        This is where new antenna types (different beam patterns, polarization sensitivities, etc.) should be added.
        Currently the output of this function is a single float.  As the specific types antennas that might be added
        in the future is hard to predict I have attempted to leave this general enough to support whatever calculations
        may be needed.  If an antenna is added, ensure to add the antenna_type label to getAcceptedAntennaTypes.

        Parameters
        ----------
        vec_neutrino_travel_dir : numpy.ndarray
            The unit vector for the direction the shower is propogating.
            This is returned in ice-frame cartesian coordinates.
        emission_wave_vector : numpy.ndarray
            The unit vector for the vector directed towards the antenna along the observation ray. 
            This is returned in ice-frame cartesian coordinates.  This should be the wave vector
            as it was emitted from the neutrino.
        detection_wave_vector : numpy.ndarray
            The unit vector for the vector directed towards the antenna along the observation ray. 
            This is returned in ice-frame cartesian coordinates.  This should be the wave vector
            as it interacts with the antenna.
        a_s : float
            This is the attenuation factor of the s-polarization.  It should contain both the
            attenuation resulting from attenuation length, as well as the net effect of the
            fresnel amplitudes over the corse of the ray's path to the antenna.
            Currently only numpy.real(a_s) is returned from refraction_libray_beta.makeLibrary,
            so a real float is expected here.
        a_p : float
            This is the attenuation factor of the p-polarization.  It should contain both the
            attenuation resulting from attenuation length, as well as the net effect of the
            fresnel amplitudes over the corse of the ray's path to the antenna.
            Currently only numpy.real(a_p) is returned from refraction_libray_beta.makeLibrary,
            so a real float is expected here.
        return_polarizations : bool
            If True then this will return the polarization vectors as well.  This only works currently for
            the 'dipole' antenna type.  (Default is False).

        Returns
        -------
        signal_reduction_factor : float
            This is the reduction factor that should be multiplied with the antenna response.  This is typically used as the 
            signal_reduction_factor input parameter of the gnosim.interaction.askaryan.quickSignalSingle calculation.
            This contains the determined factors of polarization_dot_factor, beam_pattern_factor, and attenuation_factor,
            (if intended by the particular antenna type).  Some antenna types will return filler values for the subcomponants
            of these if those don't actually go into the calculation of this number.  i.e. signal_reduction_factor could be
            1.0 even though a value of -999.0 is given for beam_pattern_factor if that particular antenna type doesn't have
            a built in beam_patter_factor.  Because of this, only signal_reduction_factor should be used in calculations, with
            the other outputs just being store for information.
        polarization_dot_factor : float
            The polarization componant of the signal_reduction factor (i.e. the factor resulting from the dot product between the
            polarization vector at the antenna and the polarization sensitivity vector set for that antenna).  This componant may 
            be calculated depending on the type of antenna chosen. If it is not calculated for a particular antenna type then a 
            value of -999.0 is returned.  To see how this factor was specifically calculated for your antenna type see the code below.
        beam_pattern_factor : float
            The beam pattern componant of the signal_reduction factor.  I.e. the reduction in signal strength due to the approach angles
            of the wave vector at the antenna.  This componant may be calculated depending on the type of antenna chosen.
            If it is not calculated for a particular antenna type then a value of -999.0 is returned.  To see how this factor was
            specifically calculated for your antenna type see the code below.
        attenuation_factor : float
            The attenuation componant of the signal_reduction factor.  The contains the affects of a_s and a_p (the attennuations for 
            each polarization of light).  Depending on the type of detector chosen this could entail simply returning one of those values,
            or accounting for the appropriate vectorized proportion of each componant on the final signal.  This componant may be calculated 
            depending on the type of antenna chosen. If it is not calculated for a particular antenna type then a value of -999.0 is returned.  
            o see how this factor was specifically calculated for your antenna type see the code below.
        dot_angle : float
            The angle from the dot product of the polarization vector and the polarization sensitivity vector.  Only works for
            detectors sensitive to polarization, will return -999.0 otherwise. 
        polarization_unit_vector_1_ice_frame: numpy.ndarray, optional
            The unit vector for the polarization as it is just before interacting with the antenna.
            This is a unit vector, magnitudes represent how the s and p polarizations have been
            reduced during ray propogation.  The magnitude of this vector originally contained information about a_s and a_p, however
            the magnitude has been split from the direction and is stored in attenuation_factor output value.
            This vector is returned in ice-frame cartesian coordinates.
        polarization_unit_vector_0_ice_frame: numpy.ndarray, optional
            The unit vector for the polarization as it is just after emission at the neutrino.
            This is a unit vector.
            This is returned in ice-frame cartesian coordinates.

        See Also
        --------
        gnosim.interaction.askaryan
        gnosim.interaction.polarization
        '''

        '''
        # TODO
        elif(self.antenna_type == 'lpda'):
            print('This is something we should do!')
            #Possible source for key information:
            #https://arxiv.org/ftp/arxiv/papers/1406/1406.0820.pdf
            #https://arianna.ps.uci.edu/sites/default/files/Allan_Hallgren_VLVvT2015.pdf
        '''


        if self.antenna_type == 'simple':

            #This is attenuation from attenuation length, and for a p polarized light.  Most like what was done with simple before.  No beam pattern.
            
            polarization_dot_factor = -999.0
            beam_pattern_factor = -999.0
            attenuation_factor = numpy.abs(a_p)
            signal_reduction_factor = attenuation_factor #ONLY USE THIS FOR CALCULATIONS, AS THE OTHER NUMBERS MIGHT BE PLACEHOLDERS (IN GENERAL).  I.E. TRY NOT TO USE BEAM_PATTERN_FACTOR (E.G.) FOR ANYTHING OTHER THAN STORING IT AFTER OUTPUT BECAUSE DEPENDING ON TYPE OF ANTENNA IT MIGHT BE -999.0

            if return_polarizations == True:
                return signal_reduction_factor, polarization_dot_factor, beam_pattern_factor, attenuation_factor, dot_angle, numpy.zeros(3), numpy.zeros(3)
            else:
                return signal_reduction_factor, polarization_dot_factor, beam_pattern_factor, attenuation_factor, dot_angle

        elif self.antenna_type == 'dipole':

            #####
            #Generic calculations helpful for many antenna types
            #####
            
            #polarization_unit_vector_0_ice_frame, k_0_ice_frame, vec_neutrino_travel_dir_ice_frame = getInitialPolarization(theta_ray_from_ant_at_neutrino,phi_ray_from_ant_at_neutrino,theta_neutrino_source_dir,phi_neutrino_source_dir)
            #polarization_unit_vector_1_ice_frame, k_1_ice_frame = gnosim.interaction.polarization.getPolarizationAtAntenna(phi_ray_from_ant_at_neutrino,theta_ray_from_ant_at_neutrino,phi_ray_from_ant_at_antenna,theta_ray_from_ant_at_antenna,phi_neutrino_source_dir,theta_neutrino_source_dir, a_s, a_p, return_k_1 = True)            #This is for a vpol antenna. which is sensitive at polls, 
            #vec_neutrino_travel_dir
            #emission_wave_vector
            #detection_wave_vector
            if return_polarizations == True:
                polarization_unit_vector_1_ice_frame, attenuation_factor, polarization_unit_vector_0_ice_frame = gnosim.interaction.polarization.getPolarizationAtAntenna(vec_neutrino_travel_dir , emission_wave_vector , detection_wave_vector , a_s , a_p, return_initial_polarization = True)            #This is for a vpol antenna. which is sensitive at polls,  
            else:
                polarization_unit_vector_1_ice_frame, attenuation_factor = gnosim.interaction.polarization.getPolarizationAtAntenna(vec_neutrino_travel_dir , emission_wave_vector , detection_wave_vector , a_s , a_p)            #This is for a vpol antenna. which is sensitive at polls, 
            #Note k_0 and k_1 are the wave vectors along the ray TOWARDS the antenna (from emission), with k_0 being at emission, and k_1 being at antenna
            #polarization_vector_1_antenna_frame = antennaFrameCoefficients(self.R_inv, polarization_unit_vector_1_ice_frame, pre_inv = True)
            polarization_vector_1_antenna_frame = self.antennaFrameCoefficients(polarization_unit_vector_1_ice_frame)
            detection_wave_vector_antenna_frame = self.antennaFrameCoefficients(detection_wave_vector)
            #Below you should define how your particular antenna interacts with polarization, as well as the beam pattern. 
            #Antennas can be rotated, so be sure to do the calculations in the correct frame.
            #####
            #Specific calculations for this antenna type
            #####

            #polarization_sensitivity_theta = 0.0    #Angle from pole (in antenna frame) the antenna is most sensitive to polarization.
            #polarization_sensitivity_phi = 0.0      #Azimuthal angle (from x in antenna frame) the antenna is most sensitive to polarization.
            dipole_polarization_axis_antenna_frame = numpy.array([0.0,0.0,1.0]) #Fully sensitive if aligned with z-axis of antenna Must be unit vector!!
            polarization_dot_factor = numpy.dot(polarization_vector_1_antenna_frame,dipole_polarization_axis_antenna_frame) 
            
            dot_angle = numpy.rad2deg(numpy.arccos(polarization_dot_factor / (numpy.linalg.norm(polarization_vector_1_antenna_frame))))

            #Calculating beam pattern from theta
            #below is the standard explicit way to do this
            #antenna_frame_theta_rad = numpy.arccos(detection_wave_vector_antenna_frame[2])# Typically it is arccos(z/r) but r is 1 for unit vector
            #beam_pattern_factor = numpy.sin(antenna_frame_theta_rad)  #not squared because wante electric field NOT power
            #Below is a slightly faster way to do this using more geometry
            #Assumes detection_wave_vector_antenna_frame is a unit vector
            beam_pattern_factor = numpy.sqrt(1.0 - detection_wave_vector_antenna_frame[2]**2.0) #where r is assumed to be 1 because working with unit vectors, #Note for many beam patterns likely want vector point TO observation, i.e. negative of this. But for this calculation is doesn't matter.
            #This is sqrt because it is being applied to the E field NOT the power (power beam pattern is sin2(theta))
            signal_reduction_factor = polarization_dot_factor*beam_pattern_factor*attenuation_factor #ONLY USE THIS FOR CALCULATIONS, AS THE OTHER NUMBERS MIGHT BE PLACEHOLDERS (IN GENERAL).  I.E. TRY NOT TO USE BEAM_PATTERN_FACTOR (E.G.) FOR ANYTHING OTHER THAN STORING IT AFTER OUTPUT BECAUSE DEPENDING ON TYPE OF ANTENNA IT MIGHT BE -999.0

            if return_polarizations == True:
                return signal_reduction_factor, polarization_dot_factor, beam_pattern_factor, attenuation_factor, dot_angle, polarization_unit_vector_1_ice_frame, polarization_unit_vector_0_ice_frame
            else:
                return signal_reduction_factor, polarization_dot_factor, beam_pattern_factor, attenuation_factor, dot_angle
        elif self.antenna_type == 'old_dipole':
            #This is how is was calculated before the polarization was added. 
            #k_1_ice_frame = gnosim.interaction.polarization.getWaveVector(phi_ray_from_ant_at_antenna,theta_ray_from_ant_at_antenna) #Note for many beam patterns likely want vector point TO observation, i.e. negative of this. But for this calculation is doesn't matter.
            #detection_wave_vector_antenna_frame = antennaFrameCoefficients(self.R_inv, k_1_ice_frame, pre_inv = True)
            detection_wave_vector_antenna_frame = self.antennaFrameCoefficients(detection_wave_vector)
            beam_pattern_factor = numpy.sqrt(1.0 - detection_wave_vector_antenna_frame[2]**2.0) #where r is assumed to be 1 because working with unit vectors, note this is not sin(theta) because this is NOT power beam pattern. This is applied to the electric field.
            polarization_dot_factor = -999.0 #Not being calculated.
            dot_angle = -999.0 #Not putting 90 because that might make people think 90 was being caculated for everything even though it just isn't calculated for this type of antenna.
            attenuation_factor = numpy.abs(a_p)

            signal_reduction_factor = attenuation_factor*beam_pattern_factor #ONLY USE THIS FOR CALCULATIONS, AS THE OTHER NUMBERS MIGHT BE PLACEHOLDERS (IN GENERAL).  I.E. TRY NOT TO USE BEAM_PATTERN_FACTOR (E.G.) FOR ANYTHING OTHER THAN STORING IT AFTER OUTPUT BECAUSE DEPENDING ON TYPE OF ANTENNA IT MIGHT BE -999.0
            
            
            if return_polarizations == True:
                print('ERROR: Polarization not supported for old_dipole antenna type.  Returning null polarization vectors.')
                return signal_reduction_factor, polarization_dot_factor, beam_pattern_factor, attenuation_factor, dot_angle, numpy.zeros(3), numpy.zeros(3) #Zeros on the end are for vectors ot being calculated but expected at oupput.
            else:
                return signal_reduction_factor, polarization_dot_factor, beam_pattern_factor, attenuation_factor, dot_angle
        elif(self.antenna_type == 'angular_dependent'):

            #####
            #Generic calculations helpful for many antenna types
            #####
            
            if return_polarizations == True:
                polarization_unit_vector_1_ice_frame, attenuation_factor, polarization_unit_vector_0_ice_frame = gnosim.interaction.polarization.getPolarizationAtAntenna(vec_neutrino_travel_dir , emission_wave_vector , detection_wave_vector , a_s , a_p, return_initial_polarization = True)            #This is for a vpol antenna. which is sensitive at polls,  
            else:
                polarization_unit_vector_1_ice_frame, attenuation_factor = gnosim.interaction.polarization.getPolarizationAtAntenna(vec_neutrino_travel_dir , emission_wave_vector , detection_wave_vector , a_s , a_p)            #This is for a vpol antenna. which is sensitive at polls, 

            polarization_vector_1_antenna_frame = self.antennaFrameCoefficients(polarization_unit_vector_1_ice_frame)
            detection_wave_vector_antenna_frame = self.antennaFrameCoefficients(detection_wave_vector)

            #####
            #Specific calculations for this antenna type
            #####

            dipole_polarization_axis_antenna_frame = numpy.array([0.0,0.0,1.0]) #Fully sensitive if aligned with z-axis of antenna Must be unit vector!!
            polarization_dot_factor = numpy.dot(polarization_vector_1_antenna_frame,dipole_polarization_axis_antenna_frame) 
            
            dot_angle = numpy.rad2deg(numpy.arccos(polarization_dot_factor / (numpy.linalg.norm(polarization_vector_1_antenna_frame))))

            beam_pattern_factor = 1.0 # Reduction is done in full antenna beam pattern
            signal_reduction_factor = polarization_dot_factor*beam_pattern_factor*attenuation_factor
            if return_polarizations == True:
                return signal_reduction_factor, polarization_dot_factor, beam_pattern_factor, attenuation_factor, dot_angle, polarization_unit_vector_1_ice_frame, polarization_unit_vector_0_ice_frame
            else:
                return signal_reduction_factor, polarization_dot_factor, beam_pattern_factor, attenuation_factor, dot_angle
            
        else:
            print('ANTENNA TYPE NOT FOUND IN ACCEPTED ANTENNAS, RETURNING 1')
            
            polarization_dot_factor = -999.0
            beam_pattern_factor = -999.0
            attenuation_factor = -999.0
            signal_reduction_factor = 1.0 #ONLY USE THIS FOR CALCULATIONS, AS THE OTHER NUMBERS MIGHT BE PLACEHOLDERS (IN GENERAL).  I.E. TRY NOT TO USE BEAM_PATTERN_FACTOR (E.G.) FOR ANYTHING OTHER THAN STORING IT AFTER OUTPUT BECAUSE DEPENDING ON TYPE OF ANTENNA IT MIGHT BE -999.0

            if return_polarizations == True:
                print('ERROR: No accepted antenna type chosen.  Returning null polarization vectors.')
                return signal_reduction_factor, polarization_dot_factor, beam_pattern_factor, attenuation_factor, dot_angle, numpy.zeros(3), numpy.zeros(3)
            else:
                return signal_reduction_factor, polarization_dot_factor, beam_pattern_factor, attenuation_factor, dot_angle

############################################################


if __name__ == "__main__":
    import yaml
    solutions = numpy.array(['direct'])
    config_file = os.environ['GNOSIM_DIR'] + '/gnosim/detector/station_config/real_config_full_station.py'
    config = yaml.load(open(config_file))
    station = Station('ARA5',config,solutions = solutions)
    station.plotStation()

    ant = station.antennas[0]



    energy_neutrino = 3.e9 # GeV
    n = 1.78
    c = gnosim.utils.constants.speed_light #m/ns
    R = 1000. #m
    cherenkov_angle = numpy.arccos(1./n)
    cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./n))

    inelasticity = 1.0

    u, V_noiseless, dominant_freq, V_noise,  SNR = gnosim.interaction.askaryan.quickSignalSingle(numpy.deg2rad(50),R,inelasticity*energy_neutrino,n,2500,0.7,ant.signal_times, ant.h_fft, ant.sys_fft, ant.freqs_response,plot_signals=True,plot_spectrum=True,plot_potential=True,include_noise = True)

    '''
    pylab.figure()
    pylab.title('Response Curves',fontsize=18)
    pylab.scatter(ant.freqs_response/1e6, 10.0 * numpy.log10(ant.h_fft), label="Antenna")
    pylab.scatter(ant.freqs_response/1e6, 10.0 * numpy.log10(ant.sys_fft), label="System")
    pylab.xlabel("Freq. (MHz)",fontsize=16)
    pylab.ylabel("Realized Gain (dBi)",fontsize=16)
    pylab.legend(fontsize=14)
    pylab.xlim(0, 1000)
    pylab.ylim(-25.0, 40.0)
    pylab.minorticks_on()
    pylab.grid(which="both")
    pylab.grid(b=True, which='major', color='k', linestyle='-')
    pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    '''

    '''
    x = 0.0
    y = 0.0
    z = -200.0
    alpha_deg = 0.0
    beta_deg = 0.0
    gamma_deg = 0.0
    antenna_type = 'dipole'
    import yaml
    config_file = os.environ['GNOSIM_DIR'] + '/gnosim/detector/station_config/real_config_full_station.py'
    config = yaml.load(open(config_file))
    lib_filler = config['antenna_definitions']['dipole0']['lib']
    frequency_low = 30
    frequency_high = 500
    solutions = numpy.array(['direct'])
    #solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2'])

    try:
        antenna = Antenna(x, y, z, alpha_deg, beta_deg, gamma_deg, antenna_type, lib_filler,solutions = solutions)
        antenna.setUpFrequencyDomain(frequency_low,frequency_high)
        a = antenna.totalElectricField(numpy.arange(10), numpy.arange(10)**2, 45., n_steps=100)
        print('This worked as intended')
    except:
        print('This should not have broken :(')

    try:
        bantenna = Antenna(x, y, z, alpha_deg, beta_deg, gamma_deg, antenna_type, lib_filler,solutions = solutions)
        b = bantenna.totalElectricField(numpy.arange(10), numpy.arange(10)**2, 45., n_steps=100)
        print('This should not have worked :(')
    except:
        print('This broke as intended!')

    station = Station(0,0,-200,config,solutions = solutions)
    station.loadLib(pre_split = True,build_lib = True)
    #station.deleteLib(verbose=True)
    #station = Station(0,0,-200,config,,pre_split = True)
    '''
