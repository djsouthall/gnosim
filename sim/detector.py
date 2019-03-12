"""
Detector object
"""

import numpy
import scipy.interpolate
import sys
import os 
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import pylab
import types
import gnosim.trace.refraction_library_beta
############################################################
# ORIENTATION TOOLS

def xRotationMatrix(theta_rad):
    '''
    Returns a 3x3 rotation matrix for rotating theta radians about the x axis.
    '''
    R = numpy.array([   [1,0,0],
                        [0,numpy.cos(theta_rad),-numpy.sin(theta_rad)],
                        [0,numpy.sin(theta_rad),numpy.cos(theta_rad)]   ])
    return R

def yRotationMatrix(theta_rad):
    '''
    Returns a 3x3 rotation matrix for rotating theta radians about the y axis.
    '''
    R = numpy.array([   [numpy.cos(theta_rad),0,numpy.sin(theta_rad)],
                        [0,1,0],
                        [-numpy.sin(theta_rad),0,numpy.cos(theta_rad)]   ])
    return R
    
def zRotationMatrix(theta_rad):
    '''
    Returns a 3x3 rotation matrix for rotating theta radians about the x axis.
    '''
    R = numpy.array([   [numpy.cos(theta_rad),-numpy.sin(theta_rad),0],
                        [numpy.sin(theta_rad),numpy.cos(theta_rad),0],
                        [0,0,1]   ])
    return R

def eulerRotationMatrix(alpha_rad, beta_rad, gamma_rad):
    '''
    This creates the rotation matrix R using the given Euler angles and a
    z-x-z extrinsic rotation.
    '''
    Rz1 = zRotationMatrix(gamma_rad)
    Rx1 = xRotationMatrix(beta_rad)
    Rz2 = zRotationMatrix(alpha_rad)
    R = numpy.dot(Rz2,numpy.dot(Rx1,Rz1))
    return R
    
def antennaFrameCoefficients(R, in_vector, pre_inv = False):
    '''
    R should be calculated in advance using R = eulerRotationMatrix(alpha_rad, beta_rad, gamma_rad)
    and passed to this function.  Not internally calculated because it is the same for a given antenna
    for each in_vector and does not need to be redundently calculated.  The inversion of the matrix
    also only needs to be done once, so there is an option to pass this function the previously 
    inverted R.  
    
    This is intended to perform the extrinsic rotation of a vector
    using the Euler angles alpha, beta, gamma.  I intend for the output vector
    to be in the frame in the basis frame defined by the given Euler angles.
    
    This returns the coefficients of the vector in the antenna frame.  
    I.e. if the vector u, given in the ice basis (x,y,z) as u = a x + b y + c z 
    is represented in the ice frame, this returns the coefficients A,B,C of the
    antenna frame basis (X,Y,Z), such that u = A X + B Y + C Z = a x + b y + c z  
    '''
    if pre_inv == True:
        out_vector = numpy.dot(R,in_vector)
    else:
        out_vector = numpy.dot(numpy.linalg.inv(R),in_vector)
    
    return out_vector   
    
def plotArrayFromConfig(config,solutions,only_station = 'all',verbose = False):
    '''
    Given a loaded config file this shouldplot the cooardinate system of each of
    the antennas in the lab frame.  
    only_station should either be 'all', to plot all stations, or a single index, 
    to plot a single station.  The index should be base 0. 

    #Eventually this should be added as a method to the station class.
    '''
    from matplotlib import markers
    fig = pylab.figure(figsize=(16,11.2))
    ax = fig.gca(projection='3d')
    xs = []
    ys = []
    zs = []
    first = True
    
    marker_exclusions = ['.']
    ms = numpy.array(list(markers.MarkerStyle.markers))[~numpy.isin(list(markers.MarkerStyle.markers),marker_exclusions)]


    stations = []
    for ii in range(0, config['stations']['n']):
        x_station, y_station, z_station = config['stations']['positions'][ii]
        station = gnosim.sim.detector.Station(x_station, y_station, z_station,config,solutions)
        #station.loadLib(pre_split = True)
        self.stations.append(station)

    for index_station, station in enumerate(stations):
        # Loop over station antennas
        if numpy.logical_or(only_station == 'all', only_station == index_station):
            station_label = 'station'+str(index_station)
            m = ms[index_station]
            for antenna_index, antenna in enumerate(station.antennas):
                alpha_deg, beta_deg, gamma_deg = antenna.orientation
                xs.append(antenna.x)
                ys.append(antenna.y)
                zs.append(antenna.z)
                if verbose == True:
                    print('alpha = ', alpha_deg ,'deg')
                    print('beta = ', beta_deg,'deg')
                    print('gamma = ', gamma_deg,'deg')
                    print('x = ', antenna.x, 'm')
                    print('y = ', antenna.y, 'm')
                    print('z = ', antenna.z, 'm')
                R = eulerRotationMatrix(numpy.deg2rad(alpha_deg), numpy.deg2rad(beta_deg), numpy.deg2rad(gamma_deg))
                basis_X = R[:,0] #x basis vector of the antenna frame in the ice basis
                basis_Y = R[:,1] #y basis vector of the antenna frame in the ice basis
                basis_Z = R[:,2] #z basis vector of the antenna frame in the ice basis
                if first == True:
                    first = False
                    ax.quiver(antenna.x, antenna.y, antenna.z, basis_X[0], basis_X[1], basis_X[2],color='r',label = 'Antenna X',linestyle='--')
                    ax.quiver(antenna.x, antenna.y, antenna.z, basis_Y[0], basis_Y[1], basis_Y[2],color='g',label = 'Antenna Y',linestyle='--')
                    ax.quiver(antenna.x, antenna.y, antenna.z, basis_Z[0], basis_Z[1], basis_Z[2],color='b',label = 'Antenna Z',linestyle='--')
                else:
                    ax.quiver(antenna.x, antenna.y, antenna.z, basis_X[0], basis_X[1], basis_X[2],color='r',linestyle='--')
                    ax.quiver(antenna.x, antenna.y, antenna.z, basis_Y[0], basis_Y[1], basis_Y[2],color='g',linestyle='--')
                    ax.quiver(antenna.x, antenna.y, antenna.z, basis_Z[0], basis_Z[1], basis_Z[2],color='b',linestyle='--')
                ax.scatter(x,y,z,label = 'S%i '%index_station + antenna_label,marker=m,s=50)
    ax.set_xlim([min(xs) - 1, max(xs) + 1])
    ax.set_ylim([min(ys) - 1, max(ys) + 1])
    ax.set_zlim([min(zs) - 1, max(zs) + 1])
    ax.set_xlabel('Ice x',fontsize=16)
    ax.set_ylabel('Ice y',fontsize=16)
    ax.set_zlabel('Ice z',fontsize=16)
    ax.view_init(elev = 30.0, azim = 45.0)
    pylab.legend(fancybox=True, framealpha=0.5,fontsize=12)
    return fig

############################################################

class Station:

    def __init__(self, x, y, z, config, station_label, solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2']),electricFieldDomain = 'time'):
        """
        
        """
        self.label = station_label
        self.config = config
        self.x = x
        self.y = y
        self.z = z
        self.electricFieldDomain = electricFieldDomain
        self.accepted_solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2'])
        self.power_calculation_sum_length = config['DAQ']['power_calculation_sum_length']
        self.power_calculation_interval = config['DAQ']['power_calculation_interval']
        self.beamforming_power_sum_byte_cap = config['DAQ']['beamforming_power_sum_byte_cap']
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

        self.antennas = [] #In the future might want 2 types of antennas, those in the phased array and those in the reconstruction array.  Prefereably stored seperately.
        self.buildStation() #Does not load libraries.  Call loadLib() for that. 

    def buildStation(self):
        #In the future might want 2 types of antennas, those in the phased array and those in the reconstruction array.  Prefereably stored seperately.
        for key in list(self.config['antenna_definitions'].keys()):
            x_antenna, y_antenna, z_antenna = self.config['antenna_definitions'][key]['position']
            if numpy.isin('orientations',list(self.config['antenna_definitions'][key].keys())) == True:                                   
                alpha_deg, beta_deg, gamma_deg =  self.config['antenna_definitions'][key]['orientation']
            else:
                alpha_deg, beta_deg, gamma_deg = [0.0,0.0,0.0]

            antenna = Antenna(x_antenna + self.x, y_antenna + self.y, z_antenna + self.z, 
                                alpha_deg, beta_deg, gamma_deg,
                                self.config['antenna_definitions'][key]['antenna_type'],
                                self.config['antenna_definitions'][key]['noise_temperature'], 
                                self.config['antenna_definitions'][key]['resistance'],
                                self.config['antenna_definitions'][key]['lib'],
                                key,
                                solutions = self.solutions)
            
            if self.electricFieldDomain == 'freq':
                antenna.setUpFrequencyDomain(self.config['antenna_definitions'][key]['frequency_low'],self.config['antenna_definitions'][key]['frequency_high'])
            elif self.electricFieldDomain == 'time':
                antenna.addTimingInfo(self.config['antenna_definitions'][key]['system_response'],self.config['antenna_definitions'][key]['antenna_response'])

            self.antennas.append(antenna)


    def deleteLib(self,verbose=False):
        '''
        This is intended to delete the library after it is used for interpolation to free up ram.
        '''
        import gc
        
        if verbose == True:
            def memTools():
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

    def loadLib(self,pre_split = False,build_lib = True):
        for antenna in self.antennas:
            antenna.loadLib(pre_split,build_lib = build_lib)
            self.solutions = self.solutions[numpy.isin(self.solutions,antenna.lib.solutions)] #will catch any inconsistencies between solutions present in library and intended list of solutions. 
    def loadConcaveHull(self):
        for antenna in self.antennas:
            antenna.loadConcaveHull()

    def getBeams(self, n , verbose = False):
        '''
        The goal of this function is to determine the beam and subbeam time delays 
        semiautomatically for the station.
        
        Currently the minimum time shift is assigned to the smallest baseline.  Thus
        every other timeshift resulting from larger baselines must be a multiple of the
        minimum baseline. i.e. all subbeam baselines must be in integer multiples of 
        the minimum baseline.  
        Currently requires all other baselines to be an integer multiple of the minimum baseline
        
        n should be average index of refraction of array (used for angle estimates)
        self.digital_sampling_periodshould be in nanoseconds (used for angle estimates)
        '''
        print('Using:\npower_calculation_sum_length = %i\npower_calculation_interval = %i\nn_baselines = %i'%(self.power_calculation_sum_length,self.power_calculation_interval,self.n_baselines))
        n_antennas = len(self.antennas)
        min_antennas_per_subbeam =  2#numpy.round(n_antennas/3) #if the number of antennas satisfying a baseline is less than this that beam won't be counted
        
        depths = numpy.array([antenna.z for antenna in self.antennas])
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
        colormap = pylab.cm.gist_ncar #nipy_spectral, Set1,Paired   
        self.beam_colors = [colormap(i) for i in numpy.linspace(0, 1,self.n_beams+1)]

    def calculateNoiseRMS(self):
        noise_rms = []
        for antenna in self.antennas:
            antenna.calculateNoiseRMS()
            noise_rms.append(antenna.noise_rms)
        self.noise_rms = numpy.array(noise_rms)

############################################################

class Antenna:
    #Currently this class is drastically underutilized.  Should transition to having this contain antenna orientations, etc. 
    #Most antenna information is currently carried through the config file which is clunky. 

    def __init__(self, x, y, z, alpha_deg, beta_deg, gamma_deg, antenna_type, noise_temperature, resistance, lib_dir, label, solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2'])):
        """
        x, y, z given relative to station center
        #When this is called in antarcticsim.py it uses the input x_antenna + x_station, y_antenna + y_station, z_antenna + z_station
        #which does not seem to me to be be a relative position, as it adds the station position?
        """
        self.x = x
        self.y = y
        self.z = z
        self.alpha_deg = alpha_deg
        self.beta_deg  = beta_deg
        self.gamma_deg = gamma_deg
        self.lib_dir = lib_dir
        self.label = label
        self.noise_temperature = noise_temperature
        self.resistance = resistance 

        self.R = eulerRotationMatrix(numpy.deg2rad(self.alpha_deg), numpy.deg2rad(self.beta_deg), numpy.deg2rad(self.gamma_deg))
        self.R_inv = numpy.linalg.inv(self.R)     

        accepted_types = numpy.array(['simple', 'dipole'])
        if numpy.isin(antenna_type,accepted_types):
            self.antenna_type = antenna_type
        else:
            print('Selection of antenna type did not match predefined values.  Using default type:')
            self.antenna_type = accepted_types[0]
            print(self.antenna_type)

        self.accepted_solutions = numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2'])
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
        if numpy.isin('lib',list(self.__dict__.keys())):
            print('Antenna library already loaded.')
        else:
        '''
        if numpy.logical_and(pre_split == False,numpy.logical_not(len(self.solutions) == len(self.accepted_solutions))):
            print('Limiting Solution Types Currently only works for pre_split = True, using default solution types.')
            self.solutions = self.accepted_solutions
        self.lib = gnosim.trace.refraction_library_beta.RefractionLibrary(self.lib_dir,solutions=self.solutions,pre_split = pre_split,build_lib = build_lib)
        self.solutions = self.lib.solutions #Catches mistakes if the refraction library has a varying number of antennas.
    def deleteLib(self,verbose=False):
        '''
        This is intended to delete the library after it is used for interpolation to free up ram.
        '''
        import gc
        if verbose == True:
            def memTools():
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
        print('Loading Hull For:',self.lib_dir)
        self.concave_hull = {}
        indir = self.lib_dir.replace('*.h5','concave_hull')
        if os.path.isdir(indir) == False:
            print('Hull not previously generated, calculating now.')
            self.lib.saveEnvelope( self.lib_dir.replace('*.h5','') )
        chull = self.lib.loadEnvelope( indir ,store_fit_data = False)
        for dkey in self.lib.data.keys():
            self.concave_hull[dkey] = {}
            self.concave_hull[dkey]['z_min'] = chull[dkey]['z_min']
            self.concave_hull[dkey]['z_max'] = chull[dkey]['z_max']
            self.concave_hull[dkey]['f_inner_r_bound'] = chull[dkey]['f_inner_r_bound']
            self.concave_hull[dkey]['f_outer_r_bound'] = chull[dkey]['f_outer_r_bound']

    def addTimingInfo(self, system_response_dir, antenna_response_dir):
        '''
        Expects the timing info that is calculated by   
        '''
        self.antenna_response_dir = antenna_response_dir
        self.system_response_dir = system_response_dir

        antenna_response = numpy.load(self.antenna_response_dir)
        electronic_response = numpy.load(self.system_response_dir)

        freqs, h_fft = numpy.hsplit(antenna_response, 2)
        freqs, sys_fft = numpy.hsplit(electronic_response, 2)
        self.h_fft = numpy.ravel(h_fft)
        self.sys_fft = numpy.ravel(sys_fft)
        self.freqs_response =  numpy.absolute(numpy.ravel(freqs).astype(float))

        t_step = 1/(2*max(self.freqs_response))*1e9 #ns
        self.signal_times = numpy.arange(-(len(self.freqs_response)-1),(len(self.freqs_response)-1))*t_step #To increase time duration of signal I should just need to upsample?


    def calculateNoiseRMS(self):
        noise_signal  = numpy.array([])
        for i in range(100):
            #Values below are mostly just filler to get just the noise (As calculated for the Askaryan pulse).  I am doing this
            #by just calculating signals with 0 energy.  The noise is overlaying on them, but because there is no signal we just
            #get the noise.  Not pretty but....
            noise_signal_i = gnosim.interaction.askaryan.quickSignalSingle( 0,1,0,1.8,\
                      0,0,0,self.signal_times,self.h_fft,self.sys_fft,self.freqs_response,\
                      plot_signals=False,plot_spectrum=False,plot_potential = False,\
                      include_noise = True, resistance = self.resistance, noise_temperature = self.noise_temperature)[3]
            noise_signal = numpy.append(noise_signal,noise_signal_i)
        self.noise_rms = numpy.std(noise_signal)

    def constructAntennaPattern(self):
        '''
        This will hopefully create whatever is needed for a given antenna type, including beam pattern, polarization sensitivity, etc.
        As this is intended to do a fair bit I am not quite yet sure how to handle it.
        '''
        if self.antenna_type == simple:
            print('I will figure out the harder version first and backsolve')
        elif self.antenna_type == dipole:
            polarization_sensitivity_theta = 0.0    #Angle from pole (in antenna frame) the antenna is most sensitive to polarization.
            polarization_sensitivity_phi = 0.0      #Azimuthal angle (from x in antenna frame) the antenna is most sensitive to polarization.



class test:
    def __init__(self,x):
        self.x = x

    def delX(self):
        del self.__dict__['x']
############################################################


if __name__ == "__main__":
    x = 0.0
    y = 0.0
    z = -200.0
    alpha_deg = 0.0
    beta_deg = 0.0
    gamma_deg = 0.0
    antenna_type = 'dipole'
    import yaml
    config_file = '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/real_config.py'
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