#!/usr/bin/env python

"""
Simulation
"""

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

sys.path.append("/home/dsouthall/Projects/GNOSim/")
import gnosim.utils.quat
import gnosim.earth.earth
import gnosim.earth.antarctic
import gnosim.trace.refraction_library_beta
import gnosim.interaction.askaryan
import gnosim.sim.detector

pylab.ion()

############################################################


class Sim:

    def __init__(self, config_file,pre_split = False):
        #pre_split False unless using a library already sorted into different
        #directories by solution type.
        self.config_file = config_file
        self.pre_split = pre_split
        #self.config = eval(''.join(open(config_file).readlines()))
        self.config = yaml.load(open(config_file))
        self.detector()     
        self.info_dtype = numpy.dtype([('eventid','i'),('station','i'),('antenna','i'),('has_solution','i'),('solution','S10'),('time','f'),('distance','f'),('theta_ant','f'),('observation_angle','f'),('electric_field','f'),('weighted_freq','f'),('a_h','f'),('a_v','f')])
        # List attributes of interest
        self.keys = ['t', 'd', 'theta', 'theta_0', 'a_v', 'a_h']
        self.n_antenna = sum([len(self.stations[s].antennas) for s in range(len(self.stations))])
        print('Total Number of Antennas = ', self.n_antenna)

    def detector(self):
        self.lib = {}
        for key in self.config['antenna_definitions'].keys():
            self.lib[key] = gnosim.trace.refraction_library_beta.RefractionLibrary(self.config['antenna_definitions'][key]['lib'],pre_split = self.pre_split)
            #if library is split, I want to invoke it here with some flag.  I will need an altered version of RefractionLibrary
        self.stations = []
        for ii in range(0, self.config['stations']['n']):
            x_station, y_station, z_station = self.config['stations']['positions'][ii]
            self.stations.append(gnosim.sim.detector.Station(x_station, y_station, z_station))
            for jj in range(0, self.config['antennas']['n']):
                antenna_type = self.config['antennas']['types'][jj]
                x_antenna, y_antenna, z_antenna = self.config['antennas']['positions'][jj]
                antenna = gnosim.sim.detector.Antenna(x_antenna + x_station, y_antenna + y_station, z_antenna + z_station, 
                                                      self.lib[self.config['antennas']['types'][jj]],
                                                      self.config['antenna_definitions'][antenna_type]['frequency_low'],
                                                      self.config['antenna_definitions'][antenna_type]['frequency_high'],
                                                      self.config)
                self.stations[ii].antennas.append(antenna)

    def event(self, energy_neutrino, phi_0, theta_0, x_0, y_0, z_0, eventid, anti=False):
            p_interact = gnosim.earth.earth.probInteract(energy_neutrino, z_0, anti=anti)
        
            # Probability for neutrino to make it through the Earth
            p_earth = gnosim.earth.earth.probSurvival(energy_neutrino, theta_0, elevation=z_0, anti=anti)

            p_detect = 0.

            # Placeholders
            inelasticity = -999.
            #electric_field_max = 0. Now per antenna, not event
            dic_max = {}
            observation_angle_max = -999.
            solution_max = -999.
            index_station_max = -999.
            index_antenna_max = -999.
            
            info = numpy.zeros(  self.n_antenna  , dtype = self.info_dtype) #was using numpy.empy, switch to 0s
            #Old indexing -> (len(self.stations)*len(self.stations[0].antennas))
            
            # Loop over stations
            for index_station in range(0, len(self.stations)):
                # Loop over station antennas
                station_label = 'station'+str(index_station)
                for index_antenna in range(0, len(self.stations[index_station].antennas)):
                    electric_field_max = 0. #This clearly makes no sense.  Reset every antenna??  Why?  So bizarre.  My info object outputs more info. 
                    x_antenna = self.stations[index_station].antennas[index_antenna].x
                    y_antenna = self.stations[index_station].antennas[index_antenna].y
                    z_antenna = self.stations[index_station].antennas[index_antenna].z
                    r = numpy.sqrt((x_0 - x_antenna)**2 + (y_0 - y_antenna)**2)
                    detector_type = ['simple0','simple1','simple2','simple3','simple4','simple5','simple6','simple7'][index_antenna] #temporary
                    flag_array = [self.in_flag_array[station_label][detector_type]['direct'][eventid],self.in_flag_array[station_label][detector_type]['cross'][eventid],self.in_flag_array[station_label][detector_type]['reflect'][eventid],self.in_flag_array[station_label][detector_type]['direct_2'][eventid],self.in_flag_array[station_label][detector_type]['cross_2'][eventid],self.in_flag_array[station_label][detector_type]['reflect_2'][eventid]]
                    #flag_array, dic_array = self.stations[index_station].antennas[index_antenna].lib.event(r, z_0)                                              
                    temporary_info = numpy.zeros(  numpy.size(self.stations[index_station].antennas[index_antenna].lib.solutions)  , dtype = self.info_dtype)
                    #temporary_info['eventid'] = eventid
                    #temporary_info['station'] = index_station
                    #temporary_info['antenna'] = index_antenna
                    has_solution = 0
                    #temporary_info['has_solution'] = 0
                    if numpy.any(flag_array):
                        has_solution = 1
                        p_detect = 1.  #This essentially flags if any detector sees any solution type.
                        #temporary_info['has_solution'] = 1
                        #want an antenna specific p_detect that flags if a particular detector sees any solution type.  
                        inelasticity = gnosim.interaction.inelasticity.inelasticity(energy_neutrino, mode='cc') # GENERALIZE THIS LATER #DS not sure why this is in the loop.  Does not depend on any loop variables, only need to calculate once.  Maybe because it was going to be 'generalized'?
                        frequency = numpy.linspace(self.stations[index_station].antennas[index_antenna].frequency_low,
                                                   self.stations[index_station].antennas[index_antenna].frequency_high,
                                                   100) # GHz
                        index_of_refraction = gnosim.earth.antarctic.indexOfRefraction(z_0, ice_model=self.config['detector_volume']['ice_model'])
                        vector_neutrino = gnosim.utils.quat.angToVec(phi_0, theta_0) # Direction neutrino came from
                        phi_ray = numpy.degrees(numpy.arctan2(y_0 - y_antenna, x_0 - x_antenna)) % 360. # deg
                        
                        for ii, solution in enumerate(self.stations[index_station].antennas[index_antenna].lib.solutions):
                            #if flag_array[ii]:
                            if self.in_flag_array[station_label][detector_type][solution][eventid]:
                                # Direction of outgoing ray from antenna to interaction vertex
                                
                                #vector_ray = gnosim.utils.quat.angToVec(phi_ray, dic_array[ii]['theta'])
                                vector_ray = gnosim.utils.quat.angToVec(phi_ray, self.in_dic_array[station_label][detector_type][solution]['theta'][eventid])
                                observation_angle = gnosim.utils.quat.angTwoVec(vector_neutrino, vector_ray) # deg
                                d = self.in_dic_array[station_label][detector_type][solution]['d'][eventid] # m
                                electric_field \
                                    = gnosim.interaction.askaryan.electricField(frequency, d, observation_angle,
                                                                                energy_neutrino, inelasticity, 
                                                                                'cc', index_of_refraction) # V m^-1 GHz^-1, dimensionless
                                electric_field *= self.in_dic_array[station_label][detector_type][solution]['a_v'][eventid] # COME BACK TO GENERALIZE THIS
                                electric_array, electric_field, weighted_freq = self.stations[index_station].antennas[index_antenna].electricField(frequency, electric_field, self.in_dic_array[station_label][detector_type][solution]['theta_ant'][eventid]) # V m^-1 #THIS WAS CHANGED THETA WAS ADDED
                                
                                temporary_info[ii] = numpy.array([(eventid,index_station,index_antenna,has_solution,solution,self.in_dic_array[station_label][detector_type][solution]['t'][eventid],self.in_dic_array[station_label][detector_type][solution]['d'][eventid],self.in_dic_array[station_label][detector_type][solution]['theta_ant'][eventid],observation_angle,electric_field,weighted_freq,self.in_dic_array[station_label][detector_type][solution]['a_h'][eventid],self.in_dic_array[station_label][detector_type][solution]['a_v'][eventid])],dtype = self.info_dtype)
                                if electric_field > electric_field_max:
                                    electric_field_max = electric_field;
                                    observation_angle_max = observation_angle
                                    solution_max = ii
                                    solution_type_max = solution
                                    max_eventid = eventid
                                    index_station_max = index_station
                                    index_antenna_max = index_antenna
                        #NEED WAY TO HANDLE SCENARIO WHERE NONE!!! ARE A SOLUTION.  PICK UP ON THIS.
                        dic_max['d'] = self.in_dic_array[station_label][detector_type][solution_type_max]['d'][max_eventid]
                        dic_max['r'] = self.in_dic_array[station_label][detector_type][solution_type_max]['r'][max_eventid]
                        dic_max['t'] = self.in_dic_array[station_label][detector_type][solution_type_max]['t'][max_eventid]
                        dic_max['theta'] = self.in_dic_array[station_label][detector_type][solution_type_max]['theta'][max_eventid]
                        dic_max['theta_ant'] = self.in_dic_array[station_label][detector_type][solution_type_max]['theta_ant'][max_eventid]
                        dic_max['a_h'] = self.in_dic_array[station_label][detector_type][solution_type_max]['a_h'][max_eventid]
                        dic_max['a_v'] = self.in_dic_array[station_label][detector_type][solution_type_max]['a_v'][max_eventid]
                        dic_max['z'] = self.in_dic_array[station_label][detector_type][solution_type_max]['z'][max_eventid]
                        info[ sum([len(self.stations[s].antennas) for s in range(0,index_station)]) + index_antenna] = temporary_info[numpy.argmax(temporary_info['electric_field'])]
                    #self.in_flag_array[station_label][antenna_label][solution] #Want a way to include this in info.  I.e. a binary has_solution flag for each antenna for each event
                    else:
                        #This event has no solution for this antenna
                        has_solution = 0
                        info[ sum([len(self.stations[s].antennas) for s in range(0,index_station)]) + index_antenna] = numpy.array([(eventid,index_station,index_antenna,has_solution,'',-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0,-999.0)],dtype = self.info_dtype)
                    

            return p_interact, p_earth, p_detect, inelasticity, electric_field_max, dic_max, observation_angle_max, solution_max, index_station_max, index_antenna_max, info
            #return p_interact, p_earth, p_detect, electric_field_direct, electric_field_crossover, electric_field_reflect,  dic_direct, dic_crossover, dic_reflect
    
    
    def make_Envelope(self, plot_hulls = False,verbose = False):
        '''
        Creates two functions for each antenna/solution type pair.  These take a depth value
        and output a corresponding bounding r value at that depth.  The pair of functions can
        be used as an inner and outer bound.  The dictionary that contains these functions also
        contains bounding depth values for each pair of functions, outside of which the functions
        are extrapolating, so these should be used as an initial gate as to whether an r-z pair
        is of a paticular solution type. These outputs are all within self.concave_hull.  This is
        currently all done on the simulation side, but this can and should eventually be part of 
        the library itself.
        
        If @plot_hulls is True than a plot of the libraries set of trace points, and the provided hull
        are produced for each solution type. 
        
        The n_bins values in self.concave_hull are tuned by eye using these plots, are vary depending
        the ray density/point spacing.  They should be double checked each time a new libary is
        created.  Try not to erase the current values, and just comment out.  
        '''
        
        legend_locs = {'direct':'upper right','cross':'upper right','reflect':'upper right','direct_2':'lower right','cross_2':'lower right','reflect_2':'lower right'}        
        self.concave_hull = {}
        for antenna_label in self.lib.keys():
            if verbose:
                print('Making Envelopes for:', antenna_label)
            self.concave_hull[antenna_label] = {'direct':{'n_bins':1000},'cross':{'n_bins':1500},'reflect':{'n_bins':2000},'direct_2':{'n_bins':500},'cross_2':{'n_bins':400},'reflect_2':{'n_bins':2000}} # worked for 120 rays
            #self.concave_hull[antenna_label] = {'direct':{'n_bins':300},'cross':{'n_bins':1500},'reflect':{'n_bins':2000},'direct_2':{'n_bins':500},'cross_2':{'n_bins':400},'reflect_2':{'n_bins':2000}} # worked for 60 rays
            for solution_type in ['direct','cross','reflect','direct_2','cross_2','reflect_2']:
                if verbose:
                    print('\tSolution Type: %10s \tNumber of points: %i'%( solution_type , len(self.lib[antenna_label].data[solution_type]['z'])))
                if (len(self.lib[antenna_label].data[solution_type]['z']) == 0):
                    print('\tNot enough points, returning 0 value hull')
                    self.concave_hull[antenna_label][solution_type]['z_min'] = 0.0001
                    self.concave_hull[antenna_label][solution_type]['z_max'] = -0.0001#Want upper bound to be smaller than lower bound so you can never be appropriately between them in error scenario. 
                    self.concave_hull[antenna_label][solution_type]['f_inner_r_bound'] = scipy.interpolate.interp1d([-1000000,1000000],[0.0001,0.0001],bounds_error=False,fill_value = 0.0001)
                    self.concave_hull[antenna_label][solution_type]['f_outer_r_bound'] = scipy.interpolate.interp1d([-1000000,1000000],[-0.0001,-0.0001],bounds_error=False,fill_value = -0.0001) #Want outer bound to be smaller than in bound so you can never be appropriately between them in error scenario. 
                else:
                    test_z = self.lib[antenna_label].data[solution_type]['z']
                    test_r = self.lib[antenna_label].data[solution_type]['r']
                    
                    z_bins = numpy.linspace(min(test_z)-1.0,max(test_z)+1.0,self.concave_hull[antenna_label][solution_type]['n_bins'])
                    z_out = numpy.zeros(self.concave_hull[antenna_label][solution_type]['n_bins']-1)
                    r_out = numpy.zeros(self.concave_hull[antenna_label][solution_type]['n_bins']-1)
                    z_in = numpy.zeros(self.concave_hull[antenna_label][solution_type]['n_bins']-1)
                    r_in = numpy.zeros(self.concave_hull[antenna_label][solution_type]['n_bins']-1)
                    
                    tz = numpy.tile(test_z,(len(z_bins)-1,1))
                    bz = numpy.tile(z_bins, (len(test_z),1)).T
                    cut = numpy.logical_and(numpy.greater_equal(tz, bz[0:-1]),numpy.less_equal(tz , bz[1:]))
                    use_in_interp = numpy.where(numpy.sum(cut,axis=1)!=0)[0]
                    
                    for bin in range(1,self.concave_hull[antenna_label][solution_type]['n_bins']):
                        if (numpy.any(numpy.isin(use_in_interp,bin-1))):
                            r_out[bin-1] = max(test_r[numpy.where(cut[bin-1,:]) ])
                            r_in[bin-1] = min(test_r[numpy.where(cut[bin-1,:]) ])
                            z_in[bin-1] = max(numpy.unique(test_z[numpy.where(cut[bin-1,:])][ numpy.where(test_r[numpy.where(cut[bin-1,:]) ] == r_in[bin-1])]))
                            z_out[bin-1] = max(numpy.unique(test_z[numpy.where(cut[bin-1,:])][ numpy.where(test_r[numpy.where(cut[bin-1,:]) ] == r_out[bin-1])]))
                            if (bin == self.concave_hull[antenna_label][solution_type]['n_bins']-1):
                                #shallowest
                                #print('Trying to adjust shallowest')
                                r_out[bin-1] = max(test_r[numpy.isclose(test_z,max(test_z),atol = 0.5)])
                                z_out[bin-1] = max(test_z)
                    #These could be calculated and stored in the original h5 file, then called to make interp1d and max/min within the library, this would save time.
                    z_out = z_out[use_in_interp]
                    r_out = r_out[use_in_interp]
                    z_in = z_in[use_in_interp]
                    r_in = r_in[use_in_interp]
                    
                    self.concave_hull[antenna_label][solution_type]['f_inner_r_bound'] = scipy.interpolate.interp1d(z_in,r_in,bounds_error=False,fill_value = (r_in[0],r_in[-1])) #fill_value=max(r_in))#,kind='cubic') #given z, give r, want big value for fill, because this is region where solution shouldn't exist, so a test of is this > f_in then solution should be false
                    self.concave_hull[antenna_label][solution_type]['f_outer_r_bound'] = scipy.interpolate.interp1d(z_out,r_out,bounds_error=False,fill_value = (r_out[0],r_out[-1]))# fill_value=min(r_out))#,kind='cubic') These make boundaries weird but I think are a necessary evil?  Unless I match each with an z_min, z_max?  Could do....,  I can give interp1d two fill values so it fits well up to min/max z
                    
                    self.concave_hull[antenna_label][solution_type]['z_min'] = min(z_in[0],z_out[0])
                    self.concave_hull[antenna_label][solution_type]['z_max'] = max(z_in[-1],z_out[-1])

                if plot_hulls:
                    fig1, ax1 = pylab.subplots()
                    library = pylab.scatter(test_r,test_z,c='b')
                    pylab.xlabel('r(m)',fontsize=20)
                    pylab.ylabel('z(m)',fontsize=20)
                    pylab.title('Convex Hull for %s'%(solution_type),fontsize=20)
                    pylab.scatter(r_out,z_out,c='m')
                    pylab.scatter(r_in,z_in,c='m')
                    lines = pylab.hlines([self.concave_hull[antenna_label][solution_type]['z_min'],self.concave_hull[antenna_label][solution_type]['z_max']],ax1.get_xlim()[0],ax1.get_xlim()[1],lw='1.5',colors='r',linestyles = 'dotted')
                    
                    plot_z = numpy.linspace(min(test_z),max(test_z),10000)
                    plot_z[0] = min(test_z)
                    plot_z[-1] = max(test_z)
                    in_plot = pylab.plot(self.concave_hull[antenna_label][solution_type]['f_inner_r_bound'](plot_z),plot_z,c='r',lw=4)
                    pylab.plot(self.concave_hull[antenna_label][solution_type]['f_outer_r_bound'](plot_z),plot_z,c='r',lw=4)
                    #plot_r = numpy.linspace(min(test_r),max(test_r),10000)
                    #plot_r[0] = min(test_r)
                    #plot_r[-1] = max(test_r)
                    #low_plot = pylab.plot(plot_r,self.lib[antenna_label].envelope[solution_type]['low'](plot_r),c='g',lw=4,linestyle = '--')
                    #pylab.plot(plot_r,lib[antenna_label].envelope[solution_type]['high'](plot_r),c='g',lw=4,linestyle = '--')
                    out_points = pylab.scatter(r_out,z_out,c='m',s=50)
                    in_points = pylab.scatter(r_in,z_in,c='c',s=50)
                    pylab.legend((library,in_plot[0],lines,out_points,in_points),('Trace Library Points','Concave Hull', 'Accepted Depth Window', 'Points Used To\nCreate Outter Bound','Points Used To\nCreate Inner Bound' ),loc = legend_locs[solution_type],fontsize=16)
                    #pylab.legend((library,low_plot[0],in_plot[0],lines,out_points,in_points),('Trace Library Points','Old Hull','New Hull', 'Accepted Depth Window', 'Points Used To\nCreate Outter Bound','Points Used To\nCreate Inner Bound' ),loc = legend_locs[solution_type],fontsize=16)
                if plot_hulls:
                    input("Press any key to clear current antenna plots and move to next")
            if plot_hulls:
                input("Press any key to clear all antenna plots and continue simulation")

            
            
    def griddata_Event(self, x_query, y_query , z_query, method = 'cubic'):
        '''
        This function takes a set of x,y,z coordinates and determines whether each set
        is within the set of solutions.  First it checks the points against the
        self.concave_hull bounding functions, and the locates which triangle each
        point is within referencing the self.delaunay grid created elsewhere.  
        Using barycentric weighting of the 3 corners of the triangle, an average
        value is calculated to estimate the coresponding information about the pair.
        
        Right now this expects r_query,z_query to be centered coordinates
        '''
        
        self.in_dic_array = {}
        self.in_flag_array = {}
        for index_station in range(0, len(self.stations)):
            station_label = 'station'+str(index_station)
            self.in_dic_array[station_label] = {}
            self.in_flag_array[station_label] = {}
            for index_antenna, antenna_label in enumerate(self.lib.keys()):
                #antenna_label = self.lib.keys()[index_antenna]
                print('Running Events Through Griddata Interpolation for:', antenna_label)
                x_antenna = self.stations[index_station].antennas[index_antenna].x
                y_antenna = self.stations[index_station].antennas[index_antenna].y
                #z_antenna = self.stations[index_station].antennas[index_antenna].z
                r_query = numpy.sqrt((x_query - x_antenna)**2 + (y_query - y_antenna)**2)
                
                if ((type(r_query) != numpy.ndarray) or (type(z_query) != numpy.ndarray)):
                    if ((type(r_query) != list) or (type(z_query) != list)):
                        r_query = numpy.array([r_query])
                        z_query = numpy.array([z_query])
                    else:
                        r_query = numpy.array(r_query)
                        z_query = numpy.array(z_query)
                
                self.in_dic_array[station_label][antenna_label] = {}
                self.in_flag_array[station_label][antenna_label] = {}
                for solution in self.lib[antenna_label].data.keys():
                    print('\tSolution Type:', solution)
                    self.in_dic_array[station_label][antenna_label][solution] = {}

                    

                    in_bound = numpy.logical_and((z_query >= self.concave_hull[antenna_label][solution]['z_min']),z_query <= self.concave_hull[antenna_label][solution]['z_max'])
                    r_in_hull = numpy.logical_and((r_query >= self.concave_hull[antenna_label][solution]['f_inner_r_bound'](z_query)),(r_query <= self.concave_hull[antenna_label][solution]['f_outer_r_bound'](z_query)))
                    has_solution = numpy.logical_and(in_bound,r_in_hull)
                    
                    if numpy.all( has_solution == False ):
                        print('No solutions found for', antenna_label, solution)
                        self.in_flag_array[station_label][antenna_label][solution] = numpy.array([False]*len(r_query))
                        for key in self.lib[antenna_label].data[solution].keys():
                            self.in_dic_array[station_label][antenna_label][solution][key] = []
                        continue

                    zm_query = numpy.ma.masked_array( z_query, mask = ~has_solution)
                    rm_query = numpy.ma.masked_array( r_query, mask = ~has_solution)
                    no_solution_index = numpy.where(~has_solution)
                    #no_solution_index = numpy.where(simplices_index == -1) #make sure this is as I want.  
                    self.in_flag_array[station_label][antenna_label][solution] = has_solution
                    
                    # New weighting method using barycentric coordinates
                                  
                    for key in self.lib[antenna_label].data[solution].keys():
                        self.in_dic_array[station_label][antenna_label][solution][key] = numpy.ma.filled(numpy.ma.masked_array(scipy.interpolate.griddata((self.lib[antenna_label].data[solution]['r'],self.lib[antenna_label].data[solution]['z']),self.lib[antenna_label].data[solution][key],(rm_query, zm_query),method=method,fill_value=-999.0),mask = ~has_solution,fill_value = -999.0))
                        #Currently can poduce nan values if griddata thinks it is extrapolating but hull says it is in region 
        print('Finished griddata_Event')
        #return self.in_dic_array[station_label], self.in_flag_array

    def throw(self, energy_neutrino=1.e9, 
              theta_0=None, phi_0=None, x_0=None, y_0=None, z_0=None, 
              anti=False, n_events=10000, detector_volume_radius=6000., detector_volume_depth=3000., 
              outfile=None,seed = None,method = 'cubic'):
              
        #seed for testing purposes (if want replicated data)
        if (seed != None):
            numpy.random.seed(seed)
            
        #Loading Hulls (or creating if hulls have not been previously determined in the necessary folder)
        self.concave_hull = {}
        #self.make_Envelope(plot_hulls=False, verbose = True)
        for lkey in self.lib.keys():
            print('Loading Hull For:', lkey)
            self.concave_hull[lkey] = {}
            indir = self.config['antenna_definitions'][lkey]['lib'].replace('*.h5','concave_hull')
            if os.path.isdir(indir) == False:
                print('Hull not previously generated, calculating now.')
                (self.lib[lkey]).saveEnvelope( self.config['antenna_definitions'][lkey]['lib'].replace('*.h5','') )
            chull = (self.lib[lkey]).loadEnvelope( indir ,store_fit_data = False)
            for dkey in self.lib[lkey].data.keys():
                self.concave_hull[lkey][dkey] = {}
                self.concave_hull[lkey][dkey]['z_min'] = chull[dkey]['z_min']
                self.concave_hull[lkey][dkey]['z_max'] = chull[dkey]['z_max']
                self.concave_hull[lkey][dkey]['f_inner_r_bound'] = chull[dkey]['f_inner_r_bound']
                self.concave_hull[lkey][dkey]['f_outer_r_bound'] = chull[dkey]['f_outer_r_bound']
        
        energy_neutrino = energy_neutrino * numpy.ones(n_events)
    
        #Randomizing direction neutrino came from (characterized by phi_0, theta_0
        phi_0 = numpy.random.uniform(0., 360., size=n_events) # deg
        theta_0 = numpy.degrees(numpy.arccos(numpy.random.uniform(1., -1., size=n_events))) # deg
        
        # Producing neutrino randomly in approximate detector volume
        # NEW CURVATURE
        alpha_max_radians = detector_volume_radius / gnosim.utils.constants.radius_earth # radians
        alpha = numpy.arccos(numpy.random.uniform(1., numpy.cos(alpha_max_radians), size=n_events)) # radians
        r = gnosim.utils.constants.radius_earth * alpha # m #Seems to be using small angle formula
        # NEW CURVATURE

        phi_vertex = numpy.random.uniform(0., 360., size=n_events) # deg
    
        x_0 = r * numpy.cos(numpy.radians(phi_vertex))
        y_0 = r * numpy.sin(numpy.radians(phi_vertex))
        z_0 = numpy.random.uniform(-1. * detector_volume_depth, 0., size=n_events) # m #maybe something to double check later, make sure doesn't give solutions outside of earth
        #TESTING FOR ERROR SCENARIO
        z_0[0] = 1000.0
        """
        pylab.figure()
        pylab.scatter(x, y)
        pylab.figure()
        pylab.hist(r, bins=40)
        pylab.figure()
        pylab.hist(z, bins=40)
        pylab.figure()
        pylab.hist(phi, bins=40)
        pylab.figure()
        pylab.hist(theta, bins=40)
        """

        p_interact = numpy.zeros(n_events)
        p_earth = numpy.zeros(n_events)
        p_detect = numpy.zeros(n_events)
        inelasticity = numpy.zeros(n_events)
        electric_field_max = numpy.zeros(n_events)
        observation_angle_max = numpy.zeros(n_events)
        solution_max = numpy.zeros(n_events)
        index_station_max = numpy.zeros(n_events)
        index_antenna_max = numpy.zeros(n_events)
        t_max = numpy.zeros(n_events)
        d_max = numpy.zeros(n_events)
        theta_ray_max = numpy.zeros(n_events)
        theta_ant_max = numpy.zeros(n_events)
        a_v_max = numpy.zeros(n_events)
        a_h_max = numpy.zeros(n_events)
        info = numpy.empty(n_events * self.n_antenna , dtype = self.info_dtype)

        
        print('About to run griddata_Event:')
        ############################################################################
        #Interpolating values from using griddata:
        ############################################################################
        
        #If memory becomes an issue this might need to be adapted to run for chunks of events 
        self.griddata_Event(x_0, y_0, z_0, method = method)   
        print('Succesfully ran griddata_Event:')
        
        ############################################################################
        #Using interpolated values for further calculations on an event/event basis:
        ############################################################################
        for ii in range(0, n_events):
            if(ii%(n_events/100) == 0):
                print ('Event (%i/%i)'%(ii, n_events)) #might want to comment out these print statements to run faster and spew less
            
            p_interact[ii], p_earth[ii], p_detect[ii], inelasticity[ii], electric_field_max[ii], dic_max, observation_angle_max[ii], solution_max[ii], index_station_max[ii], index_antenna_max[ii], info[(ii * self.n_antenna ):((ii+1) * self.n_antenna )] \
                = self.event(energy_neutrino[ii], phi_0[ii], theta_0[ii], x_0[ii], y_0[ii], z_0[ii], ii, anti=anti)

            if p_detect[ii] == 1.:
                t_max[ii] = dic_max['t']
                d_max[ii] = dic_max['d']
                theta_ray_max[ii] = dic_max['theta']
                theta_ant_max[ii] = dic_max['theta_ant']
                a_v_max[ii] = dic_max['a_v']
                a_h_max[ii] = dic_max['a_h']

        cut = (p_detect == 1.)
        cut_inv = numpy.logical_not(cut)
        
        
        if outfile:
            file = h5py.File(outfile, 'w')
            # ORIGINAL 28 MAY 2014
            #file.attrs['geometric_factor'] = (4. * numpy.pi) * (numpy.pi * detector_volume_radius**2 * detector_volume_depth) # m^3 sr
            # ORIGINAL 28 MAY 2014
            # NEW CURVATURE
            file.attrs['geometric_factor'] = (4. * numpy.pi) \
                                             * (2. * numpy.pi * gnosim.utils.constants.radius_earth**2 \
                                                * (1. - numpy.cos(detector_volume_radius / gnosim.utils.constants.radius_earth))\
                                                * detector_volume_depth) # m^3 sr
            # NEW CURVATURE

            file.attrs['config'] = self.config_file
            #file.attrs['ice_model'] = gnosim.earth.antarctic.ice_model_default
            file.attrs['ice_model'] = self.config['detector_volume']['ice_model']

            file.create_dataset('energy_neutrino', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('inelasticity', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('x_0', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('y_0', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('z_0', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('theta_0', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('phi_0', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)

            file.create_dataset('p_interact', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('p_earth', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('p_detect', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            
            file.create_dataset('index_station', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('index_antenna', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)

            file.create_dataset('electric_field', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('observation_angle', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('solution', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('t', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('d', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('theta_ray', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('theta_ant', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('a_v', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('a_h', (n_events,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
            file.create_dataset('info', ( n_events * self.n_antenna , ) , dtype=self.info_dtype, compression='gzip', compression_opts=9, shuffle=True)


            file['energy_neutrino'][...] = energy_neutrino
            file['inelasticity'][...] = inelasticity
            file['x_0'][...] = x_0
            file['y_0'][...] = y_0
            file['z_0'][...] = z_0
            file['theta_0'][...] = theta_0
            file['phi_0'][...] = phi_0

            file['p_interact'][...] = p_interact
            file['p_earth'][...] = p_earth
            file['p_detect'][...] = p_detect

            file['index_station'][...] = index_station_max
            file['index_antenna'][...] = index_antenna_max

            file['electric_field'][...] = electric_field_max
            file['observation_angle'][...] = observation_angle_max
            file['solution'][...] = solution_max
            file['t'][...] = t_max
            file['d'][...] = d_max
            file['theta_ray'][...] = theta_ray_max
            file['theta_ant'][...] = theta_ant_max
            file['a_v'][...] = a_v_max
            file['a_h'][...] = a_h_max
            file['info'][...] = info

            file.close()

############################################################

if __name__ == "__main__":

    config_file = sys.argv[1]
    energy_neutrino = float(sys.argv[2]) # GeV
    n_events = int(sys.argv[3])
    index = int(sys.argv[4])
    #detector_volume_radius = float(sys.argv[5]) # m, 1200 for Ross surface, 51000 for Minna bluff, >6000 for subterranean
    #detector_volume_depth = float(sys.argv[6]) # m, 500 for Ross and Minna, 3000 for subterranean

    #SEED FOR TESTNG:
    seed = 0
    config_file_fix = config_file.replace('/home/dsouthall/Projects/GNOSim/','')
    config_file_fix = config_file_fix.replace('gnosim/sim/ConfigFiles/Config_dsouthall/','')
    config_file_fix = config_file_fix.replace('./','')
    if (seed != None):
        outfile = '/home/dsouthall/Projects/GNOSim/Output/results_2018_Sep_%s_%.2e_GeV_%i_events_%i_seed_%i.h5'%(config_file_fix.replace('.py', ''),
                                                                    energy_neutrino,
                                                                    n_events,
                                                                    seed,
                                                                    index)
        print('!!!Using Seed!!! Seed: ', seed, '\nOutfile Name: \n', outfile)
    else:
        outfile = '/home/dsouthall/Projects/GNOSim/Output/results_2018_Sep_%s_%.2e_GeV_%i_events_%i.h5'%(config_file_fix.replace('.py', ''),
                                                                energy_neutrino,
                                                                n_events,
                                                                index)
        print('Outfile Name: \n', outfile)
    if os.path.isfile(outfile):
        print('Outfile Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(outfile))
        outfile = './' + outfile.split('/')[-1]
        while os.path.isfile(outfile):
            outfile = outfile.replace('.h5','_new.h5')
            
    #f = h5py.File(outfile, 'w')
   
    my_sim = Sim(config_file, pre_split = True)
    my_sim.throw(energy_neutrino, n_events=n_events, 
                 detector_volume_radius=my_sim.config['detector_volume']['radius'],
                 detector_volume_depth=my_sim.config['detector_volume']['depth'],
                 outfile=outfile,seed=seed)

    #python /home/dsouthall/Projects/GNOSim/sim/antarcticsim.py config energy n_events index 
    #python /home/dsouthall/Projects/GNOSim/gnosim/sim/antarcticsim.py /home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_octo_-200_polar_120_rays.py 1.0e8 50000 1 
    #f.close()

############################################################


