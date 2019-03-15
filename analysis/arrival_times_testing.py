'''
This file was made to serve as a rough wroking environment for Dan Southall.
I am hoping to test the arrival times for various solution/detector combinations
to ensure that they make sense. 

Run in python command line:
exec(open('./gnosim/sim/arrival_times_testing.py').read())
'''
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
import gnosim.trace.refraction_library_beta
from matplotlib.colors import LogNorm
pylab.ion()
import matplotlib.colors

### INPUT INFORMATION ###

#infile = 'results_2018_july_config_octo_-200_polar_1.00e+08_GeV_100000_events_3.h5'
#infile = 'results_2018_july_config_octo_-200_polar_1.00e+08_GeV_20_events_1.h5'
#results_2018_Sep_config_octo_-200_polar_360_rays_1.00e+08_GeV_1000000_events_0_seed_1.h5

infile = 'results_2018_Sep_config_octo_-200_polar_120_rays_1.00e+08_GeV_100_events_0_seed_1.h5'#'results_2018_Sep_config_octo_-200_polar_360_rays_1.00e+08_GeV_1000000_events_0_seed_1.h5'
infile2 = 'results_2018_Sep_config_octo_-200_polar_120_rays_1.00e+08_GeV_100_events_0_seed_1.h5'#'results_2018_Sep_config_octo_-200_polar_120_rays_1.00e+08_GeV_50000_events_0_seed_1.h5'#'results_2018_Aug_config_octo_-200_polar_griddata_double_1.00e+08_GeV_50000_events_0_seed_2.h5'#'results_2018_july_config_octo_-200_polar_delaunay_double_1.00e+08_GeV_50000_events_0_seed_3.h5'#'results_2018_july_config_octo_-200_polar_double_1.00e+08_GeV_50000_events_0_seed_1.h5'
#infile2 = 'results_2018_Aug_config_octo_-200_polar_griddata_double_1.00e+08_GeV_50000_events_0_seed_2.h5'

infilepath = '/home/dsouthall/Projects/GNOSim/Output/'
configfile = 'config_octo_-200_polar_360_rays.py' #just the file (no /'s), put directory below
configfilepath = '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/' #file directory
config = yaml.load(open(configfilepath + configfile))

configfile2 = 'config_octo_-200_polar_120_rays.py'
config2 = yaml.load(open(configfilepath + configfile2))

'''
lib2 = {}
for key in config2['antenna_definitions'].keys():
    lib2[key] = gnosim.trace.refraction_library_beta.RefractionLibrary(config2['antenna_definitions'][key]['lib'])

#Saving Lib
with open('Pickled_Library_Double.pkl','wb') as f:
    pickle.dump(lib2, f, pickle.HIGHEST_PROTOCOL)
'''

''' 
#Code to make a trace library and save/load the object for quicker use in the future

#Making Lib
lib = {}
for key in config['antenna_definitions'].keys():
    lib[key] = gnosim.trace.refraction_library_beta.RefractionLibrary(config['antenna_definitions'][key]['lib'])

#Saving Lib
with open('Pickled_Library.pkl','wb') as f:
    pickle.dump(lib, f, pickle.HIGHEST_PROTOCOL)

#Loading Lib
with open('Pickled_Library.pkl','rb') as f:
   lib = pickle.load(f)

'''
'''
#Loading Lib
with open('Pickled_Library.pkl','rb') as f:
   lib = pickle.load(f)
'''

'''
lib = {}
for key in config['antenna_definitions'].keys():
    lib[key] = gnosim.trace.refraction_library_beta.RefractionLibrary(config['antenna_definitions'][key]['lib'])

with open('Pickled_Library_Double.pkl','rb') as f2:
   lib2 = pickle.load(f2)
'''

reader = h5py.File(infilepath + infile, 'r')
reader2 = h5py.File(infilepath + infile2, 'r')

print('Loaded %s'%(infilepath + infile))
print('Loaded %s'%(infilepath + infile2))
# Interaction vertex has a ray-tracing solution

cut_seen = reader['p_detect'][...] == 1.
cut_seen2 = reader2['p_detect'][...] == 1.

def remove_Undetected(reader):
    '''
    Removes all events with p_detect == 0, returns new reader as dictionary
    ''' 
    new_reader = {}
    info = reader['info'][...]
    cut_seen = reader['p_detect'][...] == 1.
    for key in list(reader.keys()):
        #print(key)
        if key == 'info':
            new_reader['info'] = reader['info'][...][numpy.isin(info['eventid'],numpy.where(reader['p_detect'][...] == 1))]
        else:
            new_reader[key] = reader[key][cut_seen]
        new_reader['eventid_key'] = numpy.where([cut_seen])[0]
        numpy.any(numpy.array(list(reader.keys()))=='eventid_key')
    return new_reader
'''
print ('# Events total                      = %s'%(len(cut_seen)))
print ('# Events with ray-tracing solutions = %s'%(numpy.sum(cut_seen)))
if (len(cut_seen) != numpy.sum(cut_seen)):
    print('Removing undetected events from reader1')
    reader = remove_Undetected(reader)

print ('# Events total                      = %s'%(len(cut_seen2)))
print ('# Events with ray-tracing solutions = %s'%(numpy.sum(cut_seen2)))
if (len(cut_seen2) != numpy.sum(cut_seen2)):
    print('Removing undetected events from reader2')
    reader2 = remove_Undetected(reader2)
    
print ('# Events total                      = %s'%(len(cut_seen3)))
print ('# Events with ray-tracing solutions = %s'%(numpy.sum(cut_seen3)))
if (len(cut_seen3) != numpy.sum(cut_seen3)):
    print('Removing undetected events from reader3')
    reader3 = remove_Undetected(reader3)
    
print ('# Events total                      = %s'%(len(cut_seen4)))
print ('# Events with ray-tracing solutions = %s'%(numpy.sum(cut_seen4)))
if (len(cut_seen4) != numpy.sum(cut_seen4)):
    print('Removing undetected events from reader4')
    reader4 = remove_Undetected(reader4)
'''    
    
def timeDiffHist(reader, index_station , index_antenna, histbins = None, threshold_time = 200, match_solution_type = False):
    '''
    Computes the arrival time differences at each antenna for each event in reader and returns a hist and metadata
    If match_solution_type == True then only time differences with the same solution type as the indexed antenna
    will show up in plots
    ''' 
    info = reader['info'][...]
    timediffs = (info[info['antenna'] == index_antenna]['time'])[info['eventid']] - info['time']
    antennas = info['antenna']
    
    n_antennas = len(numpy.unique(info['antenna']))
    
    axlabels = ['S%iA%i-S%iA%i'%(index_station,index_antenna,index_station,ant) for ant in range(n_antennas)]

    #gating pairs
    if match_solution_type == True:
        solution_cut = info['solution'] == (info[info['antenna'] == index_antenna]['solution'])[info['eventid']]
    else:
        solution_cut = len(info)*[True]
        
    info , index_unique_entries = numpy.unique(info[solution_cut],return_index=True) #redefines info as only unique entries corresponding to match_solution_type request
    
    timediffs = (timediffs[solution_cut])[index_unique_entries]
    antennas = (antennas[solution_cut])[index_unique_entries]
    flagged_events = numpy.unique(info['eventid'][abs(timediffs) > threshold_time])
    #figure production
    if (histbins == None):
        histbins = [n_antennas,1000]
    fig1, ax1 = pylab.subplots()
    ax1.set_xlim(-1,n_antennas+1)
    pylab.hist2d(antennas,timediffs,histbins,norm=LogNorm())
    
    pylab.colorbar()
    pylab.ylabel('Time Difference (ns)',fontsize = 20)
    pylab.xlabel('Detector Pair',fontsize = 20)
    pylab.xticks([(n_antennas - 1.0)*(t+0.5)/n_antennas for t in range(0,n_antennas)])
    ax1.set_xticklabels(axlabels,rotation=90,fontsize=10)
    return fig1, flagged_events, antennas, timediffs, info

def timeDiffThetaAntHist(reader, index_station , index_antenna_1,index_antenna_2, histbins = None, config = None, threshold_time = 10, hist_range = None, match_solution_type = False, plot_expected = False, residual = False,solution = None, title = None,xlim = None, ylim = None, refraction_index = 1.8):
    '''
    Computes the arrival time differences at each antenna for each event in reader and returns a hist and metadata
    If match_solution_type == True then only time differences with the same solution type as the indexed antenna
    will show up in plots
    ''' 
    info = reader['info'][...]

    timediffs = (info[info['antenna'] == index_antenna_1]['time'])[info['eventid']] - info['time']
    
    og_timediffs = (info[info['antenna'] == index_antenna_1]['time'])[info['eventid']] - info['time']
    #antennas = info['antenna']

    n_antennas = len(numpy.unique(info['antenna']))
    
    axlabels = ['S%iA%i-S%iA%i'%(index_station,index_antenna_1,index_station,ant) for ant in range(n_antennas)]

    #Identifying antennas that have different solution types flagged compared to index_antenna_1
    if match_solution_type == True:
        match_solution_cut = info['solution'] == (info[info['antenna'] == index_antenna_1]['solution'])[info['eventid']]
    else:
        match_solution_cut = len(info)*[True]
    if solution == None:
        solution_cut = len(info)*[True]
    else:
        if numpy.isin(solution, ['direct','cross','reflect','direct_2','cross_2','reflect_2']):
            match_solution_cut = numpy.logical_and((info['solution']).astype('U13') == solution , match_solution_cut)
        else:
            print(solution ,'is not a valid solution type, no cut made')
    info , index_unique_entries = numpy.unique(info[match_solution_cut],return_index=True) #redefines info as only unique entries corresponding to match_solution_type request
    
    contains_both = numpy.in1d(info['eventid'], numpy.intersect1d(info['eventid'][info['antenna'] == index_antenna_1] , info['eventid'][info['antenna'] == index_antenna_2])) #cutting any events that have don't have remaining data for each queried antenna
    info = info[contains_both]
    timediffs = ((timediffs[match_solution_cut])[index_unique_entries])[contains_both]
    
    timediffs_2 = timediffs[info['antenna'] == index_antenna_2]
    theta_ant = info['theta_ant'][info['antenna'] == index_antenna_1]
    #flagged_events = numpy.unique(info['eventid'][abs(timediffs) > threshold_time])
    #figure production
    if (histbins == None):
        histbins = [360,1000]
    d = numpy.diff(numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,2][[index_antenna_1,index_antenna_2]])[0]
    c = gnosim.utils.constants.speed_light #m/ns
    if residual:
        flagged_events = numpy.unique(info['eventid'][info['antenna'] == index_antenna_2][abs(timediffs_2 - (d*refraction_index/c)* numpy.cos(numpy.deg2rad(theta_ant))) > threshold_time])
        fig1, ax1 = pylab.subplots()
        nancut = numpy.logical_and(numpy.isnan(theta_ant) == False,numpy.isnan(timediffs_2 - (d*refraction_index/c)* numpy.cos(numpy.deg2rad(theta_ant))) == False)
        pylab.hist2d(theta_ant[nancut],(timediffs_2 - (d*refraction_index/c)* numpy.cos(numpy.deg2rad(theta_ant)))[nancut] ,histbins,norm=LogNorm(),range=hist_range)
        pylab.colorbar()
        pylab.ylabel('Time(Antenna %i)  - Time(Antenna %i)  (ns)'%(index_antenna_1,index_antenna_2),fontsize = 20)
        pylab.xlabel('Theta_ant (Antenna %i)  (degrees)'%(index_antenna_1),fontsize = 20)
    else:
        flagged_events = numpy.unique(info['eventid'][info['antenna'] == index_antenna_2][abs(timediffs_2) > threshold_time])
        fig1, ax1 = pylab.subplots()
        nancut = numpy.logical_and(numpy.isnan(theta_ant) == False,numpy.isnan(timediffs_2) == False)
        print('here')
        pylab.hist2d(theta_ant[nancut],timediffs_2[nancut],histbins,norm=LogNorm())
        pylab.colorbar()
        pylab.ylabel('Time(Antenna %i)  - Time(Antenna %i)  (ns)'%(index_antenna_1,index_antenna_2),fontsize = 20)
        pylab.xlabel('Theta_ant (Antenna %i)  (degrees)'%(index_antenna_1),fontsize = 20)
       
    
    
    if (plot_expected):
        if(config == None):
            print('Set Config file with input parameter config=inputconfig to run.  Returning None')
        else:
            theta = numpy.linspace(0,180,60)
            dt = (d*refraction_index/c)* numpy.cos(numpy.deg2rad(theta))
            pylab.plot(theta,dt,'-r')
    if title != None:
        pylab.title(title)
    if xlim != None:
        pylab.xlim(xlim)
    if ylim != None:
        pylab.ylim(ylim)
    return fig1, flagged_events, og_timediffs, info
    
def thetaThetaHist(reader, index_station , index_antenna_1,index_antenna_2, histbins = None, threshold_time = 200, match_solution_type = False):
    '''
    Computes the arrival time differences at each antenna for each event in reader and returns a hist and metadata
    If match_solution_type == True then only time differences with the same solution type as the indexed antenna
    will show up in plots
    ''' 
    info = reader['info'][...]
    timediffs = (info[info['antenna'] == index_antenna_1]['time'])[info['eventid']] - info['time']
    #antennas = info['antenna']

    n_antennas = len(numpy.unique(info['antenna']))
    
    axlabels = ['S%iA%i-S%iA%i'%(index_station,index_antenna_1,index_station,ant) for ant in range(n_antennas)]

    #Identifying antennas that have different solution types flagged compared to index_antenna_1
    if match_solution_type == True:
        match_solution_cut = info['solution'] == (info[info['antenna'] == index_antenna_1]['solution'])[info['eventid']]
    else:
        match_solution_cut = len(info)*[True]
        
    info , index_unique_entries = numpy.unique(info[match_solution_cut],return_index=True) #redefines info as only unique entries corresponding to match_solution_type request
    
    contains_both = numpy.in1d(info['eventid'], numpy.intersect1d(info['eventid'][info['antenna'] == index_antenna_1] , info['eventid'][info['antenna'] == index_antenna_2])) #cutting any events that have don't have remaining data for each queried antenna
    info = info[contains_both]
    timediffs = ((timediffs[match_solution_cut])[index_unique_entries])[contains_both]
    theta_ant_1 = info['theta_ant'][info['antenna'] == index_antenna_1]
    theta_ant_2 = info['theta_ant'][info['antenna'] == index_antenna_2]
    flagged_events = numpy.unique(info['eventid'][abs(timediffs) > threshold_time])
    #figure production
    if (histbins == None):
        histbins = [360,360]
    fig1, ax1 = pylab.subplots()
    pylab.hist2d(theta_ant_1,theta_ant_2,histbins,norm=LogNorm())
    pylab.colorbar()
    pylab.ylabel('Theta_ant (Antenna %i)  (degrees)'%(index_antenna_2),fontsize = 20)
    pylab.xlabel('Theta_ant (Antenna %i)  (degrees)'%(index_antenna_1),fontsize = 20)
    return fig1, flagged_events, timediffs, info
    
def thetaThetaDiffHist(reader, index_station , index_antenna_1,index_antenna_2, histbins = None, threshold_time = 200, match_solution_type = False,vline=None):
    '''
    Computes the arrival time differences at each antenna for each event in reader and returns a hist and metadata
    If match_solution_type == True then only time differences with the same solution type as the indexed antenna
    will show up in plots
    ''' 
    info = reader['info'][...]
    timediffs = (info[info['antenna'] == index_antenna_1]['time'])[info['eventid']] - info['time']
    #antennas = info['antenna']

    n_antennas = len(numpy.unique(info['antenna']))
    
    axlabels = ['S%iA%i-S%iA%i'%(index_station,index_antenna_1,index_station,ant) for ant in range(n_antennas)]

    #Identifying antennas that have different solution types flagged compared to index_antenna_1
    if match_solution_type == True:
        match_solution_cut = info['solution'] == (info[info['antenna'] == index_antenna_1]['solution'])[info['eventid']]
    else:
        match_solution_cut = len(info)*[True]
        
    info , index_unique_entries = numpy.unique(info[match_solution_cut],return_index=True) #redefines info as only unique entries corresponding to match_solution_type request
    
    contains_both = numpy.in1d(info['eventid'], numpy.intersect1d(info['eventid'][info['antenna'] == index_antenna_1] , info['eventid'][info['antenna'] == index_antenna_2])) #cutting any events that have don't have remaining data for each queried antenna
    info = info[contains_both]
    timediffs = ((timediffs[match_solution_cut])[index_unique_entries])[contains_both]
    theta_ant_1 = info['theta_ant'][info['antenna'] == index_antenna_1]
    theta_ant_2 = info['theta_ant'][info['antenna'] == index_antenna_2]
    flagged_events = numpy.unique(info['eventid'][abs(timediffs) > threshold_time])
    #figure production
    if (histbins == None):
        histbins = [360,360]
    fig1, ax1 = pylab.subplots()
    pylab.hist2d(theta_ant_1 ,theta_ant_1 - theta_ant_2,histbins,norm=LogNorm())
    pylab.colorbar()
    pylab.xlabel('Theta_ant (Antenna %i)  (degrees)'%(index_antenna_1),fontsize = 20)
    pylab.ylabel('Theta_ant (Antenna %i) - Theta_ant (Antenna %i)  (degrees)'%(index_antenna_1,index_antenna_2),fontsize = 20)
    if not None:
        pylab.ylim(-12,3)
        pylab.vlines(numpy.linspace(0,180,vline),ax1.get_ylim()[0],ax1.get_ylim()[1],lw='0.5')
    return fig1, flagged_events, timediffs, info

    
def pathDiffHist(reader, index_station , index_antenna, config = None, histbins = None, threshold_time = 200, match_solution_type = False):
    '''
    Computes the difference in traveled distance for each photon between each antenna and index_antenna
    ''' 
    info = reader['info'][...]
    timediffs = (info[info['antenna'] == index_antenna]['time'])[info['eventid']] - info['time']
    antennas = info['antenna']
    pathdiffs = (info[info['antenna'] == index_antenna]['distance'])[info['eventid']] - info['distance']
    
    n_antennas = len(numpy.unique(info['antenna']))
    
    axlabels = ['S%iA%i-S%iA%i'%(index_station,index_antenna,index_station,ant) for ant in range(n_antennas)]

    #gating pairs
    if match_solution_type == True:
        solution_cut = info['solution'] == (info[info['antenna'] == index_antenna]['solution'])[info['eventid']]
    else:
        solution_cut = len(info)*[True]
        
    info , index_unique_entries = numpy.unique(info[solution_cut],return_index=True) #redefines info as only unique entries corresponding to match_solution_type request
    
    timediffs = (timediffs[solution_cut])[index_unique_entries]
    pathdiffs = (pathdiffs[solution_cut])[index_unique_entries]
    antennas = (antennas[solution_cut])[index_unique_entries]
    flagged_events = numpy.unique(info['eventid'][abs(timediffs) > threshold_time])

    #figure production
    if (histbins == None):
        histbins = [n_antennas,1000]
    fig1, ax1 = pylab.subplots()
    ax1.set_xlim(-1,n_antennas+1)
    pylab.hist2d(antennas,pathdiffs,histbins,norm=LogNorm())
    pylab.colorbar()
    pylab.ylabel('Difference in Path Distance (m)')
    pylab.xlabel('Detector Pair')
    pylab.xticks([(n_antennas - 1.0)*(t+0.5)/n_antennas for t in range(0,n_antennas)])
    ax1.set_xticklabels(axlabels,rotation=90,fontsize=10)
    return fig1, flagged_events, antennas, timediffs
    
def pathDiffDepthHist(reader, index_station , index_antenna, config = None, histbins = None, threshold_time = 200, match_solution_type = False, index_antenna2 = None):
    '''
    Computes the difference in traveled distance for each photon and plots it neutrino event depth
    ''' 
    info = reader['info'][...]
    timediffs = (info[info['antenna'] == index_antenna]['time'])[info['eventid']] - info['time']
    pathdiffs = (info[info['antenna'] == index_antenna]['distance'])[info['eventid']] - info['distance']
    z_neutrino = (reader['z_0'][...])[info['eventid']]
    
    #gating pairs
    if match_solution_type == True:
        solution_cut = info['solution'] == (info[info['antenna'] == index_antenna]['solution'])[info['eventid']]
    else:
        solution_cut = len(info)*[True]
    
    info , index_unique_entries = numpy.unique(info[solution_cut],return_index=True) #redefines info as only unique entries corresponding to match_solution_type request
    z_neutrino = (z_neutrino[solution_cut])[index_unique_entries]
    timediffs = (timediffs[solution_cut])[index_unique_entries]
    pathdiffs = (pathdiffs[solution_cut])[index_unique_entries]
    
    if (index_antenna2 != None):
        
        timediffs = timediffs[info['antenna'] == index_antenna2]
        pathdiffs = pathdiffs[info['antenna'] == index_antenna2]
        z_neutrino = z_neutrino[info['antenna'] == index_antenna2]
        flagged_events = ((info[(info['antenna'] == index_antenna2)])['eventid'])[abs(timediffs) > threshold_time]
        #gate on only pairs with index_antenna2
    else:
        flagged_events = numpy.unique(info['eventid'][abs(timediffs) > threshold_time])
    

    #figure production
    if (histbins == None):
        histbins = [1000,1000]
    fig1, ax1 = pylab.subplots()
    pylab.hist2d(z_neutrino,pathdiffs,histbins,norm=LogNorm())
    pylab.colorbar()
    pylab.ylabel('Difference in Path Distance (m)')
    pylab.xlabel('Neutrino Event Depth (m)')
    return fig1, flagged_events, z_neutrino, timediffs
        
def timeDirectDistanceHist(reader, index_station , index_antenna, config = None, histbins = None, threshold_time = 200, match_solution_type = False, index_antenna2 = None):
    '''
    Computes the arrival time differences and direct distance to neutrino interaction site 
    for each event and outputs a 2dhist and some meta data.
    Config should be an alreay yaml'd config file.  This is used to get antenna locations for distance calculations. 
    ''' 
    if(config == None):
        print('Set Config file with input parameter config=inputconfig to run.  Returning None')
        return
    info = reader['info'][...]
    timediffs = (info[info['antenna'] == index_antenna]['time'])[info['eventid']] - info['time']
    x_neutrino = (reader['x_0'][...])[info['eventid']]
    y_neutrino = (reader['y_0'][...])[info['eventid']]
    z_neutrino = (reader['z_0'][...])[info['eventid']]

    x_antenna = (numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,0])[info['antenna']]
    y_antenna = (numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,1])[info['antenna']]
    z_antenna = (numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,2])[info['antenna']]
    
    direct_distances = numpy.sqrt( (x_neutrino - x_antenna)**2 + (y_neutrino - y_antenna)**2 + (z_neutrino - z_antenna)**2 )
    
    
    #gating pairs
    if match_solution_type == True:
        solution_cut = info['solution'] == (info[info['antenna'] == index_antenna]['solution'])[info['eventid']]
    else:
        solution_cut = len(info)*[True]
        
    info , index_unique_entries = numpy.unique(info[solution_cut],return_index=True) #redefines info as only unique entries corresponding to match_solution_type request
    
    timediffs = (timediffs[solution_cut])[index_unique_entries]
    direct_distances = (direct_distances[solution_cut])[index_unique_entries]

    if (index_antenna2 != None):
        
        timediffs = timediffs[info['antenna'] == index_antenna2]
        direct_distances = direct_distances[info['antenna'] == index_antenna2]
        flagged_events = ((info[(info['antenna'] == index_antenna2)])['eventid'])[abs(timediffs) > threshold_time]
        #gate on only pairs with index_antenna2
    else:
        flagged_events = numpy.unique(info['eventid'][abs(timediffs) > threshold_time])
    
    #figure production
    if (histbins == None):
        histbins = [1000,1000]
    fig1, ax1 = pylab.subplots()
    pylab.hist2d(direct_distances,timediffs,histbins,norm=LogNorm())
    pylab.colorbar()
    pylab.ylabel('Time Difference (ns)')
    pylab.xlabel('Direct Distance from Event to Antenna (m)')
    return fig1, flagged_events, direct_distances, timediffs
    
def timePathDistanceHist(reader, index_station , index_antenna, config = None, histbins = None, threshold_time = 200, match_solution_type = False, index_antenna2 = None):
    '''
    Computes the arrival time differences and distance traveled by each photon 
    for each event and outputs a 2dhist and some meta data.
    ''' 

    info = reader['info'][...]
    timediffs = (info[info['antenna'] == index_antenna]['time'])[info['eventid']] - info['time']
    path_distances = info['distance']
    
    
    #gating pairs
    if match_solution_type == True:
        solution_cut = info['solution'] == (info[info['antenna'] == index_antenna]['solution'])[info['eventid']]
        
    else:
        solution_cut = len(info)*[True]
        
    info , index_unique_entries = numpy.unique(info[solution_cut],return_index=True) #redefines info as only unique entries corresponding to match_solution_type request
    
    timediffs = (timediffs[solution_cut])[index_unique_entries]
    path_distances = (path_distances[solution_cut])[index_unique_entries]
    
    if (index_antenna2 != None):
        
        timediffs = timediffs[info['antenna'] == index_antenna2]
        path_distances = path_distances[info['antenna'] == index_antenna2]
        flagged_events = ((info[(info['antenna'] == index_antenna2)])['eventid'])[abs(timediffs) > threshold_time]
        #gate on only pairs with index_antenna2
    else:
        flagged_events = numpy.unique(info['eventid'][abs(timediffs) > threshold_time])

    
    #figure production
    if (histbins == None):
        histbins = [1000,1000]
    fig1, ax1 = pylab.subplots()
    pylab.hist2d(path_distances,timediffs,histbins,norm=LogNorm())
    pylab.colorbar()
    pylab.ylabel('Time Difference (ns)')
    pylab.xlabel('Path Distance from Event to Antenna (m)')
    return fig1, flagged_events, path_distances, timediffs
    
def directDistancePathDistanceHist(reader, index_station , index_antenna, config = None, histbins = None, threshold_time = 200, match_solution_type = False, index_antenna2 = None):
    '''
    Computes path distance and direct distance between antenna and neutrino event site for each event
    and returns a 2dhist and some metadata
    ''' 
    if(config == None):
        print('Set Config file with input parameter config=inputconfig to run.  Returning None')
        return
    info = reader['info'][...]
    timediffs = (info[info['antenna'] == index_antenna]['time'])[info['eventid']] - info['time']
    x_neutrino = (reader['x_0'][...])[info['eventid']]
    y_neutrino = (reader['y_0'][...])[info['eventid']]
    z_neutrino = (reader['z_0'][...])[info['eventid']]

    x_antenna = (numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,0])[info['antenna']]
    y_antenna = (numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,1])[info['antenna']]
    z_antenna = (numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,2])[info['antenna']]
    
    direct_distances = numpy.sqrt( (x_neutrino - x_antenna)**2 + (y_neutrino - y_antenna)**2 + (z_neutrino - z_antenna)**2 )
    path_distances = info['distance']
    
    #gating pairs
    if match_solution_type == True:
        solution_cut = info['solution'] == (info[info['antenna'] == index_antenna]['solution'])[info['eventid']]
        
    else:
        solution_cut = len(info)*[True]
        
    info , index_unique_entries = numpy.unique(info[solution_cut],return_index=True) #redefines info as only unique entries corresponding to match_solution_type request
    
    timediffs = (timediffs[solution_cut])[index_unique_entries]
    path_distances = (path_distances[solution_cut])[index_unique_entries]
    direct_distances = (direct_distances[solution_cut])[index_unique_entries]

    if (index_antenna2 != None):
        
        timediffs = timediffs[info['antenna'] == index_antenna2]
        path_distances = path_distances[info['antenna'] == index_antenna2]
        direct_distances = direct_distances[info['antenna'] == index_antenna2]
        flagged_events = ((info[(info['antenna'] == index_antenna2)])['eventid'])[abs(timediffs) > threshold_time]
        #gate on only pairs with index_antenna2
    else:
        flagged_events = numpy.unique(info['eventid'][abs(timediffs) > threshold_time])
    
    #figure production
    if (histbins == None):
        histbins = [1000,1000]
    fig1, ax1 = pylab.subplots()
    pylab.hist2d(direct_distances,path_distances,histbins,norm=LogNorm())
    pylab.colorbar()
    pylab.ylabel('Path Distance from Event to Antenna (m)')
    pylab.xlabel('Direct Distance from Event to Antenna (m)')
    return fig1, flagged_events, direct_distances, path_distances, timediffs







def plotAntennaSolutions( reader , lib , eventid , index_station , antennas , point_size = 5 , plot_solution_grid = False , plot_solution_grid_alpha = 0.01 , label_points = False , verbose = False):
    '''
    I want this to plot each trace that is written to the info object for each
    antenna.  This means there should be 8 traces.  Hopefully this can make any
    divergences in path obvious.
    '''
    def getData( self ):
        return self.data
    def getSolutions( self ):
        return self.solutions
    def getAveragedTraces(self, dic, r, z):
        #DS:    This should reflect whatever RefractionLibrary.getValue( self , dic , r , z )
        #       except that it outputs the rays chosen and their respective weights.
        #       returns: theta_ant, r, z, weight
        #       each being a pair of values, one for each trace
        if self.dense_rays:
            # This scheme works well when the rays are densely packed, 
            # but completely fails when the rays are spread far apart.
            distance = numpy.sqrt((r - dic['r'])**2 + (z - dic['z'])**2)
            index_1 = numpy.argmin(distance) #finding first closest trace solution
            weight_1 = distance[index_1]**(-1) #weighting with inverse square law
            theta_ant_1 = dic['theta_ant'][index_1]
            r_1 = dic['r'][index_1]
            z_1 = dic['z'][index_1]
            
            distance[dic['theta_ant'] == dic['theta_ant'][index_1]] = 1.e10 #used to ignore first closest
            index_2 = numpy.argmin(distance) #finding second closest solution
            weight_2 = distance[index_2]**(-1) #again weighting with invese square law
            theta_ant_2 = dic['theta_ant'][index_2]
            r_2 = dic['r'][index_2]
            z_2 = dic['z'][index_2]
        else:
            # This scheme works much better when the rays are spread
            # far apart.
            distance = numpy.sqrt((r - dic['r'])**2 + (z - dic['z'])**2)
            theta_ant_1 = dic['theta_ant'][numpy.argmin(distance)]
            cut_1 = dic['theta_ant'] == theta_ant_1  
            index_1 = numpy.nonzero(cut_1)[0][numpy.argmin(numpy.fabs(dic['z'][cut_1] - z))] 
            r_1 = dic['r'][index_1]
            z_1 = dic['z'][index_1]
            
            distance[cut_1] = 1.e10
            theta_ant_2 = dic['theta_ant'][numpy.argmin(distance)]
            cut_2 = dic['theta_ant'] == theta_ant_2
            index_2 = numpy.nonzero(cut_2)[0][numpy.argmin(numpy.fabs(dic['z'][cut_2] - z))]
            r_2 = dic['r'][index_2]
            z_2 = dic['z'][index_2]
            
            weight_1 = 1. / numpy.fabs(r - dic['r'][index_1])
            weight_2 = 1. / numpy.fabs(r - dic['r'][index_2])

        return [ theta_ant_1 , theta_ant_2 ] , [ r_1 , r_2 ] , [ z_1 , z_2 ] , [ weight_1 , weight_2 ]
    
    data = {}
    data_keys = []
    for key in lib.keys():
        lib[key].getData = types.MethodType( getData , lib[key] )
        lib[key].getSolutions = types.MethodType( getSolutions , lib[key] )
        lib[key].getAveragedTraces = types.MethodType( getAveragedTraces , lib[key] )
        data[key] = lib[key].getData()
        data_keys.append(key)
    
    info = reader['info'][...]
    theta_ant = info['theta_ant'][info['eventid'] == eventid]
    event = (reader['info'])[reader['info']['eventid'] == eventid]
    
    fig, ax = pylab.subplots()
    if (numpy.size(antennas) == 1):
        antennas = [antennas]
            
    x_neutrino = ((reader['x_0'][...])[info['eventid']])[info['eventid'] == eventid]
    y_neutrino = ((reader['y_0'][...])[info['eventid']])[info['eventid'] == eventid]
    z_neutrino = ((reader['z_0'][...])[info['eventid']])[info['eventid'] == eventid]
    r_neutrino = numpy.sqrt(x_neutrino**2 + y_neutrino**2)
    
    x_antenna = ((numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,0])[info['antenna']])[info['eventid'] == eventid]
    y_antenna = ((numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,1])[info['antenna']])[info['eventid'] == eventid]
    z_antenna = ((numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,2])[info['antenna']])[info['eventid'] == eventid]
    r_antenna = numpy.sqrt(x_antenna**2 + y_antenna**2)
    
    if ( verbose ):
        print('z_neutrino: ', z_neutrino[0], 'r_neutrino:' , r_neutrino[0])
        print(event)
    
    for index_antenna in antennas:
        solution = (event['solution'][index_antenna]).decode('utf-8')
        
        ant_lib = lib[list(lib.keys())[index_antenna]]
        
        theta_ant_trace , r_snap , z_snap , trace_weight = ant_lib.getAveragedTraces( ant_lib.data[solution] , r_neutrino[index_antenna] , z_neutrino[index_antenna] )
       
        z_trace_1 = (data[data_keys[index_antenna]][solution]['z'][...])[data[data_keys[index_antenna]][solution]['theta_ant'] == theta_ant_trace[0]]
        r_trace_1 = (data[data_keys[index_antenna]][solution]['r'][...])[data[data_keys[index_antenna]][solution]['theta_ant'] == theta_ant_trace[0]]
        z_trace_2 = (data[data_keys[index_antenna]][solution]['z'][...])[data[data_keys[index_antenna]][solution]['theta_ant'] == theta_ant_trace[1]]
        r_trace_2 = (data[data_keys[index_antenna]][solution]['r'][...])[data[data_keys[index_antenna]][solution]['theta_ant'] == theta_ant_trace[1]]
        
        if ( solution != 'direct' ):
        
            z_trace_1 = numpy.append( z_trace_1 , (data[data_keys[index_antenna]]['direct']['z'][...])[data[data_keys[index_antenna]]['direct']['theta_ant'] == theta_ant_trace[0]])
            r_trace_1 = numpy.append( r_trace_1 , (data[data_keys[index_antenna]]['direct']['r'][...])[data[data_keys[index_antenna]]['direct']['theta_ant'] == theta_ant_trace[0]])
            z_trace_2 = numpy.append( z_trace_2 , (data[data_keys[index_antenna]]['direct']['z'][...])[data[data_keys[index_antenna]]['direct']['theta_ant'] == theta_ant_trace[1]])
            r_trace_2 = numpy.append( r_trace_2 ,(data[data_keys[index_antenna]]['direct']['r'][...])[data[data_keys[index_antenna]]['direct']['theta_ant'] == theta_ant_trace[1]])
            if ( (solution == 'reflect_2') ):
                z_trace_1 = numpy.append( z_trace_1 , (data[data_keys[index_antenna]]['reflect']['z'][...])[data[data_keys[index_antenna]]['reflect']['theta_ant'] == theta_ant_trace[0]])
                r_trace_1 = numpy.append( r_trace_1 , (data[data_keys[index_antenna]]['reflect']['r'][...])[data[data_keys[index_antenna]]['reflect']['theta_ant'] == theta_ant_trace[0]])
                z_trace_2 = numpy.append( z_trace_2 , (data[data_keys[index_antenna]]['reflect']['z'][...])[data[data_keys[index_antenna]]['reflect']['theta_ant'] == theta_ant_trace[1]])
                r_trace_2 = numpy.append( r_trace_2 ,(data[data_keys[index_antenna]]['reflect']['r'][...])[data[data_keys[index_antenna]]['reflect']['theta_ant'] == theta_ant_trace[1]])
            if ( (solution == 'cross_2' ) ):
                z_trace_1 = numpy.append( z_trace_1 , (data[data_keys[index_antenna]]['cross']['z'][...])[data[data_keys[index_antenna]]['cross']['theta_ant'] == theta_ant_trace[0]])
                r_trace_1 = numpy.append( r_trace_1 , (data[data_keys[index_antenna]]['cross']['r'][...])[data[data_keys[index_antenna]]['cross']['theta_ant'] == theta_ant_trace[0]])
                z_trace_2 = numpy.append( z_trace_2 , (data[data_keys[index_antenna]]['cross']['z'][...])[data[data_keys[index_antenna]]['cross']['theta_ant'] == theta_ant_trace[1]])
                r_trace_2 = numpy.append( r_trace_2 ,(data[data_keys[index_antenna]]['cross']['r'][...])[data[data_keys[index_antenna]]['cross']['theta_ant'] == theta_ant_trace[1]])
        
        trace_weight_norm = [float(i)/sum(trace_weight) for i in trace_weight]
        if (verbose):
            print('Antenna:', index_antenna, '\tSolution Type:', solution , '\ttheta_antenna:', theta_ant[index_antenna]) 
            print('Normalized weight traces: ', trace_weight_norm)   
            print('self.dense_rays: ' , ant_lib.dense_rays)
        
        if ( plot_solution_grid ):
            pylab.scatter(ant_lib.data['direct']['r'],ant_lib.data['direct']['z'],marker = ',',s=(72./fig.dpi)**2,c='k',alpha=plot_solution_grid_alpha)
            pylab.scatter(ant_lib.data['reflect']['r'],ant_lib.data['reflect']['z'],marker = ',',s=(72./fig.dpi)**2,c='k',alpha=plot_solution_grid_alpha)
            pylab.scatter(ant_lib.data['cross']['r'],ant_lib.data['cross']['z'],marker = ',',s=(72./fig.dpi)**2,c='k',alpha=plot_solution_grid_alpha)
            pylab.scatter(ant_lib.data['direct_2']['r'],ant_lib.data['direct_2']['z'],marker = ',',s=(72./fig.dpi)**2,c='k',alpha=plot_solution_grid_alpha)
            pylab.scatter(ant_lib.data['reflect_2']['r'],ant_lib.data['reflect_2']['z'],marker = ',',s=(72./fig.dpi)**2,c='k',alpha=plot_solution_grid_alpha)
            pylab.scatter(ant_lib.data['cross_2']['r'],ant_lib.data['cross_2']['z'],marker = ',',s=(72./fig.dpi)**2,c='k',alpha=plot_solution_grid_alpha)

        if label_points:
            ax.annotate(str(index_antenna),(r_snap[0],z_snap[0]),color = 'k')
            ax.annotate(str(index_antenna),(r_snap[1],z_snap[1]),color = 'k')
            ax.annotate(str(index_antenna),(r_antenna[index_antenna],z_antenna[index_antenna]),color = 'k')
        
        pylab.scatter( r_trace_1,z_trace_1 , c = 'b' , s = point_size*(trace_weight_norm[0]**2) )
        pylab.scatter( r_snap[0] , z_snap[0] , c = 'g' , s = 4*point_size )
        pylab.scatter( r_trace_2 , z_trace_2 , c = 'b' , s = point_size*(trace_weight_norm[1]**2) )
        pylab.scatter( r_snap[1] , z_snap[1] , c = 'g' , s = 4*point_size )
        pylab.scatter( r_neutrino[index_antenna] , z_neutrino[index_antenna] , marker = '*', c = 'r', s = 4*point_size )
        pylab.scatter( r_antenna[index_antenna] , z_antenna[index_antenna] , marker = 's', c = 'r', s = 4*point_size )
  
def maxTimeDiffCMap(reader,lib, index_station = 0, index_antenna = 0,  match_solution_type = False, plot_hull = False, hull = None):
    '''
    Computes the max time difference for the station and returns a scatterplot fo all points 
    with color corresponding to the time difference. 
    ''' 
    r_0 = numpy.sqrt(reader['x_0'][...]**2 + reader['y_0'][...]**2)
    z_0 = reader['z_0'][...]
    a = numpy.reshape(reader['info']['time'],(len(numpy.unique(reader['info']['eventid'])),8))
    c =  numpy.amax(a,axis = 1) - numpy.amin(a,axis = 1)
    
    pylab.figure()
    pylab.scatter(r_0,z_0,c = c,marker=',',s = 1,norm=LogNorm(vmin=0.1,vmax =10000.))
    
    #c=numpy.fabs(timediffs)
    pylab.colorbar(label='Max Time Difference Between Antennas')
    pylab.xlabel('r (m)',fontsize = 20)
    pylab.ylabel('z (m)',fontsize = 20) 
    
    if plot_hull:
        if hull != None:    
            
            r,z = ant0.makeHull(lib[lib.keys()[index_antenna]].data[hull])
            pylab.plot(r,z,c='r',linewidth=2)
        else:
            print('Need to give hull type to plot')

      
def getThetaAntBounds(lib,antenna_label,solution, r_query, z_query, theta_bounds = 1):
    '''
    Finds the closest theta ant, then returns the theta that is [theta_bounds]
    angles away.
    '''
    if len(r_query) > 100:
        print('r_query is large, breaking up:')
        thetamin = numpy.array([])
        thetamax = numpy.array([])
        index = 0
        while index < len(r_query):
            print('Processing %i:%i'%(index,min(index+100, len(r_query) - 1)))
            tmin,tmax = getThetaAntBounds(lib,antenna_label,solution, r_query[index:min(index+100, len(r_query) - 1)], z_query[index:min(index+100, len(r_query) - 1)], theta_bounds = theta_bounds)
            thetamin = numpy.append(thetamin,tmin)
            thetamax = numpy.append(thetamax,tmax)
            index += 100
    else:
        theta_ant = lib[antenna_label].data[solution]['theta_ant']
        r = lib[antenna_label].data[solution]['r']
        z = lib[antenna_label].data[solution]['z']
        t = lib[antenna_label].data[solution]['t']
        r_diff = numpy.tile(r,(numpy.size(r_query),1)) - numpy.tile(r_query,(numpy.size(r),1)).T
        z_diff = numpy.tile(z,(numpy.size(z_query),1)) - numpy.tile(z_query,(numpy.size(z),1)).T
        diff = z_diff**2 + r_diff**2
        theta = theta_ant[numpy.argmin(diff,axis=1)]
        theta_ant_list1d = numpy.sort(numpy.unique(lib[antenna_label].data[solution]['theta_ant']))
        theta_ant_list = numpy.tile(theta_ant_list1d,(numpy.size(z_query),1))
        theta_index = numpy.argmin(numpy.fabs(theta_ant_list - numpy.tile(theta,(numpy.shape(theta_ant_list)[1],1)).T) ,axis = 1)
        thetamin_ind = theta_index - theta_bounds*numpy.ones_like(theta_index)
        thetamax_ind = theta_index + theta_bounds*numpy.ones_like(theta_index)
        thetamin_ind[thetamin_ind < 0] = 0
        thetamax_ind[thetamax_ind > numpy.shape(theta_ant_list)[1] - 1] = (numpy.shape(theta_ant_list)[1] - 1)
        thetamin = theta_ant_list1d[thetamin_ind]
        thetamax = theta_ant_list1d[thetamax_ind]
    return thetamin,thetamax

def findBetterRay(origin, r_match, z_match, theta_ant_lower, theta_ant_upper, ice_model = 'antarctic', phi_0 = 0 , atol = 0.1, max_attempts = 10, t_max=50000., verbose = False, plot = False):
    '''
    Given an initial guess for theta_ant (as bounds upper and lower), this uses
    the bisection method to determine a more accurate value corrsponding to the
    initial input.  phi_0 set to 0 because right now this only wants proximity in
    the r,z plane, meaning the specific contributions from x,y don't contribute.
    This would need to be changed if an ice model was used that is not purely
    z dependant. 
    '''
    def returnLine(r_in,z_in,theta_in):
         m = 1./numpy.tan(numpy.deg2rad(theta_in))
         b = 1.0*z_in - m * (1.0*r_in)
         #print('z = %f * r + %f'%(m,b))
         def Line(x_in):
             return m*x_in + b
         return Line
         
         
    if verbose == True:
        time_start = time.perf_counter()
    ice = gnosim.earth.ice.Ice(ice_model)
    t_steps = (numpy.linspace(0.5,numpy.sqrt(5),max_attempts)**2)[::-1]
    t_step = t_steps[0]
    attempt = -1
    #theta_ant_lower = 0.7*theta_ant_lower
    x_lower, y_lower, z_lower, t_lower, d_lower, phi_lower, theta_lower, a_p_lower, a_s_lower, index_reflect_air_lower, index_reflect_water_lower = gnosim.trace.refraction_library_beta.rayTrace(origin, phi_0, theta_ant_lower,ice, t_max=t_max, r_limit = max(100,1.01*r_match), t_step=t_step)
    r_lower = numpy.sqrt( (x_lower)**2 + (y_lower)**2)
    sep_lower = numpy.sqrt( (r_lower - r_match)**2 + (z_lower - z_match)**2)
    lower_min_index = min(numpy.where(sep_lower == min(sep_lower))[0])
    l_lower = returnLine(r_lower[lower_min_index],z_lower[lower_min_index],theta_lower[lower_min_index])
    
    #theta_ant_upper = min(180.,1.3*theta_ant_upper)
    x_upper, y_upper, z_upper, t_upper, d_upper, phi_upper, theta_upper, a_p_upper, a_s_upper, index_reflect_air_upper, index_reflect_water_upper = gnosim.trace.refraction_library_beta.rayTrace(origin, phi_0, theta_ant_upper,ice, t_max=t_max, r_limit = max(100,1.01*r_match), t_step=t_step)
    r_upper = numpy.sqrt( (x_upper)**2 + (y_upper)**2)
    sep_upper = numpy.sqrt( (r_upper - r_match)**2 + (z_upper - z_match)**2)
    #print('r',r_match,'z',z_match)
    #print('theta_ant_upper',theta_ant_upper)
    #print('theta_ant_lower',theta_ant_lower)
    #print('len(sep_upper)',len(sep_upper))
    #print('min(sep_upper)',min(sep_upper))
    #print('numpy.where(sep_upper == min(sep_upper))[0]',numpy.where(sep_upper == min(sep_upper))[0])
    upper_min_index = min(numpy.where(sep_upper == min(sep_upper))[0])
    l_upper = returnLine(r_upper[upper_min_index],z_upper[upper_min_index],theta_upper[upper_min_index])
    
    if sep_upper[upper_min_index] < sep_lower[lower_min_index]:
        minimum_min_sep = sep_upper[upper_min_index]
        theta_ant_out = theta_ant_upper
        x_out     = x_upper[upper_min_index]
        y_out     = y_upper[upper_min_index]
        z_out     = z_upper[upper_min_index]
        r_out     = r_upper[upper_min_index]
        t_out     = t_upper[upper_min_index]
        d_out     = d_upper[upper_min_index]
        phi_out   = phi_upper[upper_min_index]
        theta_out = theta_upper[upper_min_index]
        a_p_out   = a_p_upper[upper_min_index]
        a_s_out   = a_s_upper[upper_min_index]
        index_reflect_air_out = index_reflect_air_upper
        index_reflect_water_out = index_reflect_water_upper
    else:
        minimum_min_sep = sep_lower[lower_min_index]
        theta_ant_out = theta_ant_lower
        x_out     = x_lower[lower_min_index]
        y_out     = y_lower[lower_min_index]
        z_out     = z_lower[lower_min_index]
        r_out     = r_lower[lower_min_index]
        t_out     = t_lower[lower_min_index]
        d_out     = d_lower[lower_min_index]
        phi_out   = phi_lower[lower_min_index]
        theta_out = theta_lower[lower_min_index]
        a_p_out   = a_p_lower[lower_min_index]
        a_s_out   = a_s_lower[lower_min_index]
        index_reflect_air_out = index_reflect_air_lower
        index_reflect_water_out = index_reflect_water_lower

    if plot == True:
        fig = pylab.figure()
        pylab.title('Ray Finding Attempts')
        pylab.xlabel('r(m)')
        pylab.ylabel('z(m)')
        pylab.scatter(r_match,z_match,c='r',marker='*',label='Point to fit')
        pylab.scatter(r_upper,z_upper,marker='^',c='k',label='Upper Bound: %f.3'%(theta_ant_upper))
        pylab.scatter(r_lower,z_lower,marker='v',c='k',label='Lower Bound: %f.3'%(theta_ant_lower))
        pylab.scatter(r_upper[index_reflect_air_upper],z_upper[index_reflect_air_upper],marker='^',c='r',label='air: %f.3'%(theta_ant_upper))
        pylab.scatter(r_upper[index_reflect_water_upper],z_upper[index_reflect_water_upper],marker='^',c='m',label='water: %f.3'%(theta_ant_upper))
        pylab.scatter(r_lower[index_reflect_air_lower],z_lower[index_reflect_air_lower],marker='v',c='r',label='air: %f.3'%(theta_ant_lower))
        pylab.scatter(r_lower[index_reflect_water_lower],z_lower[index_reflect_water_lower],marker='v',c='m',label='water: %f.3'%(theta_ant_lower))
    while (attempt+1 < max_attempts):
        attempt = attempt + 1
        t_step = t_steps[attempt]
        if verbose == True:
            print('[%.2f]On attempt %i with bounds %f to %f'%(time.perf_counter() - time_start, attempt, theta_ant_lower , theta_ant_upper))
        theta_ant_mid = (theta_ant_lower + theta_ant_upper) / 2.0
        x, y, z, t, d, phi, theta, a_p, a_s, index_reflect_air, index_reflect_water = gnosim.trace.refraction_library_beta.rayTrace(origin, phi_0, theta_ant_mid,ice, t_max=t_max, r_limit = max(100,1.01*r_match), t_step=t_step)
        r = numpy.sqrt( (x)**2 + (y)**2)
        sep = numpy.sqrt( (r - r_match)**2 + (z - z_match)**2)
        min_sep = min(sep)
        min_sep_index = min(numpy.where(sep == min_sep)[0])
        
        if min_sep < minimum_min_sep:
            minimum_min_sep = min_sep
            theta_ant_out = theta_ant_mid
            x_out     = x[min_sep_index]
            y_out     = y[min_sep_index]
            z_out     = z[min_sep_index]
            r_out     = r[min_sep_index]
            t_out     = t[min_sep_index]
            d_out     = d[min_sep_index]
            phi_out   = phi[min_sep_index]
            theta_out = theta[min_sep_index]
            a_p_out   = a_p[min_sep_index]
            a_s_out   = a_s[min_sep_index]
            index_reflect_air_out = index_reflect_air
            index_reflect_water_out = index_reflect_water
        
        l = returnLine(r[min_sep_index],z[min_sep_index],theta[min_sep_index])
        if plot == True:
            pylab.scatter(r,z,marker=',',label='Theta Ant: %f.3'%(theta_ant_mid))
        if verbose == True:
            print('[%.2f]Min Sep = %f, with Theta of %f'%(time.perf_counter() - time_start, min_sep, theta[min_sep_index]))
        
        if numpy.logical_or((min_sep < atol),(attempt+1 == max_attempts)):
            if verbose == True:
                if (min_sep < atol):
                    print('[%.2f]Min Sep %f less than atol %f'%(time.perf_counter() - time_start,min_sep, atol))
                if (attempt+1 == max_attempts):
                    print('[%.2f]Max attempts reached'%(time.perf_counter() - time_start))
                print('[%.2f]Making final adjustment'%(time.perf_counter() - time_start))
            x, y, z, t, d, phi, theta, a_p, a_s, index_reflect_air, index_reflect_water = gnosim.trace.refraction_library_beta.rayTrace(origin, phi_0, theta_ant_mid,ice, t_max=t_max, r_limit = max(100.0,1.01*r_match), t_step=t_step/2.0)
            r = numpy.sqrt( (x)**2 + (y)**2)
            sep = numpy.sqrt( (r - r_match)**2 + (z - z_match)**2)
            if plot == True:
                pylab.scatter(r,z,c='g',marker=',',label='Chosen Theta Ant: %f.3'%(theta_ant_mid))
                pylab.legend()
                pylab.scatter(r_match,z_match,c='r',marker='*')
            min_sep = min(sep)
            min_sep_index = min(numpy.where(sep == min_sep)[0])
            
            if min_sep < minimum_min_sep:
                minimum_min_sep = min_sep
                theta_ant_out = theta_ant_mid
                x_out     = x[min_sep_index]
                y_out     = y[min_sep_index]
                z_out     = z[min_sep_index]
                r_out     = r[min_sep_index]
                t_out     = t[min_sep_index]
                d_out     = d[min_sep_index]
                phi_out   = phi[min_sep_index]
                theta_out = theta[min_sep_index]
                a_p_out   = a_p[min_sep_index]
                a_s_out   = a_s[min_sep_index]
                index_reflect_air_out = index_reflect_air
                index_reflect_water_out = index_reflect_water
            
            if verbose == True:
                print('[%.2f]Complete'%(time.perf_counter() - time_start))
            break
        #print('l_lower(r_match):',l_lower(r_match))
        #print('l(r_match):',l(r_match))
        #print('l_upper(r_match):',l_upper(r_match))
        #print('z_match:',z_match)
        if numpy.sign(l_lower(r_match) - z_match) == numpy.sign(l(r_match) - z_match):
            theta_ant_lower = theta_ant_mid
            x_lower     = x
            y_lower     = y
            z_lower     = z
            r_lower     = r
            sep_lower   = sep
            lower_min_index = min_sep_index
            l_lower     = l
            t_lower     = t
            d_lower     = d
            phi_lower   = phi
            theta_lower = theta
            a_p_lower   = a_p
            a_s_lower   = a_s
            index_reflect_air_lower = index_reflect_air
            index_reflect_water_lower = index_reflect_water
        elif numpy.sign(l_upper(r_match) - z_match) == numpy.sign(l(r_match) - z_match):
            theta_ant_upper = theta_ant_mid
            x_upper     = x
            y_upper     = y
            z_upper     = z
            r_upper     = r
            sep_upper   = sep
            upper_min_index = min_sep_index
            l_upper     = l
            t_upper     = t
            d_upper     = d
            phi_upper   = phi
            theta_upper = theta
            a_p_upper   = a_p
            a_s_upper   = a_s
            index_reflect_air_upper = index_reflect_air
            index_reflect_water_upper = index_reflect_water
        elif (attempt+1 < max_attempts):
            if verbose == True:
                print('Theta bisection method error: Theta out of bounds, adjusting bounds by 0.5 deg') 
            if numpy.logical_and((l_upper(r_match) > z_match) , (l_lower(r_match) > z_match) ):
                #must lower one the bottom one:
                if (l_upper(r_match) > l_lower(r_match) ):
                    #lower 'lower'
                    if (index_reflect_air_lower == 0) == (index_reflect_water_lower==0):
                        theta_ant_lower = numpy.min([180.0,theta_ant_lower + 0.5])
                    else:
                        theta_ant_lower = numpy.max([0.0,theta_ant_lower - 0.5])
                else:
                    #lower 'upper'
                    if (index_reflect_air_upper == 0) == (index_reflect_water_upper==0):
                        theta_ant_upper = numpy.min([180.0,theta_ant_upper + 0.5])
                    else:
                        theta_ant_upper = numpy.max([0.0,theta_ant_upper - 0.5])
            else:
                #both are lower, must raise the upper one
                if (l_upper(r_match) > l_lower(r_match) ):
                    #raise 'upper' recall raise means mean theta smaller in direct cases
                    if (index_reflect_air_upper == 0) == (index_reflect_water_upper==0):
                        #direct or double reflect
                        theta_ant_upper = numpy.max([0.0,theta_ant_upper - 0.5])
                    else:
                        #single reflect
                        theta_ant_upper = numpy.min([180.0,theta_ant_upper + 0.5])
                    
                else:
                    #raise 'lower'
                    if (index_reflect_air_lower == 0) == (index_reflect_water_lower==0):
                        #direct or double reflect
                        theta_ant_lower = numpy.max([0.0,theta_ant_lower - 0.5])
                    else:
                        #single reflect
                        theta_ant_lower = numpy.min([180.0,theta_ant_lower + 0.5])
                
            x_lower, y_lower, z_lower, t_lower, d_lower, phi_lower, theta_lower, a_p_lower, a_s_lower, index_reflect_air_lower, index_reflect_water_lower = gnosim.trace.refraction_library_beta.rayTrace(origin, phi_0, theta_ant_lower,ice, t_max=t_max, r_limit = max(100,1.01*r_match), t_step=t_step)
            r_lower = numpy.sqrt( (x_lower)**2 + (y_lower)**2)
            sep_lower = numpy.sqrt( (r_lower - r_match)**2 + (z_lower - z_match)**2)
            lower_min_index = min(numpy.where(sep_lower == min(sep_lower))[0])
            l_lower = returnLine(r_lower[lower_min_index],z_lower[lower_min_index],theta_lower[lower_min_index])
            
            x_upper, y_upper, z_upper, t_upper, d_upper, phi_upper, theta_upper, a_p_upper, a_s_upper, index_reflect_air_upper, index_reflect_water_upper = gnosim.trace.refraction_library_beta.rayTrace(origin, phi_0, theta_ant_upper,ice, t_max=t_max, r_limit = max(100,1.01*r_match), t_step=t_step)
            r_upper = numpy.sqrt( (x_upper)**2 + (y_upper)**2)
            sep_upper = numpy.sqrt( (r_upper - r_match)**2 + (z_upper - z_match)**2)
            upper_min_index = min(numpy.where(sep_upper == min(sep_upper))[0])
            l_upper = returnLine(r_upper[upper_min_index],z_upper[upper_min_index],theta_upper[upper_min_index])
        else:
            if verbose == True:
                print('Theta bisection method error: Theta out of bounds')
                print('theta_ant_lower:', theta_ant_lower)
                print('theta_ant_mid:', theta_ant_mid)
                print('theta_ant_upper:', theta_ant_upper)
                print('theta[min_sep_index]:',theta[min_sep_index])
                print('Breaking at attempt:' ,attempt)
            if plot == True:
                pylab.legend()
                pylab.scatter(r_match,z_match,c='r',marker='*')
            break
    if plot:
        pylab.scatter(r_out,z_out,c='r',marker='o',label='Best Fit')
        pylab.legend()
    #return x[min_sep_index],y[min_sep_index],z[min_sep_index],t[min_sep_index],d[min_sep_index],phi[min_sep_index],theta[min_sep_index],a_p[min_sep_index],a_s[min_sep_index],index_reflect_air,index_reflect_water
    return r_out,z_out,t_out,d_out,theta_out,theta_ant_out,a_p_out,a_s_out

def flaggedTimeDiffLocation(reader,flagged_events,timediffs,title = ''):
    if len(flagged_events) == 0:
        print('No flagged events passed to the function.')
        return
    r = numpy.sqrt(reader['x_0'][...]**2 + reader['y_0'][...]**2)[flagged_events]
    z = reader['z_0'][...][flagged_events]

    pylab.figure()
    pylab.title(title,fontsize=20)

    pylab.scatter(r,z,c = timediffs[flagged_events],norm=LogNorm())
    pylab.colorbar(label='Time Difference Between Antennas (ns)')
    pylab.ylabel('z(m)',fontsize=20)
    pylab.xlabel('r(m)',fontsize=20)
    pylab.ylim(-3010,10)
    pylab.xlim(-10,6310)
    
    
def residualLocation(reader,eventid,residual,title = '',cnorm = None):
    r = numpy.sqrt(reader['x_0'][...]**2 + reader['y_0'][...]**2)[eventid]
    z = reader['z_0'][...][eventid]

    pylab.figure()
    pylab.title(title,fontsize=20)

    pylab.scatter(r,z,c = residual,norm = cnorm)
    pylab.colorbar(label='Residual (ns)')
    pylab.ylabel('z(m)',fontsize=20)
    pylab.xlabel('r(m)',fontsize=20)
    pylab.ylim(-3010,10)
    pylab.xlim(-10,6310)
    
def flaggedSolutionLocation(reader,flagged_events,index_antenna,title = ''):
    if len(flagged_events) == 0:
        print('No flagged events passed to the function.')
        return
    r = numpy.sqrt(reader['x_0'][...]**2 + reader['y_0'][...]**2)[flagged_events]
    z = reader['z_0'][...][flagged_events]
    color_key = {'direct':'r', 'cross':'darkgreen', 'reflect':'blue', 'direct_2':'gold', 'cross_2':'lawngreen', 'reflect_2':'purple'}
    pylab.figure()
    pylab.title(title,fontsize=20)
    sols = numpy.core.defchararray.decode(reader['info']['solution'][reader['info']['antenna'] == index_antenna][flagged_events])
    for ckey in color_key.keys():
        pylab.scatter(r[sols == ckey],z[sols == ckey],c = color_key[ckey],label = ckey)
        pylab.ylabel('z(m)',fontsize=20)
        pylab.xlabel('r(m)',fontsize=20)
        pylab.ylim(-3010,10)
        pylab.xlim(-10,6310)
    pylab.legend(fontsize=20)
    
def electricFieldThreshold( snr , gain, noise_temperature, bandwidth, frequency, verbose = True):
    '''
    Resistance assumed to be 50 Ohms
    noise_temperature (K)
    Bandwidth (GHz)
    Gain (dBi)
    frequency (GHz)
    Calculates the electric field threshold using the noise_temperature, resistance, and BW
    to determine a V_RMS, and the signal to noise, gain, and frequency to calculate
    the electric field using the antenna factor formula.

    Currently uncertain about if the gain is being correctly utilized in this formula.
    If an array of frequencies is given this should be able to output an array of 
    correspondng thresholds.  This would be helpful if the threshold should be event
    by event depending on the dominant E field frequency. 
    '''
    V_rms = numpy.sqrt(gnosim.utils.constants.boltzmann * noise_temperature * 50.0 * bandwidth * gnosim.utils.constants.GHz_to_Hz)
    thresh = snr * V_rms * 9.73 * frequency / (gnosim.utils.constants.speed_light * gnosim.utils.rf.amplitude(gain))  #f/c gives 1/m because f (GHz) and c (m/ns)
    if verbose == True:
        print('V_rms = ', V_rms*1000000,'uV')
        print('Bandwidth = %.2f MHz'%(bandwidth*1000))
        print('Thresh = %0.3e V/m'%(thresh))
    return thresh


def timeDiffAtAngle(reader, index_station , index_antenna_1,index_antenna_2, electricThresh = None, theta_ant_bounds = [0.0,180.0], mode = 'path', histbins = None, config = None, threshold_time = 10, hist_range = None, match_solution_type = False, solution = None, title = None,xlim = None, ylim = None, refraction_index = 1.8):
    '''
    Computes the arrival time differences at each antenna for each event in reader and returns a hist and metadata
    If match_solution_type == True then only time differences with the same solution type as the indexed antenna
    will show up in plots
    ''' 
    info = reader['info'][...]
    
    x_neutrino = (reader['x_0'][...])[info['eventid']]
    y_neutrino = (reader['y_0'][...])[info['eventid']]
    z_neutrino = (reader['z_0'][...])[info['eventid']]

    x_antenna = (numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,0])[info['antenna']]
    y_antenna = (numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,1])[info['antenna']]
    z_antenna = (numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,2])[info['antenna']]
    
    direct_distances = numpy.sqrt( (x_neutrino - x_antenna)**2 + (y_neutrino - y_antenna)**2 + (z_neutrino - z_antenna)**2 )
    timediffs_og = (info[info['antenna'] == index_antenna_1]['time'])[info['eventid']] - info['time'] #easiest to do before slimming info
    timediffs = timediffs_og
    additional_string = ''
    if solution != None:
        additional_string = additional_string + ' (%s)'%(solution)
    if electricThresh != None:
        additional_string = additional_string + ' (Mean E-Field Thresh = %0.2e V/m)'%(numpy.mean(electricThresh))
    
    #APPLYING CUTS:
    #FIRST THOSE THAT APPLY ON THE ANTENNA BY ANTENNA SCALE (len is n_antenna*n_events)
    #ignore any events with nan numbers (currently default output of griddata when extrapolating (i.e. hull failed)
    cut = [True]*len(info)
    for field in info.dtype.names:
        if (info[field].dtype.type is not numpy.string_):
            cut = numpy.logical_and(cut,  (numpy.isnan(info[field]) == False)  )  #HOW ARE NANS STILL GETTING THROUGH>??
    cut = numpy.logical_and(cut,  (numpy.isnan(timediffs) == False)  )
    #NOW AT THE EVENT LEVEL (len = n_events)
    
    #Match Solution Type (SHOULD BE DONE FIRST OR IT MIGHT BREAK BECAUSE OF THE WAY THE CALCULATION IS DONE):
    if match_solution_type == True:
        cut = numpy.logical_and(cut,  info['solution'] == (info[info['antenna'] == index_antenna_1]['solution'])[info['eventid']]  )
    #Within Angular Range
    cut = numpy.logical_and(cut,  numpy.logical_and(numpy.less_equal(info['theta_ant'][...] ,max(theta_ant_bounds)),numpy.greater_equal(info['theta_ant'][...] ,min(theta_ant_bounds)))  )
    #Selecting Solution Type:
    if solution != None:
        if numpy.isin(solution, ['direct','cross','reflect','direct_2','cross_2','reflect_2']):
            cut = numpy.logical_and(cut,  info['solution'][...] == solution.encode() )
        else:
            print('Not cutting on solution type.  Input %s not in list of solutions expected'%(solution))
    if electricThresh != None:
        #I believe this assues BOTH pass the threshold.  Maybe I don't want it this way?
        cut = numpy.logical_and(cut, numpy.greater(info['electric_field'][...] , electricThresh))
    if sum(cut) == 0:
        print('No events pass all cuts for (%0.2f $\leq \\theta \leq$ %0.2f) %s'%(min(theta_ant_bounds),max(theta_ant_bounds), additional_string))
        return 0,0
    #Ignore repeated entries (if there are any)
    unique_info,unique_cut = numpy.unique(info,return_index=True) #redefines info as only unique entries corresponding to match_solution_type request 
    cut = numpy.logical_and(cut,  numpy.isin(numpy.arange(0,len(info)),unique_cut)  )
    #Select Second Antenna
    
    
    info2 = info[numpy.logical_and(cut, info['antenna'] == index_antenna_2 )]
    info1 = info[numpy.logical_and(numpy.isin(info['eventid'],info2['eventid']),info['antenna'] == index_antenna_1)]
    #info = info[cut]
    cut = numpy.logical_and(cut, info['antenna'] == index_antenna_2 )
    timediffs = timediffs[cut]
    #'''
    direct_distances = direct_distances[cut]
    path_distances = info2['distance']
    theta_ant = info2['theta_ant']
    #residual calculation:
    d = numpy.diff(numpy.add(config['antennas']['positions'] , config['stations']['positions'][index_station])[:,2][[index_antenna_1,index_antenna_2]])[0]
    c = gnosim.utils.constants.speed_light #m/ns
    residual = timediffs - (d*refraction_index/c)* numpy.cos(numpy.deg2rad(theta_ant))
    #flagged_events = info[abs(residual) > threshold_time]
    flagged_events = info2[abs(residual) > threshold_time]['eventid']
    #flagged_events = info2[residual < -0.05]['eventid']
    if (histbins == None):
        histbins = [6300/4.0,10000]

    fig1, ax1 = pylab.subplots()
    if mode == 'path':
        if hist_range == None:
            pylab.hist2d( path_distances , residual , histbins , norm=LogNorm() , range = hist_range )
        else:
            pylab.hist2d( path_distances , residual , histbins , norm=LogNorm() , range = hist_range )
        pylab.colorbar()
        pylab.ylabel('Residual for S%iA%i-S%iA%i (ns)'%(index_station,index_antenna_1,index_station,index_antenna_2),fontsize = 20)
        pylab.xlabel('Path Distance To Event (m)',fontsize = 20)
        if title != None:
            pylab.title(title,fontsize = 20)
        else:
            pylab.title('Residual Time from Plane Wave Solution v.s. Path Distance to Event \n(%0.2f $\leq \\theta \leq$ %0.2f) %s'%(min(theta_ant_bounds),max(theta_ant_bounds),additional_string),fontsize = 20)
                
    else:
        if hist_range == None:
            pylab.hist2d( direct_distances , residual , histbins , norm=LogNorm() , range = hist_range )
        else:
            pylab.hist2d( direct_distances , residual , histbins , norm=LogNorm() , range = hist_range )
        pylab.colorbar()
        pylab.ylabel('Residual for S%iA%i-S%iA%i (ns)'%(index_station,index_antenna_1,index_station,index_antenna_2),fontsize = 20)
        pylab.xlabel('Direct Distance To Event (m)',fontsize = 20)
        if title != None:
            pylab.title(title,fontsize = 20)
        else:
            pylab.title('Residual Time from Plane Wave Solution v.s. Direct Distance to Event \n(%0.2f $\leq \\theta \leq$ %0.2f) %s'%(min(theta_ant_bounds),max(theta_ant_bounds),additional_string),fontsize = 20)
    if xlim != None:
        pylab.xlim(xlim)
    if ylim != None:
        pylab.ylim(ylim)
    #'''
    pylab.subplots_adjust(right = 1.0)
    return info,info1,info2, residual, flagged_events, timediffs_og
#########################

pylab.close("all")
info = reader['info'][...]
info2 = reader2['info'][...]
index_antenna_1 = 0
index_antenna_2 = 7
refraction_index = numpy.mean(gnosim.earth.antarctic.indexOfRefraction(numpy.linspace(-200,-207,8),ice_model=gnosim.earth.antarctic.ice_model_default))


sol = 'direct'
p1 = False
p2 = False
p3 = False
p4 = False
p5 = False
p6 = False
p7 = False
p8 = False
p9 = False
###############

if p1:
    plotAntennaSolutions( reader, lib , 0, index_antenna_1, index_antenna_2 ) 
    plotAntennaSolutions( reader2, lib2 , 0, index_antenna_1, index_antenna_2 ) 

if p2:
    threshold_time = 1.0
    residual = True
    expected = False
    histbins = [720,10000]
    xlim = (20,180)
    ylim = (-2,2)
    hist_range = [[0,180],[-10,10]]
    #timeDiffThetaAntHist(reader, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,plot_expected = expected,config=config,residual=residual,histbins = [360,2000])
    #pylab.ylim(-100,100)
    fig1, flagged_events1, timediffs1, info1 = timeDiffThetaAntHist(reader, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,plot_expected = expected,config=config,residual=residual,histbins = [360,10000],hist_range = hist_range,title = 'all',xlim = xlim,ylim = ylim,refraction_index=refraction_index,threshold_time=threshold_time)
    fig2, flagged_events2, timediffs2, info2 = timeDiffThetaAntHist(reader2, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,plot_expected = expected,config=config2,residual=residual,histbins = [360,10000],hist_range = hist_range,title = 'all',xlim = xlim,ylim = ylim,refraction_index=refraction_index,threshold_time=threshold_time)
    residual = False
    expected = True
    ylim = None
    timeDiffThetaAntHist(reader, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,plot_expected = expected,config=config,residual=residual,histbins = [360,10000],title = 'all',xlim = xlim,ylim = ylim,refraction_index=refraction_index,threshold_time=threshold_time)
    timeDiffThetaAntHist(reader2, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,plot_expected = expected,config=config2,residual=residual,histbins = [360,10000],title = 'all',xlim = xlim,ylim = ylim,refraction_index=refraction_index,threshold_time=threshold_time)
    #timeDiffThetaAntHist(reader2, 0 , index_antenna_1,index_antenna_2, match_solution_type = False,plot_expected = True,config=config2,residual=True,histbins = [720,20000],hist_range = hist_range,title = 'all',xlim = xlim,ylim = ylim)
    #timeDiffThetaAntHist(reader2, 0 , index_antenna_1,index_antenna_2, match_solution_type = False,plot_expected = expected,config=config2,residual=residual,histbins = [720,20000],hist_range = hist_range,title = 'all',xlim = xlim,ylim = ylim)
    #pylab.ylim(-100,100)
    
    #residual = False
    #expected = True
    #timeDiffThetaAntHist(reader2, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,plot_expected = expected,config=config2,residual=residual,histbins = histbins,title = 'all',xlim = xlim,ylim = ylim)
    '''
    for sol in ['direct','cross','reflect','direct_2','cross_2','reflect_2']:
        residual = True
        expected=False
        timeDiffThetaAntHist(reader2, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,plot_expected = expected,config=config2,residual=residual,histbins = histbins,solution = sol,title = sol,xlim = xlim,ylim = ylim)
        #residual = False
        #expected = True
        #timeDiffThetaAntHist(reader2, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,plot_expected = expected,config=config2,residual=residual,histbins = histbins,solution = sol,title = sol,xlim = xlim,ylim = ylim)
    '''   
    
    
if p3:
    vline = 360
    thetaThetaDiffHist(reader, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,vline=vline)
    thetaThetaDiffHist(reader2, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,vline=vline/3)
    

if p4:
    timeDiffThetaAntHist(reader, 0 , index_antenna_1,index_antenna_2, match_solution_type = False,histbins = [500,500])
    timeDiffThetaAntHist(reader2, 0 ,index_antenna_1,index_antenna_2, match_solution_type = False,histbins = [500,500])

if p5:
    timeDiffThetaAntHist(reader, 0 , index_antenna_1,index_antenna_2, match_solution_type = True,histbins = [500,500],config=config)
    timeDiffThetaAntHist(reader2, 0 ,index_antenna_1,index_antenna_2, match_solution_type = True,histbins = [500,500],config=config2)
    
if p6:
    pathDiffHist(reader, 0 , index_antenna_1, config = None, histbins = None, threshold_time = 200, match_solution_type = False)
    pathDiffHist(reader2, 0 , index_antenna_1, config = None, histbins = None, threshold_time = 200, match_solution_type = False)

if p7:
    pathDiffHist(reader, 0 , index_antenna_1, config = None, histbins = None, threshold_time = 200, match_solution_type = True)
    pathDiffHist(reader2, 0 , index_antenna_1, config = None, histbins = None, threshold_time = 200, match_solution_type = True)

if p8:
    maxTimeDiffCMap(reader , lib , plot_hull = False , hull = None)
    maxTimeDiffCMap(reader2 , lib2 , plot_hull = False , hull = None)
########################
if p9:
    flaggedTimeDiffLocation(reader,flagged_events1,timediffs1)
    flaggedSolutionLocation(reader,flagged_events1,index_antenna_1)

    flaggedTimeDiffLocation(reader2,flagged_events2,timediffs2)
    flaggedSolutionLocation(reader2,flagged_events2,index_antenna_1)


hist_range = [[0,9000],[-2,2]]
ylim = [-0.5,0.75]
histbins = [1000,2000]
#ethresh = electricFieldThreshold( snr , gain, noise_temperature, bandwidth, frequency, verbose = True)
ethresh = electricFieldThreshold( 1.0 , 2.0, 320.0, 0.7, 0.2, verbose = True)
#ethresh = None
info,info1,info2, residual, flagged_events, timediffs = timeDiffAtAngle(reader, 0 , index_antenna_1,index_antenna_2, electricThresh = ethresh, theta_ant_bounds = [0.0,180.0], histbins = histbins, config = config, threshold_time = 2, hist_range = hist_range, match_solution_type = True,  solution = 'direct', title = None,xlim = None, ylim = ylim, refraction_index = refraction_index)
#flaggedSolutionLocation(reader,numpy.unique(info1['eventid']),index_antenna_1)
flaggedTimeDiffLocation(reader,flagged_events,timediffs)
norm = matplotlib.colors.Normalize(vmin = -0.6, vmax = 3.0)
residualLocation(reader,numpy.unique(info1['eventid']),residual,title = 'All Events',cnorm = norm)
residualLocation(reader,flagged_events,residual[numpy.isin(info1['eventid'],flagged_events)],title = 'Events with Residual < -0.05',cnorm = norm)
residualLocation(reader,flagged_events,residual[numpy.isin(info1['eventid'],flagged_events)],title = 'Events with Residual < -0.05',cnorm = None)
'''

angles = numpy.linspace(0,180,7)
for ang_index in range(len(angles)-1):
    theta_ant_bounds = [angles[ang_index],angles[ang_index+1]]
    info, flagged_events = timeDiffAtAngle(reader, 0 , index_antenna_1,index_antenna_2, electricThresh = ethresh, theta_ant_bounds = theta_ant_bounds, histbins = histbins, config = config, threshold_time = 10, hist_range = hist_range, match_solution_type = True,  solution = None, title = None,xlim = None, ylim = ylim, refraction_index = refraction_index)

info, flagged_events = timeDiffAtAngle(reader, 0 , index_antenna_1,index_antenna_2, electricThresh = ethresh, theta_ant_bounds = [0.0,180.0], histbins = histbins, config = config, threshold_time = 10, hist_range = hist_range, match_solution_type = True,  solution = None, title = None,xlim = None, ylim = ylim, refraction_index = refraction_index)
info, flagged_events = timeDiffAtAngle(reader, 0 , index_antenna_1,index_antenna_2, electricThresh = ethresh, theta_ant_bounds = [0.0,90.0], histbins = histbins, config = config, threshold_time = 10, hist_range = hist_range, match_solution_type = True,  solution = None, title = None,xlim = None, ylim = ylim, refraction_index = refraction_index)
info, flagged_events = timeDiffAtAngle(reader, 0 , index_antenna_1,index_antenna_2, electricThresh = ethresh, theta_ant_bounds = [90,180.0], histbins = histbins, config = config, threshold_time = 10, hist_range = hist_range, match_solution_type = True,  solution = None, title = None,xlim = None, ylim = ylim, refraction_index = refraction_index)

for solution in ['direct','cross','reflect','direct_2','cross_2','reflect_2']:
    info, flagged_events = timeDiffAtAngle(reader, 0 , index_antenna_1,index_antenna_2, electricThresh = ethresh, theta_ant_bounds = [0.0,180.0], histbins = histbins, config = config, threshold_time = 10, hist_range = hist_range, match_solution_type = True,  solution = solution, title = None,xlim = None, ylim = ylim, refraction_index = refraction_index) 
    '''
