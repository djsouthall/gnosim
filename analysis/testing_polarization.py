'''
This test that the polarization makes sense. 
'''

import sys
import numpy
import h5py
import matplotlib
import pylab
import yaml
import os
import os.path
import glob
import gnosim.sim.antarcticsim
import gnosim.trace.refraction_library
import matplotlib
pylab.ion()
  
if __name__ == "__main__":
    pylab.close('all')
    #TODO:  Make this have seperate plots per antenna incase they are oriented different and have different expected shapes.
    #Parameters
    #infile = '/home/dsouthall/scratch-midway2/results_2019_April15_real_config_antarctica_180_rays_signed_fresnel_3.00e+09_GeV_1000000_events_1_seed_1.h5'#'/home/dsouthall/scratch-midway2/April8/results_2019_April_real_config_antarctica_180_rays_signed_fresnel_1.00e+08_GeV_1000000_events_0_seed_1.h5'#'/home/dsouthall/scratch-midway2/results_2019_April_real_config_full_station_3.00e+09_GeV_10000_events_5_seed_1.h5'#'/home/dsouthall/scratch-midway2/results_2019_April_real_config_full_station_3.00e+09_GeV_10000_events_1_seed_1.h5'#results_2019_Mar_config_dipole_octo_-200_antarctica_180_rays_3.00e+09_GeV_10000_events_1.h5
    infile = '/home/dsouthall/scratch-midway2/April8/results_2019_April_real_config_antarctica_180_rays_signed_fresnel_1.00e+08_GeV_1000000_events_0_seed_1.h5'#'/home/dsouthall/scratch-midway2/results_2019_April_real_config_full_station_3.00e+09_GeV_10000_events_5_seed_1.h5'#'/home/dsouthall/scratch-midway2/results_2019_April_real_config_full_station_3.00e+09_GeV_10000_events_1_seed_1.h5'#results_2019_Mar_config_dipole_octo_-200_antarctica_180_rays_3.00e+09_GeV_10000_events_1.h5
    library_dir =  '/home/dsouthall/Projects/GNOSim/library_-173_antarctica_180_rays_signed_fresnel'
    try:
        print('Pre_loaded = %i, skipping loading',pre_loaded)
    except:
        reader = h5py.File(infile , 'r')
        cut = reader['info']['has_solution'][...]
        info = reader['info'][cut]
        events_with_pretrigger = numpy.unique(info[info['pre_triggered']]['eventid'])
        cut = numpy.isin(info['eventid'],events_with_pretrigger)
        info = info[cut] #only events with solutions where at least one solution met pre_trigger
        #test_lib = gnosim.trace.refraction_library.RefractionLibrary(library_dir+'/*.h5',pre_split = True)

        reader2 = h5py.File(infile , 'r')
        cut = reader2['info']['has_solution'][...]
        info2 = reader2['info'][cut]
        pre_loaded = True


    fig = pylab.figure()

    pylab.title('Signal Reduction Factor v.s. Calculated Dot Angle\nfor Direct and Cross Solutions',fontsize=24)
    ax = fig.gca()
    s = 1.0
    cut = info['solution'] != b'reflect'
    beam_factor  = numpy.sin(numpy.deg2rad(info[cut]['theta_ant']%180.0))
    #angles = info[cut]['pol_dot_angle']
    angles = numpy.rad2deg(numpy.arccos(numpy.divide(info[cut]['signal_reduction_factor'],numpy.multiply(info[cut]['a_s'],beam_factor))))
    pylab.scatter(angles,info[cut]['signal_reduction_factor'],s = s,c='b',label='All Pretriggered Events')
    pylab.scatter(angles[info[cut]['triggered']],info[cut]['signal_reduction_factor'][info[cut]['triggered']],s = s,c='r',label='Triggered Events')
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    pylab.ylabel('Signal Reduction Factor',fontsize=20)
    pylab.xlabel('Polarization Dot Angle (Deg)',fontsize=20)
    pylab.legend(loc='upper right',fontsize=20)
    pylab.tick_params(labelsize=16)

    fig = pylab.figure()
    pylab.hist2d(angles,info[cut]['signal_reduction_factor'],norm=matplotlib.colors.LogNorm(),bins=(100,100))
    pylab.ylabel('Signal Reduction Factor',fontsize=20)
    pylab.xlabel('Polarization Dot Angle (Deg)',fontsize=20)
    pylab.colorbar()

    '''
    info2 = info2[info2['antenna'] == 0]
    numpy.savetxt('direct_-173m.txt',(info2[info2['solution'] == b'direct']['theta_ant']), delimiter = ',')
    numpy.savetxt('cross_-173m.txt',(info2[info2['solution'] == b'cross']['theta_ant']), delimiter = ',')
    numpy.savetxt('reflect_-173m.txt',(info2[info2['solution'] == b'reflect']['theta_ant']), delimiter = ',')
    numpy.savetxt('total_-173m.txt',(info2['theta_ant']), delimiter = ',')
    '''
    fig = pylab.figure()
    pylab.hist(info2[info2['solution'] == b'direct']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='b',label='direct')
    pylab.hist(info2[info2['solution'] == b'cross']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='g',label='cross')
    pylab.hist(info2[info2['solution'] == b'reflect']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='r',label='reflect')
    pylab.legend(fontsize=16)
    pylab.xlabel('Theta_ant (deg)',fontsize=16)
    pylab.ylabel('Counts',fontsize=16)
    pylab.tick_params(labelsize=16)

    fig = pylab.figure()
    angle_low = 85.0
    angle_high = 95.0
    ang_cut = numpy.logical_and(info['theta_ant'] > angle_low, info['theta_ant'] < angle_high)
    cut = numpy.logical_and(numpy.logical_and(info['antenna'] == 0, info['solution'] == b'direct'),ang_cut)
    r = numpy.sqrt(reader['x_0'][...][info[cut]['eventid']]**2 + reader['y_0'][...][info[cut]['eventid']]**2)
    z = reader['z_0'][...][info[cut]['eventid']]
    c = info[cut]['theta_ant']
    pylab.scatter(r,z,c=c,s=1)
    pylab.colorbar()

    fig = pylab.figure()
    cut = info['solution'] != b'reflect'
    pylab.hist2d(info['theta_ant'][cut],angles,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm())
    pylab.colorbar()
    pylab.xlabel('theta_ant',fontsize=16)
    pylab.ylabel('Polarization Dot Angle (Deg)',fontsize=16)
    pylab.tick_params(labelsize=16)
    pylab.plot(numpy.arange(180),90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**3+90.0,c='r',label='+-90*sin(theta)+90')
    pylab.plot(numpy.arange(180),-90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**3+90.0,c='r')
    pylab.legend()

    fig = pylab.figure()
    cut = info['solution'] != b'reflect'
    cut2 = info['observation_angle'][cut] > 90.0#numpy.fabs(info['observation_angle'][cut]-55.8) > 20.0
    pylab.scatter(info['theta_ant'][cut][cut2],angles[cut2],c = info['observation_angle'][cut][cut2])
    pylab.colorbar()
    pylab.xlabel('theta_ant',fontsize=16)
    pylab.ylabel('Polarization Dot Angle (Deg)',fontsize=16)
    pylab.tick_params(labelsize=16)
    pylab.legend()


    #pylab.plot(numpy.arange(180),numpy.sqrt(0.5)*90*numpy.sin(numpy.deg2rad(numpy.arange(180)))+90.0,c='b')
    #pylab.plot(numpy.arange(180),-numpy.sqrt(0.5)*90*numpy.sin(numpy.deg2rad(numpy.arange(180)))+90.0,c='b')

    #pylab.plot(numpy.arange(180),numpy.sqrt(0.5)*90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**2+90.0,c='g')
    #pylab.plot(numpy.arange(180),-numpy.sqrt(0.5)*90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**2+90.0,c='g')
    '''
    solutions = numpy.array(['direct','cross','reflect'])
    for solution in solutions:
        r = test_lib.data[solution]['r']
        z = test_lib.data[solution]['z']
        c = test_lib.data[solution]['theta_ant']
        pylab.figure()
        pylab.scatter(r,z,c=c,label=solution,s=1)
        pylab.colorbar()
        pylab.xlabel('r (m)')
        pylab.ylabel('z (m)')

    '''








































    '''
    def mcLibArea(lib,ray_window,solutions = numpy.array(['direct','cross','reflect'])):
        #Might not need this
        #Calculates the area of the hull (in the r-z plane) using a monte carlo area calculator
        #and the hull algorithms of the library.
        fig_array = []
        for solution in solutions:
            rays = numpy.unique(test_lib.data[solution]['theta_ant'])
            windowed_cut = numpy.zeros_like(rays,dtype=bool)
            windowed_cut[0:ray_window] = True
            n_rolls = sum(~windowed_cut)
            for i in range(n_rolls):
                print('Rolling')
                dic_cut = numpy.isin(test_lib.data[solution]['theta_ant'],rays[windowed_cut])
                r = lib.data[solution]['r'][dic_cut]
                z = lib.data[solution]['z'][dic_cut]
                theta_ant = lib.data[solution]['theta_ant'][dic_cut]
                fig = pylab.figure()
                pylab.scatter(r,z,c=theta_ant,label=solution,s=1)
                pylab.colorbar()
                pylab.xlabel('r (m)')
                pylab.ylabel('z (m)')
                fig_array.append(fig)
                windowed_cut = numpy.roll(windowed_cut,1)
        return fig_array
    figs = mcLibArea(test_lib,178,solutions=numpy.array(['direct']))
    '''