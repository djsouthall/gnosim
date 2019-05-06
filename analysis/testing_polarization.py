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
from matplotlib import gridspec
pylab.ion()
  
if __name__ == "__main__":
    pylab.close('all')
    figsize = (20.,12.5)#(16.,11.2)
    #TODO:  Make this have seperate plots per antenna incase they are oriented different and have different expected shapes.
    #Parameters
    #infile = '/home/dsouthall/scratch-midway2/results_2019_April15_real_config_antarctica_180_rays_signed_fresnel_3.00e+09_GeV_1000000_events_1_seed_1.h5'#'/home/dsouthall/scratch-midway2/April8/results_2019_April_real_config_antarctica_180_rays_signed_fresnel_1.00e+08_GeV_1000000_events_0_seed_1.h5'#'/home/dsouthall/scratch-midway2/results_2019_April_real_config_full_station_3.00e+09_GeV_10000_events_5_seed_1.h5'#'/home/dsouthall/scratch-midway2/results_2019_April_real_config_full_station_3.00e+09_GeV_10000_events_1_seed_1.h5'#results_2019_Mar_config_dipole_octo_-200_antarctica_180_rays_3.00e+09_GeV_10000_events_1.h5
    #infile = '/home/dsouthall/scratch-midway2/April8/results_2019_April_real_config_antarctica_180_rays_signed_fresnel_1.00e+08_GeV_1000000_events_0_seed_1.h5'#'/home/dsouthall/scratch-midway2/results_2019_April_real_config_full_station_3.00e+09_GeV_10000_events_5_seed_1.h5'#'/home/dsouthall/scratch-midway2/results_2019_April_real_config_full_station_3.00e+09_GeV_10000_events_1_seed_1.h5'#results_2019_Mar_config_dipole_octo_-200_antarctica_180_rays_3.00e+09_GeV_10000_events_1.h5
    infile = '/home/dsouthall/scratch-midway2/results_2019_testingMay1_real_config_antarctica_180_rays_signed_fresnel_3.00e+09_GeV_1000000_events_1_seed_1.h5'
    try:
        print('Pre_loaded = %i, skipping loading',pre_loaded)
    except:
        reader = h5py.File(infile , 'r')
        cut = reader['info']['has_solution'][...]
        info = reader['info'][cut]
        events_with_pretrigger = numpy.unique(info[info['pre_triggered']]['eventid'])
        cut = numpy.isin(info['eventid'],events_with_pretrigger)
        info = info[cut] #only events with solutions where at least one solution met pre_trigger

        reader2 = h5py.File(infile , 'r')
        cut = reader2['info']['has_solution'][...]
        info2 = reader2['info'][cut]
        pre_loaded = True


    fig = pylab.figure(figsize=figsize)
    fig.canvas.set_window_title('Red Fact v Pol Dot (Scatter)')

    pylab.title('Signal Reduction Factor v.s. Calculated Dot Angle',fontsize=24)
    ax = fig.gca()
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    s = 1.0
    beam_factor  = numpy.sin(numpy.deg2rad(info['theta_ant']%180.0))
    angles = info['pol_dot_angle']
    pylab.scatter(angles,info['signal_reduction_factor'],s = s,c='b',label='All Pretriggered Events')
    pylab.scatter(angles[info['triggered']],info['signal_reduction_factor'][info['triggered']],s = s,c='r',label='Triggered Events')
    pylab.ylabel('Signal Reduction Factor',fontsize=20)
    pylab.xlabel('Polarization Dot Angle (Deg)',fontsize=20)
    pylab.legend(loc='upper right',fontsize=20)
    pylab.tick_params(labelsize=16)

    fig = pylab.figure(figsize=figsize)
    fig.canvas.set_window_title('Red Fact v Pol Dot (Color)')
    ax = fig.gca()
    pylab.hist2d(angles,info['signal_reduction_factor'],norm=matplotlib.colors.LogNorm(),bins=(100,100))
    pylab.ylabel('Signal Reduction Factor',fontsize=20)
    pylab.xlabel('Polarization Dot Angle (Deg)',fontsize=20)
    pylab.colorbar()

    fig = pylab.figure(figsize=figsize)
    fig.canvas.set_window_title('Theta Ant')
    ax = fig.gca()
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    pylab.hist(info2[info2['solution'] == b'direct']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='b',label='direct')
    pylab.hist(info2[info2['solution'] == b'cross']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='g',label='cross')
    pylab.hist(info2[info2['solution'] == b'reflect']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='r',label='reflect')
    pylab.legend(fontsize=16)
    pylab.xlabel('Theta_ant (deg)',fontsize=16)
    pylab.ylabel('Counts',fontsize=16)
    pylab.tick_params(labelsize=16)

    ##############################################################

    fig = pylab.figure(figsize=figsize)
    fig.canvas.set_window_title('Ang Range')
    ax = fig.gca()
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    angle_low = 85.0
    angle_high = 95.0
    ang_cut = numpy.logical_and(info['theta_ant'] > angle_low, info['theta_ant'] < angle_high)
    cut = numpy.logical_and(numpy.logical_and(info['antenna'] == 0, info['solution'] == b'direct'),ang_cut)
    r = numpy.sqrt(reader['x_0'][...][info[cut]['eventid']]**2 + reader['y_0'][...][info[cut]['eventid']]**2)
    z = reader['z_0'][...][info[cut]['eventid']]
    c = info[cut]['theta_ant']
    pylab.scatter(r,z,c=c,s=1)
    pylab.colorbar()

    ##############################################################

    fig = pylab.figure(figsize=figsize)
    fig.canvas.set_window_title('Pol Dot v Theta Ant')
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4]) 
    ax1 = pylab.subplot(gs[0])
    ax2 = pylab.subplot(gs[1],sharex=ax1)
    pylab.hist2d(info['theta_ant'],angles,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm())
    pylab.colorbar(label='Counts')
    pylab.xlabel('$\\theta_\mathrm{ant}$ (deg)',fontsize=16)
    pylab.ylabel('Polarization Dot Angle (Deg)',fontsize=16)
    pylab.tick_params(labelsize=16)
    pylab.plot(numpy.arange(180),90*numpy.sin(numpy.deg2rad(numpy.arange(180)))+90.0,c='r',label='+-90*sin(theta)+90')
    pylab.plot(numpy.arange(180),-90*numpy.sin(numpy.deg2rad(numpy.arange(180)))+90.0,c='r')
    x=numpy.arange(181)
    y1=numpy.zeros_like(x)
    y1[x <= 90] = x[x<=90] + 90
    y1[x > 90] = -x[x>90] + 270
    y2 = 180.0-y1
    pylab.plot(x,y1,c='r',linestyle='-.',label='Linear (Slope = +-1)')
    pylab.plot(x,y2,c='r',linestyle='-.')
    #pylab.plot(numpy.arange(180),90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**2+90.0,c='b',label='+-90*sin(theta)**2+90')
    #pylab.plot(numpy.arange(180),-90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**2+90.0,c='b')
    #pylab.plot(numpy.arange(180),90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**3+90.0,c='g',label='+-90*sin(theta)**3+90')
    #pylab.plot(numpy.arange(180),-90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**3+90.0,c='g')
    pylab.legend()

    pylab.subplot(gs[0])
    ax1.minorticks_on()
    ax1.grid(b=True, which='major', color='k', linestyle='-')
    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    pylab.hist(info2[info2['solution'] == b'direct']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='b',label='direct')
    pylab.hist(info2[info2['solution'] == b'cross']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='g',label='cross')
    pylab.hist(info2[info2['solution'] == b'reflect']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='r',label='reflect')
    ax1.set_yscale('log')
    pylab.legend(fontsize=16)
    pylab.xlabel('$\\theta_\mathrm{ant}$ (deg)',fontsize=16)
    pylab.ylabel('Counts',fontsize=16)
    pylab.tick_params(labelsize=16)

    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    ax1.set_position([pos2.x0,pos1.y0,pos2.width,pos1.height])


    ##############################################################


    fig = pylab.figure(figsize=figsize)
    fig.canvas.set_window_title('Attenuation v D')
    ax = pylab.subplot(3,1,1)
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    cut = info['solution'] == b'direct'
    pylab.scatter(info['distance'][cut] ,info['attenuation_factor'][cut] ,c=180.0-info['theta_ray'][cut], label = 'Direct')
    pylab.ylabel('Attenuation Factor')
    pylab.xlabel('Path Distance (m)')
    cbar = pylab.colorbar(label='Emission Theta Ray (deg)')
    pylab.legend()

    ax = pylab.subplot(3,1,2)
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    cut = info['solution'] == b'cross'
    pylab.scatter(info['distance'][cut] ,info['attenuation_factor'][cut] ,c=180.0-info['theta_ray'][cut], label = 'Cross')
    pylab.ylabel('Attenuation Factor')
    pylab.xlabel('Path Distance (m)')
    cbar = pylab.colorbar(label='Emission Theta Ray (deg)')
    pylab.legend()

    ax = pylab.subplot(3,1,3)
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    cut = info['solution'] == b'reflect'
    pylab.scatter(info['distance'][cut] ,info['attenuation_factor'][cut] ,c=180.0-info['theta_ray'][cut], label = 'Reflect')
    pylab.ylabel('Attenuation Factor')
    pylab.xlabel('Path Distance (m)')
    cbar = pylab.colorbar(label='Emission Theta Ray (deg)')
    pylab.legend()

    ##############################################################################

    fig = pylab.figure(figsize=figsize)
    fig.canvas.set_window_title('Em v Det Pol Angles')
    ax1 = pylab.subplot(3,1,1)
    cut = info['solution'] == b'direct'
    ax1.minorticks_on()
    ax1.grid(b=True, which='major', color='k', linestyle='-')
    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    emission_angles = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))[cut]
    detection_angles = numpy.rad2deg(numpy.arccos( info['detection_polarization_vector'][:,2] ))[cut]
    pylab.hist2d(emission_angles,detection_angles,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm(),label='direct')
    pylab.plot(numpy.arange(180),numpy.arange(180),c='r',linewidth=1,label='Exact Linear')
    pylab.colorbar(label='Direct ounts')
    pylab.legend()
    pylab.xlabel('Emission Theta (deg)',fontsize=16)
    pylab.ylabel('Detection Theta (deg)',fontsize=16)
    pylab.tick_params(labelsize=16)

    ax = pylab.subplot(3,1,2,sharex=ax1)
    cut = info['solution'] == b'cross'
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    emission_angles = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))[cut]
    detection_angles = numpy.rad2deg(numpy.arccos( info['detection_polarization_vector'][:,2] ))[cut]
    pylab.hist2d(emission_angles,detection_angles,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm(),label='cross')
    pylab.plot(numpy.arange(180),numpy.arange(180),c='r',linewidth=1,label='Exact Linear')
    pylab.colorbar(label='Cross Counts')
    pylab.legend()
    pylab.xlabel('Emission Theta (deg)',fontsize=16)
    pylab.ylabel('Detection Theta (deg)',fontsize=16)
    pylab.tick_params(labelsize=16)

    ax = pylab.subplot(3,1,3,sharex=ax1)
    cut = info['solution'] == b'reflect'
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    emission_angles = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))[cut]
    detection_angles = numpy.rad2deg(numpy.arccos( info['detection_polarization_vector'][:,2] ))[cut]
    pylab.hist2d(emission_angles,detection_angles,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm(),label='reflect')
    pylab.plot(numpy.arange(180),numpy.arange(180),c='r',linewidth=1,label='Exact Linear')
    pylab.colorbar(label='Reflect Counts')
    pylab.legend()
    pylab.xlabel('Emission Theta (deg)',fontsize=16)
    pylab.ylabel('Detection Theta (deg)',fontsize=16)
    pylab.tick_params(labelsize=16)


    ##############################################################################
    '''
    fig = pylab.figure(figsize=figsize)
    fig.canvas.set_window_title('Em v Det Pol Angles')
    ax1 = pylab.subplot(3,1,1)
    ax1.minorticks_on()
    ax1.grid(b=True, which='major', color='k', linestyle='-')
    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    emission_angles = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))
    detection_angles = numpy.rad2deg(numpy.arccos( info['detection_polarization_vector'][:,2] ))
    pylab.hist2d(emission_angles,detection_angles,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm())
    pylab.plot(numpy.arange(180),numpy.arange(180),c='r',label='Exact Linear')
    pylab.colorbar(label='Counts')
    pylab.xlabel('Emission Theta (deg)',fontsize=16)
    pylab.ylabel('Detection Theta (deg)',fontsize=16)
    pylab.tick_params(labelsize=16)

    ax = pylab.subplot(3,1,2,sharex = ax1)
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    emission_angles = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))
    detection_angles = numpy.rad2deg(numpy.arccos( info['detection_polarization_vector'][:,2] ))

    pylab.scatter(emission_angles,detection_angles,s=1,c=180.0-info['theta_ant'])
    #pylab.plot(numpy.arange(180),numpy.arange(180),c='r',label='Exact Linear')
    pylab.colorbar(label='Wave Theta Emission (Toward Antenna)')
    pylab.xlabel('Polarization Emission Theta (deg)',fontsize=16)
    pylab.ylabel('Polarization Detection Theta (deg)',fontsize=16)
    pylab.tick_params(labelsize=16)

    ax = pylab.subplot(3,1,3,sharex = ax1)
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    emission_angles = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))
    detection_angles = numpy.rad2deg(numpy.arccos( info['detection_polarization_vector'][:,2] ))
    pylab.scatter(emission_angles,detection_angles,s=1,c=info['theta_ant'])
    #pylab.plot(numpy.arange(180),numpy.arange(180),c='r',label='Exact Linear')
    pylab.colorbar(label='Distance Travelled')
    pylab.xlabel('Emission Theta (deg)',fontsize=16)
    pylab.ylabel('Detection Theta (deg)',fontsize=16)
    pylab.tick_params(labelsize=16)
    '''
    #pylab.plot(numpy.arange(180),90*numpy.sin(numpy.deg2rad(numpy.arange(180)))+90.0,c='r',label='+-90*sin(theta)+90')
    #pylab.plot(numpy.arange(180),-90*numpy.sin(numpy.deg2rad(numpy.arange(180)))+90.0,c='r')
    #pylab.plot(numpy.arange(180),90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**2+90.0,c='b',label='+-90*sin(theta)**2+90')
    #pylab.plot(numpy.arange(180),-90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**2+90.0,c='b')
    #pylab.plot(numpy.arange(180),90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**3+90.0,c='g',label='+-90*sin(theta)**3+90')
    #pylab.plot(numpy.arange(180),-90*numpy.sin(numpy.deg2rad(numpy.arange(180)))**3+90.0,c='g')

    fig = pylab.figure(figsize=figsize)
    fig.canvas.set_window_title('Diff Em v Det Pol Angles')
    dz = numpy.cos(numpy.deg2rad(info['theta_ant'])) - numpy.cos(numpy.deg2rad(info['theta_ray']))
    dzp = info['detection_polarization_vector'][:,2] - info['emission_polarization_vector'][:,2]
    pylab.scatter(dzp,dz)



































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
                fig = pylab.figure(figsize=figsize)
                fig.canvas.set_window_title('Test')
                pylab.scatter(r,z,c=theta_ant,label=solution,s=1)
                pylab.colorbar()
                pylab.xlabel('r (m)')
                pylab.ylabel('z (m)')
                fig_array.append(fig)
                windowed_cut = numpy.roll(windowed_cut,1)
        return fig_array
    figs = mcLibArea(test_lib,178,solutions=numpy.array(['direct']))
    '''