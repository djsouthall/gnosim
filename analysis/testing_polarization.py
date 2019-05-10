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
    show_Red_Fact_v_Pol_Dot = False
    show_Ang_Range = False
    show_Pol_Dot_v_Theta_Ant = True
    show_Emission_Pol_Dot_v_Theta_Ant = True
    show_Emission_Pol_Dot_v_Theta_Emit = True
    show_Attenuation_v_D = False
    show_Em_v_Det_Pol_Angles = False
    show_Diff_Em_v_Det_Pol_Angles = False
    #####################################################################

    figsize = (20.,12.5)#(16.,11.2)
    #TODO:  Make this have seperate plots per antenna incase they are oriented different and have different expected shapes.
    #Parameters
    infile = os.environ['GNOSIM_DATA'] + '/May3/results_2019_testing_May3_real_config_antarctica_180_rays_signed_fresnel_3.16e+09_GeV_1000000_events_0_seed_1.h5'
    try:
        print('Pre_loaded = %i, skipping loading',pre_loaded)
    except:
        reader = h5py.File(infile , 'r')
        cut = reader['info']['has_solution'][...]
        info = reader['info'][cut]
        x = reader['x_0'][...]
        y = reader['y_0'][...]
        r = numpy.sqrt(x**2+y**2)
        z = reader['z_0'][...]

        #events_with_pretrigger = numpy.unique(info[info['pre_triggered']]['eventid'])
        #cut = numpy.isin(info['eventid'],events_with_pretrigger)
        #info = info[cut] #only events with solutions where at least one solution met pre_trigger

    #####################################################################

    if show_Red_Fact_v_Pol_Dot == True:
        fig = pylab.figure(figsize=figsize)
        fig.canvas.set_window_title('Red Fact v Pol Theta (Color)')
        ax = fig.gca()
        pylab.hist2d(info['pol_dot_angle'],info['signal_reduction_factor'],norm=matplotlib.colors.LogNorm(),bins=(100,100))
        pylab.ylabel('Signal Reduction Factor',fontsize=20)
        pylab.xlabel('Polarization Dot Angle (Deg)',fontsize=20)

    #####################################################################

    if show_Ang_Range == True:
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

    #####################################################################

    if show_Pol_Dot_v_Theta_Ant == True:
        fig = pylab.figure(figsize=figsize)
        fig.canvas.set_window_title('Pol Theta v Theta Ant')
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4]) 
        ax1 = pylab.subplot(gs[0])
        ax2 = pylab.subplot(gs[1],sharex=ax1)
        pylab.hist2d(info['theta_ant'],info['pol_dot_angle'],bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm())
        pylab.colorbar(label='Counts')
        pylab.xlabel('$\\theta_\mathrm{ant}$ (deg)',fontsize=16)
        pylab.ylabel('Polarization Dot Angle (Deg)',fontsize=16)
        pylab.tick_params(labelsize=16)

        x_window=numpy.arange(181)
        y_window=numpy.zeros_like(x_window)
        y_window[x_window <= 90] = x_window[x_window<=90] + 90
        y_window[x_window > 90] = -x_window[x_window>90] + 270
        y_window2 = 180.0-y_window
        pylab.plot(x_window,y_window,c='r',linestyle='-.',label='Linear (Slope = +-1)')
        pylab.plot(x_window,y_window2,c='r',linestyle='-.')

        pylab.legend()

        pylab.subplot(gs[0])
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', color='k', linestyle='-')
        ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        pylab.hist(info[info['solution'] == b'direct']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='b',label='direct')
        pylab.hist(info[info['solution'] == b'cross']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='g',label='cross')
        pylab.hist(info[info['solution'] == b'reflect']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='r',label='reflect')
        ax1.set_yscale('log')
        pylab.legend(fontsize=16)
        pylab.xlabel('$\\theta_\mathrm{ant}$ (deg)',fontsize=16)
        pylab.ylabel('Counts',fontsize=16)
        pylab.tick_params(labelsize=16)

        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax1.set_position([pos2.x0,pos1.y0,pos2.width,pos1.height])

    #####################################################################

    if show_Emission_Pol_Dot_v_Theta_Ant == True:
        fig = pylab.figure(figsize=figsize)
        fig.canvas.set_window_title('Emission Pol Theta v Theta Ant')
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4]) 
        ax1 = pylab.subplot(gs[0])
        ax2 = pylab.subplot(gs[1],sharex=ax1)

        emission_polarization_theta = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))

        pylab.hist2d(info['theta_ant'],emission_polarization_theta,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm())
        pylab.colorbar(label='Counts')
        pylab.xlabel('$\\theta_\mathrm{ant}$ (deg)',fontsize=16)
        pylab.ylabel('Emission Polarization Theta (Deg)',fontsize=16)
        pylab.tick_params(labelsize=16)

        x_window=numpy.arange(181)
        y_window=numpy.zeros_like(x_window)
        y_window[x_window <= 90] = x_window[x_window<=90] + 90
        y_window[x_window > 90] = -x_window[x_window>90] + 270
        y_window2 = 180.0-y_window
        pylab.plot(x_window,y_window,c='r',linestyle='-.',label='Linear (Slope = +-1)')
        pylab.plot(x_window,y_window2,c='r',linestyle='-.')

        pylab.legend()

        pylab.subplot(gs[0])
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', color='k', linestyle='-')
        ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        pylab.hist(info[info['solution'] == b'direct']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='b',label='direct')
        pylab.hist(info[info['solution'] == b'cross']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='g',label='cross')
        pylab.hist(info[info['solution'] == b'reflect']['theta_ant'],bins=180,range = (0.,180.),alpha=0.5,color='r',label='reflect')
        ax1.set_yscale('log')
        pylab.legend(fontsize=16)
        pylab.xlabel('$\\theta_\mathrm{ant}$ (deg)',fontsize=16)
        pylab.ylabel('Counts',fontsize=16)
        pylab.tick_params(labelsize=16)

        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax1.set_position([pos2.x0,pos1.y0,pos2.width,pos1.height])

    #####################################################################

    if show_Emission_Pol_Dot_v_Theta_Emit == True:
        fig = pylab.figure(figsize=figsize)
        fig.canvas.set_window_title('Emission Pol Theta v Theta WaveEmit')
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4]) 
        ax1 = pylab.subplot(gs[0])
        ax2 = pylab.subplot(gs[1],sharex=ax1)

        emission_polarization_theta = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))
        emission_wave_theta = 180.0 - info['theta_ray']
        pylab.hist2d(emission_wave_theta,emission_polarization_theta,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm())
        pylab.colorbar(label='Counts')
        pylab.xlabel('$\\theta_\mathrm{Wave Emit}$ (deg)',fontsize=16)
        pylab.ylabel('Emission Polarization Theta (Deg)',fontsize=16)
        pylab.tick_params(labelsize=16)

        x_window=numpy.arange(181)
        y_window=numpy.zeros_like(x_window)
        y_window[x_window <= 90] = x_window[x_window<=90] + 90
        y_window[x_window > 90] = -x_window[x_window>90] + 270
        y_window2 = 180.0-y_window
        pylab.plot(x_window,y_window,c='r',linestyle='-.',label='Linear (Slope = +-1)')
        pylab.plot(x_window,y_window2,c='r',linestyle='-.')

        pylab.legend()

        pylab.subplot(gs[0])
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', color='k', linestyle='-')
        ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        pylab.hist(emission_wave_theta[info['solution'] == b'direct'],bins=180,range = (0.,180.),alpha=0.5,color='b',label='direct')
        pylab.hist(emission_wave_theta[info['solution'] == b'cross'],bins=180,range = (0.,180.),alpha=0.5,color='g',label='cross')
        pylab.hist(emission_wave_theta[info['solution'] == b'reflect'],bins=180,range = (0.,180.),alpha=0.5,color='r',label='reflect')
        ax1.set_yscale('log')
        pylab.legend(fontsize=16)
        pylab.xlabel('$\\theta_\mathrm{Wave Emit}$ (deg)',fontsize=16)
        pylab.ylabel('Counts',fontsize=16)
        pylab.tick_params(labelsize=16)

        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax1.set_position([pos2.x0,pos1.y0,pos2.width,pos1.height])

        pylab.figure()
        pylab.hist(z[info[emission_wave_theta > 90.0]['eventid']],bins=100)
        pylab.xlabel('z')
        pylab.ylabel('Counts for emission_wave_theta > 90.0')

        interest = numpy.where(z < - 200)[0][numpy.isin(numpy.where(z < - 200)[0], numpy.unique(info[emission_wave_theta > 90.0]['eventid']))]
        interest_info = info[numpy.isin(info['eventid'],interest)]
        print(180.0-interest_info['theta_ray'])

        pylab.figure()
        pylab.scatter(r[interest],z[interest])
        pylab.xlabel('r(m)')
        pylab.ylabel('z(m)')


    #####################################################################

    if show_Attenuation_v_D == True:
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

    #####################################################################

    if show_Em_v_Det_Pol_Angles == True:
        fig = pylab.figure(figsize=figsize)
        fig.canvas.set_window_title('Em v Det Pol Angles')
        ax1 = pylab.subplot(3,1,1)
        cut = info['solution'] == b'direct'
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', color='k', linestyle='-')
        ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        emission_polarization_theta = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))[cut]
        detection_angles = numpy.rad2deg(numpy.arccos( info['detection_polarization_vector'][:,2] ))[cut]
        pylab.hist2d(emission_polarization_theta,detection_angles,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm(),label='direct')
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

        emission_polarization_theta = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))[cut]
        detection_angles = numpy.rad2deg(numpy.arccos( info['detection_polarization_vector'][:,2] ))[cut]
        pylab.hist2d(emission_polarization_theta,detection_angles,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm(),label='cross')
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

        emission_polarization_theta = numpy.rad2deg(numpy.arccos( info['emission_polarization_vector'][:,2] ))[cut]
        detection_angles = numpy.rad2deg(numpy.arccos( info['detection_polarization_vector'][:,2] ))[cut]
        pylab.hist2d(emission_polarization_theta,detection_angles,bins=(100,100),range=[[0,180],[0,180]],norm=matplotlib.colors.LogNorm(),label='reflect')
        pylab.plot(numpy.arange(180),numpy.arange(180),c='r',linewidth=1,label='Exact Linear')
        pylab.colorbar(label='Reflect Counts')
        pylab.legend()
        pylab.xlabel('Emission Theta (deg)',fontsize=16)
        pylab.ylabel('Detection Theta (deg)',fontsize=16)
        pylab.tick_params(labelsize=16)

    #####################################################################

    if show_Diff_Em_v_Det_Pol_Angles == True:
        fig = pylab.figure(figsize=figsize)
        fig.canvas.set_window_title('Diff Em v Det Pol Angles')
        dz = numpy.cos(numpy.deg2rad(info['theta_ant'])) - numpy.cos(numpy.deg2rad(info['theta_ray']))
        dzp = info['detection_polarization_vector'][:,2] - info['emission_polarization_vector'][:,2]
        pylab.scatter(dzp,dz)


