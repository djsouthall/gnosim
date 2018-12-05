"""
This is/was used for the development of the orientation of antennas (specifically dipole antennas)
such that the antennas can be given different orientations down in the simulation and thus have 
beam patterns pointing in differing directions.  

What I want to do:
Use Euler angles to define the orientation of the dipole antenna wrt to the normal xyz coordinates
used in the sumulation.  Incoming rays should have a vector define in xyz, that can then be translated
to the XYZ frame f the dipole, and then used to determine where in the beam pattern they are located.

I aim to use Euler angles as defined on wikipedia and using extrinsic rotations.  A copy of the
relevant portion of the wiki has been saved as EulerAngleDefinition.pdf for reference. 

Tools that would be useful to develop here are:
    - A printout of th antenna location, perhaps in the form of a 3d plot with the vectors of xyz
      and XYZ plotted.
    - The actual beam pattern plotted on this same plot?  Might be hard
"""

import numpy
import pylab
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
import scipy.signal
import time
import math
sys.path.append("/home/dsouthall/Projects/GNOSim/")
from matplotlib import gridspec
import pandas
from mpl_toolkits.mplot3d import Axes3D

import gnosim.utils.constants
import gnosim.interaction.inelasticity
import gnosim.utils.quat
import gnosim.earth.earth
import gnosim.earth.antarctic
import gnosim.trace.refraction_library_beta
from gnosim.trace.refraction_library_beta import *
import gnosim.interaction.askaryan
import gnosim.sim.detector
pylab.ion()

############################################################

import cProfile, pstats, io

def profile(fnc):
    """
    A decorator that uses cProfile to profile a function
    This is lifted from https://osf.io/upav8/
    
    Required imports:
    import cProfile, pstats, io
    
    To use, decorate function of interest by putting @profile above
    its definition.
    """
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s)
        ps.strip_dirs().sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

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

@profile
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
    
def plotArrayFromConfig(config,only_station = 'all',verbose = False):
    '''
    Given a loaded config file this shouldplot the cooardinate system of each of
    the antennas in the lab frame.  
    only_station should either be 'all', to plot all stations, or a single index, 
    to plot a single station.  The index should be base 0. 
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
    for index_station in range(0, config['stations']['n']):
        # Loop over station antennas
        if numpy.logical_or(only_station == 'all', only_station == index_station):
            station_label = 'station'+str(index_station)
            m = ms[index_station]
            for antenna_index, antenna_label in enumerate(config['antennas']['types']):
                alpha_deg, beta_deg, gamma_deg = config['antennas']['orientations'][antenna_index]
                x, y, z = numpy.array(config['antennas']['positions'][antenna_index]) + numpy.array(config['stations']['positions'][index_station])
                xs.append(x)
                ys.append(y)
                zs.append(z)
                if verbose == True:
                    print('alpha = ', alpha_deg ,'deg')
                    print('beta = ', beta_deg,'deg')
                    print('gamma = ', gamma_deg,'deg')
                    print('x = ', x, 'm')
                    print('y = ', y, 'm')
                    print('z = ', z, 'm')
                R = eulerRotationMatrix(numpy.deg2rad(alpha_deg), numpy.deg2rad(beta_deg), numpy.deg2rad(gamma_deg))
                basis_X = R[:,0] #x basis vector of the antenna frame in the ice basis
                basis_Y = R[:,1] #y basis vector of the antenna frame in the ice basis
                basis_Z = R[:,2] #z basis vector of the antenna frame in the ice basis
                if first == True:
                    first = False
                    ax.quiver(x, y, z, basis_X[0], basis_X[1], basis_X[2],color='r',label = 'Antenna X',linestyle='--')
                    ax.quiver(x, y, z, basis_Y[0], basis_Y[1], basis_Y[2],color='g',label = 'Antenna Y',linestyle='--')
                    ax.quiver(x, y, z, basis_Z[0], basis_Z[1], basis_Z[2],color='b',label = 'Antenna Z',linestyle='--')
                else:
                    ax.quiver(x, y, z, basis_X[0], basis_X[1], basis_X[2],color='r',linestyle='--')
                    ax.quiver(x, y, z, basis_Y[0], basis_Y[1], basis_Y[2],color='g',linestyle='--')
                    ax.quiver(x, y, z, basis_Z[0], basis_Z[1], basis_Z[2],color='b',linestyle='--')
                ax.scatter(x,y,z,label = 'S%i '%index_station + antenna_label,marker=m,s=50)
    ax.set_xlim([min(xs) - 1, max(xs) + 1])
    ax.set_ylim([min(ys) - 1, max(ys) + 1])
    ax.set_zlim([min(zs) - 1, max(zs) + 1])
    ax.set_xlabel('Ice x',fontsize=16)
    ax.set_ylabel('Ice y',fontsize=16)
    ax.set_zlabel('Ice z',fontsize=16)
    pylab.legend(fancybox=True, framealpha=0.5,fontsize=12)
    return fig
############################################################

if __name__ == "__main__":
    pylab.close('all')
    
    plot_test1 = True
    plot_test2 = True
    speed_test1 = True
    
    if speed_test1 == True:
        alpha_deg = 20.0
        beta_deg = 40.0
        gamma_deg = 15.0
        
        ray_phi_deg = 35.0
        ray_theta_deg = 90.0 #horizontal at 90
        
        ray_x = numpy.sin(numpy.deg2rad(ray_theta_deg)) * numpy.cos(numpy.deg2rad(ray_phi_deg))
        ray_y = numpy.sin(numpy.deg2rad(ray_theta_deg)) * numpy.sin(numpy.deg2rad(ray_phi_deg))
        ray_z = numpy.cos(numpy.deg2rad(ray_theta_deg))
        
        ray_vector = numpy.array([ray_x,ray_y,ray_z])
        
        R = eulerRotationMatrix(numpy.deg2rad(alpha_deg), numpy.deg2rad(beta_deg), numpy.deg2rad(gamma_deg)) 
        antennaFrameCoefficients(R, ray_vector, pre_inv = False)
        R_inv = numpy.linalg.inv(R)
        antennaFrameCoefficients(R_inv, ray_vector, pre_inv = True)
    
    
    if plot_test1 == True:
        alpha_deg = 20.0
        beta_deg = 40.0
        gamma_deg = 15.0
        
        basis_x = numpy.array([1,0,0])
        basis_y = numpy.array([0,1,0])
        basis_z = numpy.array([0,0,1])
        
        ray_phi_deg = 35.0
        ray_theta_deg = 90.0 #horizontal at 90
        
        ray_x = numpy.sin(numpy.deg2rad(ray_theta_deg)) * numpy.cos(numpy.deg2rad(ray_phi_deg))
        ray_y = numpy.sin(numpy.deg2rad(ray_theta_deg)) * numpy.sin(numpy.deg2rad(ray_phi_deg))
        ray_z = numpy.cos(numpy.deg2rad(ray_theta_deg))
        
        ray_vector = numpy.array([ray_x,ray_y,ray_z])
        #Need to get ray_X, ray_Y, ray_Z and then get ray_THETA to check it against the beam pattern. 
        R = eulerRotationMatrix(numpy.deg2rad(alpha_deg), numpy.deg2rad(beta_deg), numpy.deg2rad(gamma_deg))
        basis_X = R[:,0] #x basis vector of the antenna frame in the ice basis
        basis_Y = R[:,1] #y basis vector of the antenna frame in the ice basis
        basis_Z = R[:,2] #z basis vector of the antenna frame in the ice basis
        
        #Below are the coefficients of the vector in the antenna frame.  
        #I.e. if the vector u = a x + b y + c z in the ice frame, this returns A,B,C 
        #corresponding to the same vector u but written in the antenna basis u = A X + B Y + C Z = a x + b y + c z  
        antenna_frame_coefficients = antennaFrameCoefficients(R, ray_vector)  
        
        ray_VECTOR = basis_X*antenna_frame_coefficients[0] + basis_Y*antenna_frame_coefficients[1] + basis_Z*antenna_frame_coefficients[2] #This should be identical to ray_vector, but was constructed using a different basis. 
        
        fig = pylab.figure(figsize=(11.2,11.2)) #square my screensize
        gs_upper = gridspec.GridSpec(2, 2) #should only call left plots.  pylab.subplot(gs_left[0]),pylab.subplot(gs_left[2]),...
        gs_lower = gridspec.GridSpec(2, 3,width_ratios=[0.25,0.5,0.25])
        
        #Ice Frame
        ax = fig.add_subplot(gs_upper[0],projection='3d')
        ax.quiver(0,0,0,1,0,0,label='x',color='r')
        ax.quiver(0,0,0,0,1,0,label='y',color='g')
        ax.quiver(0,0,0,0,0,1,label='z',color='b')
        ax.quiver(ray_vector[0],ray_vector[1],ray_vector[2],-ray_vector[0],-ray_vector[1],-ray_vector[2],color='k',label='Ray')
        pylab.legend(fancybox=True, framealpha=0.5)
        pylab.title('Ice Frame')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_xlabel('Ice x',fontsize=16)
        ax.set_ylabel('Ice y',fontsize=16)
        ax.set_zlabel('Ice z',fontsize=16)
        #Antenna Frame
        ax = fig.add_subplot(gs_upper[1],projection='3d')
        ax.quiver(0,0,0,1,0,0,label='X',color='r',linestyle='--')
        ax.quiver(0,0,0,0,1,0,label='Y',color='g',linestyle='--')
        ax.quiver(0,0,0,0,0,1,label='Z',color='b',linestyle='--')
        ax.quiver(antenna_frame_coefficients[0],antenna_frame_coefficients[1],antenna_frame_coefficients[2],-antenna_frame_coefficients[0],-antenna_frame_coefficients[1],-antenna_frame_coefficients[2],color='y',linestyle = '--',label='Ray')
        pylab.legend(fancybox=True, framealpha=0.5)
        pylab.title('Antenna Frame')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_xlabel('Antenna x',fontsize=16)
        ax.set_ylabel('Antenna y',fontsize=16)
        ax.set_zlabel('Antenna z',fontsize=16)
        #Overlap (Ice Frame)
        ax = fig.add_subplot(gs_lower[4],projection='3d')
        ax.quiver(0,0,0,1,0,0,label='x',color='r')
        ax.quiver(0,0,0,0,1,0,label='y',color='g')
        ax.quiver(0,0,0,0,0,1,label='z',color='b')
        ax.quiver(ray_vector[0],ray_vector[1],ray_vector[2],-ray_vector[0],-ray_vector[1],-ray_vector[2],color='k',label='Ray')
        
        ax.quiver(0,0,0,basis_X[0],basis_X[1],basis_X[2],label='X',color='r',linestyle='--')
        ax.quiver(0,0,0,basis_Y[0],basis_Y[1],basis_Y[2],label='Y',color='g',linestyle='--')
        ax.quiver(0,0,0,basis_Z[0],basis_Z[1],basis_Z[2],label='Z',color='b',linestyle='--')
        ax.quiver(ray_VECTOR[0],ray_VECTOR[1],ray_VECTOR[2],-ray_VECTOR[0],-ray_VECTOR[1],-ray_VECTOR[2],color='y',linestyle = '--',label='Ray')
        
        pylab.legend(fancybox=True, framealpha=0.5)
        pylab.title('Ice Frame')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_xlabel('Ice x',fontsize=16)
        ax.set_ylabel('Ice y',fontsize=16)
        ax.set_zlabel('Ice z',fontsize=16)
        fig.suptitle('Euler Angles: $\\alpha = $%0.2g$^\circ$, $\\beta = $%0.2g$^\circ$, $\gamma = $%0.2g$^\circ$'%(alpha_deg,beta_deg,gamma_deg))
        pylab.subplots_adjust(left = 0.00, bottom = 0.05, right = 0.97, top = 0.95, wspace = 0.00, hspace = 0.06)
        
    if plot_test2 == True:
        test_config = yaml.load(open('/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/test_dipole_octo_-200_polar_120_rays.py'))
        plotArrayFromConfig(test_config)
        plotArrayFromConfig(test_config,only_station=0)
############################################################
