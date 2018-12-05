"""
Detector object
"""

import numpy
import scipy.interpolate
import sys
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

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

class Station:

    def __init__(self, x, y, z):
        """
        
        """
        self.x = x
        self.y = y
        self.z = z

        self.antennas = []

############################################################

class Antenna:
    

    def __init__(self, x, y, z, lib, frequency_low, frequency_high, config_file):
        """
        x, y, z given relative to station center
        #When this is called in antarcticsim.py it uses the input x_antenna + x_station, y_antenna + y_station, z_antenna + z_station
        #which does not seem to me to be be a relative position, as it adds the station position?
        """
    
        self.x = x
        self.y = y
        self.z = z
        
        self.lib = lib

        self.frequency_low = frequency_low
        self.frequency_high = frequency_high
        self.config_file = sys.argv[1] #DS:  Shouldn't this be the config_file input? That input is never used...  I know that argv[1] is the config from running in command line from antarctic sim, but it seems weird to call that here and not the input

        #print(self.config_file['antenna_type'][antenna_type])
        # Bandwidth
        # Gain
        # etc.
        
        """
        #For Dipole:
        def electricField(self, frequency, electric_field, theta, n_steps=100): #dipole
            f = scipy.interpolate.interp1d(frequency, electric_field, bounds_error=False, fill_value=0.)
            delta_frequency = (self.frequency_high - self.frequency_low) / n_steps
            frequency_array = numpy.linspace(self.frequency_low, self.frequency_high, n_steps)
            gainfactor = (numpy.cos((numpy.pi/2)*numpy.cos(numpy.deg2rad(theta)))/numpy.sin(numpy.deg2rad(theta)))  #DS: What is this doing?
            return (numpy.sum(f(frequency_array)) * delta_frequency)*gainfactor # V m^-1
        """
        #For Simple:
        
    '''
    def electricField(self, frequency, electric_field, theta_ant, n_steps=100):
        f = scipy.interpolate.interp1d(frequency, electric_field, bounds_error=False, fill_value=0.)
        delta_frequency = (self.frequency_high - self.frequency_low) / n_steps
        frequency_array = numpy.linspace(self.frequency_low, self.frequency_high, n_steps)
        return numpy.sum(f(frequency_array)) * delta_frequency * numpy.sin(numpy.deg2rad(theta_ant)) # V m^-1
    '''
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

############################################################
