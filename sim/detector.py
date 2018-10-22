"""
Detector object
"""

import numpy
import scipy.interpolate
import sys

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
