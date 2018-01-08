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
        """
    
        self.x = x
        self.y = y
        self.z = z
        
        self.lib = lib

        self.frequency_low = frequency_low
        self.frequency_high = frequency_high
        self.config_file = sys.argv[1]

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
            gainfactor = (numpy.cos((numpy.pi/2)*numpy.cos(numpy.deg2rad(theta)))/numpy.sin(numpy.deg2rad(theta)))
            return (numpy.sum(f(frequency_array)) * delta_frequency)*gainfactor # V m^-1
        """
        #For Simple:
        
    
    def electricField(self, frequency, electric_field, theta_ant, n_steps=100):
        f = scipy.interpolate.interp1d(frequency, electric_field, bounds_error=False, fill_value=0.)
        delta_frequency = (self.frequency_high - self.frequency_low) / n_steps
        frequency_array = numpy.linspace(self.frequency_low, self.frequency_high, n_steps)
        return numpy.sum(f(frequency_array)) * delta_frequency * numpy.sin(numpy.deg2rad(theta_ant)) # V m^-1



############################################################
