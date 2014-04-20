"""
Detector object
"""

import numpy
import scipy.interpolate

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
    
    def __init__(self, x, y, z, lib, frequency_low, frequency_high):
        """
        x, y, z given relative to station center
        """

        self.x = x
        self.y = y
        self.z = z
        
        self.lib = lib

        self.frequency_low = frequency_low
        self.frequency_high = frequency_high

        # Bandwidth
        # Gain
        # etc.

    def electricField(self, frequency, electric_field, n_steps=100):
        f = scipy.interpolate.interp1d(frequency, electric_field, bounds_error=False, fill_value=0.)
        delta_frequency = (self.frequency_high - self.frequency_low) / n_steps
        frequency_array = numpy.linspace(self.frequency_low, self.frequency_high, n_steps)
        return numpy.sum(f(frequency_array)) * delta_frequency # V m^-1

############################################################
