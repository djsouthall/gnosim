"""
Radio frequency utils.
"""

import numpy

import gnosim.utils.constants

############################################################

def decibel(amplitude):
    """
    Amplitude of voltage to decibels
    """
    power = amplitude**2
    return 10. * numpy.log10(power)

############################################################

def amplitude(decibel):
    """
    Decibels to amplitude of voltage
    """
    power = 10.**(0.1 * decibel)
    return numpy.sqrt(power)

############################################################

def thermalNoise(resistance, temp, bandwidth):
    """
    Resistance (Ohms)
    Temperature (K)
    Bandwidth (GHz)
    """
    return numpy.sqrt(4. * gnosim.utils.constants.boltzmann * temp * resistance * bandwidth * gnosim.utils.constants.GHz_to_Hz)

############################################################
