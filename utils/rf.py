'''
Radio frequency utils.
'''

import numpy

import gnosim.utils.constants

############################################################

def decibel(amplitude):
    '''
    Converts the amplitude of voltage to decibels.  Note that you should be careful using this,
    as definitions of decibels are complicated.

    Paramters
    ---------
    amplitude : float
        The amplitude of the signal.  Given in Voltage.
    
    Returns
    -------
    decibel : float
        The amplitude of the signal.  Given in decibels.
    '''
    power = amplitude**2
    return 10. * numpy.log10(power)

############################################################

def amplitude(decibel):
    '''
    Converts the decibels to amplitude of voltage.

    Paramters
    ---------
    decibel : float
        The amplitude of the signal.  Given in decibels.
    
    Returns
    -------
    amplitude : float
        The amplitude of the signal.  Given in Voltage.
    '''
    power = 10.**(0.1 * decibel)
    return numpy.sqrt(power)

############################################################

def thermalNoise(resistance, temp, bandwidth):
    '''
    Calculates the thermal noise level for a given resistance, temperature, and bandwidth.
    Note that this does not contain the factor of 4 under the square root that is present in many sources.
    I cannot say I entirely understand why, but at some point it was impressed upon me that it should not be there
    for out applications.  In our case, the system response is scaled to achieve a particular noise level that
    matches the observed noise anyways, so this does not matter.

    Parameters
    ----------
    resistance : float
        The resistance, given in Ohms.
    temp : float
        The temperature.  Given in K.
    Bandwidth : float
        The bandwidth for which the noise should be calculated.

    Returns
    -------
    noise : float
        The rms of the thermal noise.  Given in V.
    '''
    #noise = numpy.sqrt(4. * gnosim.utils.constants.boltzmann * temp * resistance * bandwidth * gnosim.utils.constants.GHz_to_Hz)
    noise = numpy.sqrt(gnosim.utils.constants.boltzmann * temp * resistance * bandwidth * gnosim.utils.constants.GHz_to_Hz)
    return noise

############################################################
