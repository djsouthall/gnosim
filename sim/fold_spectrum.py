"""
Fold an acceptance curve with a model spectrum to obtain the expected number of events per year as a function of energy threshold.
"""

import numpy
import scipy.interpolate
import pylab

import gnosim.utils.constants

pylab.ion()

############################################################

model_dict = {'kotera_2010_optimistic': [[100000, 2.1910855798922374e-10],
                                         [152724.26579169, 4.1246263829013647e-10],
                                         [227154.75856012452, 7.381283196139171e-10],
                                         [337859.10882615007, 1.1937766417144407e-9],
                                         [440228.506036475, 1.5768929001678596e-9],
                                         [604796.3972190483, 2.0309176209047387e-9],
                                         [923670.8571873846, 2.751442488979455e-9],
                                         [1790084.6282355362, 3.634457212917857e-9],
                                         [4520353.656360231, 4.230323952886772e-9],
                                         [7088832.467204189, 4.124626382901365e-9],
                                         [13738237.958832638, 3.634457212917857e-9],
                                         [22122162.91070441, 3.0445035410462135e-9],
                                         [34692043.72218362, 2.486591218599803e-9],
                                         [52983169.06283702, 2.751442488979455e-9],
                                         [92367085.71873847, 3.92108803822976e-9],
                                         [188739182.21350917, 6.341584770237437e-9],
                                         [428730033.6654485, 1.0000000000000042e-8],
                                         [809181559.4438767, 1.2879236254336313e-8],
                                         [1410669534.971237, 1.4990778107927092e-8],
                                         [2525217850.0721936, 1.576892900167863e-8],
                                         [4402285060.364741, 1.4990778107927092e-8],
                                         [7474174246.317503, 1.3547779883625934e-8],
                                         [13738237958.832497, 1.1348672281080458e-8],
                                         [24592608589.1482, 9.037409390018221e-9],
                                         [41753189365.60392, 7.017038286703837e-9],
                                         [67233575364.99307, 5.179474679231223e-9],
                                         [108263673387.40474, 3.4551072945922323e-9],
                                         [174332882219.99802, 2.0309176209047387e-9],
                                         [280721620394.117, 1e-9],
                                         [464158883361.2753, 3.5436477855785615e-10],
                                         [604796397219.0459, 1.7448505784296298e-10],
                                         [809181559443.8734, 7.017038286703836e-11],
                                         [973880672847.494, 3.5436477855785615e-11],
                                         [999999999999.9918, 3.3687790490147177e-11]],
              'kotera_2010_pessimistic':[[100000, 1.3547779883625935e-11],
                                         [130299.43385750223, 1.744850578429637e-11],
                                         [183809.44176677923, 2.615666785272682e-11],
                                         [266248.61440398474, 3.92108803822976e-11],
                                         [428730.0336654511, 6.341584770237436e-11],
                                         [654774.7961440256, 9.75014308321881e-11],
                                         [1373823.7958832637, 1.6173023172161422e-10],
                                         [2332470.136161077, 2.1910855798922283e-10],
                                         [4066262.4152903343, 2.8942661247167636e-10],
                                         [9236708.571873847, 3.92108803822976e-10],
                                         [18380944.176677886, 4.338730126091887e-10],
                                         [37558880.90680076, 5.050061921841308e-10],
                                         [62101694.18915603, 6.670767806717186e-10],
                                         [97388067.28474961, 1.000000000000004e-9],
                                         [161026202.75609425, 1.7448505784296298e-9],
                                         [252521785.00721934, 2.7514424889794662e-9],
                                         [396005437.6977427, 4.230323952886772e-9],
                                         [708883246.7204174, 6.183135888417604e-9],
                                         [1111672881.5539234, 7.1968567300115284e-9],
                                         [1610262027.5609293, 7.381283196139172e-9],
                                         [2271547585.601236, 6.841712731578671e-9],
                                         [3657786820.89141, 5.179474679231223e-9],
                                         [5159928433.650846, 3.8231169414637754e-9],
                                         [7674630429.274264, 2.3638851566381917e-9],
                                         [11721022975.334745, 1.2243683313417999e-9],
                                         [15272426579.168938, 7.570435770161489e-10],
                                         [20981623055.342274, 3.823116941463775e-10],
                                         [28825053029.682068, 1.789564074636749e-10],
                                         [38566204211.63457, 7.963406789959582e-11],
                                         [51599284336.50825, 3.1225219107668166e-11],
                                         [69036769328.63962, 1.1639493066120234e-11],
                                         [97388067284.7494, 3.0445035410462136e-12],
                                         [126896100316.79182, 1.000000000000004e-12]]}

############################################################

def readModel(key):
    """
    Linear interpolation in log-space, i.e., power-law interpolation
    """
    energy, e2dnde = zip(*model_dict[key])
    energy = numpy.array(energy)
    differential_intensity = numpy.array(e2dnde) / energy**2
    f = scipy.interpolate.interp1d(numpy.log10(energy), numpy.log10(differential_intensity))
    return f

############################################################

def foldSpectrum(energy_acceptance, acceptance, model_key='kotera_2010_pessimistic', n_steps=10000):
    """
    Energy acceptance [GeV]
    Acceptance [m^2 sr]
    #Energy model [GeV]
    #Intensity model []
    """
    
    #energy_acceptance = 10**numpy.arange(7., 12.5, 0.5)
    #acceptance = 1.e4 * numpy.ones(len(energy_acceptance))
    #acceptance = 10**(4. + 0. * (numpy.log10(energy_acceptance) - 9))

    f_model = readModel(model_key) # GeV, GeV^-1 cm^-2 s^-1 sr^-1

    f_acceptance = scipy.interpolate.interp1d(numpy.log10(energy_acceptance), numpy.log10(acceptance)) # GeV, m^2 sr

    energy_min = max(numpy.min(energy_acceptance), 10**numpy.min(f_model.x))
    energy_max = min(numpy.max(energy_acceptance), 10**numpy.max(f_model.x))
    energy_array = numpy.exp(numpy.linspace(numpy.log(energy_min), numpy.log(energy_max), n_steps)) # GeV
    delta_log_energy = (numpy.log(energy_max) - numpy.log(energy_min)) / n_steps

    differential_intensity_array = 10**f_model(numpy.log10(energy_array)) * gnosim.utils.constants.cm_to_m**(-2) * gnosim.utils.constants.yr_to_s # GeV^-1 m^-2 yr^-1 sr^-1
    acceptance_array = 10**f_acceptance(numpy.log10(energy_array)) # m^2 sr
    rate_array = acceptance_array * differential_intensity_array * energy_array * delta_log_energy # yr^-1
    rate_threshold = numpy.cumsum(rate_array[::-1])[::-1]

    rate = rate_array / numpy.log10(numpy.exp(delta_log_energy)) # yr^-1

    #pylab.figure()
    #pylab.xscale('log')
    #pylab.yscale('log')
    #pylab.plot(energy_array, rate)

    return energy_array, rate_threshold, rate

############################################################
