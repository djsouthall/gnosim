"""
Nucleon interaction cross sections for high-energy neutrinos from 10^4 GeV to 10^12 GeV. 
Both charged current (CC) and neutral current (NC) cross sections for neutrinos and anti-neutrinos are implemented.

Source: Connolly et al. 2011, arXiv:1102.0691
"""

import numpy
import pylab

import gnosim.utils.constants

pylab.ion()

############################################################

#mode_dict = {'upper': {'nc': [-1.456, 32.23, -32.32, 5.881, -49.41],
#                       'cc': [-1.456, 33.47, -33.02, 6.026, -142.8],
#                       'nc_anti': [-2.945, 143.2, -76.70, 11.75, -142.8],
#                       'cc_anti': [-2.945, 144.5, -77.44, 11.90, -142.8]},
#             'lower': {'nc': [-15.35, 16.16, 37.71, -8.801, -253.1],
#                       'cc': [-15.35, 13.86, 39.84, -9.205, -253.1],
#                       'nc_anti': [-13.08, 15.17, 31.19, -7.757, -216.1],
#                       'cc_anti': [-13.08, 12.48, 33.52, -8.191, -216.1]}}


############################################################

def inelasticityArray(energy_neutrino, mode):
    """
    energy_neutrino = neutrino energy (GeV) (an array)
    mode = specifies the neutrino type (nu or anti-nu) and interaction type (charged current or neutral current)
    
    Returns:
    inelasticity
    """
    
    epsilon = numpy.log10(energy_neutrino) # Neutrino energy in GeV

    r_1 = numpy.random.random(numpy.size(energy_neutrino))
    r_2 = numpy.random.random(numpy.size(energy_neutrino))

    f_0 = 0.128
    f_1 = -0.197
    f_2 = 21.8
    f = f_0 * numpy.sin(f_1 * (epsilon - f_2))

    '''
    if numpy.size(t_ns) == 1:
        if t_ns > 0:
            return (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns)/0.057) + (1. + 2.87*numpy.fabs(t_ns))**(-3.0))
        else:
            return (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns)/0.030) + (1. + 3.05*numpy.fabs(t_ns))**(-3.5)) 
    else:
        ra = numpy.zeros_like(t_ns)
        ra[t_ns > 0] = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns > 0])/0.057) + (1. + 2.87*numpy.fabs(t_ns[t_ns > 0]))**(-3.0))
        ra[t_ns <= 0] = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns <= 0])/0.030) + (1. + 3.05*numpy.fabs(t_ns[t_ns <= 0]))**(-3.5)) 
        return ra
    '''
    if numpy.size(f) == 1:
        if r_1 < f:
            # Low-inelasticity region
            a_0 = 0.
            a_1 = 0.0941
            a_2 = 4.72
            a_3 = 0.456

            y_min = 0.
            y_max = 1.e-3

            c_1 = a_0 - a_1 * numpy.exp(-1. * (epsilon - a_2) / a_3)
            b_0 = 2.55
            b_1 = -0.0949
            c_2 = b_0 + b_1 * epsilon

            #y_0 = c_1 + (r_2 * (y_max - c_1)**(-1. / (c_2 + 1.)) + (1. - r_2) * (y_min - c_1)**(-1. / (c_2 + 1.)))**(c_2 / (c_2 - 1.))
            y_0 = c_1 + (r_2 * (y_max - c_1)**((-1. / c_2) + 1.) + (1. - r_2) * (y_min - c_1)**((-1. / c_2) + 1.))**(c_2 / (c_2 - 1.))

        else:
            # High-inelasticity region
            mode_dict = {'cc_anti': [-0.0026, 0.085, 4.1, 1.7],                                                                                        
                         'cc': [-0.008, 0.26, 3.0, 1.7],                                                                                        
                         'nc_anti': [-0.005, 0.23, 3.0, 1.7],                                                                                   
                         'nc': [-0.005, 0.23, 3.0, 1.7]}
            a_0, a_1, a_2, a_3 = mode_dict[mode]

            y_min = 1.e-3
            y_max = 1.

            c_1 = a_0 - a_1 * numpy.exp(-1. * (epsilon - a_2) / a_3)

            y_0 = ((y_max - c_1)**r_2 / (y_min - c_1)**(r_2 - 1.)) + c_1

        return y_0
    else:
        y_0 = numpy.zeros_like(f)
        '''
        ra = numpy.zeros_like(t_ns)
        ra[t_ns > 0] = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns > 0])/0.057) + (1. + 2.87*numpy.fabs(t_ns[t_ns > 0]))**(-3.0))
        ra[t_ns <= 0] = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns <= 0])/0.030) + (1. + 3.05*numpy.fabs(t_ns[t_ns <= 0]))**(-3.5)) 
        return ra
        '''
        r_1_less_cut = numpy.greater(f,r_1)
        #y_0[r_1_less_cut]
        #if r_1 < f:
        # Low-inelasticity region
        a_0 = 0.
        a_1 = 0.0941
        a_2 = 4.72
        a_3 = 0.456

        y_min = 0.
        y_max = 1.e-3

        c_1 = a_0 - a_1 * numpy.exp(-1. * (epsilon[r_1_less_cut] - a_2) / a_3)
        b_0 = 2.55
        b_1 = -0.0949
        c_2 = b_0 + b_1 * epsilon[r_1_less_cut]

        #y_0 = c_1 + (r_2 * (y_max - c_1)**(-1. / (c_2 + 1.)) + (1. - r_2) * (y_min - c_1)**(-1. / (c_2 + 1.)))**(c_2 / (c_2 - 1.))
        y_0[r_1_less_cut] = c_1 + (r_2[r_1_less_cut] * (y_max - c_1)**((-1. / c_2) + 1.) + (1. - r_2[r_1_less_cut]) * (y_min - c_1)**((-1. / c_2) + 1.))**(c_2 / (c_2 - 1.))

        #else:
        # High-inelasticity region
        mode_dict = {'cc_anti': [-0.0026, 0.085, 4.1, 1.7],                                                                                        
                     'cc': [-0.008, 0.26, 3.0, 1.7],                                                                                        
                     'nc_anti': [-0.005, 0.23, 3.0, 1.7],                                                                                   
                     'nc': [-0.005, 0.23, 3.0, 1.7]}
        a_0, a_1, a_2, a_3 = mode_dict[mode]

        y_min = 1.e-3
        y_max = 1.

        c_1 = a_0 - a_1 * numpy.exp(-1. * (epsilon[~r_1_less_cut] - a_2) / a_3)

        y_0[~r_1_less_cut] = ((y_max - c_1)**r_2[~r_1_less_cut] / (y_min - c_1)**(r_2[~r_1_less_cut] - 1.)) + c_1

        return y_0
def inelasticity(energy_neutrino, mode):
    """
    energy_neutrino = neutrino energy (GeV)
    mode = specifies the neutrino type (nu or anti-nu) and interaction type (charged current or neutral current)
    
    Returns:
    inelasticity
    """
    
    epsilon = numpy.log10(energy_neutrino) # Neutrino energy in GeV

    r_1, r_2 = numpy.random.random(2)

    f_0 = 0.128
    f_1 = -0.197
    f_2 = 21.8
    f = f_0 * numpy.sin(f_1 * (epsilon - f_2))

    if r_1 < f:
        # Low-inelasticity region
        a_0 = 0.
        a_1 = 0.0941
        a_2 = 4.72
        a_3 = 0.456

        y_min = 0.
        y_max = 1.e-3

        c_1 = a_0 - a_1 * numpy.exp(-1. * (epsilon - a_2) / a_3)
        b_0 = 2.55
        b_1 = -0.0949
        c_2 = b_0 + b_1 * epsilon

        #y_0 = c_1 + (r_2 * (y_max - c_1)**(-1. / (c_2 + 1.)) + (1. - r_2) * (y_min - c_1)**(-1. / (c_2 + 1.)))**(c_2 / (c_2 - 1.))
        y_0 = c_1 + (r_2 * (y_max - c_1)**((-1. / c_2) + 1.) + (1. - r_2) * (y_min - c_1)**((-1. / c_2) + 1.))**(c_2 / (c_2 - 1.))

    else:
        # High-inelasticity region
        mode_dict = {'cc_anti': [-0.0026, 0.085, 4.1, 1.7],                                                                                        
                     'cc': [-0.008, 0.26, 3.0, 1.7],                                                                                        
                     'nc_anti': [-0.005, 0.23, 3.0, 1.7],                                                                                   
                     'nc': [-0.005, 0.23, 3.0, 1.7]}
        a_0, a_1, a_2, a_3 = mode_dict[mode]

        y_min = 1.e-3
        y_max = 1.

        c_1 = a_0 - a_1 * numpy.exp(-1. * (epsilon - a_2) / a_3)

        y_0 = ((y_max - c_1)**r_2 / (y_min - c_1)**(r_2 - 1.)) + c_1

    return y_0

############################################################

if __name__ == "__main__":
    
    E = numpy.ones(100)*3e9
    y_0 = inelasticityArray(E, mode='cc')
    print(y_0)
    '''
    energy_neutrino = 1.e6 # GeV
    #mode = 'cc'
    mode_array = ['cc', 'nc', 'cc_anti', 'nc_anti']
    n_trials = 100000

    y_array = {}
    for mode in mode_array:
        y_array[mode] = []
        for ii in range(0, n_trials):
            y_array[mode].append(inelasticity(energy_neutrino, mode))

    pylab.figure()
    for mode in mode_array:
        pylab.hist(y_array[mode], bins=40, normed=True, histtype='step', label=mode)
    #pylab.title(r'Energy Neutrino = %.e GeV, CC $\nu$N'%(energy_neutrino))
    pylab.title(r'Energy Neutrino = %.e GeV'%(energy_neutrino))
    pylab.xlabel('Inelasticity')
    pylab.ylabel('PDF')
    pylab.legend(loc='upper right')

    #print y_array
    '''
############################################################
