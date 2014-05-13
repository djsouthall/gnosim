import numpy
import h5py
import pylab

import gnosim.utils.constants
import gnosim.utils.bayesian_efficiency
import gnosim.earth.earth
import gnosim.earth.greenland

pylab.ion()

############################################################

def getAcceptance(infile, cos_theta_bins, electric_field_threshold = 1.e-4, earth=True):
    reader = h5py.File(infile, 'r')

    energy_neutrino = reader['energy_neutrino'][0] # GeV

    interaction_length, interaction_length_anti \
        = gnosim.earth.earth.interactionLength(gnosim.utils.constants.density_water * gnosim.utils.constants.mass_proton,
                                               energy_neutrino)
    interaction_length = numpy.sqrt(interaction_length * interaction_length_anti) # m

    n_bins = len(cos_theta_bins) - 1

    volumetric_acceptance = numpy.zeros(n_bins)
    volumetric_acceptance_error_low = numpy.zeros(n_bins)
    volumetric_acceptance_error_high = numpy.zeros(n_bins)
    acceptance = numpy.zeros(n_bins)
    acceptance_error_low = numpy.zeros(n_bins)
    acceptance_error_high = numpy.zeros(n_bins)
    n_volume = numpy.zeros(n_bins)

    # THIS INFORMATION SHOULD BE STORED AS AN ATTRIBUTE IN THE RESULTS FILE!!!!
    if 'ross' in infile:
        ice_model = 'test'
    else:
        ice_model = 'empirical'
    print 'ICE_MODEL = %s'%(ice_model)

    for ii in range(0, n_bins):

        theta_min = numpy.degrees(numpy.arccos(cos_theta_bins[ii]))
        theta_max = numpy.degrees(numpy.arccos(cos_theta_bins[ii + 1]))

        if theta_min > theta_max:
            theta_temp = theta_max
            theta_max = theta_min
            theta_min = theta_temp

        print '  Theta %.3f -- %.3f deg'%(theta_min, theta_max)

        n_total = numpy.sum(numpy.all([reader['theta_0'][...] >= theta_min,
                                       reader['theta_0'][...] <= theta_max],
                                      axis=0))
        
        n_pass = numpy.sum(numpy.all([reader['theta_0'][...] >= theta_min,
                                      reader['theta_0'][...] <= theta_max,
                                      reader['electric_field'][...] > electric_field_threshold],
                                     axis=0))

        #print n_total, n_pass, float(n_pass) / n_total, reader.attrs['geometric_factor']

        #n_volume = numpy.sum(reader['p_earth'][...] \
        #                     * reader['p_detect'][...] \
        #                     * (reader['theta_0'][...] >= theta_min) \
        #                     * (reader['theta_0'][...] <= theta_max))

        efficiency, (efficiency_low, efficiency_high) = gnosim.utils.bayesian_efficiency.confidenceInterval(n_total, n_pass)

        if earth:
            # Including effect of Earth attenuation
            volumetric_acceptance[ii] = numpy.sum(reader['p_earth'][...] \
                                                  * (reader['theta_0'][...] >= theta_min) \
                                                  * (reader['theta_0'][...] <= theta_max) \
                                                  * (gnosim.earth.greenland.density(reader['z_0'][...], ice_model=ice_model) / gnosim.utils.constants.density_water) \
                                                  * (reader['electric_field'][...] > electric_field_threshold)) \
                * (reader.attrs['geometric_factor'] / (n_total * n_bins)) * gnosim.utils.constants.km_to_m**-3 # km^3 sr
        else:
            # Ignore Effect of Earth attenuation
            volumetric_acceptance[ii] = numpy.sum((reader['theta_0'][...] >= theta_min) \
                                                      * (reader['theta_0'][...] <= theta_max) \
                                                      * (gnosim.earth.greenland.density(reader['z_0'][...], ice_model=ice_model) / gnosim.utils.constants.density_water) \
                                                      * (reader['electric_field'][...] > electric_field_threshold)) \
                    * (reader.attrs['geometric_factor'] / (n_total * n_bins)) * gnosim.utils.constants.km_to_m**-3 # km^3 sr

        volumetric_acceptance_error_low[ii] = volumetric_acceptance[ii] * ((efficiency - efficiency_low) / efficiency)
        volumetric_acceptance_error_high[ii] = volumetric_acceptance[ii] * ((efficiency_high - efficiency) / efficiency)
    
        acceptance[ii] = volumetric_acceptance[ii] * gnosim.utils.constants.km_to_m**3 / interaction_length # m^2 sr 
        acceptance_error_low[ii] = volumetric_acceptance_error_low[ii] * gnosim.utils.constants.km_to_m**3 / interaction_length
        acceptance_error_high[ii] = volumetric_acceptance_error_high[ii] * gnosim.utils.constants.km_to_m**3 / interaction_length

    reader.close()
    
    return volumetric_acceptance, volumetric_acceptance_error_low, volumetric_acceptance_error_high, \
        acceptance, acceptance_error_low, acceptance_error_high

############################################################

dic_infiles = {'2': 'results_empirical_trials/config_simple_-2_1.00e+09_GeV_1000000_events.h5',
               '30': 'results_empirical_trials/config_simple_-30_1.00e+09_GeV_1000000_events.h5',
               '100': 'results_empirical_trials/config_simple_-100_1.00e+09_GeV_1000000_events.h5',
               'Ross': 'results_ross_trials/config_simple_-2_1.00e+09_GeV_1000000_events.h5'}

cos_theta_bins = numpy.linspace(-1., 1., 21) # Must be equally spaced
cos_theta_centers = 0.5 * (cos_theta_bins[0: -1] + cos_theta_bins[1:])

dic_data = {}
for key in dic_infiles.keys():
    print key

    volumetric_acceptance, volumetric_acceptance_error_low, volumetric_acceptance_error_high, \
        acceptance, acceptance_error_low, acceptance_error_high = getAcceptance(dic_infiles[key], cos_theta_bins)

    dic_data[key] = {}
    dic_data[key]['acceptance'] = acceptance
    dic_data[key]['acceptance_error_low'] = acceptance_error_low
    dic_data[key]['acceptance_error_high'] = acceptance_error_high

    #break

pylab.figure()
#pylab.xscale('log')
#pylab.yscale('log')
xerr = 2. * numpy.ones(len(cos_theta_centers)) / (2. * len(cos_theta_centers))
#for key in dic_data.keys():
for key in ['2', '30', '100', 'Ross']:
#for key in ['2', '30', '100']:
    if key in ['Ross']:
        label = 'Ross'
    else:
        label = 'Depth = %i m'%(int(key))
    pylab.errorbar(cos_theta_centers, dic_data[key]['acceptance'], 
                   xerr=[xerr, xerr],
                   yerr=[dic_data[key]['acceptance_error_low'], dic_data[key]['acceptance_error_high']], 
                   #linestyle='none',
                   label=label)

    print key, numpy.sum(dic_data[key]['acceptance'])

pylab.xlabel('Cos(Theta)')
pylab.ylabel(r'A$\Omega$ (m$^2$ sr)')
pylab.title(r'E$_{\nu}$ = %.2e GeV'%(1.e9))
pylab.legend(loc='upper left')
pylab.xlim([-1., 1])
#pylab.ylim([1.e-1, 1.e6])
