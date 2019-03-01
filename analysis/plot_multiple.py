import glob
import numpy
import h5py
import pylab

import gnosim.utils.constants
import gnosim.utils.bayesian_efficiency
import gnosim.earth.ice
import gnosim.earth.earth

pylab.ion()

############################################################

def volumetricAcceptance(infile, electric_field_threshold = 1.e-4):
    reader = h5py.File(infile, 'r')
    ice = gnosim.earth.ice.Ice(reader.attrs['ice_model'],suppress_fun = True)
    efficiency, (efficiency_low, efficiency_high) \
        = gnosim.utils.bayesian_efficiency.confidenceInterval(reader['p_interact'].shape[0], 
                                                              numpy.sum(reader['electric_field'][...] > electric_field_threshold))
    
    volumetric_acceptance = numpy.sum(reader['p_earth'][...] \
                                      * (ice.density(reader['z_0'][...]) / gnosim.utils.constants.density_water)
                                      * (reader['electric_field'][...] > electric_field_threshold)) \
        * (reader.attrs['geometric_factor'] / reader['p_interact'].shape[0]) * gnosim.utils.constants.km_to_m**-3 # km^3 sr

    volumetric_acceptance_error_low = volumetric_acceptance * ((efficiency - efficiency_low) / efficiency)
    volumetric_acceptance_error_high = volumetric_acceptance * ((efficiency_high - efficiency) / efficiency)
    
    return reader['energy_neutrino'][0], reader.attrs['config'], volumetric_acceptance, volumetric_acceptance_error_low, volumetric_acceptance_error_high

############################################################

infiles = glob.glob('results_empirical/*.h5')

label_dic = {'config_simple_-2.py': 'Depth = 2 m',
             'config_simple_-30.py': 'Depth = 30 m',
             'config_simple_-100.py': 'Depth = 100 m'}

dic = {}

for infile in infiles:
    energy_neutrino, config, volumetric_acceptance, volumetric_acceptance_error_low, volumetric_acceptance_error_high \
        = volumetricAcceptance(infile)

    #print infile, energy_neutrino, config, volumetric_acceptance, volumetric_acceptance_error_low, volumetric_acceptance_error_high

    print infile, config

    if config not in dic.keys():
        dic[config] = {'energy_neutrino': [],
                       'volumetric_acceptance': [],
                       'volumetric_acceptance_error_low': [],
                       'volumetric_acceptance_error_high': []}

    dic[config]['energy_neutrino'].append(energy_neutrino)
    dic[config]['volumetric_acceptance'].append(volumetric_acceptance)
    dic[config]['volumetric_acceptance_error_low'].append(volumetric_acceptance_error_low)
    dic[config]['volumetric_acceptance_error_high'].append(volumetric_acceptance_error_high)

for key in dic.keys():
    index = numpy.argsort(dic[key]['energy_neutrino'])
    dic[key]['energy_neutrino'] = numpy.array(dic[key]['energy_neutrino'])[index]
    dic[key]['volumetric_acceptance'] = numpy.array(dic[key]['volumetric_acceptance'])[index] # km^3 sr
    dic[key]['volumetric_acceptance_error_low'] = numpy.array(dic[key]['volumetric_acceptance_error_low'])[index] # km^3 sr
    dic[key]['volumetric_acceptance_error_high'] = numpy.array(dic[key]['volumetric_acceptance_error_high'])[index] # km^3 sr

    interaction_length, interaction_length_anti \
        = gnosim.earth.earth.interactionLength(gnosim.utils.constants.density_water * gnosim.utils.constants.mass_proton,
                                               dic[key]['energy_neutrino'])

    interaction_length = numpy.sqrt(interaction_length * interaction_length_anti)

    dic[key]['acceptance'] = dic[key]['volumetric_acceptance'] * gnosim.utils.constants.km_to_m**3 / interaction_length # m^2 sr
    dic[key]['acceptance_error_low'] = dic[key]['volumetric_acceptance_error_low'] * gnosim.utils.constants.km_to_m**3 / interaction_length # m^2 sr
    dic[key]['acceptance_error_high'] = dic[key]['volumetric_acceptance_error_high'] * gnosim.utils.constants.km_to_m**3 / interaction_length # m^2 sr

# Volumetric Acceptance

pylab.figure()
pylab.xscale('log')
pylab.yscale('log')
for key in dic.keys():
    pylab.errorbar(dic[key]['energy_neutrino'], dic[key]['volumetric_acceptance'], 
                   yerr=[dic[key]['volumetric_acceptance_error_low'], dic[key]['volumetric_acceptance_error_high']], 
                   label=label_dic[key])
pylab.xlabel('Energy Neutrino (GeV)')
pylab.ylabel(r'V$\Omega$ Water Equivalent (km$^3$ sr)')
pylab.legend(loc='lower right')
pylab.xlim([10**6.5, 10**12])
pylab.ylim([1.e-3, 1.e3])

# Acceptance

pylab.figure()
pylab.xscale('log')
pylab.yscale('log')
for key in dic.keys():
    pylab.errorbar(dic[key]['energy_neutrino'], dic[key]['acceptance'], 
                   yerr=[dic[key]['acceptance_error_low'], dic[key]['acceptance_error_high']], 
                   label=label_dic[key])
pylab.xlabel('Energy Neutrino (GeV)')
pylab.ylabel(r'A$\Omega$ (m$^2$ sr)')
pylab.legend(loc='lower right')
pylab.xlim([10**6.5, 10**12])
pylab.ylim([1.e-1, 1.e6])
