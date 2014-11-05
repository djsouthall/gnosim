import numpy
import pylab

import gnosim.sim.acceptance
import gnosim.sim.fold_spectrum

#pylab.ion()

############################################################

energy_neutrino_array = 10**numpy.arange(7., 12.1, 0.5) # GeV
dic_configurations = {'-100': r'-100 m, 3 km, 37 $\times$ 3 yr',
                      '-100 x4': r'-100 m, 3 km, 37 $\times$ 3 yr, x4',
                      '-100 x20': r'-100 m, 3 km, 37 $\times$ 3 yr, x20'}
                      #'-30': r'-30 m, 3 km',
                      #'-2': r'-2 m, 3 km',
                      #'0': r'0 m, Ross, 961 $\times$ 3 yr',
                      #'1000': r'1000 m, Ross, 1 $\times$ 3 yr',
                      #'5000': r'5000 m, Ross',
                      #'6000': r'6000 m, 1.5 km, 1 $\times$ 0.3 yr',
                      #'6000_ross': r'6000 m, Ross, 1 $\times$ 0.3 yr',
                      #'38000': r'38000 m, 1.5 km, 1 $\times$ 0.1 yr'}

dic_adjust = {'-100': 37.,
              '-100 x4': 37.,
              '-100 x20': 37.,
              '-30': 37.,
              '-2': 37.,
              '0': 961.,
              '1000': 1.,
              #'5000': 1.,
              '6000': 1.,
              '6000_ross': 1.,
              '38000': 1.}

dic_time = {'-100': 3.,
            '-100 x4': 3.,
            '-100 x20': 3.,
            '-30': 3.,
            '-2': 3.,
            '0': 3.,
            '1000': 3.,
            #'5000': 3.,                          
            '6000': 0.3,
            '6000_ross': 0.3,
            '38000': 0.1}

dic_threshold = {'-100': 1.e-4,
                 '-100 x4': 1.e-4 / 4.,
                 '-100 x20': 1.e-4 / 20.}

#order = ['-100', '-30', '-2', '0', '1000', '5000', '38000']
#order = ['-100', '-30', '-2', '0', '38000']
#order = ['-100', '0', '1000', '6000', '6000_ross', '38000']
order = ['-100', '-100 x4', '-100 x20']

##### FIRST BY ENERGY #####

dic_data = {}

for key in dic_configurations:
#for key in ['-2']:

    print key
    dic_data[key] = {'volumetric_acceptance': numpy.zeros(len(energy_neutrino_array)), 
                     'volumetric_acceptance_error_low': numpy.zeros(len(energy_neutrino_array)), 
                     'volumetric_acceptance_error_high': numpy.zeros(len(energy_neutrino_array)), 
                     'acceptance': numpy.zeros(len(energy_neutrino_array)), 
                     'acceptance_error_low': numpy.zeros(len(energy_neutrino_array)), 
                     'acceptance_error_high': numpy.zeros(len(energy_neutrino_array))}

    for ii, energy_neutrino in enumerate(energy_neutrino_array):
        if key in ['0', '1000', '5000']:
            #infile = 'results_2014_may_29/config_simple_%i_%.2e_GeV_100000_events_0.h5'%(int(key), energy_neutrino)
            infile = 'results_2014_may_29/config_simple_%i_%.2e_GeV_1000000_events.h5'%(int(key), energy_neutrino)
        if key in ['-2', '-30', '-100', '-100 x4', '-100 x20', '38000']: 
            infile = 'results_2014_jun_5/config_simple_%i_%.2e_GeV_1000000_events.h5'%(int(key.split()[0]), energy_neutrino)
        if key in ['6000']:
            infile = 'results_2014_aug_7/config_simple_%i_%.2e_GeV_1000000_events.h5'%(int(key), energy_neutrino)
        if key in ['6000_ross']:
            infile = 'results_2014_aug_7/config_simple_%s_%.2e_GeV_1000000_events.h5'%(key, energy_neutrino)

        dic_data[key]['volumetric_acceptance'][ii], \
            dic_data[key]['volumetric_acceptance_error_low'][ii], \
            dic_data[key]['volumetric_acceptance_error_high'][ii], \
            dic_data[key]['acceptance'][ii], \
            dic_data[key]['acceptance_error_low'][ii], \
            dic_data[key]['acceptance_error_high'][ii] = gnosim.sim.acceptance.acceptance(infile, 
                                                                                          electric_field_threshold=dic_threshold[key], 
                                                                                          mode_reflections='all') # all, direct, reflect


pylab.figure()
pylab.xscale('log')
pylab.yscale('log')
for key in order:
#for key in ['-2']:
    #pylab.errorbar(energy_neutrino_array, dic_data[key]['acceptance'] / energy_neutrino_array, 
    #               yerr=[dic_data[key]['acceptance_error_low'] / energy_neutrino_array, dic_data[key]['acceptance_error_high'] / energy_neutrino_array], 
    #               label=dic_configurations[key])
    print key
    print energy_neutrino_array
    print dic_data[key]['acceptance']
    pylab.errorbar(energy_neutrino_array, dic_adjust[key] * dic_data[key]['acceptance'],
                   yerr=[dic_adjust[key] * dic_data[key]['acceptance_error_low'], dic_adjust[key] * dic_data[key]['acceptance_error_high']],    
                   label=dic_configurations[key])
    
    #pylab.errorbar(energy_neutrino_array, dic_data[key]['volumetric_acceptance'],
    #               yerr=[dic_data[key]['volumetric_acceptance_error_low'], dic_data[key]['volumetric_acceptance_error_high']],    
    #               label=dic_configurations[key])
    
    #pylab.errorbar(energy_neutrino_array, dic_adjust[key] * dic_time[key] * dic_data[key]['volumetric_acceptance'],
    #               yerr=[dic_adjust[key] * dic_time[key] * dic_data[key]['volumetric_acceptance_error_low'], 
    #                     dic_adjust[key] * dic_time[key] * dic_data[key]['volumetric_acceptance_error_high']],    
    #               label=dic_configurations[key])

pylab.xlabel('Energy Neutrino (GeV)')
pylab.ylabel(r'A$\Omega$ (m$^2$ sr)')
#pylab.ylabel(r'V$\Omega$ (km$^3$ sr)')
#pylab.ylabel(r'V$\Omega$ $\times$ Livetime (km$^3$ sr yr)')
pylab.legend(loc='upper left', fontsize=10, frameon=False, title='Elevation, Ice Model, Livetime Factor')
#pylab.xlim([10**6.5, 10**12])
#pylab.ylim([1.e0, 1.e6])

pylab.savefig('/home/bechtol/public_html/temp/fig1.pdf')

##### NOW BY ZENTIH ANGLE #####
"""
cos_theta_bins = numpy.linspace(-1., 1., 21) # Must be equally spaced
cos_theta_centers = 0.5 * (cos_theta_bins[0: -1] + cos_theta_bins[1:])
energy_neutrino = 1.e9

dic_data = {}

for key in dic_configurations:

    print key
    dic_data[key] = {'volumetric_acceptance': [],
                     'volumetric_acceptance_error_low': [],
                     'volumetric_acceptance_error_high': [],
                     'acceptance': [],
                     'acceptance_error_low': [],
                     'acceptance_error_high': []}

    if key in ['0', '1000', '5000']:
        infile = 'results_2014_may_29/config_simple_%i_%.2e_GeV_1000000_events.h5'%(int(key), energy_neutrino)
    if key in ['-2', '-30', '-100', '38000']:
        infile = 'results_2014_jun_5/config_simple_%i_%.2e_GeV_1000000_events.h5'%(int(key), energy_neutrino)

    dic_data[key]['volumetric_acceptance'], \
        dic_data[key]['volumetric_acceptance_error_low'], \
        dic_data[key]['volumetric_acceptance_error_high'], \
        dic_data[key]['acceptance'], \
        dic_data[key]['acceptance_error_low'], \
        dic_data[key]['acceptance_error_high'] = gnosim.sim.acceptance.acceptance(infile,
                                                                                  cos_theta_bins=cos_theta_bins,
                                                                                  electric_field_threshold=1.e-4,
                                                                                  mode_reflections='all') # all, direct, reflect

pylab.figure()
#pylab.yscale('log')
for key in order:
    print key
    #print dic_data[key]['acceptance']
    pylab.errorbar(cos_theta_centers, dic_adjust[key] * dic_data[key]['volumetric_acceptance'] / (2. / len(cos_theta_centers)),
                   xerr=[(1. / len(cos_theta_centers)) * numpy.ones(len(cos_theta_centers)),
                         (1. / len(cos_theta_centers)) * numpy.ones(len(cos_theta_centers))],
                   yerr=[dic_adjust[key] * dic_data[key]['volumetric_acceptance_error_low'] / (2. / len(cos_theta_centers)),
                         dic_adjust[key] * dic_data[key]['volumetric_acceptance_error_high'] / (2. / len(cos_theta_centers))],
                   label=dic_configurations[key])

pylab.xlabel('Cos(Zenith Angle)')  
#pylab.ylabel(r'$\frac{d({\rm V}\Omega)}{d\cos({\rm Zenith Angle})}$ $\times$ Livetime (km$^3$ sr yr)')
pylab.ylabel(r'V$\Omega$ $\times$ Livetime (km$^3$ sr yr)')
pylab.legend(loc='upper left')
#pylab.xlim([10**6.5, 10**12])
#pylab.ylim([1.e-1, 1.e6])
"""
##### NOW BY NUMBER OF EVENTS #####
"""
dic_model = {'kotera_2010_optimistic': 'Kotera et al. 2010 Optimistic',
             'kotera_2010_pessimistic': 'Kotera et al. 2010 Pessimistic'}

for key_model in dic_model.keys():
    pylab.figure()
    pylab.xscale('log')
    pylab.yscale('log')
    for key in order:
        energy_plot, rate_threshold_plot, rate_plot = gnosim.sim.fold_spectrum.foldSpectrum(energy_neutrino_array, dic_adjust[key] * dic_time[key] * dic_data[key]['acceptance'], model_key=key_model)
        #pylab.plot(energy_plot, rate_threshold_plot, label=dic_configurations[key])
        pylab.plot(energy_plot, rate_plot, label=dic_configurations[key])

    pylab.xlabel('Energy Neutrino (GeV)')
    #pylab.ylabel(r'Rate above Energy Threshold (yr$^{-1}$)')
    #pylab.ylabel('Total above Energy Threshold (Duration of Experiment)')
    pylab.ylabel('dN/dlog(E) (Duration of Experiment)')
    pylab.title(dic_model[key_model])
    pylab.legend(loc='upper right', fontsize=10, frameon=False, title='Elevation, Ice Model, Livetime Factor')
    pylab.xlim([10**7., 10**12])
    pylab.ylim([1.e-1, 1.e3])
"""
##### NOW BY NUMBER OF EVENTS FOR DIFFERENT THRESHOLDS #####

dic_model = {'kotera_2010_optimistic': 'Kotera et al. 2010 Optimistic',
             'kotera_2010_pessimistic': 'Kotera et al. 2010 Pessimistic'}

for key_model in dic_model.keys():
    pylab.figure()
    pylab.xscale('log')
    pylab.yscale('log')
    for key in order:
        energy_plot, rate_threshold_plot, rate_plot = gnosim.sim.fold_spectrum.foldSpectrum(energy_neutrino_array, dic_adjust[key] * dic_time[key] * dic_data[key]['acceptance'], model_key=key_model)
        pylab.plot(energy_plot, rate_threshold_plot, label=dic_configurations[key])                                                                                                                       
        #pylab.plot(energy_plot, rate_plot, label=dic_configurations[key])

    pylab.xlabel('Energy Neutrino (GeV)')
    #pylab.ylabel(r'Rate above Energy Threshold (yr$^{-1}$)')                                                                                                                                              
    pylab.ylabel('Total above Energy Threshold (Duration of Experiment)')
    #pylab.ylabel('dN/dlog(E) (Duration of Experiment)')
    pylab.title(dic_model[key_model])
    pylab.legend(loc='upper right', fontsize=10, frameon=False, title='Elevation, Ice Model, Livetime Factor')
    pylab.xlim([10**7., 10**12])
    pylab.ylim([1.e-1, 1.e3])

    pylab.savefig('/home/bechtol/public_html/temp/%s.pdf'%(key_model))
