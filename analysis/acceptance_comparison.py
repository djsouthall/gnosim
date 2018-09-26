import os
import numpy
import pylab
import matplotlib
import collections

import gnosim.sim.acceptance
import gnosim.sim.fold_spectrum

pylab.ion()

############################################################

params = {
    #'backend': 'eps',
    'axes.labelsize': 20,
    #'text.fontsize': 12,           
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    #'xtick.major.size': 3,      # major tick size in points
    #'xtick.minor.size': 1.5,    # minor tick size in points
    #'xtick.major.size': 3,      # major tick size in points
    #'xtick.minor.size': 1.5,    # minor tick size in points
    'text.usetex': True,
    #'figure.figsize': fig_size,
    'font.family':'serif',
    'font.serif':'Computer Modern Roman',
    'font.size': 12
    }
matplotlib.rcParams.update(params)

############################################################

# All-flavor
icecube_acceptance_raw = 10**numpy.array([[5.678513731825525, 0.25369458128078826],
                                          [5.72621501388701, 0.4901477832512322],
                                          [5.83133450583732, 0.9556650246305418],
                                          [5.869557605226927, 1.125615763546798],
                                          [5.975369458128079, 1.3768472906403941],
                                          [6.071679253841808, 1.5689655172413794],
                                          [6.168108422133267, 1.7241379310344827],
                                          [6.3225526632022095, 1.9236453201970445],
                                          [6.622106209761494, 2.211822660098522],
                                          [6.950906037864982, 2.4482758620689653],
                                          [7.550777115481031, 2.7881773399014778],
                                          [8.64406280589223, 3.416256157635468],
                                          [9.13759679126511, 3.6674876847290636],
                                          [9.689312971024297, 3.911330049261083],
                                          [10.28954216637354, 4.14039408866995],
                                          [10.686527610877228, 4.273399014778325],
                                          [11.00611187597985, 4.362068965517241]])
icecube_energy = numpy.array(zip(*icecube_acceptance_raw)[0])
icecube_acceptance = 4. * numpy.pi * numpy.array(zip(*icecube_acceptance_raw)[1]) # m^2 sr
icecube_volumetric_acceptance = gnosim.sim.acceptance.acceptanceToVolumetricAcceptance(icecube_acceptance, icecube_energy) # km^3 sr

# Single flavor
icecube_acceptance_contained_raw = 10**numpy.array([[4.299401197604791, -2.9891107078039925],
                                                    [4.359281437125748, -2.390199637023594],
                                                    [4.44311377245509, -1.8021778584392019],
                                                    [4.6017964071856285, -1.0399274047186937],
                                                    [4.763473053892215, -0.5281306715063527],
                                                    [4.922155688622755, -0.07078039927404722],
                                                    [5.083832335329341, 0.2558983666061705],
                                                    [5.2395209580838324, 0.5063520871143372],
                                                    [5.479041916167665, 0.7676950998185115],
                                                    [5.802395209580839, 0.996370235934664],
                                                    [6.2814371257485035, 1.2577132486388383],
                                                    [7.000000000000001, 1.551724137931034]])
icecube_energy_contained = numpy.array(zip(*icecube_acceptance_contained_raw)[0])
icecube_acceptance_contained = 3. * 4. * numpy.pi * numpy.array(zip(*icecube_acceptance_contained_raw)[1]) # m^2 sr
icecube_volumetric_acceptance_contained = gnosim.sim.acceptance.acceptanceToVolumetricAcceptance(icecube_acceptance_contained, icecube_energy_contained) # km^3 sr

############################################################

save = True

energy_neutrino_array = 10**numpy.arange(5., 12.1, 1.) # GeV
energy_neutrino_array = numpy.concatenate([[10**4.5], energy_neutrino_array])

dic_configurations = {'-100': r'-100 m, 3 km, 37 stations', # $\times$ 3 yr
                      '-100 x4': r'-100 m, 3 km, 37 stations, x4', # $\times$ 3 yr
                      '-100 x20': r'-100 m, 3 km, 37 stations, x20'} # $\times$ 3 yr
                      #'-30': r'-30 m, 3 km',
                      #'-2': r'-2 m, 3 km',
                      #'0': r'0 m, Ross, 961 $\times$ 3 yr',
                      #'1000': r'1000 m, Ross, 1 $\times$ 3 yr',
                      #'5000': r'5000 m, Ross',
                      #'6000': r'6000 m, 1.5 km, 1 $\times$ 0.3 yr',
                      #'6000_ross': r'6000 m, Ross, 1 $\times$ 0.3 yr',
                      #'38000': r'38000 m, 1.5 km, 1 $\times$ 0.1 yr'}
dic_configurations = {'-100': '10 stations, 16 unphased antennas',
                      '-100 x4': '10 stations, 16-antenna phased arrays',
                      '-100 x20': '10 stations, 400-antenna phased arrays'}

dic_adjust = {'-100': 10.,
              '-100 x4': 10.,
              '-100 x20': 10.,
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

dic_threshold = {'-100': 1.46e-4,
                 '-100 x4': 1.46e-4 / 4.,
                 '-100 x20': 1.46e-4 / 20.}

#dic_color = {'-100': 'steelblue',
#             '-100 x4': 'forestgreen',
#             '-100 x20': 'firebrick'}
dic_color = {'-100': 'orange',
             '-100 x4': 'tomato',
             '-100 x20': 'firebrick'}
#dic_color = {'-100': 'deepskyblue',
#             '-100 x4': 'mediumslateblue',
#             '-100 x20': 'purple'}

dic_linestyle = {'-100': '-',
                 '-100 x4': '-', # --
                 '-100 x20': '-'}

dic_linewidth = {'-100': 4,
                 '-100 x4': 4,
                 '-100 x20': 4}

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
            infile = 'results_2014_jun_5/config_simple_%i_%.2e_GeV_10000000_events.h5'%(int(key.split()[0]), energy_neutrino)
            if not os.path.exists(infile):
                infile = 'results_2014_jun_5/config_simple_%i_%.2e_GeV_1000000_events.h5'%(int(key.split()[0]), energy_neutrino)
        if key in ['6000']:
            infile = 'results_2014_aug_7/config_simple_%i_%.2e_GeV_1000000_events.h5'%(int(key), energy_neutrino)
        if key in ['6000_ross']:
            infile = 'results_2014_aug_7/config_simple_%s_%.2e_GeV_1000000_events.h5'%(key, energy_neutrino)

        print infile

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

pylab.plot(1.e-6 * icecube_energy_contained, icecube_volumetric_acceptance_contained, linestyle='-', color='black', linewidth=4, label='IceCube, contained event analysis')
pylab.plot(1.e-6 * icecube_energy, icecube_volumetric_acceptance, linestyle='--', color='black', linewidth=4, label='IceCube, extremely high energy search')

for key in order:
#for key in ['-2']:
    #pylab.errorbar(energy_neutrino_array, dic_data[key]['acceptance'] / energy_neutrino_array, 
    #               yerr=[dic_data[key]['acceptance_error_low'] / energy_neutrino_array, dic_data[key]['acceptance_error_high'] / energy_neutrino_array], 
    #               label=dic_configurations[key])
    print key
    print energy_neutrino_array
    print dic_data[key]['acceptance']

    cut = dic_data[key]['acceptance'] > 1.e-20

    #pylab.errorbar(energy_neutrino_array[cut], dic_adjust[key] * dic_data[key]['acceptance'][cut],
    #               yerr=[dic_adjust[key] * dic_data[key]['acceptance_error_low'][cut], dic_adjust[key] * dic_data[key]['acceptance_error_high'][cut]],    
    #               ecolor=dic_color[key], color='none')
    
    # THIS VERSION CURRENTLY USED FOR PAPER
    #pylab.plot(1.e-6 * energy_neutrino_array[cut], dic_adjust[key] * dic_data[key]['acceptance'][cut], 
    #           linestyle=dic_linestyle[key], color=dic_color[key], linewidth=dic_linewidth[key], label=dic_configurations[key])
    # THIS VERSION CURRENTLY USED FOR PAPER

    pylab.plot(1.e-6 * energy_neutrino_array[cut], dic_adjust[key] * dic_data[key]['volumetric_acceptance'][cut], 
               linestyle=dic_linestyle[key], color=dic_color[key], linewidth=dic_linewidth[key], label=dic_configurations[key])

    #pylab.errorbar(1.e-6 * energy_neutrino_array[cut], dic_adjust[key] * dic_data[key]['volumetric_acceptance'][cut],
    #               yerr=[dic_data[key]['volumetric_acceptance_error_low'][cut], dic_data[key]['volumetric_acceptance_error_high'][cut]],    
    #               label=dic_configurations[key])
    
    #pylab.errorbar(energy_neutrino_array, dic_adjust[key] * dic_time[key] * dic_data[key]['volumetric_acceptance'],
    #               yerr=[dic_adjust[key] * dic_time[key] * dic_data[key]['volumetric_acceptance_error_low'], 
    #                     dic_adjust[key] * dic_time[key] * dic_data[key]['volumetric_acceptance_error_high']],    
    #               label=dic_configurations[key])

pylab.xlabel('Neutrino Energy (PeV)')#, labelpad=10)
#pylab.ylabel(r'A$\Omega$ (m$^2$ sr)', fontsize='large') # THIS VERSION CURRENTLY USED FOR PAPER
pylab.ylabel(r'V$\Omega$ (km$^3$ sr)')
#pylab.ylabel(r'V$\Omega$ $\times$ Livetime (km$^3$ sr yr)')
pylab.legend(loc='lower right', fontsize=15, frameon=False)# title='Elevation, Ice Model, Livetime Factor')
#pylab.xlim([10**5.0, 10**12.])
pylab.xlim([10**(-2.0), 10**6.])
#pylab.ylim([1.e0, 1.e6])
pylab.draw()
pylab.tight_layout()

if save:
    pylab.savefig('acceptance_comparison.pdf')

#pylab.savefig('/home/bechtol/public_html/temp/fig1.pdf')

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
"""
dic_model = {'kotera_2010_optimistic': 'Kotera et al. 2010 Optimistic',
             'kotera_2010_pessimistic': 'Kotera et al. 2010 Pessimistic',
             'icecube_2014_power_law': 'IceCube 2014 Power Law',
             'icecube_2014_exp_cutoff': 'IceCube 2014 Exponential Cutoff'}

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
    pylab.ylabel('Total above Energy Threshold (3 Years)')
    #pylab.ylabel('dN/dlog(E) (Duration of Experiment)')
    pylab.title(dic_model[key_model])
    pylab.legend(loc='upper right', fontsize=12, frameon=False, title='Elevation, Ice Model, Livetime Factor')
    pylab.xlim([10**5., 10**12])
    pylab.ylim([1.e-1, 1.e3])

    #pylab.savefig('/home/bechtol/public_html/temp/%s.pdf'%(key_model))
"""
############################################################

def histogramWeight(energy, rate, energy_bins):
    weight_array = numpy.zeros(len(energy_bins) - 1)
    for ii in range(1, len(energy_bins)):
        cut = numpy.logical_and(energy >= energy_bins[ii - 1], energy <= energy_bins[ii])
        weight_array[ii - 1] = numpy.sum(rate[cut])
    return weight_array

############################################################

dic_zorder = {'-100': 2,
              '-100 x4': 1,
              '-100 x20': 0}

dic_model = {'kotera_2010_optimistic': 'Kotera et al. 2010, Optimistic Cosmogenic',
             'kotera_2010_pessimistic': 'Kotera et al. 2010, Pessimistic Cosmogenic',
             'kotera_2010_iron_high_emax': r'Kotera et al. 2010, High E$_{\rm max}$ Iron Cosmogenic',
             'kotera_2010_iron_low_emax': r'Kotera et al. 2010, Low E$_{\rm max}$ Iron Cosmogenic',
             'icecube_2014_power_law': 'IceCube 2014, Power Law',
             'icecube_2014_exp_cutoff': 'IceCube 2014, Power Law w/ Exponential Cutoff at 1 PeV'}

energy_bins = 10.**numpy.arange(5, 12.1, 0.5) # Was 1 bin per decade
#energy_bins = 10.**numpy.arange(4.5, 12.1, 0.5)
#energy_bins = numpy.concatenate([[10**4.5], energy_bins])

for key_model in dic_model.keys():
    pylab.figure()
    pylab.xscale('log')
    #pylab.yscale('log')

    print '\n===== %s =====\n'%(key_model)

    for key in order:
        energy_plot, rate_threshold_plot, rate_plot = gnosim.sim.fold_spectrum.foldSpectrum(energy_neutrino_array, dic_adjust[key] * dic_time[key] * dic_data[key]['acceptance'], model_key=key_model)
        #pylab.plot(energy_plot, rate_threshold_plot, label=dic_configurations[key])                                                                                                                       
        #pylab.plot(energy_plot, rate_plot, label=dic_configurations[key])
        values = pylab.hist(1.e-6 * energy_plot, bins=1.e-6 * energy_bins, weights=rate_plot, label=dic_configurations[key], color=dic_color[key], zorder=dic_zorder[key])[0]

        #print key_model, key
        #print values, numpy.sum(values)

        #print '\n%s'%(dic_configurations[key])
        print '%40s   %10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f'%(dic_configurations[key],
                                                                         values[0], values[1], values[2], values[3], values[4], values[5], values[6], numpy.sum(values))

    pylab.xlabel('Neutrino Energy (PeV)')#, labelpad=10)
    #pylab.ylabel(r'Rate above Energy Threshold (yr$^{-1}$)')                                                                                                                                              
    pylab.ylabel('Number of Detected Events (3 Years)')
    #pylab.ylabel('dN/dlog(E) (Duration of Experiment)')
    pylab.title(dic_model[key_model], fontsize=20)
    pylab.legend(loc='upper right', fontsize=16, frameon=False)# title='Elevation, Ice Model, Livetime Factor')
    pylab.xlim([10**5, 10**12])
    pylab.xlim([10**(-1.), 10**6])
    #pylab.xlim([5., 12.])
    #pylab.xticks([5, 6, 7, 8, 9, 10, 11, 12], [r'10^{5}', r'10^{6}', r'10^{7}', r'10^{8}', r'10^{9}', r'10^{10}', r'10^{11}', r'10^{12}'])
    #pylab.ylim([1.e-1, 1.e3])
    #pylab.ylim([0., 30.])
    pylab.ylim([0., 1.5 * numpy.max(values)])
    pylab.draw()
    pylab.tight_layout()
    
    if save:
        pylab.savefig('hist_detected_events_linear_scale_%s.pdf'%(key_model))

############################################################

"""
dic_model_linestyle = {'kotera_2010_optimistic': '-',
                       'kotera_2010_pessimistic': '--',
                       'kotera_2010_iron_high_emax': '-',
                       'kotera_2010_iron_low_emax': '--',
                       'icecube_2014_power_law': ':',
                       'icecube_2014_exp_cutoff': '-.'}
"""

dic_model_linestyle = collections.OrderedDict([('kotera_2010_optimistic', '-'),
                                               ('kotera_2010_pessimistic', '--'),
                                               #('kotera_2010_iron_high_emax', '-'),
                                               #('kotera_2010_iron_low_emax', '--'),
                                               ('icecube_2014_power_law', ':'),
                                               ('icecube_2014_exp_cutoff', '-.')])

pylab.figure()
pylab.xscale('log')
pylab.yscale('log')

for key_model in dic_model_linestyle.keys():
    print key_model
    energy, e2dNdE = zip(*gnosim.sim.fold_spectrum.model_dict[key_model])
    energy = numpy.array(energy)
    if 'iron' in key_model:
        color = 'red'
    else:
        color = 'black'
    pylab.plot(1.e-6 * energy, e2dNdE, label=dic_model[key_model], linewidth=2, color=color, linestyle=dic_model_linestyle[key_model])
    
pylab.legend(loc='upper center', frameon=False, fontsize=15) # Normally 14
#pylab.xlim(1.e5, 1.e12)
pylab.xlim(10**(-1.), 1.e6)
pylab.ylim(10**(-10.5), 10**(-6.5))
pylab.xlabel('Neutrino Energy (PeV)')#, labelpad=10)
pylab.ylabel(r'E$^{2}$ dN/dE (GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
pylab.draw()
pylab.tight_layout()

if save:
      pylab.savefig('model_comparison.pdf')     
