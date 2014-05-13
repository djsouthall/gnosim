import sys
import os
import numpy
import h5py
import pylab

import gnosim.utils.constants
import gnosim.utils.bayesian_efficiency
import gnosim.earth.greenland

pylab.ion()

infile = sys.argv[1]

reader = h5py.File(infile, 'r')

title = r'E$_{\nu}$ = %.2e GeV, Depth = %.1f, %i Events'%(float(os.path.basename(infile).split('_')[3]),
                                                          -1. * float(os.path.basename(infile).split('_')[2]),
                                                          int(os.path.basename(infile).split('_')[5]))

cut_seen = reader['p_detect'][...] == 1.
cut_unseen = numpy.logical_not(cut_seen)
electric_field_threshold = 1.e-4
cut_detected = numpy.logical_and(reader['electric_field'][...] > electric_field_threshold, reader['p_earth'][...] >= 0.9)

print numpy.sum(cut_seen), len(cut_seen)

r = numpy.sqrt(reader['x_0'][...]**2 + reader['y_0'][...]**2)
"""
pylab.figure()
pylab.scatter(r[cut_unseen], reader['z_0'][cut_unseen], c='gray', edgecolors='none')
pylab.scatter(r[cut_seen], reader['z_0'][cut_seen], c=numpy.log10(reader['electric_field'][cut_seen]), edgecolors='none')
colorbar = pylab.colorbar()
colorbar.set_label(r'Log(Electric Field) V m$^{-1}$')
pylab.xlabel('Radius (m)')
pylab.ylabel('Elevation (m)')
pylab.title(title)

pylab.figure()
pylab.yscale('log')
pylab.scatter(numpy.cos(numpy.radians(reader['theta_0'][cut_seen])), reader['electric_field'][cut_seen], c=reader['p_earth'][cut_seen], edgecolors='none')
colorbar = pylab.colorbar()
colorbar.set_label('Probability Earth')
pylab.xlabel('Cos(Theta)')
pylab.ylabel(r'Electric Field (V m$^{-1}$)')
pylab.title(title)

pylab.figure()
pylab.scatter(reader['d'][cut_seen], reader['a_v'][cut_seen], edgecolors='none')
pylab.xlabel('Distance (m)')
pylab.ylabel('Voltage Attenuation')
pylab.title(title)

pylab.figure()
pylab.yscale('log')
pylab.scatter(reader['d'][cut_seen], reader['electric_field'][cut_seen], edgecolors='none')
pylab.xlabel('Distance (m)')
pylab.ylabel(r'Electric Field (V m$^{-1}$)')
pylab.title(title)

pylab.figure()
pylab.yscale('log')
pylab.scatter(reader['theta_ant'][cut_seen], reader['electric_field'][cut_seen], c=reader['p_earth'][cut_seen], edgecolors='none')
colorbar = pylab.colorbar()
colorbar.set_label('Probability Earth')
pylab.xlabel('Theta Antenna (deg)')
pylab.ylabel(r'Electric Field (V m$^{-1}$)')
pylab.title(title)

# Cumulative distribution

cut_cdf = numpy.logical_and(cut_seen, reader['p_earth'][...] > 0.5)
electric_field_cdf = numpy.sort(reader['electric_field'][cut_cdf])[::-1]
cdf = numpy.linspace(0, 1, len(electric_field_cdf))
pylab.figure()
pylab.xscale('log')
pylab.yscale('log')
pylab.plot(electric_field_cdf, cdf)
pylab.xlabel(r'Electric Field (V m$^{-1}$)')
pylab.ylabel('CDF')
x_min, x_max = pylab.xlim()
pylab.xlim([x_max, x_min])
pylab.title(title)

# Theta ray distribution

pylab.figure()
pylab.hist(numpy.cos(numpy.radians(reader['theta_ray'][cut_seen])), weights=reader['p_earth'][cut_seen], bins=numpy.linspace(-1, 1, 41), normed=True)
pylab.xlabel('Cos(Theta Ray)')
pylab.ylabel('PDF')
pylab.title(title)
pylab.xlim([-1., 1.])

# Theta ray distribution

pylab.figure()
pylab.hist(numpy.cos(numpy.radians(reader['theta_0'][cut_seen])), weights=reader['p_earth'][cut_seen], bins=numpy.linspace(-1, 1, 41), normed=True)
pylab.xlabel('Cos(Theta)')
pylab.ylabel('PDF')
pylab.title(title)
pylab.xlim([-1., 1.])

pylab.figure()
pylab.scatter(r[cut_detected], reader['z_0'][cut_detected], c=reader['theta_0'][cut_detected], edgecolors='none', vmin=0., vmax=90.)
colorbar = pylab.colorbar()
colorbar.set_label(r'Theta (deg)')
pylab.xlabel('Radius (m)')
pylab.ylabel('Elevation (m)')
pylab.title(title)
pylab.xlim([-1000., 5000.])
"""
# Observation Angle

pylab.figure()
pylab.hist(reader['observation_angle'][cut_seen], bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='red', label='Visible')
pylab.hist(reader['observation_angle'][cut_detected], bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='blue', label='Detected')
pylab.xlabel('Observation Angle (deg)')
pylab.ylabel('PDF')
pylab.title(title)
pylab.legend(loc='upper right')
"""
pylab.figure()
pylab.yscale('log')
pylab.scatter(reader['observation_angle'][cut_seen], reader['electric_field'][cut_seen], c=reader['d'][cut_seen], edgecolors='none')
colorbar = pylab.colorbar()
colorbar.set_label(r'Distance (m)')
pylab.xlabel('Observation Angle (deg)')
pylab.ylabel(r'Electric Field (V m$^{-1}$)')
pylab.title(title)
pylab.xlim([0., 180.])
"""
# Solutions

pylab.figure()
pylab.hist(reader['solution'][cut_seen], bins=numpy.arange(6 + 1), alpha=0.5, color='red', normed=True, label='Visible')
pylab.hist(reader['solution'][cut_detected], bins=numpy.arange(6 + 1), alpha=0.5, color='blue', normed=True, label='Detected')
pylab.xlabel('Solution')
pylab.ylabel('PDF')
pylab.title(title)
pylab.legend(loc='upper right')

# Acceptance

electric_field_threshold = 1.e-4
efficiency, (efficiency_low, efficiency_high) \
    = gnosim.utils.bayesian_efficiency.confidenceInterval(reader['p_interact'].shape[0], 
                                                          numpy.sum(reader['electric_field'][...] > electric_field_threshold))



volumetric_acceptance = numpy.sum(reader['p_earth'][...] \
                                  * (gnosim.earth.greenland.density(reader['z_0'][...]) / gnosim.utils.constants.density_water)
                                  * (reader['electric_field'][...] > electric_field_threshold) \
    * reader.attrs['geometric_factor']) / float(reader['p_interact'].shape[0]) * gnosim.utils.constants.km_to_m**-3 # km^3 sr

print 'Volumetric Acceptance = %.2e km^3 sr water equivalent'%(volumetric_acceptance)
print 'Efficiency = %.2e (%.2e -- %.2e)'%(efficiency, efficiency_low, efficiency_high)
