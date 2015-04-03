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

# Interaction vertex has a ray-tracing solution
cut_seen = reader['p_detect'][...] == 1. 
cut_unseen = numpy.logical_not(cut_seen)

# Interaction vertex has ray-tracing solution, and at least 50% probability to survive Earth passage
cut_detectable = numpy.logical_and(reader['p_detect'][...] == 1., reader['p_earth'][...] >= 0.5) 

if len(sys.argv) == 3:
    electric_field_threshold = float(sys.argv[2])
else:
    electric_field_threshold = 1.e-4 # 1.e-5
print 'Electric field threshold = %.2e'%(electric_field_threshold)

# Interaction vertex has ray-tracing solution, and at least 50% probability to survive Earth passage, and electric field threshold crossed
cut_detected = numpy.logical_and(reader['electric_field'][...] > electric_field_threshold, reader['p_earth'][...] >= 0.5)

cut_no_bottom = numpy.logical_and(reader['solution'][...] >= 0, reader['solution'][...] <= 2)
cut_bottom = numpy.logical_and(reader['solution'][...] >= 3, reader['solution'][...] <= 5)

cut_detected_no_bottom = numpy.logical_and(cut_detected, cut_no_bottom)
cut_detected_bottom = numpy.logical_and(cut_detected, cut_bottom)

#print numpy.sum(cut_seen), numpy.sum(cut_detected), len(cut_seen)

print '# Events with ray-tracing solutions = %i'%(numpy.sum(cut_seen))
print '# Events detectable                 = %i'%(numpy.sum(cut_detectable))
print '# Events detected                   = %i'%(numpy.sum(cut_detected))
print '# Events total                      = %i'%(len(cut_seen))

r = numpy.sqrt(reader['x_0'][...]**2 + reader['y_0'][...]**2)
"""
pylab.figure()
pylab.scatter(reader['d'][cut_seen], reader['a_v'][cut_seen])


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
"""
"""
pylab.figure()
pylab.yscale('log')
pylab.scatter(reader['d'][cut_detectable], reader['electric_field'][cut_detectable], edgecolors='none')
pylab.xlabel('Distance (m)')
pylab.ylabel(r'Electric Field (V m$^{-1}$)')
pylab.title(title)
"""
"""
pylab.figure()
pylab.yscale('log')
pylab.scatter(reader['d'][cut_detected], reader['electric_field'][cut_detected], c=reader['observation_angle'][cut_detected], edgecolors='none')
colorbar = pylab.colorbar()                                                                                                                                 
colorbar.set_label('Observation Angle (deg)')
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


pylab.figure()
pylab.hist(reader['theta_ant'][cut_detected], bins=numpy.linspace(0., 180., 19), normed=True)
pylab.xlabel('Theta Ant (deg)')
pylab.ylabel('PDF')
"""
"""
# Cumulative distribution

#cut_cdf = numpy.logical_and(cut_seen, reader['p_earth'][...] > 0.5)
#electric_field_cdf = numpy.sort(reader['electric_field'][cut_cdf])[::-1]
electric_field_cdf = numpy.sort(reader['electric_field'][cut_detectable])[::-1]
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
"""
"""
pylab.figure()
pylab.scatter(r[cut_detected_no_bottom], reader['z_0'][cut_detected_no_bottom], c=reader['theta_0'][cut_detected_no_bottom], edgecolors='none', vmin=0., vmax=90.)
colorbar = pylab.colorbar()
colorbar.set_label(r'Theta (deg)')
pylab.xlabel('Radius (m)')
pylab.ylabel('Elevation (m)')
pylab.title(title)
#pylab.xlim([-1000., 5000.])
"""
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

pylab.figure()

bins = numpy.linspace(0, 5000., 51)

#pylab.yscale('log')
#pylab.hist(reader['d'][cut_detected], bins=41, normed=True, color='blue', weights=reader['d'][cut_detected]**(-2))
normed = False
values = pylab.hist(reader['d'][cut_detected], bins=bins, normed=normed, color='blue', alpha=0.5, label='Detected')[0]
v = pylab.hist(reader['d'][cut_detectable], bins=bins, normed=normed, color='black', alpha=1., histtype='step', label='Ray Length')[0]
#pylab.hist(reader['d'][cut_seen], bins=bins, normed=False, color='gray', alpha=0.1)
r3 = numpy.sqrt(reader['x_0'][...]**2 + reader['y_0'][...]**2 + (reader['z_0'][...] + 100.)**2)

#pylab.hist(r, bins=bins, normed=normed, color='green', alpha=1., histtype='step', label='Projected Distance to Vertex')
pylab.hist(r3[cut_detectable], bins=bins, normed=normed, color='red', alpha=1., histtype='step', label='Distance to Vertex')

#pylab.hist(reader['observation_angle'][cut_detected], bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='blue', label='Detected')
pylab.xlabel('Distance (m)')
pylab.ylabel('PDF')
pylab.title(title)
pylab.legend(loc='upper left')

pylab.figure()
pylab.plot(bins[1:], values / v)

"""

# Solutions

pylab.figure()
pylab.hist(reader['solution'][cut_seen], bins=numpy.arange(6 + 1), alpha=0.5, color='red', normed=True, label='Visible')
pylab.hist(reader['solution'][cut_detected], bins=numpy.arange(6 + 1), alpha=0.5, color='blue', normed=True, label='Detected')
pylab.xlabel('Solution')
pylab.ylabel('PDF')
pylab.title(title)
pylab.legend(loc='upper right')
"""
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
