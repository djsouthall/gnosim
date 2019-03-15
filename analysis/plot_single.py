import sys
import os
import numpy
import h5py
import pylab

sys.path.append("/home/dsouthall/Projects/GNOSim/")
import gnosim.utils.constants
import gnosim.utils.bayesian_efficiency
import gnosim.earth.ice

pylab.ion()


infile = sys.argv[1]

reader = h5py.File(infile, 'r')


#title = 'E$_{\nu}$ = %.2e GeV, Depth = %.1f, %i Events'%(float(os.path.basename(infile).split('_')[3]),
 #                                                         -1. * float(os.path.basename(infile).split('_')[2]),
  #                                                       int(os.path.basename(infile).split('_')[5]))
infilefix = infile.replace('results_2017_may_','')
infilefix2 = infilefix.replace('_events_1.h5', ' ')
title=infilefix2 + " Plot"


# Interaction vertex has a ray-tracing solution
cut_seen = reader['p_detect'][...] == 1.
cut_unseen = numpy.logical_not(cut_seen)

# Interaction vertex has ray-tracing solution, and at least 50% probability to survive Earth passage
cut_detectable = numpy.logical_and(reader['p_detect'][...] == 1., reader['p_earth'][...] >= 0.5)

if len(sys.argv) == 3:
    electric_field_threshold = float(sys.argv[2])
else:
    electric_field_threshold = 1.e-4 # 1.e-5
print ('Electric field threshold = %.2e'%(electric_field_threshold))

# Interaction vertex has ray-tracing solution, and at least 50% probability to survive Earth passage, and electric field threshold crossed
cut_detected = numpy.logical_and(reader['electric_field'][...] > electric_field_threshold, reader['p_earth'][...] >= 0.5)

cut_no_bottom = numpy.logical_and(reader['solution'][...] >= 0, reader['solution'][...] <= 2)
cut_bottom = numpy.logical_and(reader['solution'][...] >= 3, reader['solution'][...] <= 5)

cut_detected_no_bottom = numpy.logical_and(cut_detected, cut_no_bottom)
cut_detected_bottom = numpy.logical_and(cut_detected, cut_bottom)

#print (numpy.sum(cut_seen), numpy.sum(cut_detected), len(cut_seen))

print ('# Events with ray-tracing solutions = %s'%(numpy.sum(cut_seen)))
print ('# Events detectable                 = %s'%(numpy.sum(cut_detectable)))
print ('# Events detected                   = %s'%(numpy.sum(cut_detected)))
print ('# Events total                      = %s'%(len(cut_seen)))
#UNCOMMENT ME

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
pylab.ylabel('Electric Field (V m$^{-1}$)')
pylab.title(title)
"""
"""
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
"""
pylab.figure()
pylab.yscale('log')
pylab.scatter(reader['d'][cut_detected], reader['electric_field'][cut_detected], c=reader['observation_angle'][cut_detected], edgecolors='none')
colorbar = pylab.colorbar()                                                                                                                                 
colorbar.set_label('Observation Angle (deg)')
pylab.xlabel('Distance (m)')
pylab.ylabel(r'Electric Field (V m$^{-1}$)')
pylab.title(title)



"""
pylab.figure()
pylab.yscale('log')
pylab.scatter(reader['theta_ant'][cut_seen], reader['electric_field'][cut_seen], c=reader['p_earth'][cut_seen], edgecolors='none')
colorbar = pylab.colorbar()
colorbar.set_label('Probability Earth')
pylab.xlabel('Theta Antenna (deg)')
pylab.ylabel(r'Electric Field (V m$^{-1}$)')
pylab.title(title)
"""
"""
#one-two shift
q = reader['t'][cut_detected]
print(q)
print(len(q))
t1 = numpy.zeros([len(q)])

for number in range(0, 462):
	t1[number] = (q[number] - q[(number+462)])

t2 = numpy.zeros([len(q)])

for number in range(0, 462):
	t2[number] = (q[number+462] - q[(number+462+462)])

print(len(reader['theta_0'][cut_detected]))
print(len(t1))

pylab.figure()
pylab.scatter(t1, reader['theta_0'][cut_detected])
pylab.scatter(t2, reader['theta_0'][cut_detected], color='red')


#plot time to one antennae minus time to other vs theta
pylab.figure()
#print(h5py.hdf5read(infile, 'index_antenna'))
#pylab.scatter(reader['t'][cut_detected], reader['theta_0'][cut_detected])
pylab.scatter(reader['theta_0'][cut_detected], reader['index_antenna'][cut_detected], c=reader['t'][cut_detected], edgecolors='none')
colorbar = pylab.colorbar()
pylab.xlabel('Theta')
pylab.ylabel('Antenna Index')
"""

"""
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

"""
# Theta ray distribution

pylab.figure()
pylab.hist(numpy.cos(numpy.radians(reader['theta_ray'][cut_seen])), weights=reader['p_earth'][cut_seen], bins=numpy.linspace(-1, 1, 41), normed=True)
pylab.xlabel('Cos(Theta Ray)')
pylab.ylabel('PDF')
pylab.title(title)
pylab.xlim([-1., 1.])



#PLOT EVERYTHING WRT THETA
"""
#Theta vs d, for detected
pylab.figure()
pylab.hist2d(reader['d'][cut_detected], reader['theta_ant'][cut_detected], bins=(75,75))
pylab.xlabel('Distance Traveled')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " Distance/Theta For Detected")

#Theta vs d, for seen
pylab.figure()
pylab.hist2d(reader['d'][cut_seen], reader['theta_ant'][cut_seen], bins=(75,75))
pylab.xlabel('Distance Traveled')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " Distance/Theta For  Seen")

#Theta vs a_v, for detected
pylab.figure()
pylab.hist2d(numpy.log(reader['a_v'][cut_detected]), reader['theta_ant'][cut_detected], bins=(25,75))
pylab.xlabel('Vertical Electric Field')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " a_v/Theta For Detected")

#Theta vs a_v, for seen
pylab.figure()
pylab.hist2d(numpy.log(reader['a_v'][cut_seen]), reader['theta_ant'][cut_seen], bins=(25,75))
pylab.xlabel('Vertical Electric Field')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " a_v/Theta For Seen")

if numpy.isin('a_h',list(reader.keys())):
  #Theta vs a_h, for detected
  pylab.figure()
  pylab.hist2d(numpy.log(reader['a_h'][cut_detected]), reader['theta_ant'][cut_detected], bins=(25,75))
  pylab.xlabel('Sanity Check -- horizontal EF')
  pylab.ylabel('Theta Antenna')
  pylab.colorbar()
  pylab.title(title + " a_h/Theta For Detected")

  #Theta vs a_h, for seen
  pylab.figure()
  pylab.hist2d(numpy.log(reader['a_h'][cut_seen]), reader['theta_ant'][cut_seen], bins=(25,75))
  pylab.xlabel('Sanity Check -- Horizontal EF')
  pylab.ylabel('Theta Antenna')
  pylab.colorbar() 
  pylab.title(title + " a_h/Theta For Seen")
elif numpy.isin('a_s',list(reader.keys())):
  #Theta vs a_s, for detected
  pylab.figure()
  pylab.hist2d(numpy.log(reader['a_s'][cut_detected]), reader['theta_ant'][cut_detected], bins=(25,75))
  pylab.xlabel('Sanity Check -- horizontal EF')
  pylab.ylabel('Theta Antenna')
  pylab.colorbar()
  pylab.title(title + " a_s/Theta For Detected")

  #Theta vs a_s, for seen
  pylab.figure()
  pylab.hist2d(numpy.log(reader['a_s'][cut_seen]), reader['theta_ant'][cut_seen], bins=(25,75))
  pylab.xlabel('Sanity Check -- Horizontal EF')
  pylab.ylabel('Theta Antenna')
  pylab.colorbar() 
  pylab.title(title + " a_s/Theta For Seen")

#Theta vs electric field, for detected
pylab.figure()
pylab.hist2d(numpy.log(reader['electric_field'][cut_detected]), reader['theta_ant'][cut_detected], bins=(50,50))
pylab.xlabel('Electric Field')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " Electric Field/Theta For Detected")

#Theta vs Electric Field, for seen
pylab.figure()
pylab.hist2d(numpy.log(reader['electric_field'][cut_seen]), reader['theta_ant'][cut_seen], bins=(50,50))
pylab.xlabel('Electric Field')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " Electric Field/Theta For Seen")

#Theta vs solution, for detected
pylab.figure()
pylab.hist2d(reader['solution'][cut_detected], reader['theta_ant'][cut_detected], bins=(25,25))
pylab.xlabel('Solution')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " Solution/Theta For Detected")

#Theta vs Solution, for seen
pylab.figure()
pylab.hist2d(reader['solution'][cut_seen], reader['theta_ant'][cut_seen], bins=(25,25))
pylab.xlabel('Solution')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " Solution/Theta For Seen")

#Theta vs time, for detected
pylab.figure()
pylab.hist2d(reader['t'][cut_detected], reader['theta_ant'][cut_detected], bins=(75,75))
pylab.xlabel('Time')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " Time/Theta For Detected")


#Theta vs time, for seen
pylab.figure()
pylab.hist2d(reader['t'][cut_seen], reader['theta_ant'][cut_seen], bins=(75,75))
pylab.xlabel('Time')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " Time/Theta For Seen")
"""

#Theta vs Time with solution cuts!

#Solution 0
zeroTimeCutdetected = []
zeroThetaCutdetected = []
for event in numpy.arange(0, len(reader['solution'][cut_detected])):
  if reader['solution'][cut_detected][event] == 0:
    zeroTimeCutdetected.append(reader['t'][cut_detected][event])
    zeroThetaCutdetected.append(reader['theta_ant'][cut_detected][event])
"""
pylab.figure()
pylab.hist2d(zeroTimeCutdetected, zeroThetaCutdetected, bins=(75,75))
pylab.xlabel('Time for Solution Zero')
pylab.ylabel('Theta Antenna  for Solution Zero')
pylab.colorbar()
pylab.title(title + " Time/Theta Detected for Solution Zero")
"""
"""
pylab.figure()
pylab.hist(zeroThetaCutdetected, bins=20,  normed=True)
pylab.title(title+" Theta Detected for Solution 0, Normalized")
pylab.xlabel('Angle')
pylab.ylabel('PDF')
"""

#Solution 1
oneTimeCutdetected = []
oneThetaCutdetected = []
for event in numpy.arange(0, len(reader['solution'][cut_detected])):
  if reader['solution'][cut_detected][event] == 1:
    oneTimeCutdetected.append(reader['t'][cut_detected][event])
    oneThetaCutdetected.append(reader['theta_ant'][cut_detected][event])
"""
pylab.figure()
pylab.hist2d(oneTimeCutdetected, oneThetaCutdetected, bins=(75,75))
pylab.xlabel('Time for Solution One')
pylab.ylabel('Theta Antenna  for Solution One')
pylab.colorbar()
pylab.title(title + " Time/Theta Detected for Solution One")
"""
"""
pylab.figure()
pylab.hist(oneThetaCutdetected, bins=20,  normed=True)
pylab.title(title+" Theta Detected for Solution 1, Normalized")
pylab.xlabel('Angle')
pylab.ylabel('PDF')
"""

#Solution 2
twoTimeCutdetected = []
twoThetaCutdetected = []
for event in numpy.arange(0, len(reader['solution'][cut_detected])):
  if reader['solution'][cut_detected][event] == 2:
    twoTimeCutdetected.append(reader['t'][cut_detected][event])
    twoThetaCutdetected.append(reader['theta_ant'][cut_detected][event])
"""
pylab.figure()
pylab.hist2d(twoTimeCutdetected, twoThetaCutdetected, bins=(75,75))
pylab.xlabel('Time for Solution Two')
pylab.ylabel('Theta Antenna  for Solution Two')
pylab.colorbar()
pylab.title(title + " Time/Theta Detected for Solution Two")
"""
pylab.figure()
pylab.hist(twoThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='red', label='Solution 2')
pylab.hist(oneThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='blue', label='Solution 1')
pylab.hist(zeroThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='green', label='Solution 0')
pylab.title(title+" Theta Detected for All Solutions, Normalized")
pylab.xlabel('Angle')
pylab.ylabel('PDF')
pylab.legend(loc='upper left')

thetacutdetected = numpy.append(zeroThetaCutdetected, oneThetaCutdetected)
thetacutdetected = numpy.append(thetacutdetected, twoThetaCutdetected)
zeroThetaCutdetected = numpy.asarray(zeroThetaCutdetected)
oneThetaCutdetected = numpy.asarray(oneThetaCutdetected)
twoThetaCutdetected = numpy.asarray(twoThetaCutdetected)


weights0 = zeroThetaCutdetected/float((len(thetacutdetected))**2)
weights1 = oneThetaCutdetected/float((len(thetacutdetected))**2)
weights2 = twoThetaCutdetected/float((len(thetacutdetected))**2)


pylab.figure()
pylab.hist(zeroThetaCutdetected, bins=numpy.linspace(0., 180., 41),  alpha=0.5, weights=weights0, color='blue', label='Solution 0')
pylab.hist(oneThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0.5, weights=weights1, color='green', label='Solution 1')
pylab.hist(twoThetaCutdetected, bins=numpy.linspace(0., 180., 41),  alpha=0.5, weights=weights2, color='red', label='Solution 2')
pylab.title("Energy 10^9 GeV, Depth of 200 Meters, Theta Detected, Normalized")
pylab.xlabel('Angle')
pylab.ylabel('PDF')
pylab.legend(loc='upper right')


pylab.figure()
pylab.hist(twoThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0, normed=True, color='red')
pylab.hist(oneThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0, normed=True, color='blue')
pylab.hist(zeroThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='green', label='Solution 0')
pylab.title(title+" Theta Detected for Solution 0, Normalized")
pylab.xlabel('Angle')
pylab.ylabel('PDF')
pylab.legend(loc='upper right')

pylab.figure()
pylab.hist(twoThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0, normed=True, color='red')
pylab.hist(oneThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='blue', label='Solution 1')
pylab.hist(zeroThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0, normed=True, color='green')
pylab.title(title+" Theta Detected for Solution 1, Normalized")
pylab.xlabel('Angle')
pylab.ylabel('PDF')
pylab.legend(loc='upper right')

pylab.figure()
pylab.hist(twoThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='red', label='Solution 2')
pylab.hist(oneThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0, normed=True, color='blue')
pylab.hist(zeroThetaCutdetected, bins=numpy.linspace(0., 180., 41), alpha=0, normed=True, color='green')
pylab.title(title+" Theta Detected for Solution 2, Normalized")
pylab.xlabel('Angle')
pylab.ylabel('PDF')
pylab.legend(loc='upper right')


#Others
threeTimeCutdetected = []
threeThetaCutdetected = []
for event in numpy.arange(0, len(reader['solution'][cut_detected])):
  if reader['solution'][cut_detected][event] == 3:
    threeTimeCutdetected.append(reader['t'][cut_detected][event])
    threeThetaCutdetected.append(reader['theta_ant'][cut_detected][event])
fourTimeCutdetected = []
fourThetaCutdetected = []
for event in numpy.arange(0, len(reader['solution'][cut_detected])):
  if reader['solution'][cut_detected][event] == 4:
    fourTimeCutdetected.append(reader['t'][cut_detected][event])
    fourThetaCutdetected.append(reader['theta_ant'][cut_detected][event])
fiveTimeCutdetected = []
fiveThetaCutdetected = []
for event in numpy.arange(0, len(reader['solution'][cut_detected])):
  if reader['solution'][cut_detected][event] == 5:
    fiveTimeCutdetected.append(reader['t'][cut_detected][event])
    fiveThetaCutdetected.append(reader['theta_ant'][cut_detected][event])

#Plot of Electric Field Squared vs Angle
pylab.figure()
pylab.hist2d(numpy.log((reader['electric_field'][cut_detected]**2)), reader['theta_ant'][cut_detected], bins=(50,50))
pylab.ylabel('Angle')
pylab.xlabel('Log of the Square of the Electric Field')
pylab.title(title+ " Log of the Square of the Electric Field vs Theta")




#At what angle do we see things from the surface?
FromSurfaceThetaCutdetected = []
for event in numpy.arange(0, len(reader['z_0'][cut_detected])):
  if reader['z_0'][cut_detected][event] >= -200:
    FromSurfaceThetaCutdetected.append(reader['theta_ant'][cut_detected][event])

FromSurfaceThetaCutdetected=numpy.asarray(FromSurfaceThetaCutdetected)
twoThetaCutdetected=numpy.asarray(twoThetaCutdetected)
fiveThetaCutdetected=numpy.asarray(fiveThetaCutdetected)
weightedup1 = FromSurfaceThetaCutdetected/(float((len(FromSurfaceThetaCutdetected)+len(twoThetaCutdetected)+len(fiveThetaCutdetected))**2))
weightedup2 = twoThetaCutdetected/(float((len(FromSurfaceThetaCutdetected)+len(twoThetaCutdetected)+len(fiveThetaCutdetected))**2))
weightedup3 = fiveThetaCutdetected/(float((len(FromSurfaceThetaCutdetected)+len(twoThetaCutdetected)+len(fiveThetaCutdetected))**2))
pylab.figure()
pylab.hist(FromSurfaceThetaCutdetected, bins=numpy.linspace(0., 180., 41),  alpha=0.5, weights=weightedup1, color='royalblue', label='Generated Above')
pylab.hist(twoThetaCutdetected, bins=numpy.linspace(0., 180., 41),  alpha=0.5, weights=weightedup2, color='honeydew', label='Reflected Solution')
pylab.hist(fiveThetaCutdetected, bins=numpy.linspace(0., 180., 41),  alpha=0.5, weights=weightedup3, color='orangered', label='Reflected Solution 2')
pylab.title(title+" Theta From Above")
pylab.xlabel('Angle')
pylab.ylabel('PDF')
pylab.legend(loc='upper right')


#Electric Field Distribution From 80 to 120 Degrees
PopularAngleElectricField = []
PoplularAngleTheta = []
for event in numpy.arange(0, len(reader['theta_ant'][cut_detected])):
  if reader['theta_ant'][cut_detected][event] >= 80:
  	if reader['theta_ant'][cut_detected][event] <= 120:
	    PopularAngleElectricField.append(reader['electric_field'][cut_detected][event])
	    PoplularAngleTheta.append(reader['theta_ant'][cut_detected][event])
pylab.figure()
pylab.hist2d(numpy.log(PopularAngleElectricField), PoplularAngleTheta, bins=(50,50))
pylab.xlabel('Electric Field')
pylab.ylabel('Theta Antenna')
pylab.colorbar()
pylab.title(title + " Electric Field/Theta For Detected, Popular Angle")


"""
#Overlay
pylab.figure()
pylab.scatter(twoTimeCutdetected, twoThetaCutdetected, color='orangered', label='Solution 2')
pylab.scatter(oneTimeCutdetected, oneThetaCutdetected, color='skyblue', label='Solution 1')
pylab.scatter(zeroTimeCutdetected, zeroThetaCutdetected, color='greenyellow', label='Solution 0')
pylab.scatter(threeTimeCutdetected, threeThetaCutdetected, color='orchid', label='Solution 3')
pylab.scatter(fourTimeCutdetected, fourThetaCutdetected, color='gold', label='Solution 4')
pylab.scatter(fiveTimeCutdetected, fiveThetaCutdetected, color='navy', label='Solution 5')
pylab.xlabel('Time')
pylab.ylabel('Theta Antenna')
pylab.legend()
pylab.title(title + " Time/Theta Detected OverLay by Solution")
"""
"""
#Now, for seen
#Solution 1
oneTimeCutseen = []
oneThetaCutseen = []
for event in numpy.arange(0, len(reader['solution'][cut_seen])):
  if reader['solution'][cut_seen][event] == 1:
    oneTimeCutseen.append(reader['t'][cut_seen][event])
    oneThetaCutseen.append(reader['theta_ant'][cut_seen][event])
#Solution 2
twoTimeCutseen = []
twoThetaCutseen = []
for event in numpy.arange(0, len(reader['solution'][cut_seen])):
  if reader['solution'][cut_seen][event] == 2:
    twoTimeCutseen.append(reader['t'][cut_seen][event])
    twoThetaCutseen.append(reader['theta_ant'][cut_seen][event])
#Solution 3
threeTimeCutseen = []
threeThetaCutseen = []
for event in numpy.arange(0, len(reader['solution'][cut_seen])):
  if reader['solution'][cut_seen][event] == 3:
    threeTimeCutseen.append(reader['t'][cut_seen][event])
    threeThetaCutseen.append(reader['theta_ant'][cut_seen][event])
#Solution 4
fourTimeCutseen = []
fourThetaCutseen = []
for event in numpy.arange(0, len(reader['solution'][cut_seen])):
  if reader['solution'][cut_seen][event] == 4:
    fourTimeCutseen.append(reader['t'][cut_seen][event])
    fourThetaCutseen.append(reader['theta_ant'][cut_seen][event])
#Solution 5
fiveTimeCutseen = []
fiveThetaCutseen = []
for event in numpy.arange(0, len(reader['solution'][cut_seen])):
  if reader['solution'][cut_seen][event] == 5:
    fiveTimeCutseen.append(reader['t'][cut_seen][event])
    fiveThetaCutseen.append(reader['theta_ant'][cut_seen][event])
#Solution 0
zeroTimeCutseen = []
zeroThetaCutseen = []
for event in numpy.arange(0, len(reader['solution'][cut_seen])):
  if reader['solution'][cut_seen][event] == 0:
    zeroTimeCutseen.append(reader['t'][cut_seen][event])
    zeroThetaCutseen.append(reader['theta_ant'][cut_seen][event])

#Overlay
pylab.figure()
pylab.scatter(twoTimeCutseen, twoThetaCutseen, color='orangered', label='Solution 0')
pylab.scatter(oneTimeCutseen, oneThetaCutseen, color='skyblue', label='Solution 1')
pylab.scatter(zeroTimeCutseen, zeroThetaCutseen, color='greenyellow', label='Solution 2')
pylab.scatter(threeTimeCutseen, threeThetaCutseen, color='orchid', label='Solution 3')
pylab.scatter(fourTimeCutseen, fourThetaCutseen, color='gold', label='Solution 4')
pylab.scatter(fiveTimeCutseen, fiveThetaCutseen, color='navy', label='Solution 5')
pylab.xlabel('Time')
pylab.ylabel('Theta Antenna')
pylab.legend()
pylab.title(title + " Time/Theta Seen OverLay by Solution")

"""


"""
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




# theta_ant

pylab.figure()
pylab.hist(reader['theta_ant'][cut_seen], bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='red', label='Visible')
pylab.hist(reader['theta_ant'][cut_detected], bins=numpy.linspace(0., 180., 41), alpha=0.5, normed=True, color='blue', label='Detected')
pylab.xlabel('Theta Antenna')
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

pylab.figure()

bins = numpy.linspace(0, 5000., 51)

pylab.yscale('log')
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

infilefix = infile.replace('Output/','Output/Graphs/')

pylab.savefig(infilefix.replace('.h5','_eff.png'))
pylab.figure()
#pylab.plot(bins[1:], values/v  )
pylab.plot(bins[1:], values)

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


ice = gnosim.earth.ice.Ice(reader.attrs['ice_model'],suppress_fun = True)
volumetric_acceptance = numpy.sum(reader['p_earth'][...] \
                                  * (ice.density(reader['z_0'][...]) / gnosim.utils.constants.density_water)
                                  * (reader['electric_field'][...] > electric_field_threshold) \
    * reader.attrs['geometric_factor']) / float(reader['p_interact'].shape[0]) * gnosim.utils.constants.km_to_m**-3 # km^3 sr

print ('Volumetric Acceptance = %.2e km^3 sr water equivalent'%(volumetric_acceptance))
print ('Efficiency = %.2e (%.2e -- %.2e)'%(efficiency, efficiency_low, efficiency_high))


input("press any key to exit")
