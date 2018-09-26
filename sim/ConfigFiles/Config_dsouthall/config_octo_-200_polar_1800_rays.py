#This is done with surface jitter on the order of a few mm
{'detector_volume': {'ice_model': 'polar',
                     'radius': 6300.,
                     'depth': 3000.},
 'antenna_type':{'antenna_type':'simple'}, #or simple, as the case may be
 'stations': {'n': 1,
              'positions': [[0., 0., -200.]]},
 'antennas': {'n': 8,
              'positions': [[0., 0., 0.], [0., 0., -1.], [0., 0., -2.], [0., 0., -3.], [0., 0., -4.], [0., 0., -5.], [0., 0., -6.], [0., 0., -7.]],
              'types': ['simple0','simple1','simple2','simple3','simple4','simple5','simple6','simple7']},
 'antenna_definitions': {'simple0': {'lib': '/home/dsouthall/Projects/GNOSim/library_-200_polar_1800_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.} ,# K
                        'simple1': {'lib': '/home/dsouthall/Projects/GNOSim/library_-201_polar_1800_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple2': {'lib': '/home/dsouthall/Projects/GNOSim/library_-202_polar_1800_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple3': {'lib': '/home/dsouthall/Projects/GNOSim/library_-203_polar_1800_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple4': {'lib': '/home/dsouthall/Projects/GNOSim/library_-204_polar_1800_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple5': {'lib': '/home/dsouthall/Projects/GNOSim/library_-205_polar_1800_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple6': {'lib': '/home/dsouthall/Projects/GNOSim/library_-206_polar_1800_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple7': {'lib': '/home/dsouthall/Projects/GNOSim/library_-207_polar_1800_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.}}
}

