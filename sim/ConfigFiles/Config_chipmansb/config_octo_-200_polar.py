{'detector_volume': {'ice_model': 'polar',
                     'radius': 6300.,
                     'depth': 3000.},
 'antenna_type':{'antenna_type':'simple'}, #or simple, as the case may be
 'stations': {'n': 1,
              'positions': [[0., 0., -200.]]},
 'antennas': {'n': 8,
              'positions': [[0., 0., 0.], [0., 0., -1.], [0., 0., -2.], [0., 0., -3.], [0., 0., -4.], [0., 0., -5.], [0., 0., -6.], [0., 0., -7.]],
              'types': ['simple0','simple1','simple2','simple3','simple4','simple5','simple6','simple7']},
 'antenna_definitions': {'simple0': {'lib': '/home/chipmansb/library_-200_polar/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.} ,# K
                        'simple1': {'lib': '/home/chipmansb/library_-201_polar/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple2': {'lib': '/home/chipmansb/library_-202_polar/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple3': {'lib': '/home/chipmansb/library_-203_polar/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple4': {'lib': '/home/chipmansb/library_-204_polar/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple5': {'lib': '/home/chipmansb/library_-205_polar/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple6': {'lib': '/home/chipmansb/library_-206_polar/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.},
                        'simple7': {'lib': '/home/chipmansb/library_-207_polar/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 300.}}
}

