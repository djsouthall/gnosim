#This is done with surface jitter on the order of a few mm
{'detector_volume': {'ice_model': 'antarctica',
                     'radius': 6200.,
                     'depth': 2950.},#making volume slightly smaller than throw valume to avoid edge effects of interpolation
 'DAQ': {   'sampling_rate_GHz': 1.5,
            'sampling_bits': 7},
 'antenna_type':{'antenna_type':'dipole'}, #or dipole, as the case may be
 'stations': {'n': 1,
              'positions': [[0., 0., -200.]]},
 'antennas': {'n': 8,
              'positions': [[0., 0., 0.], [0., 0., -1.], [0., 0., -2.], [0., 0., -3.], [0., 0., -4.], [0., 0., -5.], [0., 0., -6.], [0., 0., -7.]],
              'types': ['dipole0','dipole1','dipole2','dipole3','dipole4','dipole5','dipole6','dipole7'],
              'orientations': [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]},#alpha, beta, gamma Euler angles in degrees for zxz rotation, all zeros to match ice frame 
 'antenna_definitions': {'dipole0': {'lib': '/home/dsouthall/Projects/GNOSim/library_-200_polar_120_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole1': {'lib': '/home/dsouthall/Projects/GNOSim/library_-201_polar_120_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole2': {'lib': '/home/dsouthall/Projects/GNOSim/library_-202_polar_120_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole3': {'lib': '/home/dsouthall/Projects/GNOSim/library_-203_polar_120_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole4': {'lib': '/home/dsouthall/Projects/GNOSim/library_-204_polar_120_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole5': {'lib': '/home/dsouthall/Projects/GNOSim/library_-205_polar_120_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole6': {'lib': '/home/dsouthall/Projects/GNOSim/library_-206_polar_120_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole7': {'lib': '/home/dsouthall/Projects/GNOSim/library_-207_polar_120_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50}}
}

