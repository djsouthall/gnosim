#This is done with surface jitter on the order of a few mm
{'detector_volume': {'ice_model': 'antarctica',
                     'radius': 6200.,
                     'depth': 2950.},#making volume slightly smaller than throw valume to avoid edge effects of interpolation
 'DAQ': {   'sampling_rate_GHz': 1.5,
            'sampling_bits': 7},
 'antenna_type':{'antenna_type':'dipole'}, #or dipole, as the case may be
 'stations': {'n': 1,
              'positions': [[0., 0., -173.]]},
 'antennas': {'n': 7,
              'positions': [[0., 0., 0.], [0., 0., -1.], [0., 0., -2.], [0., 0., -3.], [0., 0., -4.], [0., 0., -6.], [0., 0., -8.]],
              'types': ['dipole0','dipole1','dipole2','dipole3','dipole4','dipole5','dipole6'],
              'orientations': [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]}, #alpha, beta, gamma Euler angles in degrees for zxz rotation, all zeros to match ice frame
 'antenna_definitions': {'dipole0': {'lib': '/home/dsouthall/Projects/GNOSim/library_-173_polar_180_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K  #Temperatures need to be double checked in code to see if they are being used
                                    'resistance':50},
                        'dipole1': {'lib': '/home/dsouthall/Projects/GNOSim/library_-174_polar_180_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole2': {'lib': '/home/dsouthall/Projects/GNOSim/library_-175_polar_180_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole3': {'lib': '/home/dsouthall/Projects/GNOSim/library_-176_polar_180_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole4': {'lib': '/home/dsouthall/Projects/GNOSim/library_-177_polar_180_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole5': {'lib': '/home/dsouthall/Projects/GNOSim/library_-179_polar_180_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50},
                        'dipole6': {'lib': '/home/dsouthall/Projects/GNOSim/library_-181_polar_180_rays/*.h5',
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8, # GHz
                                    'temp': 320.,#K
                                    'resistance':50}}
}

