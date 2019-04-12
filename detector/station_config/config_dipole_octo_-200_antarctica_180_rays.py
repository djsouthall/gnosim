{   'detector_volume': {'ice_model': 'antarctica',
                        'radius': 6200.,
                        'depth': 2950.},#making volume slightly smaller than throw valume to avoid edge effects of interpolation
    'stations':{
            'station0':{
                'DAQ': {
                        'sampling_rate_GHz': 1.5,
                        'sampling_bits': 7,
                        'scale_noise_to': 3,
                        'n_beams':15,
                        'n_baselines':2,
                        'power_calculation_sum_length':16,  #How long each power sum window is
                        'power_calculation_interval':8, #How frequent each power sum window begins
                        'beamforming_power_sum_bit_cap':5 #FPGA does beamforming with only 5 bits despite signals being at 7, so they are capped here:
                        },
                'position' : [0., 0., -200.], #The antenna positions are relative to this.
                'phased_antennas' : { #Antennas in this should be in one vertical string/line, they are used for the phased array calculations.
                        'dipole0': {
                                    'antenna_type':'dipole',
                                    'lib': '/home/dsouthall/Projects/GNOSim/library_-200_antarctica_180_rays_signed_fresnel/*.h5',
                                    'position':[0.0,0.0,-0.0],
                                    'orientation':[0.0,0.0,0.0], #alpha, beta, gamma Euler angles in degrees for zxz rotation, all zeros to match ice frame
                                    'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                    'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                    'noise_temperature': 320.,#K
                                    'resistance':50, #Ohm
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8
                                    },
                        'dipole1': {
                                    'antenna_type':'dipole',
                                    'lib': '/home/dsouthall/Projects/GNOSim/library_-201_antarctica_180_rays_signed_fresnel/*.h5',
                                    'position':[0.0,0.0,-1.0],
                                    'orientation':[0.0,0.0,0.0],
                                    'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                    'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                    'noise_temperature': 320.,#K
                                    'resistance':50, #Ohm
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8
                                    },
                        'dipole2': {
                                    'antenna_type':'dipole',
                                    'lib': '/home/dsouthall/Projects/GNOSim/library_-202_antarctica_180_rays_signed_fresnel/*.h5',
                                    'position':[0.0,0.0,-2.0],
                                    'orientation':[0.0,0.0,0.0],
                                    'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                    'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                    'noise_temperature': 320.,#K
                                    'resistance':50, #Ohm
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8
                                    },
                        'dipole3': {
                                    'antenna_type':'dipole',
                                    'lib': '/home/dsouthall/Projects/GNOSim/library_-203_antarctica_180_rays_signed_fresnel/*.h5',
                                    'position':[0.0,0.0,-3.0],
                                    'orientation':[0.0,0.0,0.0],
                                    'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                    'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                    'noise_temperature': 320.,#K
                                    'resistance':50, #Ohm
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8
                                    },
                        'dipole4': {
                                    'antenna_type':'dipole',
                                    'lib': '/home/dsouthall/Projects/GNOSim/library_-204_antarctica_180_rays_signed_fresnel/*.h5',
                                    'position':[0.0,0.0,-4.0],
                                    'orientation':[0.0,0.0,0.0],
                                    'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                    'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                    'noise_temperature': 320.,#K
                                    'resistance':50, #Ohm
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8
                                    },
                        'dipole5': {
                                    'antenna_type':'dipole',
                                    'lib': '/home/dsouthall/Projects/GNOSim/library_-205_antarctica_180_rays_signed_fresnel/*.h5',
                                    'position':[0.0,0.0,-5.0],
                                    'orientation':[0.0,0.0,0.0],
                                    'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                    'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                    'noise_temperature': 320.,#K
                                    'resistance':50, #Ohm
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8
                                    },
                        'dipole6': {
                                    'antenna_type':'dipole',
                                    'lib': '/home/dsouthall/Projects/GNOSim/library_-206_antarctica_180_rays_signed_fresnel/*.h5',
                                    'position':[0.0,0.0,-6.0],
                                    'orientation':[0.0,0.0,0.0],
                                    'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                    'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                    'noise_temperature': 320.,#K
                                    'resistance':50, #Ohm
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8
                                    },
                        'dipole7': {
                                    'antenna_type':'dipole',
                                    'lib': '/home/dsouthall/Projects/GNOSim/library_-207_antarctica_180_rays_signed_fresnel/*.h5',
                                    'position':[0.0,0.0,-7.0],
                                    'orientation':[0.0,0.0,0.0],
                                    'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                    'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                    'noise_temperature': 320.,#K
                                    'resistance':50, #Ohm
                                    'frequency_low': 0.1, # GHz
                                    'frequency_high': 0.8
                                    }
                            }, #end 'phased_antennas'

                'reconstruction_antennas' : { #Antennas in this should be in one vertical string/line, they are used for the phased array calculations.
                            } #end 'reconstruction_antennas'

                    }, #end 'ARA5'

                } #end 'stations'
}


