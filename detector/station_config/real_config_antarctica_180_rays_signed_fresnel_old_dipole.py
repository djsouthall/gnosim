{   'detector_volume': {'ice_model': 'antarctica',
                        'radius': 6200.,
                        'depth': 2950.},#making volume slightly smaller than throw valume to avoid edge effects of interpolation
    'stations':{
            'ARA5':{
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
                'position' : [0., 0., -173.], #The antenna positions are relative to this.
                'phased_antennas' : { #Antennas in this should be in one vertical string/line, they are used for the phased array calculations.
                        'PA00': {
                                'antenna_type':'old_dipole',
                                'lib': '/home/dsouthall/Projects/GNOSim/gnosim/trace/libraries/library_-173_antarctica_180_rays_signed_fresnel/*.h5',
                                'position':[0.0,0.0,-0.0],
                                'orientation':[0.0,0.0,0.0], #alpha, beta, gamma Euler angles in degrees for zxz rotation, all zeros to match ice frame
                                'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                'noise_temperature': 320.,#K  #Temperatures need to be double checked in code to see if they are being used
                                'resistance':50.0, #Ohm
                                'frequency_low': 0.1, # GHz
                                'frequency_high': 0.8
                                },
                        'PA01': {
                                'antenna_type':'old_dipole',
                                'lib': '/home/dsouthall/Projects/GNOSim/gnosim/trace/libraries/library_-174_antarctica_180_rays_signed_fresnel/*.h5',
                                'position':[0.0,0.0,-1.0],
                                'orientation':[0.0,0.0,0.0],
                                'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                'noise_temperature': 320.,#K
                                'resistance':50.0, #Ohm
                                'frequency_low': 0.1, # GHz
                                'frequency_high': 0.8
                                },
                        'PA02': {
                                'antenna_type':'old_dipole',
                                'lib': '/home/dsouthall/Projects/GNOSim/gnosim/trace/libraries/library_-175_antarctica_180_rays_signed_fresnel/*.h5',
                                'position':[0.0,0.0,-2.0],
                                'orientation':[0.0,0.0,0.0],
                                'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                'noise_temperature': 320.,#K
                                'resistance':50.0, #Ohm
                                'frequency_low': 0.1, # GHz
                                'frequency_high': 0.8
                                },
                        'PA03': {
                                'antenna_type':'old_dipole',
                                'lib': '/home/dsouthall/Projects/GNOSim/gnosim/trace/libraries/library_-176_antarctica_180_rays_signed_fresnel/*.h5',
                                'position':[0.0,0.0,-3.0],
                                'orientation':[0.0,0.0,0.0],
                                'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                'noise_temperature': 320.,#K
                                'resistance':50.0, #Ohm
                                'frequency_low': 0.1, # GHz
                                'frequency_high': 0.8
                                },
                        'PA04': {
                                'antenna_type':'old_dipole',
                                'lib': '/home/dsouthall/Projects/GNOSim/gnosim/trace/libraries/library_-177_antarctica_180_rays_signed_fresnel/*.h5',
                                'position':[0.0,0.0,-4.0],
                                'orientation':[0.0,0.0,0.0],
                                'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                'noise_temperature': 320.,#K
                                'resistance':50.0, #Ohm
                                'frequency_low': 0.1, # GHz
                                'frequency_high': 0.8
                                },
                        'PA05': {
                                'antenna_type':'old_dipole',
                                'lib': '/home/dsouthall/Projects/GNOSim/gnosim/trace/libraries/library_-179_antarctica_180_rays_signed_fresnel/*.h5',
                                'position':[0.0,0.0,-6.0],
                                'orientation':[0.0,0.0,0.0],
                                'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                'noise_temperature': 320.,#K
                                'resistance':50.0, #Ohm
                                'frequency_low': 0.1, # GHz
                                'frequency_high': 0.8
                                },
                        'PA06': {
                                'antenna_type':'old_dipole',
                                'lib': '/home/dsouthall/Projects/GNOSim/gnosim/trace/libraries/library_-181_antarctica_180_rays_signed_fresnel/*.h5',
                                'position':[0.0,0.0,-8.0],
                                'orientation':[0.0,0.0,0.0],
                                'system_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_vpol.npy',
                                'antenna_response':'/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_vpol.npy',
                                'noise_temperature': 320.,#K
                                'resistance':50.0, #Ohm
                                'frequency_low': 0.1, # GHz
                                'frequency_high': 0.8
                                }
                            }, #end 'phased_antennas'
                'reconstruction_antennas' : { #Antennas in this should be in one vertical string/line, they are used for the phased array calculations.
                            } #end 'reconstruction_antennas'

                    }, #end 'ARA5'

                } #end 'stations'
}

