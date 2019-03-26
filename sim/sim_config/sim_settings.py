{'outfile_dir'           :   '/home/dsouthall/scratch-midway2/', #Where the data files will be saved.
'station_config_file'   :   '/home/dsouthall/Projects/GNOSim/gnosim/detector/station_config/real_config_antarctica_180_rays_signed_fresnel.py', #The address of the station configuration file
'solutions'             :   ['direct', 'cross', 'reflect'],#The solutions to work with in the simulation #numpy.array(['direct', 'cross', 'reflect', 'direct_2', 'cross_2', 'reflect_2'])
'image_extension'       :   'svg',  #The file type for any saved images.
'image_path_root'       :   '/home/dsouthall/public_html/images/', #The location of the directory for which a new subfolder with the name of the simulation will be created for output images.
'electricFieldDomain'   :   'time', #Selects the Askaryan calculation type.  Either 'time' (highly recommended) or 'freq'.  'freq' is not fully supported and outdated, will use the frequency domain calculation that was in early builds of GNOSim rather than the new time domain calculation.
'do_beamforming'        :   True,   #Enables a beamforming trigger, v.s. a normal per-antenna threshold.
'include_noise'         :   True,   #Enables the inclusion of noise.
'summed_signals'        :   True,   #Enables signals to be joined together (i.e. a direct and reflected pulses will be read out in same time stream).
'pre_split'             :   True,   #Enables loading of ray tracing libraries to occur from pre split directories.
'plot_geometry'         :   False,  #Enables plotting of the rays connecting antennas and the neutrino for triggered event.  Slows down simulation, cannot be enabled when multithreading is enabled.
'plot_signals'          :   False,  #Enables plotting of the waveforms for triggered event.  Slows down simulation, cannot be enabled when multithreading is enabled.
'trigger_threshold'     :   11500,  #The threshold, with units specified below.  
'trigger_threshold_units' : 'fpga', #The units for the trigger.  See gnosim.antarcticsim.getAcceptedTriggerUnits()
'use_interp_threading'  :   True,   #Enables multithreading during interpolation (calculations are run for all events at once in arrays for interpolation, however this multithreads how many antennas are being run simulataneously).
'use_event_threading'   :   True,   #Enables multithreading during event by event calculations (each event is run in a seperate thread).
'output_all_solutions'  :   True,   #Enables the output of metadata about each solution type for each antenna for each event, rather than just the solution type with the max field strength per antenna.
'save_signals'          :   False,  #Enables the saving of the waveforms to the output file.
'pre_trigger_angle'     :   10.0    #Sets a pre-trigger angle.  See docstrings in antarcticsim for more information.
}


