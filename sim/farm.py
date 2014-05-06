import os
import numpy

config_file_array = ['config_simple_-2.py',
                     'config_simple_-30.py',
                     'config_simple_-100.py']
#energy_neutrino_array = 10**numpy.arange(7., 12, 0.5) # GeV
energy_neutrino_array = 10**numpy.array([9.]) # GeV
n_events = 100000
n_trials = 10
jobname = 'gnosim'

for config_file in config_file_array:
    for energy_neutrino in energy_neutrino_array:
        for index in range(0, n_trials):
            batch = 'sbatch --account=kicp --partition=kicp-ht --job-name=%s '%(jobname)
            command = 'sim.py %s %.2e %i %i'%(config_file, energy_neutrino, n_events, index)
            command_queue = batch + command
            print command
            #print '(%i/%i) %s'%(ii, len(files), command_queue)
            #os.system('./' + command) # Run locally
            os.system(command_queue) # Submit to queue

            #break
