import os
import subprocess
import time
import numpy

username = 'bechtol'

config_file_array = ['config_simple_6000_ross.py']
#config_file_array = ['config_simple_-2.py',
#                     'config_simple_-30.py',
#                     'config_simple_-100.py']
"""
config_file_array = ['config_simple_38000.py',
                     'config_simple_5000.py',
                     'config_simple_1000.py',
                     'config_simple_0.py',
                     'config_simple_-2.py',
                     'config_simple_-30.py',
                     'config_simple_-100.py']
"""
energy_neutrino_array = 10**numpy.arange(7., 12.1, 0.5) # GeV
#energy_neutrino_array = 10**numpy.array([9.]) # GeV
n_events = 100000
n_trials = 10 # 1, 10
jobname = 'gnosim'

for config_file in config_file_array:
    for energy_neutrino in energy_neutrino_array:
        for index in range(0, n_trials):
            batch = 'sbatch --account=kicp --partition=kicp-ht --job-name=%s '%(jobname)
            command = 'sim.py %s %.2e %i %i'%(config_file, energy_neutrino, n_events, index)
            command_queue = batch + command
            print command

            outfile = 'results_2014_aug_7/%s_%.2e_GeV_%i_events_%i.h5'%(config_file.replace('.py', ''),
                                                                        energy_neutrino,
                                                                        n_events,
                                                                        index)
            if os.path.exists(outfile):
                print 'Outfile %s already exists, skipping...'%(outfile)
                continue

            #print '(%i/%i) %s'%(ii, len(files), command_queue)
            #os.system('./' + command) # Run locally
            os.system(command_queue) # Submit to queue

            #break
            
            # Avoid overwhelming the queue with jobs
            while True:

                # Clean up log files
                n_output = subprocess.Popen('ls slurm*.out | wc', shell=True, 
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0]
                if n_output.isdigit():
                    os.system('rm slurm*.out')

                n_submitted = int(subprocess.Popen('squeue -u %s | wc\n'%username, shell=True, 
                                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate()[0].split()[0]) - 1
                
                # Check to see whether to enter holding pattern
                if n_submitted < 200:
                    break
                else:
                    print '%i jobs already in queue, waiting ...'%(n_submitted), time.asctime(time.localtime())
                    time.sleep(60)
            
