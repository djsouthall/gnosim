import os
import subprocess
import time
import numpy
import yaml

#Parameters
#----------
username = 'dsouthall'

simulation_config_file_array = ['/home/dsouthall/Projects/GNOSim/gnosim/sim/sim_config/sim_settings_old_dipole.py'] #List of all config files to use.  Simulations will be run for each config file.
energy_neutrino_array =  10**numpy.arange(7., 12.1, 0.5)#10**numpy.arange(6.5, 11.1, 0.5) # GeV
cpu_per_task = 16
mem_per_cpu = 1500 #1000 = 1GB, looks like the total MaxRSS for 1M events was 13GB total, so 2000-3000 per cpu for that.  
partition = 'broadwl'
n_events = 1000000
n_trials = numpy.ones(len(energy_neutrino_array),dtype=int) #should be same length as energy_neutrino.  Can allow for different number of trials for different energies.
n_trials[0:2] = 5
use_seed = True
seeds = numpy.arange(max(n_trials))#[0,1,2,3,4]#IF USING SEEDS THERE NEEDS TO BE n_trials DIFFERENT SEEDS SO YOU DON'T RUN THE SAME SEED MULTIPLE TIMES
jobname = 'gnosim'

#Script
#------
for sim_config_file in simulation_config_file_array:
    sim_config = yaml.load(open(sim_config_file))
    station_config_file = sim_config['station_config_file']
    station_config_file_fix = station_config_file.split('/')[-1].replace('.py','')
    if use_seed:
        if len(seeds) != max(n_trials):
            print('WARNING, NUMBER OF TRIALS EXCEEDS SEEDS PROVIDED SKIPPING EVERYTHING')
            break
    for index_energy, energy_neutrino in enumerate(energy_neutrino_array):
        for index, index_name in enumerate(range(1, n_trials[index_energy]+1)):
            batch = 'sbatch --partition=%s --cpus-per-task=%i --mem-per-cpu=%i --job-name=%s '%(partition,cpu_per_task,mem_per_cpu, jobname)
            if use_seed == True:
                command = '/home/dsouthall/Projects/GNOSim/gnosim/sim/antarcticsim.py %s %.2e %i %i %i'%(sim_config_file, energy_neutrino, n_events, index_name, seeds[index])
            else:
                command = '/home/dsouthall/Projects/GNOSim/gnosim/sim/antarcticsim.py %s %.2e %i %i'%(sim_config_file, energy_neutrino, n_events, index_name)
            command_queue = batch + command
            print(command)

            outpath = '/home/dsouthall/scratch-midway2/'

            #The formatting below should match that used in ANTARCTICSIM!
            if use_seed == True:
                outfile = '%s_%s_%.2e_GeV_%i_events_seed_%i_%i.h5'%(  sim_config['outfile_name_root'],
                                                                      station_config_file_fix,
                                                                      energy_neutrino,
                                                                      n_events,
                                                                      seeds[index],
                                                                      index_name)
            else:
                outfile = '%s_%s_%.2e_GeV_%i_events_%i.h5'%(  sim_config['outfile_name_root'],
                                                              station_config_file_fix,
                                                              energy_neutrino,
                                                              n_events,
                                                              index_name)
  
            
            if os.path.exists(outpath+outfile):
                print('Outfile %s already exists, skipping...'%(outfile))
                continue

            #print '(%i/%i) %s'%(ii, len(files), command_queue)
            #os.system('/home/dsouthall/Projects/GNOSim/gnosim/sim/' + command) # Run locally
            os.system(command_queue) # Submit to queue

            #break
            
            # Avoid overwhelming the queue with jobs
            while False:

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
                    print('%i jobs already in queue, waiting ...'%(n_submitted), time.asctime(time.localtime()))
                    time.sleep(60)
            
