#!/usr/bin/env python3
'''
Purpose and Description
-----------------------
This script allows you to submit many simulations to midway at once over a range of energies.  

Running
-------
It is recommended that you run this in a command line interface like ipython with the %run command or using the 
command exec(open('./script_path/script.py').read()) .

Main Parameters
---------------
username : str
    The username to use when submittng jobs to midway.
cpu_per_task : int
    This sets the --cpu_per_task value in the sbatch.  If you are using multithreading it is recommended to use a high
    number of cpus.  You should also request enough mem_per_cpu for the total amount of memory to be sufficient to run
    the simulation.  Common values used in development: 16 
mem_per_cpu :
    This sets the --mem_per_cpu value in the sbatch.  If you are using multithreading it is recommended to use a high
    number of cpus.  You should also request enough mem_per_cpu for the total amount of memory to be sufficient to run
    the simulation.  Common values used in development: 1500
partition : str
    The partition of midway you want to run the simulations on.
jobname : str
    The name of the job that will show up on squeue/midway.
n_events : int
    The number of events to throw for each simulation.  Common values used in development: 1000000
simulation_config_file_array : list of str
    A list of all config files to use.  Simulations will be run for each config file, allowing for multiple
    configurations to be run in succession with ease.
energy_neutrino_array : numpy.ndarray of floats
    The energies of neutrinos for which to run each simulation.  A simulation will be run n_trials number of times 
    for each energy in this array (all of which will be repeated for each simulation config file in the above list).
n_trials : numpy.ndarray of int
    The number of trials to run for each energy.  For high energies usually a single trial is sufficient, but for low energies
    (energy_neutrino < 1e8 #GeV) then you may need more trials to get sufficient stats for calculations.  
    Common values used in development: 1 for energy_neutrino >= 1e8 #GeV, 5 for energy_neutrino < 1e8 #GeV
use_seed : bool
    Enables the use of seeded simulations.  The seeds are set by seeds.  Internally in antarcticsim this seed is applied to the
    simulation when throw is called.  Then throw will generate seeds for each event which are then applied locally at the beginning of
    each event calculation.  This allows for accurate reproduction of random things such as noise per event, and also avoids confusion
    in ordering of calling the random variable when multithreading.  Note that each event will still be seeded if use_seed is set to false
    but the thing generating those seeds will not be seeded, so the events would have different seeds between simulations (though still have
    tracked seeds so the events can be indpendently reproduced).
seeds : numpy.ndarray of int
    The seeds to use for each trial.  Note that these should be different for each trial of a particular energy (such that you don't run the
    EXACT same soulation n_trials number of times), but should be consistently applied across all energies such that results are directly
    comparable.  
'''

import os
import subprocess
import time
import numpy
import yaml


if __name__ == "__main__":

    ###------------###
    ### Parameters ###
    ###------------###

    username = 'dsouthall'
    cpu_per_task = 16
    mem_per_cpu = 1500 #1000 = 1GB, looks like the total MaxRSS for 1M events was 13GB total, so 2000-3000 per cpu for that.  
    partition = 'broadwl'
    jobname = 'gnosim'

    n_events = 1000000
    simulation_config_file_array = ['/home/dsouthall/Projects/GNOSim/gnosim/sim/sim_config/sim_settings.py'] #List of all config files to use.  Simulations will be run for each config file.
    energy_neutrino_array =  10**numpy.arange(7., 12.1, 0.5)#10**numpy.arange(6.5, 11.1, 0.5) # GeV
    
    n_trials = numpy.ones(len(energy_neutrino_array),dtype=int) #should be same length as energy_neutrino.  Can allow for different number of trials for different energies.
    n_trials[energy_neutrino_array < 1e8] = 5 #Running more trials for lower energy events for more statistics
    
    use_seed = True
    seeds = numpy.arange(max(n_trials))#[0,1,2,3,4]#IF USING SEEDS THERE NEEDS TO BE n_trials DIFFERENT SEEDS SO YOU DON'T RUN THE SAME SEED MULTIPLE TIMES

    ###--------###
    ### Script ###
    ###--------###

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
                
