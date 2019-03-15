import os
import subprocess
import time
import numpy

'''
#If you need to only redo some of the sims, put them in this list
only_do = numpy.array(['results_2018_Aug_config_simple_-100_low_200MHz_high_800MHz_1.00e+09_GeV_5000000_events_seed_3_4.h5',\
                       'results_2018_Aug_config_simple_-100_low_200MHz_high_800MHz_1.00e+10_GeV_5000000_events_seed_0_1.h5',\
                       'results_2018_Aug_config_simple_-100_low_200MHz_high_800MHz_1.00e+10_GeV_5000000_events_seed_1_2.h5',\
                       'results_2018_Aug_config_simple_-100_low_200MHz_high_800MHz_1.00e+10_GeV_5000000_events_seed_2_3.h5',\
                       'results_2018_Aug_config_simple_-100_low_200MHz_high_800MHz_1.00e+10_GeV_5000000_events_seed_3_4.h5'])
#If you need to skip some of the sims, put them in this list
dont_do = numpy.array(['results_2018_Aug_config_simple_-2_low_50MHz_high_200MHz_1.00e+05_GeV_5000000_events_seed_0_1.h5',\
                       'results_2018_Aug_config_simple_-2_low_50MHz_high_200MHz_3.16e+04_GeV_5000000_events_seed_0_1.h5',\
                       'results_2018_Aug_config_simple_-2_low_50MHz_high_200MHz_3.16e+04_GeV_5000000_events_seed_2_3.h5',\
                       'results_2018_Aug_config_simple_-2_low_50MHz_high_200MHz_3.16e+04_GeV_5000000_events_seed_1_2.h5',\
                       'results_2018_Aug_config_simple_-2_low_50MHz_high_200MHz_1.00e+04_GeV_5000000_events_seed_0_1.h5',\
                       'results_2018_Aug_config_simple_-2_low_50MHz_high_200MHz_1.00e+04_GeV_5000000_events_seed_2_3.h5',\
                       'results_2018_Aug_config_simple_-2_low_50MHz_high_200MHz_1.00e+04_GeV_5000000_events_seed_1_2.h5',\
                       'results_2018_Aug_config_simple_-2_low_50MHz_high_200MHz_1.00e+04_GeV_5000000_events_seed_3_4.h5'])

'''
username = 'dsouthall'
'''
config_file_array = ['/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_simple_-1_polar_BW_160-600MHz.py',\
                     '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_simple_-1_polar_BW_450-750MHz.py',\
                     '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_simple_-2_polar_BW_160-600MHz.py',\
                     '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_simple_-2_polar_BW_450-750MHz.py']
'''
'''
config_file_array = ['/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_simple_-1_polar_BW_450-750MHz.py',\
                     '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_simple_-2_polar_BW_450-750MHz.py']
'''

config_file_array = ['/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/real_config.py']
'''
config_file_array = ['/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/real_config.py',\
                     '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_dipole_octo_-200_polar_120_rays.py']
'''
#energy_neutrino_array = 10**numpy.arange(4., 4.4, 0.5) # GeV

#energy_neutrino_array = 10**numpy.arange(7., 12.1, 0.5) # GeV
#energy_neutrino_array = 10**numpy.array([9.]) # GeV
#energy_neutrino_array = numpy.array([1e7,1e10])#numpy.rint(10**numpy.arange(4,10.5,0.5)) # GeV
energy_neutrino_array =  10**numpy.arange(7., 12.1, 0.5)#10**numpy.arange(6.5, 11.1, 0.5) # GeV
cpu_per_task = 16
mem_per_cpu = 1500 #1000 = 1GB, looks like the total MaxRSS for 1M events was 13GB total, so 2000-3000 per cpu for that.  
n_events = 1000000
n_trials = 1 # 1, 10
use_seed = True
seeds = numpy.arange(n_trials)#[0,1,2,3,4]#IF USING SEEDS THERE NEEDS TO BE n_trials DIFFERENT SEEDS SO YOU DON'T RUN THE SAME SEED MULTIPLE TIMES
jobname = 'gnosim'

#sinteracsinteractive  --time=08:00:00tive --cpus-per-task=3 --time=08:00:00

        
for config_file in config_file_array:
    if use_seed:
        if len(seeds) != n_trials:
            print('WARNING, NUMBER OF TRIALS EXCEEDS SEEDS PROVIDED SKIPPING EVERYTHING')
            break
    for energy_neutrino in energy_neutrino_array:
        for index, index_name in enumerate(range(1, n_trials+1)):
            batch = 'sbatch --cpus-per-task=%i --mem-per-cpu=%i --job-name=%s '%(cpu_per_task,mem_per_cpu, jobname)
            if use_seed == True:
                command = '/home/dsouthall/Projects/GNOSim/gnosim/sim/antarcticsim.py %s %.2e %i %i %i'%(config_file, energy_neutrino, n_events, index_name, seeds[index])
            else:
                command = '/home/dsouthall/Projects/GNOSim/gnosim/sim/antarcticsim.py %s %.2e %i %i'%(config_file, energy_neutrino, n_events, index_name)
            command_queue = batch + command
            print(command)

            outpath = '/home/dsouthall/scratch-midway2/'
            if use_seed == True:
                outfile = 'results_2019_Mar_%s_%.2e_GeV_%i_events_seed_%i_%i.h5'%(config_file.split('/')[-1].replace('.py',''),
                                                                            energy_neutrino,
                                                                            n_events,
                                                                            seeds[index],
                                                                            index_name)
            else:
                outfile = 'results_2019_Mar_%s_%.2e_GeV_%i_events_%i.h5'%(config_file.split('/')[-1].replace('.py',''),
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
            
