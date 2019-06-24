#!/usr/bin/env python3

'''
This script is useful for when you want to generate multiple ray tracing libraries at once by farming them
out to midway.
'''

import sys
import subprocess
import os
import numpy


if __name__ == '__main__':
    #z_array = numpy.array([-13.0,-14.0,-15.0,-16.0,-17.0,-18.0,-19.0,-20.0,-13.0 - 20.0,-14.0 - 20.0,-15.0 - 20.0,-16.0 - 20.0,-17.0 - 20.0,-18.0 - 20.0,-19.0 - 20.0,-20.0 - 20.0,-13.0 - 40.0,-14.0 - 40.0,-15.0 - 40.0,-16.0 - 40.0,-17.0 - 40.0,-18.0 - 40.0,-19.0 - 40.0,-20.0 - 40.0,-13.0 - 60.0,-14.0 - 60.0,-15.0 - 60.0,-16.0 - 60.0,-17.0 - 60.0,-18.0 - 60.0,-19.0 - 60.0,-20.0 - 60.0,-13.0 - 80.0,-14.0 - 80.0,-15.0 - 80.0,-16.0 - 80.0,-17.0 - 80.0,-18.0 - 80.0,-19.0 - 80.0,-20.0 - 80.0]) #The list of depths for which to throw rays (or load libraries if make_library == False).
    z_array = numpy.unique(numpy.array([-173.0-0.0, -173.0-1.0, -173.0-2.0, -173.0-3.0, -173.0-4.0, -173.0-6.0, -173.0-8.0, -173.0-7.0, -173.0-27.0, -173.0-7.0, -173.0-27.0, -173.0-7.0, -173.0-27.0, -173.0-7.0, -173.0-27.0])) #The list of depths for which to throw rays (or load libraries if make_library == False).
    username = 'dsouthall'
    partition = 'broadwl'
    jobname = 'gnolib'

    ###--------###
    ### Script ###
    ###--------###

    for z in z_array:
        batch = 'sbatch --partition=%s --job-name=%s '%(partition, jobname)
        command = os.environ['GNOSIM_DIR'] + '/gnosim/trace/refraction_library.py %f'%(z)
        command_queue = batch + command
        print(command)
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