#!/usr/bin/env python3
import glob
import sys
import numpy
import h5py
import time

infiles = glob.glob('./*.h5')
merge_cut = numpy.array([len(infile.split('merged')) > 1 for infile in infiles])
merged_files = numpy.array(infiles)[merge_cut]
infiles = numpy.array(infiles)[~merge_cut]
file_roots = numpy.unique([infile.replace(infile.split('_events')[-1],'') for infile in infiles])
sys.stdout.flush()

for file_root in file_roots:
    print('On setup:',file_root)
    sys.stdout.flush()
    dic_attrs = {}
    dic_data = {}
    runs = glob.glob(file_root + '*.h5')
    n_events = int(runs[0].split('_')[-5])
    writer_name = file_root.replace(str(n_events)+'_events_seed',str(n_events*len(runs))+'_events_') + '_merged.h5'
    
    #print(file_root.replace(str(n_events)+'_events_seed',str(n_events*len(runs))+'_events_') + 'merged.h5')
    #print(numpy.isin(file_root.replace(str(n_events)+'_events_seed',str(n_events*len(runs))+'_events_') + 'merged.h5', merged_files))
    if numpy.isin(writer_name, merged_files):
        print(writer_name,'already exists, skipping')
        sys.stdout.flush()
        continue
    writer = h5py.File(writer_name, 'w')
    print('n_events per run = ',n_events)
    sys.stdout.flush()
    print('n_events total = ',n_events*len(runs))
    sys.stdout.flush()
    dtypes = {}
    print('Will try adding the following runs:')
    sys.stdout.flush()
    print(runs)
    sys.stdout.flush()
    for index, run in enumerate(runs):
        print('Run:',run)
        sys.stdout.flush()
    for index, run in enumerate(runs):
        print('Adding run:',run)
        sys.stdout.flush()
        reader = h5py.File(run, 'r')
        for key in reader.attrs.keys():
            out_key = key + '_' + str(index)
            if out_key not in dic_attrs.keys():
                writer.attrs[out_key] = reader.attrs[key]
        for key in reader.keys():
            #TODO: if only one infile (or if just the first?), this should be done differently, as the element by element adjustments are not needed.
            print('\tOn key:',key)
            sys.stdout.flush()
            if key == 'signals':
                if key not in list(writer.keys()):
                    writer.create_group('signals')
            elif key not in list(writer.keys()):
                writer.create_dataset(key, (len(reader[key][...])*len(runs),), dtype=reader[key][...].dtype, compression='gzip', compression_opts=9, shuffle=True)
            if key == 'info':
                len_key = len(reader[key][...])
                temp_info = reader[key][...]
                temp_info['eventid'] += index*n_events #this is right btw
                left = index*len_key
                right = (index+1)*len_key
                writer[key][left:right] = temp_info
            elif key == 'signals':
                for signal_key in list(reader['signals'].keys()):
                    event_label = signal_key.replace(signal_key.split('event')[-1], str(int(signal_key.split('event')[-1]) + index*n_events)) #reader['signals'][signal_keys]
                    print('\t\t' + event_label)
                    sys.stdout.flush()
                    writer['signals'].create_group(event_label)
                    for station_label in list(reader['signals'][signal_key].keys()):
                        writer['signals'][event_label].create_dataset(station_label, numpy.shape(reader['signals'][signal_key][station_label][...]), dtype='f', compression='gzip', compression_opts=9, shuffle=True)  
                        writer['signals'][event_label][station_label][...] = reader['signals'][signal_key][station_label][...]
            else:
                len_key = len(reader[key][...])
                left = index*len_key
                right = (index+1)*len_key
                writer[key][left:right] = reader[key][...]
        reader.close()
    
    '''
    print('Writing output:',file_root.replace(str(n_events)+'_events_seed',str(n_events*len(runs))+'_events_') + 'merged.h5')
    for key in dic_attrs.keys():
        writer.attrs[key] = dic_attrs[key]

    for key in dic_data.keys():
        print('\tOn key:',key)
        if key == 'signals':
            pass
        else:
            start_time = time.time()
            dic_data[key] = numpy.concatenate(dic_data[key])
            writer.create_dataset(key, (len(dic_data[key]),), dtype=dtypes[key], compression='gzip', compression_opts=9, shuffle=True)
            writer[key][...] = dic_data[key]
    print('\Attempting to close file')
    '''
    print('Closing writer!')
    sys.stdout.flush()
    writer.close()
    print('Finished!')
    sys.stdout.flush()
    
'''
self.file['signals'].create_group(event_label)
for index_station in range(self.config['stations']['n']):
    station_label = 'station%i'%index_station
    self.file['signals'][event_label].create_dataset(station_label, numpy.shape(signals_out[station_label]), dtype='f', compression='gzip', compression_opts=9, shuffle=True)  
    self.file['signals'][event_label][station_label][...] = signals_out[station_label]

'''

