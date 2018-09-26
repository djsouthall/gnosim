#!/usr/bin/env python
import glob
import sys
import numpy
import h5py

infiles = glob.glob('./*.h5')
merge_cut = numpy.array([len(infile.split('merged')) > 1 for infile in infiles])
merged_files = numpy.array(infiles)[merge_cut]
infiles = numpy.array(infiles)[~merge_cut]
file_roots = numpy.unique([infile.replace(infile.split('seed')[-1],'') for infile in infiles])

for file_root in file_roots:
    print('On setup:',file_root)
    dic_attrs = {}
    dic_data = {}
    runs = glob.glob(file_root + '*.h5')
    n_events = int(runs[0].split('_')[-5])
    writer_name = file_root.replace(str(n_events)+'_events_seed',str(n_events*len(runs))+'_events_') + 'merged.h5'
    
    #print(file_root.replace(str(n_events)+'_events_seed',str(n_events*len(runs))+'_events_') + 'merged.h5')
    #print(numpy.isin(file_root.replace(str(n_events)+'_events_seed',str(n_events*len(runs))+'_events_') + 'merged.h5', merged_files))
    if numpy.isin(writer_name, merged_files):
        print(writer_name,'already exists, skipping')
        continue
    writer = h5py.File(file_root.replace(str(n_events)+'_events_seed',str(n_events*len(runs))+'_events_') + 'merged.h5', 'w')
    print('n_events per run = ',n_events)
    print('n_events total = ',n_events*len(runs))
    for run in runs:
        print('Adding run:',run)
        reader = h5py.File(run, 'r')
        for key in reader.attrs.keys():
            if key not in dic_attrs.keys():
                dic_attrs[key] = reader.attrs[key]
        for key in reader.keys():
            if key not in dic_data.keys():
                dic_data[key] = []
            dic_data[key].append(reader[key][...])
        reader.close()
    
    
    print('Writing output:',file_root.replace(str(n_events)+'_events_seed',str(n_events*len(runs))+'_events_') + 'merged.h5')
    for key in dic_attrs.keys():
        writer.attrs[key] = dic_attrs[key]

    for key in dic_data.keys():
        print('\tOn key:',key)
        dic_data[key] = numpy.concatenate(dic_data[key])
        writer.create_dataset(key, (len(dic_data[key]),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
        writer[key][...] = dic_data[key]
    print('\Attempting to close file')
    writer.close()

