"""
Script to merge output results files.
"""

import glob
import sys
import numpy
import h5py

############################################################

if len(sys.argv) != 3:
    sys.exit('  Usage: python merge_outfiles.py <infile search string using "STAR" as wildcard> <outfile>')

infile_search = sys.argv[1].replace('STAR', '*')
outfile = sys.argv[2]

infiles = glob.glob(infile_search)

print 'Found %s infiles...'%(len(infiles))

sys.exit('DONE')

dic_attrs = {}
dic_data = {}
for infile in infiles:
    print infile

    reader = h5py.File(infile, 'r')

    for key in reader.attrs.keys():
        if key not in dic_attrs.keys():
            dic_attrs[key] = reader.attrs[key]

    for key in reader.keys():
        if key not in dic_data.keys():
            dic_data[key] = []
        dic_data[key].append(reader[key][...])

    reader.close()

print dic_attrs.keys()
print dic_data.keys()

raw_input('WAIT')

writer = h5py.File(outfile, 'w')

for key in dic_attrs.keys():
    writer.attrs[key] = dic_attrs[key]

for key in dic_data.keys():
    print key
    dic_data[key] = numpy.concatenate(dic_data[key])
    writer.create_dataset(key, (len(dic_data[key]),), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
    writer[key][...] = dic_data[key]

writer.close()

############################################################
