import glob
import h5py
import numpy
infiles = glob.glob('./*h5')
snrs = [float(infile.split('_')[-1].replace('.h5','').replace('p','.')) for infile in infiles]
infiles= numpy.array(infiles)[numpy.argsort(snrs)]
snrs = numpy.sort(snrs)
depths = -numpy.array([float(infile.split('_')[4].replace('.h5','').replace('p','.')) for infile in infiles])
overlap = []
for snr in numpy.unique(snrs):
    snrcut = numpy.where(snrs == snr)[0]
    depthsort = numpy.argsort(depths)
    overlap.append( depthsort[numpy.isin(depthsort,snrcut)])
infiles = infiles[numpy.concatenate(overlap)]
extra_text = 'BW-450-750MHz-oct22'#'BW-50-200MHz'#
header = 'Depth, Gain (dBi), SNR, Neutrino Energy (GeV), VA (km^3 sr), VA Error (km^3 sr), Threshold (mV/m)'
print(header)
keys = ['energy_neutrino','volumetric_acceptance', 'error', 'mean_electric_threshold']
big_csv = open('./all_VA_data%s.csv'%('-'+extra_text),'w')
big_csv.write(header+'\n')
for infile in infiles:
    csv = open(infile.replace('.h5','-'+extra_text+'.csv'),'w')
    reader = h5py.File(infile, 'r')
    snr = float(infile.split('_')[-1].replace('.h5','').replace('p','.'))
    csv.write(header+'\n')
    energy_sort = numpy.argsort(reader['energy_neutrino'])
    for i in energy_sort:#range(len(reader[list(reader.keys())[0]])):
        row = [reader.attrs['z_0'],reader.attrs['gain'],snr]
        for key in keys:
            if key == 'mean_electric_threshold':
                row.append(1000*reader[key][i])
            else:
                row.append(reader[key][i])
        row = str(row).replace('[','').replace(']','')+'\n'
        print(row)
        csv.write(row)
        big_csv.write(row)
    reader.close()
    csv.close()
big_csv.close()
