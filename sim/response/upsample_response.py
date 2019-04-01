'''
Use this to upsample a response to get it to the desired time step.  For the Askaryan calculatons the time step should be ~less than
0.01ns for the calculation to yield accurate results.
'''



import sys
import numpy
import pylab
import os
pylab.ion()

def upsampleResponses(up_sample_factor,system_response_dir,antenna_response_dir, save = False):
    '''
    Calculates the times used for signal calculations based on the response functions
    (assumed to have the same frequency step).  up_sample_factor is not exact, as the
    number of points is rounded to a factor of 2 to ensure future ifft's are as fast
    as possible.
    
    This also returns the newly padded responses and freqs.

    Parameters
    ----------
    up_sample_factor : int
        Characterizes how much calculateTimes should up sample the responses.  This is no longer used, as it
        is expected that signal responses are up sampled in advance. up_sample_factor is not exact, as the
        number of points is rounded to a factor of 2 to ensure future ifft's are as fast as possible.
    system_response_dir : str
        The directory/file containing the system response. Only present if self.addTimingInfo() is run.
    antenna_response_dir : str
        The directory/file containing the antenna response. Only present if self.addTimingInfo() is run.
    save : bool, optional
        Enables saving of the up sampled response.

    Returns
    -------
    u : numpy.ndarray of floats
        The observer times for which to calculate the Askaryan radiation.  Should span both negative and positive times 
        to get the full details of the pulse.  Given in ns. 
    h_fft : numpy.ndarray of cfloats
        The values for the antenna response. (Should have units of m, i.e. effective height).
    sys_fft : numpy.ndarray of cfloats
        The values for the system response. (Should be unitless).
    freqs : numpy.ndarray of floats
        The values for the frequencies corresponding to the above responses. 
    '''
    antenna_response = numpy.load(antenna_response_dir)
    electronic_response = numpy.load(system_response_dir)
    freqs, h_fft = numpy.hsplit(antenna_response, 2)
    freqs, sys_fft = numpy.hsplit(electronic_response, 2)
    h_fft = numpy.ravel(h_fft)
    sys_fft = numpy.ravel(sys_fft)
    freqs =   numpy.absolute(numpy.ravel(freqs).astype(float))
    
    freq_step = freqs[1]-freqs[0] #1/(n_points*t_step*1e-9) #Hz
    print('1len(freqs) = ',len(freqs))
    possible_lengths = 2**numpy.arange(0,25) + 1
    n_points_freq = possible_lengths[possible_lengths >= up_sample_factor*len(h_fft)][0]  #Want 2^n events in time domain, so 2^n  #upsamples to the closest power of two to upsample*original_length
    freqs = numpy.arange(n_points_freq)*freq_step
    print('2len(freqs) = ',len(freqs))
    h_fft = numpy.append(h_fft,numpy.zeros(n_points_freq - len(h_fft)))
    sys_fft = numpy.append(sys_fft,numpy.zeros(n_points_freq - len(sys_fft)))
    #response_fft = numpy.multiply(h_fft,sys_fft)
    
    if save == True:
        system_response_out_dir = system_response_dir.replace('.npy','_new.npy')
        if os.path.isfile(system_response_out_dir):
            print('Outfile Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(system_response_out_dir))
            while os.path.isfile(system_response_out_dir):
                system_response_out_dir = system_response_out_dir.replace('.npy','_new.npy')
        print('Saving system response to', system_response_out_dir)
        numpy.save(system_response_out_dir,list(zip(freqs,sys_fft)))

        antenna_response_out_dir = antenna_response_dir.replace('.npy','_new.npy')
        if os.path.isfile(antenna_response_out_dir):
            print('Outfile Name %s is taken, saving in current directory and appending \'_new\' if necessary'%(antenna_response_out_dir))
            while os.path.isfile(antenna_response_out_dir):
                antenna_response_out_dir = antenna_response_out_dir.replace('.npy','_new.npy')
        print('Saving antenna response to', antenna_response_out_dir)
        numpy.save(antenna_response_out_dir,list(zip(freqs,h_fft)))

    t_step = 1/(2*max(freqs))*1e9 #ns
    u = numpy.arange(-(n_points_freq-1),(n_points_freq-1))*t_step #To increase time duration of signal I should just need to upsample?
    return u, h_fft, sys_fft, freqs


if __name__ == "__main__":
    pylab.close('all')
    antenna_response_dir = './gnosim/sim/response/ara_antenna_response_vpol.npy'
    system_response_dir = './gnosim/sim/response/ara_system_response_vpol.npy'

    u_old, h_fft_old, sys_fft_old, freqs_old = upsampleResponses(1,system_response_dir,antenna_response_dir,save=False)
    u_new, h_fft_new, sys_fft_new, freqs_new = upsampleResponses(40,system_response_dir,antenna_response_dir,save=True)

    #####

    sys_gain_old = numpy.absolute(sys_fft_old)**2
    sys_gain_old *= 2.0 # Because the rfft is missing half the power

    h_gain_old = numpy.absolute(h_fft_old)
    h_gain_old *= (1.74 / 3e8) * numpy.array(freqs_old)
    h_gain_old *= h_gain_old 
    h_gain_old *= 2.0 # Because the rfft is missing half the power
    h_gain_old *= 4.0 * numpy.pi
    pylab.figure()
    pylab.title('OLD')
    pylab.scatter(freqs_old, 10.0 * numpy.log10(h_gain_old), label="Antenna")
    pylab.scatter(freqs_old, 10.0 * numpy.log10(sys_gain_old), label="Elect.")
    pylab.xlabel("Freq. [Hz]")
    pylab.ylabel("Realized Gain [dBi]")
    pylab.legend()
    pylab.xlim(0, 1000e6)
    pylab.ylim(-15.0, 80.0)
    pylab.minorticks_on()
    pylab.grid(which="both")

    #####
    
    sys_gain_new = numpy.absolute(sys_fft_new)**2
    sys_gain_new *= 2.0 # Because the rfft is missing half the power

    h_gain_new = numpy.absolute(h_fft_new)
    h_gain_new *= (1.74 / 3e8) * numpy.array(freqs_new)
    h_gain_new *= h_gain_new 
    h_gain_new *= 2.0 # Because the rfft is missing half the power
    h_gain_new *= 4.0 * numpy.pi
    pylab.figure()
    pylab.title('NEW')
    pylab.scatter(freqs_new, 10.0 * numpy.log10(h_gain_new), label="Antenna")
    pylab.scatter(freqs_new, 10.0 * numpy.log10(sys_gain_new), label="Elect.")
    pylab.xlabel("Freq. [Hz]")
    pylab.ylabel("Realized Gain [dBi]")
    pylab.legend()
    pylab.xlim(0, 1000e6)
    pylab.ylim(-15.0, 80.0)
    pylab.minorticks_on()
    pylab.grid(which="both")

    