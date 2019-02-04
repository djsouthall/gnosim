import sys
import numpy
import matplotlib
import pylab
pylab.ion()



if __name__ == "__main__":
    #pylab.close('all')
    # Antenna response
    file_ant = numpy.load("ara_antenna_response_v6.npy")

    # Electronics response
    file_elect = numpy.load("ara_system_response_v6.npy")

    freqs, h_fft = numpy.hsplit(file_ant, 2)
    freqs, sys_fft = numpy.hsplit(file_elect, 2)
    h_fft = numpy.ravel(h_fft)
    sys_fft = numpy.ravel(sys_fft)
    freqs = numpy.ravel(freqs)

    print(len(h_fft))

    sys_gain = numpy.absolute(sys_fft)**2
    sys_gain *= 2.0 # Because the rfft is missing half the power

    h_gain = numpy.absolute(h_fft)
    h_gain *= (1.74 / 3e8) * numpy.array(freqs)
    h_gain *= h_gain 
    h_gain *= 2.0 # Because the rfft is missing half the power
    h_gain *= 4.0 * numpy.pi


    pylab.figure(figsize=(16.,11.2))
    pylab.plot(freqs, 10.0 * numpy.log10(h_gain), label="Antenna", color = 'limegreen')
    pylab.plot(freqs, 10.0 * numpy.log10(sys_gain), label="Elect.", color = 'darkorange')
    pylab.xlabel("Freq. [Hz]",fontsize=16)
    pylab.ylabel("Realized Gain [dBi]",fontsize=16)
    pylab.legend(fontsize=14)
    pylab.xlim(0, 1000e6)
    pylab.ylim(-75.0, 75.0)
    pylab.minorticks_on()
    pylab.grid(which="both")
    #pylab.show()

    h = numpy.fft.irfft(h_fft)
    sys = numpy.fft.irfft(sys_fft)

    t = [i / 1.5 for i in range(len(h))]
    pylab.figure(figsize=(16.,11.2))
    pylab.plot(t, numpy.fft.fftshift(h))
    pylab.xlabel("Time [ns]")
    pylab.ylabel("Signal [m]") # For h, its actually in meters, unless the fft gives it trouble
    #pylab.show()

