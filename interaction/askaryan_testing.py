"""
Practical and Accurate Calculations of Askaryan Radiation
Source: Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283
"""

import numpy
import pylab
import sys
import numpy
import h5py
import pylab
import json
import yaml
import os
import os.path
import glob
import scipy
import scipy.signal

sys.path.append("/home/dsouthall/Projects/GNOSim/")

import gnosim.utils.constants
import gnosim.interaction.inelasticity
import gnosim.utils.quat
import gnosim.earth.earth
import gnosim.earth.antarctic
import gnosim.trace.refraction_library_beta
import gnosim.interaction.askaryan
import gnosim.sim.detector
pylab.ion()

############################################################

def RA(Energy_GeV,t_ns):
    '''
    Ideally this would be a function of index of refraction as well, right now
    this model is parameterized specifically for n = 1.78
    I beliee this should return units of V s (i.e. it is disgned to output SI,
    not something in terms of ns)
    '''
    Energy_TeV = 0.001*Energy_GeV
    if numpy.size(t_ns) == 1:
        if t_ns > 0:
            return (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns)/0.057) + (1. + 2.87*numpy.fabs(t_ns))**(-3.0))
        else:
            return (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns)/0.030) + (1. + 3.05*numpy.fabs(t_ns))**(-3.5)) 
    else:
        ra = numpy.zeros_like(t_ns)
        ra[t_ns > 0] = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns > 0])/0.057) + (1. + 2.87*numpy.fabs(t_ns[t_ns > 0]))**(-3.0))
        ra[t_ns <= 0] = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns <= 0])/0.030) + (1. + 3.05*numpy.fabs(t_ns[t_ns <= 0]))**(-3.5)) 
        return ra


def Q(x,a=4.85,b=0.60,loc=0,scale=1.0,random_params=False):
        '''
        b is included to make the python variant of the gamma function align with
        that defined in: 
        Coherent radio pulses from GEANT generated electromagnetic showers in ice  -  Phys. Rev. D 65, 103002
        The mean a an b values they found are a = 4.85, b = 0.60 with sig_a = 1.01, sig_b = 0.14
        Perhaps we would want to pull randomly from gaussion distros for each of 
        these params for each event. 
        
        Note random_params = True it will overwrite any a or b passed to the function. 
        
        '''
        if random_params:
            a = numpy.random.normal(loc=4.85,scale=1.01,size=None)
            b = numpy.random.normal(loc=0.60,scale=0.14,size=None)
        return b * scipy.stats.gamma.pdf( b * x , a = a , loc = loc , scale = scale )
        
def excessProjectedTrackLength(Q,int_min=-100.,int_max=100.,n_steps = 1000):
    '''
    integration_length in meters.
    This is meant to go from Q to LQ following the paper Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283
    for motivating LQ, and the paper Phys. Rev. D 65, 103002 for the shape of Q.
    
    In the paper this was normalized to 1.  Still trying figure out what this
    should actually be normalized to.  The scaling may not matter as Q is ultimately
    is in the numerator of a convolution and LQ is in the denomenator.  So the scaling
    may be already accounted for by that division.  
    '''
    LQ,error = scipy.integrate.quad(Q,int_min,int_max)
    return LQ
    
def F_p(Energy_GeV,t_ns,n,LQ):
    '''
    This is the form factor as in Eq15 of Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283
    This is to be calculated at the cherenkov angle (which in the paper was for n=1.78)
    And then convolved with the charge profile Q to determine the vector potential. 
    '''
    
    cherenkov_angle = numpy.arccos(1./n)
    #mu_0 = gnosim.utils.constants.mu_0 # m kg s^-2 A^-2
    prefactor = 4. * numpy.pi /( gnosim.utils.constants.mu_0 * numpy.sin(cherenkov_angle))
    ra = RA(Energy_GeV,t_ns)  
    #print(ra)
    return prefactor * ra / LQ
    
def vector_Potential(theta_obs_rad,R,Energy_GeV,n,plot = False,u_steps = 50000):
    '''
    This should do it all?
    '''
    cherenkov_angle = numpy.arccos(1./n)
    LQ = excessProjectedTrackLength(Q)
    alpha = (1. - n*numpy.cos(theta_obs_rad))/gnosim.utils.constants.speed_light #scaling factor of u substitution units of ns/m
    #print('alpha %f\t theta %f\ttheta/theta_c %f'%(alpha,numpy.rad2deg(theta_obs_rad),numpy.rad2deg(theta_obs_rad)/numpy.rad2deg(cherenkov_angle)))
    #u isa time variable with units in ns.  It is a u substitution variable for the convolution in Eq 17. 
    u_min = -150.
    u_max = 150.
    #u_steps = 100000
    
    u = numpy.linspace(u_min,u_max,u_steps)
    u_step = abs(u[1]-u[0])
    
    fp = F_p(Energy_GeV,u,n,LQ)
    fp = numpy.multiply(scipy.signal.tukey(len(fp),alpha=0.05),fp)
    fp = numpy.pad(fp,pad_width=int(len(fp)/2),mode='constant')
    #if alpha == 0:
    if abs(alpha) < 0.001:
        A = ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) * LQ * fp / (4. * numpy.pi * R) ) 
        u = numpy.linspace(-(len(fp)/2)*u_step,(len(fp)/2)*u_step,len(fp))
        u_step = abs(u[1]-u[0])
    else:
        #For calculation Q(u/alpha) below here is my explaination:
        #The scale factor is added to modify this function by scaling the z' input
        #such that the convolution described in Eq17 of arXiv:1106.6283.
        #Essentially in trying to understand how to implement the convolution described
        #by that equation I decided a u substitution had to be done where 
        #u = z' * alpha where alpha = (1-n*cos(theta))/c.  
        #The convolution is then 1/alpha INT du Q(u/alpha)*Fp(dt - u).  The scale
        #factor is meant to account for the scaling in Q to make it appear more
        #like a convolution: 1/alpha INT du Q'(u)*Fp(dt - u), where Q' scales u. 
        #The scaling factor not being one implies the input x is some value of ns
        #that was converted from meters using the scale factor.
        
        q = Q(u/alpha)
        q = numpy.multiply(scipy.signal.tukey(len(q),alpha=0.05),q)
        q = numpy.pad(q,pad_width=int(len(q)/2),mode='constant')
        
        u = numpy.linspace(-(len(fp)/2)*u_step,(len(fp)/2)*u_step,len(fp))
        u_step = abs(u[1]-u[0])
        
        #freq_step = 1./u_step
        #freq = numpy.linspace(-(len(fp)/2)*freq_step,(len(fp)/2)*freq_step,len(fp))
        #freq2 = numpy.fft.fftfreq(len(fp),d=u_step)
        #freq3 = numpy.fft.fftfreq(len(fp),d=u_step*1e-9)

        fourier_fp = scipy.fftpack.fft(fp)
        fourier_q = scipy.fftpack.fft(q)
        '''
        pylab.figure()
        pylab.subplot(311)
        pylab.plot(freq,fourier_fp,label='freq')
        pylab.legend()
        pylab.subplot(312)
        pylab.plot(freq2,fourier_fp,label='freq2')
        pylab.legend()
        pylab.subplot(313)
        pylab.plot(freq3,fourier_fp,label='freq3')
        pylab.legend()
        pylab.figure()
        '''
        
        if plot == True:
            pylab.figure()
            
            pylab.subplot(211)
            pylab.plot(u,fp,label='fp')
            pylab.legend()
            pylab.subplot(212)
            pylab.title('alpha = %0.3f, $\\theta$ = %0.2f deg'%(alpha,numpy.rad2deg(theta_obs_rad)))
            pylab.plot(u,q,label='q')
            pylab.legend()
            
            pylab.figure()
            pylab.subplot(211)
            pylab.plot(freq,fourier_fp,label='fourier_fp')
            pylab.legend()
            pylab.subplot(212)
            pylab.plot(freq,fourier_q,label='fourier_q')
            pylab.legend()
        
        convo = numpy.multiply(fourier_fp,fourier_q)
        
        inverse_convo = scipy.fftpack.fftshift(scipy.fftpack.ifft(convo))
        A = ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) / (4. * numpy.pi * R ) ) * (inverse_convo / ( abs(alpha) * len(inverse_convo) / (max(u) - min(u)))) #abs alpha because in the u sub the bounds swap when alpha < 0, but then alpha <0 so the two negatives cancel.
        #print(alpha)
        #A = ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) * LQ * fp / (4. * numpy.pi * R) )
    #print('theta  =  %0.2f \t max(abs(A))  =  %g \t sum(A)  =  %g'%(numpy.rad2deg(theta_obs_rad),max(abs(A)),sum(A)*(u_step)))
    return A,u#q,fp,integral,fourier_fp,fourier_q

def electricField(theta_obs_rad,R,Energy_GeV,n):
    '''
    '''
    A,u = vector_Potential(theta_obs_rad,R,Energy_GeV,n)
    E = - numpy.divide(numpy.gradient(A),numpy.gradient(u)) * 1e9
    return  E , u

def electricToVoltage(u,E,gain):
    '''
    This is probably wrong and is just a quick pile of code I through down to
    later adapt and correct.  Using the antenna factor works to get a single
    frequency electric field to a voltage, but it is unclear if this is the
    appropriate way to handle a generic electric field. 
    
    '''
    #AF = E/V
    freqs = scipy.fft.fftfreq(len(E),d=(u[1]-u[0])*1e-9)
    antenna_factor =  9.73 * freqs / (gnosim.utils.constants.speed_light * gnosim.utils.rf.amplitude(gain))
    V = scipy.fftpack.fft.ifft(scipy.fftpack.fft(E)*antenna_factor)
    return V
    
def signalResponse(u,E):
    '''
    This is meant to convolve the signal with the response function and the
    system electronics.  
    
    Right now I am making this for convolution with electric field, but perhaps
    it is more proper to do this with a voltage? 
    '''
    antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response.npy')
    electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_elect_response.npy')
    freqs, h_fft = numpy.hsplit(antenna_response, 2)
    freqs, sys_fft = numpy.hsplit(electronic_response, 2)
    h_fft = numpy.ravel(h_fft)
    sys_fft = numpy.ravel(sys_fft)
    
    
    
############################################################

if __name__ == "__main__":
    pylab.close('all')
    energy_neutrino = 3.e9 # GeV
    n = 1.78
    R = 1000. #m
    cherenkov_angle = numpy.arccos(1./n)
    '''
    u_steps = numpy.round(numpy.linspace(1000,100000,40))*2
    sums = []
    angle = 50.
    for us in u_steps:
        A,u = vector_Potential(numpy.deg2rad(angle),R,energy_neutrino,n,u_steps = us)
        sums.append(numpy.absolute(sum(A)*(u[1] - u[0])))
    pylab.plot(u_steps,sums)
    '''  
    steps = 2000000
    A1,u = vector_Potential(cherenkov_angle,R,energy_neutrino,n,u_steps = steps)
    A2,u = vector_Potential(cherenkov_angle+0.00001,R,energy_neutrino,n,u_steps = steps)
    A3,u = vector_Potential(cherenkov_angle-0.00001,R,energy_neutrino,n,u_steps = steps)
    
    pylab.figure()
    pylab.plot(u,A1,label='cherenkov')
    pylab.plot(u,A2,label='cherenkov+0.00001')
    pylab.plot(u,A2,label='cherenkov-0.00001')
    print('sum(numpy.absolute(A2))/sum(numpy.absolute(A1))',sum(numpy.absolute(A2))/sum(numpy.absolute(A1)))
    print('sum(numpy.absolute(A3))/sum(numpy.absolute(A1))',sum(numpy.absolute(A3))/sum(numpy.absolute(A1)))
    A,u = vector_Potential(cherenkov_angle,R,energy_neutrino,n)
    pylab.figure()
    pylab.semilogy(u,R*abs(numpy.absolute(A)))
    pylab.xlim([-2,2])
    pylab.xlabel('$\delta t$')
    pylab.ylabel('R |A(t)| (V s)')
    
    #'''
    #angles = numpy.linspace(0.9*numpy.rad2deg(cherenkov_angle),1.1*numpy.rad2deg(cherenkov_angle),4)
    #angles = numpy.append(angles,numpy.rad2deg(cherenkov_angle))
    angles = numpy.linspace(5,85,20) 
    #angles = [numpy.rad2deg(cherenkov_angle)+0.01,numpy.rad2deg(cherenkov_angle)]
    pylab.figure()
    pylab.title('Vector Potential for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R))
    pylab.ylabel('RA (V s)')
    pylab.xlabel('t (ns)')
    for angle in angles:
        A,u = vector_Potential(numpy.deg2rad(angle),R,energy_neutrino,n)
        pylab.plot(u,R*A,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend() 
    
    pylab.figure()
    pylab.title('Electric Field for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R))
    pylab.ylabel('RE (V)')
    pylab.xlabel('t (ns)')
    for angle in angles:
        E,u = electricField(numpy.deg2rad(angle),R,energy_neutrino,n)
        pylab.plot(u,R*E,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend()
    
    A,u = vector_Potential(cherenkov_angle,R,energy_neutrino,n)
    pylab.figure()
    pylab.semilogy(u,R*abs(numpy.absolute(A)))
    pylab.xlim([-2,2])
    pylab.xlabel('$\delta t$')
    pylab.ylabel('R |A(t)| (V s)')
    #pylab.semilogy(u,abs(RA(energy_neutrino,u)))
    #'''
    '''
    test_angle = 70.
    E,u = electricField(numpy.deg2rad(test_angle),R,energy_neutrino,n)
    #E_fft = 
    antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response.npy')
    electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_elect_response.npy')
    freqs, h_fft = numpy.hsplit(antenna_response, 2)
    freqs, sys_fft = numpy.hsplit(electronic_response, 2)
    h_fft = numpy.ravel(h_fft)
    sys_fft = numpy.ravel(sys_fft)
    pylab.figure()
    pylab.plot(freqs, 20.0 * numpy.log10(numpy.absolute(h_fft)),label='h_fft')
    pylab.plot(freqs, 20.0 * numpy.log10(numpy.absolute(sys_fft)),label='sys_fft')
    pylab.xlabel("Freq. [Hz]")
    pylab.ylabel("dB")
    pylab.xlim(0, 1000e6)
    pylab.legend()
    '''  
############################################################
