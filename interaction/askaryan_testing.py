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
    print(len(t_ns))
    if numpy.size(t_ns) == 1:
        if t_ns > 0:
            return (-4.5e-14) * 1000. * Energy_GeV * ( numpy.exp(- numpy.fabs(t_ns)/0.057) + (1. + 2.87*numpy.fabs(t_ns))**(-3))
        else:
            return (-4.5e-14) * 1000. * Energy_GeV * ( numpy.exp(- numpy.fabs(t_ns)/0.030) + (1. + 3.05*numpy.fabs(t_ns))**(-3.5)) 
    else:
        ra = numpy.zeros_like(t_ns)
        print(len(ra))
        print(sum(t_ns > 0))
        print(sum(t_ns <= 0))
        ra[t_ns > 0] = (-4.5e-14) * 1000. * Energy_GeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns > 0])/0.057) + (1. + 2.87*numpy.fabs(t_ns[t_ns > 0]))**(-3))
        
        ra[t_ns <= 0] = (-4.5e-14) * 1000. * Energy_GeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns <= 0])/0.030) + (1. + 3.05*numpy.fabs(t_ns[t_ns <= 0]))**(-3.5)) 
        print(len(ra))
        #BROKEN Things are doing what I want!!
        return ra


def Q(x,a=4.85,b=0.60,loc=0,scale=1.0,random_params=False,x_scale_factor=1.0):
        '''
        b is included to make the python variant of the gamma function align with
        that defined in: 
        Coherent radio pulses from GEANT generated electromagnetic showers in ice  -  Phys. Rev. D 65, 103002
        The mean a an b values they found are a = 4.85, b = 0.60 with sig_a = 1.01, sig_b = 0.14
        Perhaps we would want to pull randomly from gaussion distros for each of 
        these params for each event. 
        
        Note random_params = True it will overwrite any a or b passed to the function. 
        
        The scale factor is added to modify this function by scaling the z' input
        such that the convolution described in Eq17 of arXiv:1106.6283.
        Essentially in trying to understand how to implement the convolution described
        by that equation I decided a u substitution had to be done where 
        u = z' * alpha where alpha = (1-n*cos(theta))/c.  
        The convolution is then 1/alpha INT du Q(u/alpha)*Fp(dt - u).  The scale
        factor is meant to account for the scaling in Q to make it appear more
        like a convolution: 1/alpha INT du Q'(u)*Fp(dt - u), where Q' scales u. 
        The scaling factor not being one implies the input x is some value of ns
        that was converted from meters using the scale factor. Example:
        
        '''
        x = x/x_scale_factor
        if random_params:
            a = numpy.random.normal(loc=4.85,scale=1.01,size=None)
            b = numpy.random.normal(loc=0.60,scale=0.14,size=None)
        return b * scipy.stats.gamma.pdf( b * x , a = a , loc = loc , scale = scale )
        
def excessProjectedTrackLength(Q,int_min=0.,int_max=25.,n_steps = 100):
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
    #cherenkov_angle = 1./1.78 #Maybe it should be this for consistency in calculation (as RA assumed n  1.78)
    mu_0 = gnosim.utils.constants.mu_0 # m kg s^-2 A^-2
    prefactor = 4. * numpy.pi /( mu_0 * numpy.sin(cherenkov_angle))
    ra = RA(Energy_GeV,t_ns)  
    #print(ra)
    return prefactor * ra / LQ
    
def vector_Potential(theta_obs_rad,R,Energy_GeV,n,plot = False):
    '''
    This should do it all?
    '''
    cherenkov_angle = numpy.arccos(1./n)
    LQ = 1.#excessProjectedTrackLength(Q,int_min=0.,int_max=25.,n_steps = 100)
    
    
    alpha = (1. - n*numpy.cos(theta_obs_rad))/gnosim.utils.constants.speed_light #scaling factor of u substitution units of ns/m
    #Fourier transforms to be multiplied
    #Number of sample points
    #N = 1000
    #sample spacing
    #dz = 1.0/10. #m
    #z = numpy.linspace(-dz * N/2.,dz * N/2.,N) #m
    #
    #u = alpha * z #scaled z, in nanoseconds
    #du = alpha * dz #scaled z, in nanoseconds
    u = numpy.linspace(-100,100,50000)
    fp = F_p(Energy_GeV,u,n,LQ)
    #pylab.figure()
    q = Q(u,x_scale_factor=alpha)
    #pylab.plot(q,label='no tukey')
    q = numpy.multiply(scipy.signal.tukey(len(q),alpha=0.05),q)
    #pylab.plot(q,label='tukey')
    
    integral = 1./alpha * scipy.signal.convolve(q,f_p,mode='full',method='fft')
    #A = ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) / (4. * numpy.pi * R) ) * integral
    fourier_fp = scipy.fftpack.fft(fp)
    fourier_q = scipy.fftpack.fft(q)
    
    convo = numpy.multiply(fourier_fp,fourier_q)
    
    inverse_convo = scipy.fftpack.fftshift(scipy.fftpack.ifft(convo))
    
    A = ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) / (4. * numpy.pi * R) ) * inverse_convo
    '''
    pylab.plot(inverse_convo)
    pylab.figure()
    pylab.plot(u,q)
    pylab.title('q')
    pylab.figure()
    pylab.plot(u,f_p)
    pylab.figure()
    pylab.plot(u,A)
    print(inverse_convo)
    print(u)
    
    
    pylab.figure()
    pylab.plot(u,q)
    pylab.title('q')
    pylab.figure()
    pylab.plot(u,f_p)
    pylab.title('fp')
    pylab.figure()
    print(len(u**(-1)))
    print(len(fourier_fp))
    print(len(numpy.real(fourier_fp)))
    pylab.plot(u**(-1),numpy.real(fourier_fp))
    pylab.title('real fourier fp')
    pylab.figure()
    pylab.plot(u**(-1),numpy.real(fourier_q))
    pylab.title('real fourier q')
    #convolution = 
    '''
    return A,u#q,fp,integral,fourier_fp,fourier_q
    

############################################################

if __name__ == "__main__":
    pylab.close('all')
    energy_neutrino = 1.e8 # GeV
    t = numpy.linspace(-2,2,10000) #ns
    n = 1.78
    R = 1000. #m
    cherenkov_angle = numpy.arccos(1./n)
    
    angles = numpy.linspace(15,65,10)
    pylab.figure()
    pylab.title('Vector Potential for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R))
    for angle in angles:
        A,u = vector_Potential(numpy.deg2rad(angle),R,t,energy_neutrino,n)
        pylab.plot(u,A,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend() 
    
    pylab.figure()
    pylab.title('Electric Field for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R))
    for angle in angles:
        A,u = vector_Potential(numpy.deg2rad(angle),R,t,energy_neutrino,n)
        pylab.plot(u,-numpy.gradient(A),label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend()
       
############################################################
