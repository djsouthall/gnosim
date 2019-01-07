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
import scipy.misc
import time
import math
sys.path.append("/home/dsouthall/Projects/GNOSim/")
from matplotlib import gridspec
import pandas

import gnosim.utils.constants
import gnosim.interaction.inelasticity
import gnosim.utils.quat
import gnosim.earth.earth
import gnosim.earth.antarctic
import gnosim.trace.refraction_library_beta
from gnosim.trace.refraction_library_beta import *
import gnosim.interaction.askaryan
import gnosim.sim.detector
import gnosim.sim.fpga
pylab.ion()

############################################################

import cProfile, pstats, io

def profile(fnc):
    """
    A decorator that uses cProfile to profile a function
    This is lifted from https://osf.io/upav8/
    
    Required imports:
    import cProfile, pstats, io
    
    To use, decorate function of interest by putting @profile above
    its definition.
    """
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s)
        ps.strip_dirs().sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def loadSignalResponse(mode='v2'):
    '''
    Returns the fft's of the signals, and their frequencies.  
    Eventually these file locations should be part of the config file such that
    individual response functions could be input per antenna.  
    '''
    if mode == 'v1':
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_elect_response.npy')
        
    elif mode == 'v2':
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_v2.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_elect_response_v2.npy')
    else:
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_v2.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_elect_response_v2.npy')
    freqs, h_fft = numpy.hsplit(antenna_response, 2)
    freqs, sys_fft = numpy.hsplit(electronic_response, 2)
    h_fft = numpy.ravel(h_fft)
    sys_fft = numpy.ravel(sys_fft)
    return h_fft,sys_fft,freqs[:,0]

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
    
    #cherenkov_angle = numpy.arccos(1./n)
    #mu_0 = gnosim.utils.constants.mu_0 # m kg s^-2 A^-2
    prefactor = 4. * numpy.pi /( gnosim.utils.constants.mu_0 * math.sqrt(1-1/n**2))#note that math.sqrt(1-1/1.78**2) is a faster form of numpy.sin(cherenkov_angle)
    ra = RA(Energy_GeV,t_ns)  
    #print(ra)
    return prefactor * ra / LQ
    
def vectorPotentialTimeDomain(theta_obs_rad,R,Energy_GeV,n,u,plot = False):
    '''
    This should do it all?
    '''
    #cherenkov_angle = numpy.arccos(1./n)
    LQ = excessProjectedTrackLength(Q)
    alpha = (1. - n*numpy.cos(theta_obs_rad))/gnosim.utils.constants.speed_light #scaling factor of u substitution units of ns/m

    fp = F_p(Energy_GeV,u,n,LQ)
    fp = numpy.multiply(scipy.signal.tukey(len(fp),alpha=0.05),fp)

    if abs(alpha) < 0.001:
        A = ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) * LQ * fp / (4. * numpy.pi * R) ) 
        #u = u_step * numpy.linspace(-(len(fp)-1)/2,(len(fp)-1)/2,len(fp))
        A_fft = numpy.fft.rfft(A)
        #A = numpy.fft.fftshift(A)
        #time_step = (u[1]-u[0]) #ns
        #freq_step = 1/(len(u)*(time_step*1e-9))
        #max_freq = 1/(2*(time_step*1e-9))
        #freqs = numpy.arange(len(A_fft))*freq_step #might be off by 1 step somewhere
        #E_fft = -1j*2*numpy.pi*numpy.multiply(A_fft , freqs)
        #E = numpy.fft.irfft(E_fft,n=len(u)) #/ (len(E_fft) / (max(u) - min(u)))
        #E = numpy.fft.fftshift(E)
        
        if plot == True:
            pylab.figure()
            pylab.subplot(211)
            pylab.title('alpha = %0.3f, $\\theta$ = %0.2f deg'%(alpha,numpy.rad2deg(theta_obs_rad)),fontsize=20)
            pylab.plot(u,fp,label='fp')
            pylab.ylabel('$F_p$ ($Amps$)',fontsize=16)
            pylab.xlim(-10,50)
            pylab.subplot(212)
            pylab.plot(u,R*A,label='q')
            #pylab.ylabel('$R|A|$ ',fontsize=16)
            pylab.semilogy(u,numpy.fabs(R*A),label='RA')
            pylab.xlabel('$\Delta t$',fontsize=16)
            pylab.xlim(-10,50)
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
        #q = numpy.pad(q,pad_width=int(len(q)/2),mode='constant')
        fp_fft = numpy.fft.rfft(fp)
        q_fft = numpy.fft.rfft(q)

        convo = numpy.multiply(fp_fft,q_fft)
        
        #time_step = (u[1]-u[0]) #ns
        #freq_step = 1/(len(u)*(time_step*1e-9))
        #max_freq = 1/(2*(time_step*1e-9))
        #freqs = numpy.arange(len(convo))*freq_step #might be off by 1 step somewhere
        #freqs = numpy.arange(0,max_freq+freq_step,freq_step)
        #deriv_convo = 1j*2*numpy.pi*numpy.multiply(convo , freqs)
        #print('max_freq =',max_freq)
        #print('max(freqs) =',max(freqs))
        #print('len(freqs)=',len(freqs))
        #print('len(convo)=',len(convo)) 
        
        inverse_convo_A = numpy.fft.irfft(convo,n=len(u))
        inverse_convo_A = numpy.fft.fftshift(inverse_convo_A) #might not need to do this, but it moves padding after signal to before signal
        
        #inverse_convo_E = numpy.fft.irfft(deriv_convo,n=len(u))
        #inverse_convo_E = numpy.fft.fftshift(inverse_convo_E)
        
        #inverse_convo = scipy.fftpack.fftshift(scipy.fftpack.ifft(convo)) #maybe could divide my 2*pi*f here to get directly 
        A = ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) / (4. * numpy.pi * R ) ) * (inverse_convo_A / ( abs(alpha) * len(inverse_convo_A) / (max(u) - min(u)))) #abs alpha because in the u sub the bounds swap when alpha < 0, but then alpha <0 so the two negatives cancel.
        #E = -( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) / (4. * numpy.pi * R ) ) * (inverse_convo_E / ( abs(alpha) * len(inverse_convo_E) / (max(u) - min(u))))# the extra negative comes from the E being -dA/dt
        if plot == True:
            pylab.figure()
            pylab.subplot(311)
            pylab.title('alpha = %0.3f, $\\theta$ = %0.2f deg'%(alpha,numpy.rad2deg(theta_obs_rad)),fontsize=20)
            pylab.plot(u,fp,label='fp')
            pylab.ylabel('$F_p$ ($Amps$)',fontsize=16)
            pylab.xlim(-10,50)
            pylab.subplot(312)
            pylab.plot(u,q,label='q')
            pylab.ylabel('$Q (arb)$ ',fontsize=16)
            pylab.xlim(-10,50)
            pylab.subplot(313)
            #pylab.semilogy(u,R*A,label='RA')
            pylab.semilogy(u,numpy.fabs(R*numpy.absolute(A)),label='RA')
            pylab.ylabel('$R|A|$ ',fontsize=16)
            pylab.xlabel('$\Delta t$',fontsize=16)
            pylab.xlim(-10,50)
            pylab.subplots_adjust(left=0.08, bottom=0.05, right=0.98, top=0.97, wspace=None, hspace=None)
    return A , u

def electricFieldTimeDomainRaw(theta_obs_rad,R,Energy_GeV,n,u,plot = False,deriv_mode = 'time'):
    '''
    Calculates the time domain electric field using the method from 
    Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283.  This stage has not 
    accounted for any system responses and is just the signal as emitted.  
    
    '''
    A, u = vectorPotentialTimeDomain(theta_obs_rad,R,Energy_GeV,n,u)
    if deriv_mode == 'freq':
        A_fft = numpy.fft.rfft(A)
        time_step = (u[1]-u[0]) #ns
        freq_step = 1/(len(u)*(time_step*1e-9))
        max_freq = 1/(2*(time_step*1e-9))
        freqs = numpy.arange(len(A_fft))*freq_step #might be off by 1 step somewhere
        E_fft = -1j*2*numpy.pi*numpy.multiply(A_fft , freqs)
        E = numpy.fft.irfft(E_fft)
    else:
        E = - numpy.divide(numpy.gradient(A),numpy.gradient(u)) * 1e9
    if plot == True:
            pylab.figure()
            pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
            pylab.ylabel('$R|\\vec{E}|$ (V)',fontsize=16)
            pylab.xlabel('t (ns)',fontsize=16)
            pylab.plot(u,R*E,label = '$R|\\vec{E}_{raw}|$ ')
    return  E , u

def electricFieldTimeDomainSignal(theta_obs_rad,R,Energy_GeV,n,h_fft=None,sys_fft=None,freqs=None,plot=False,return_pos = False,mode='v2',up_sample_factor=10,deriv_mode = 'time'):  
    '''
    Calculates the full electric field, including response function calculations.
    '''
    if any([numpy.size(h_fft) ==1,numpy.size(sys_fft)==1,numpy.size(freqs)==1]):
        h_fft,sys_fft,freqs = loadSignalResponse(mode=mode)
    freqs = numpy.absolute(freqs)
    t_step_old = 1/(2*max(freqs))*1e9 #ns
    n_old = len(freqs)
    
    h_fft = numpy.append(h_fft,numpy.zeros(up_sample_factor*len(h_fft)))
    sys_fft = numpy.append(sys_fft,numpy.zeros(up_sample_factor*len(sys_fft)))
    freqs = numpy.arange(len(sys_fft))*(freqs[1]-freqs[0])
    t_step = 1/(2*max(freqs))*1e9 #ns
    n_new = len(freqs)
    
    tmin = 500 #the time window will not go under this number
    #n_points = int(max(tmin//t_step,2*(len(sys_fft)-1)))#n_points in time
    n_points = int(2*(len(sys_fft)-1))#n_points in time
    freq_step = 1/(n_points*t_step*1e-9) #Hz
    '''
    if n_points != 2*(len(sys_fft)-1):
        #need to pad in time domain to make final signal reach further temporally
        h = numpy.fft.irfft(h_fft)
        sys = numpy.fft.irfft(sys_fft) 
        h = numpy.append(h,numpy.zeros(n_points - len(h)))
        sys = numpy.append(sys,numpy.zeros(n_points - len(sys)))
        h_fft = numpy.fft.rfft(h)
        sys_fft = numpy.fft.rfft(sys) 
    '''    
    f = numpy.arange(len(h_fft))*freq_step
    response_fft = numpy.multiply(h_fft,sys_fft)
        
    u = numpy.arange(-n_points/2,n_points/2)*t_step
    
    E_sig, u = electricFieldTimeDomainRaw(theta_obs_rad,R,Energy_GeV,n,u,plot=plot,deriv_mode = deriv_mode)
    E_fft = numpy.fft.rfft(E_sig)
    V_fft = numpy.multiply(E_fft,response_fft)
    V = numpy.fft.irfft(V_fft,n=len(u))
    
    if plot == True:
        pylab.figure()
        pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
        pylab.plot(u,V)
        pylab.ylabel('Signal (V)',fontsize=16)
        pylab.xlabel('t (ns)',fontsize=16)
        
        pylab.figure()
        pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
        pylab.plot(f/1e6,20.0 * numpy.log10(numpy.absolute(E_fft)),label='Raw Signal (fft)')
        pylab.plot(f/1e6,20.0 * numpy.log10(numpy.absolute(sys_fft)),label='System Response')
        pylab.plot(f/1e6,20.0 * numpy.log10(numpy.absolute(h_fft)),label='Antenna Response')
        pylab.plot(f/1e6,20.0 * numpy.log10(numpy.absolute(V_fft)),label='Processed Signal (fft)')
        
        pylab.xlabel('Freq. [MHz]',fontsize=16)
        pylab.ylabel('dB',fontsize=16)
        pylab.ylim(-50,100)
        pylab.xlim(0,1000)
        pylab.legend()
    if return_pos == True:
        V = V[u>=0]
        u = u[u>=0]
    dominant_freq = f[numpy.argmax(numpy.absolute(V_fft))]
    return V, u, dominant_freq


def addSignals(u_in,V_in,plot=False):
    '''
    u_in should be an array of times with dimensions (n_signal , n_timestep )
    u is assumed to be in order, i.e. u[0] is the min of each row and u[-1] is the max.
    Each row of u is also assumed to have the same time step.  
    V_in should be an array of electric fields with dimensions (n_signal , n_timestep )
    Not that the timing of signals may be shifted by up to a u-step here to align 
    descretized timing values.  There may be a more elegant want do this if this
    added wiggle becomes a problem. 
    '''
    if numpy.shape(u_in)[0] <= 1:
        return V_in.flatten(),u_in.flatten()
    else:
        u_step = u_in[0,1]-u_in[0,0]
        u_out_min = min(u_in[:,0])
        u_out_max = max(u_in[:,-1])
        u_out = numpy.arange(u_out_min,u_out_max+u_step,u_step)
        V_out = numpy.zeros_like(u_out)
        if plot == True:
            pylab.figure()    
            ax = pylab.subplot(numpy.shape(V_in)[0]+1,1,numpy.shape(V_in)[0]+1)
            pylab.xlim((u_out_min,u_out_max))
        for i in range(numpy.shape(V_in)[0]):
            V = V_in[i]
            u = u_in[i]
            
            if len(u) == 0:
                u = u_out
                V = numpy.zeros_like(u_out)   
            #print('Lengths:')
            #print(len(V))
            #print(len(u))
            #print('%i:%i ->%i'%(numpy.argmin(abs(u_out - min(u))),numpy.argmin(abs(u_out - min(u)))+len(u),len(V_out[numpy.argmin(abs(u_out - min(u))):numpy.argmin(abs(u_out - min(u)))+len(u)])))
            left_index = numpy.argmin(abs(u_out - u[0]))
            right_index = left_index + len(V)
            #print('left_index',left_index)
            #print('right_index',right_index)
            cut = numpy.arange(left_index,right_index)
            #print(len(cut))
            V_out[cut] += V
            if plot == True:
                pylab.subplot(numpy.shape(V_in)[0]+1,1,i+1,sharex=ax)
                pylab.plot(u,V,label='Signal %i'%(i))
                pylab.ylabel('V (V)',fontsize=16)
                pylab.xlabel('t-t_emit (ns)',fontsize=16)
                pylab.legend(fontsize=14)
        if plot == True:
            pylab.subplot(numpy.shape(V_in)[0]+1,1,numpy.shape(V_in)[0]+1)
            pylab.plot(u_out,V_out,label='Total Signal')
            pylab.ylabel('V (V)',fontsize=16)
            pylab.xlabel('t-t_emit (ns)',fontsize=16)
            pylab.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
            pylab.legend(fontsize=14)
        return V_out,u_out

def calculateTimes(up_sample_factor=20,h_fft=None,sys_fft=None,freqs=None,mode='v2'):
    '''
    Calculates the times used for signal calculations based on the response functions
    (assumed to have the same frequency step).  up_sample_factor is not exact, as the
    number of points is rounded to a factor of 2 to ensure future ifft's are as fast
    as possible.
    
    This also returns the newly padded responses and freqs.
    '''
    #Loading in response function and setting frequency / time steps
    if any([numpy.size(h_fft) ==1,numpy.size(sys_fft)==1,numpy.size(freqs)==1]):
        h_fft,sys_fft,freqs = loadSignalResponse(mode=mode)

    if up_sample_factor <= 0:
        up_sample_factor = 1

    freqs = numpy.absolute(freqs)
    freq_step = freqs[1]-freqs[0] #1/(n_points*t_step*1e-9) #Hz
    possible_lengths = 2**numpy.arange(0,25)
    n_points_freq = possible_lengths[possible_lengths >= up_sample_factor*len(h_fft)][0] + 1 #Want 2^n events in time domain, so 2^n  #upsamples to the closest power of two to upsample*original_length
    freqs = numpy.arange(n_points_freq)*freq_step
    
    h_fft = numpy.append(h_fft,numpy.zeros(n_points_freq - len(h_fft)))
    sys_fft = numpy.append(sys_fft,numpy.zeros(n_points_freq - len(sys_fft)))
    response_fft = numpy.multiply(h_fft,sys_fft)
    
    t_step = 1/(2*max(freqs))*1e9 #ns
    u = numpy.arange(-(n_points_freq-1),(n_points_freq-1))*t_step #To increase time duration of signal I should just need to upsample?
    return u, h_fft, sys_fft, freqs

def quickSignalSingle(theta_obs_rad,R,Energy_GeV,n,t_offset,attenuation,beam_pattern_factor,u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential = False,include_noise = False, resistance = 50, temperature = 320):  
    '''
    This should do the entire calculation, mostly in the frequency domain. 
    Expects u, h_fft, sys_fft, freqs to all come straight from calculateTimes.
    These are the same for a given up_sample and response so don't need to be calculated
    every signal. 
    
    
    Return pattern:
    if include_noise == True:
        return V_noiseless, u, dominant_freq, V_noise,  SNR
    else:
        return V_noiseless, u, dominant_freq
        
    SNR is calculated as the ratio of the peak to peak/2 over rms(noise), squared 
    (ratio of powers)
    '''
    t_step = u[1]-u[0] #ns
    
    #Calculating the vector potential
    #cherenkov_angle = numpy.arccos(1./n)
    LQ = 1#excessProjectedTrackLength(Q,int_min=min(u),int_max=max(u),n_steps = len(u)) #can probably set to 1 for now as this Q is normalized?
    alpha = (1. - n*numpy.cos(theta_obs_rad))/gnosim.utils.constants.speed_light #scaling factor of u substitution units of ns/m

    #calling the below was slow, so I put it just calculate it in this function
    #fp = F_p(Energy_GeV,u,n,LQ)
    #fp = numpy.multiply(scipy.signal.tukey(len(fp),alpha=0.05),fp)
    ra = RA(Energy_GeV,u)  
    fp = (4. * numpy.pi /(LQ * gnosim.utils.constants.mu_0 * math.sqrt(1-1/n**2))) * ra #note that math.sqrt(1-1/1.78**2) is a faster form of numpy.sin(cherenkov_angle) = numpy.sin(numpy.arccos(1/n))
    fp = numpy.multiply(scipy.signal.tukey(len(fp),alpha=0.05),fp)
    fp_fft = numpy.fft.rfft(fp)
    
    if abs(alpha) < 0.001:
        #print('alpha < 0.001')
        A_fft = fp_fft * ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) * LQ / (4. * numpy.pi * R) ) 
        
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
        
        
        #calling the below was slow, so I put it just as a line of code with force a/b parameters.  
        #q = Q(u/alpha)
        #q = numpy.multiply(scipy.signal.tukey(len(q),alpha=0.05),q)
        
        q = 0.60 * scipy.stats.gamma.pdf( 0.60 * u/alpha , a = 4.85 ) #This seems to be rate limiting.  If you think of faster way to calculate you can save some time.
        q = numpy.multiply(scipy.signal.tukey(len(q),alpha=0.05),q)
        q_fft = numpy.fft.rfft(q)
        A_fft = numpy.multiply(fp_fft,q_fft) * ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) / (4. * numpy.pi * R ) ) * ( t_step/abs(alpha) ) #the t_step already accounts for scaling required with irffting.  Might not want here?  unsure
        
        A = numpy.fft.irfft(A_fft,n=len(u))
        A = numpy.fft.fftshift(A)
        if plot_potential == True:
            pylab.figure()
            pylab.subplot(311)
            pylab.title('alpha = %0.3f, $\\theta$ = %0.2f deg'%(alpha,numpy.rad2deg(theta_obs_rad)),fontsize=20)
            pylab.plot(u,fp,label='fp')
            pylab.ylabel('$F_p$ ($Amps$)',fontsize=16)
            #pylab.xlim(-10,50)
            pylab.subplot(312)
            pylab.plot(u,q,label='q')
            pylab.ylabel('$Q (arb)$ ',fontsize=16)
            #pylab.xlim(-10,50)
            pylab.subplot(313)
            pylab.semilogy(u,numpy.fabs(R*numpy.absolute(A)),label='RA')
            pylab.ylabel('$R|A|$ ',fontsize=16)
            pylab.xlabel('$\Delta t$',fontsize=16)
            #pylab.xlim(-10,50)
    #calculating E_raw_fft    
    E_raw_fft = -1j*2*numpy.pi*numpy.multiply(A_fft , freqs) #negitive sign because E = -dA/dt
    
    #Accouning for beam pattern
    E_raw_fft *= beam_pattern_factor
    
    #Accounting for attenuation
    E_raw_fft *= attenuation #Want to do before noise is added.  Noise is not attenuated by 
    #Adding antenna response
    E_antenna_fft = numpy.multiply(E_raw_fft, h_fft) 
    V_fft_noiseless = numpy.multiply(E_antenna_fft,sys_fft)
    
    V_noiseless = numpy.fft.irfft(V_fft_noiseless,n=len(u))
    
    if include_noise == True:
        #The state is reset after this is called to avoid the random queue being shifted when using noise v.s. when not using noise
        #rand_state = numpy.random.get_state()
        
        #SNR Peak to Peak calculation and noiseless signal calculation
        
        V_noiseless_sorted = numpy.sort(V_noiseless)
        p2p_half = (V_noiseless_sorted[-1] - V_noiseless_sorted[0]) / 2
        
        #calculating noise
        bandwidth = freqs[-1]/1e9 #Calculating full band noise, response cuts out stuff we don't see
        V_rms = numpy.sqrt(gnosim.utils.constants.boltzmann * temperature * resistance * bandwidth * gnosim.utils.constants.GHz_to_Hz)
        sigma = V_rms 
        
        #Noise in Polar
        #noise_phase = numpy.random.uniform(-numpy.pi,numpy.pi,size = len(freqs))
        #noise_amp = numpy.random.normal(loc = 0.0, scale = sigma , size = len(freqs)) #these might need the normalization factor of *numpy.sqrt(len(u)/2) if used at some point for some reason
        
        #Noise in Cartesian
        noise_cartesian = numpy.sqrt(len(u)/2)*(numpy.random.normal(loc = 0.0, scale = sigma , size = len(freqs)) + 1j*numpy.random.normal(loc = 0.0, scale = sigma , size = len(freqs))) # the *numpy.sqrt(len(u)/2) factor is to handle some normalization issues
        V_fft_just_noise = numpy.multiply(noise_cartesian,sys_fft)
        #Multiplying in system noise to get V_fft
        V_fft_noise = numpy.add(V_fft_noiseless,V_fft_just_noise)
        V_noise = numpy.fft.irfft(V_fft_noise,n=len(u))
        #numpy.random.set_state(rand_state)
    
    
    if abs(alpha) >= 0.001:
        V_noiseless = numpy.fft.fftshift(V_noiseless) #This centres V so it occurs at t=0. and ensures it is located temporaly in the same place for if the calculation was done using the exception or on cone or not
        if include_noise == True:
            V_noise = numpy.fft.fftshift(V_noise)
    
    if include_noise == True:
        V_rms_measured = numpy.sqrt(numpy.mean(V_noise[u < 0]**2)) #This is the 'measured' V_rms, rather than the set.  This is after system response
        SNR = (p2p_half/V_rms_measured)**2
        SNR_dB = 10*numpy.log10( SNR )#dB, using 10log10 because input is power ratio 
        #print('SNR', SNR)
    else:
        SNR = 0
        print('No noise included.  Cannot perform SNR calculation.')

    if plot_signals==True:
        E_raw = numpy.fft.irfft(E_raw_fft,n=len(u)) #should just be the signal, as A_fft already divided by scaling factor of t_step?
        A = numpy.fft.irfft(A_fft,n=len(u))
        if abs(alpha) >= 0.001:
            E_raw = numpy.fft.fftshift(E_raw) #This centres E so it occurs at t=0. and ensures it is located temporaly in the same place for if the calculation was done using the exception or on cone or not
            A = numpy.fft.fftshift(A) #This centres A so it occurs at t=0. and ensures it is located temporaly in the same place for if the calculation was done using the exception or on cone or not
            
        pylab.figure()
        if include_noise == True:
            pylab.subplot(411)
            pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f\tt_step = %g ns'%(energy_neutrino,numpy.rad2deg(theta_obs_rad),n,t_step),fontsize=20)
            pylab.ylabel('R*|A| (V s)')
            pylab.xlabel('t (ns)')
            #pylab.scatter(u,R*numpy.absolute(A),s=1)
            pylab.plot(u,R*numpy.absolute(A))
            
            pylab.subplot(412)
            pylab.ylabel('$R \cdot E_{raw}$ (V)')
            pylab.xlabel('t (ns)')
            #pylab.scatter(u,R*E_raw,s=1)
            pylab.plot(u,R*E_raw)
            
            pylab.subplot(413)
            pylab.ylabel('Noiseless Signal Voltage (V)')
            pylab.xlabel('t (ns)')
            #pylab.scatter(u,V,s=1)
            pylab.plot(u,V_noiseless)
            
            pylab.subplot(414)
            pylab.ylabel('Signal Voltage (V)')
            pylab.xlabel('t (ns)')
            #pylab.scatter(u,V,s=1)
            pylab.plot(u,V_noise)
        else:
            pylab.subplot(311)
            pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f\tt_step = %g ns'%(energy_neutrino,numpy.rad2deg(theta_obs_rad),n,t_step),fontsize=20)
            pylab.ylabel('R*|A| (V s)')
            pylab.xlabel('t (ns)')
            #pylab.scatter(u,R*numpy.absolute(A),s=1)
            pylab.plot(u,R*numpy.absolute(A))
            
            pylab.subplot(312)
            pylab.ylabel('$R \cdot E_{raw}$ (V)')
            pylab.xlabel('t (ns)')
            #pylab.scatter(u,R*E_raw,s=1)
            pylab.plot(u,R*E_raw)
            
            pylab.subplot(313)
            pylab.ylabel('Noiseless Signal Voltage (V)')
            pylab.xlabel('t (ns)')
            #pylab.scatter(u,V,s=1)
            pylab.plot(u,V_noiseless)
    if plot_spectrum == True:
        pylab.figure()
        pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
        pylab.plot(freqs/1e6,20.0 * numpy.log10(numpy.absolute(E_raw_fft)),label='Raw Signal (fft)')
        pylab.plot(freqs/1e6,20.0 * numpy.log10(numpy.absolute(sys_fft)),label='System Response')
        pylab.plot(freqs/1e6,20.0 * numpy.log10(numpy.absolute(h_fft)),label='Antenna Response')
        if include_noise == True:
            pylab.plot(freqs/1e6,20.0 * numpy.log10(numpy.absolute(V_fft_noise)),label='Processed Signal (fft)')
        else:
            pylab.plot(freqs/1e6,20.0 * numpy.log10(numpy.absolute(V_fft_noiseless)),label='Processed Signal (fft)')
        
        pylab.xlabel('Freq. [MHz]',fontsize=16)
        pylab.ylabel('dB',fontsize=16)
        #pylab.ylim(-50,100)
        pylab.xlim(0,1500)
        pylab.legend()
    
    if include_noise == True:
        dominant_freq = freqs[numpy.argmax(numpy.absolute(V_fft_noise))]
        return V_noiseless, u + t_offset, dominant_freq, V_noise,  SNR
    else:
        dominant_freq = freqs[numpy.argmax(numpy.absolute(V_fft_noiseless))]
        return V_noiseless, u + t_offset, dominant_freq

def signalsFromInfo(eventid,reader,u_signal,n,h_fft,sys_fft,freqs,include_noise = False,resistance = 50, temperature = 320):
    #quickSignalSingle(theta_obs_rad,R,Energy_GeV,n,t_offset,attenuation,
    #           u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,
    #           plot_potential = False,out_dom_freq = False,include_noise = False, 
    #           resistance = 50, temperature = 320)
    if any([numpy.size(h_fft) ==1,numpy.size(sys_fft)==1,numpy.size(freqs)==1]):
        h_fft,sys_fft,freqs = loadSignalResponse(mode=mode)
    energy_neutrino = reader['energy_neutrino'][...][eventid]
    info = reader['info'][...]
    event_info = info[info['eventid'] == eventid]
    event_info = event_info[event_info['has_solution'] ==1]
    if len(event_info) == 0:
        print('No solution for event %i'%(eventid))
    else:
        #print(info.dtype)
        #print(event_info)
        thetas = event_info['observation_angle']
        Rs = event_info['distance']
        t_offset = event_info['time']
        av = event_info['a_v']
        ah = event_info['a_h']
        inelasticity = reader['inelasticity'][...][eventid]
        p_interact = reader['p_interact'][...][eventid]
        p_earth = reader['p_earth'][...][eventid]
        V = []
        u = []
        f = []
        
        Vd = []
        ud = []
        
        noise_signal = quickSignalSingle( 0,1,numpy.unique(energy_neutrino),n,\
                          0,0,0,u_signal,h_fft,sys_fft,freqs,\
                          plot_signals=False,plot_spectrum=False,plot_potential = False,\
                          include_noise = True, resistance = 50, temperature = 320)[3]
        noise_rms = numpy.std(noise_signal)
        
        #if case to handle if info has dtype label of beam_pattern_factor
        
        if numpy.isin('beam_pattern_factor',numpy.array(event_info.dtype.names)) == True:
            beam_pattern_factor = event_info['beam_pattern_factor']
        else:
            beam_pattern_factor = 1.0
        for index in range(len(Rs)):
            if include_noise == True:
                _Vi, ui, fi,Vi,SNRi = quickSignalSingle(numpy.deg2rad(thetas[index]),Rs[index],inelasticity*energy_neutrino,n,t_offset[index],av[index],beam_pattern_factor[index],u_signal, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential = False,include_noise = include_noise,resistance = resistance, temperature = temperature)  
                #in this case I would want Vi to be the noisy signal, not _Vi which is the clean signal.
            else:
                Vi, ui, fi = quickSignalSingle(numpy.deg2rad(thetas[index]),Rs[index],inelasticity*energy_neutrino,n,t_offset[index],av[index],beam_pattern_factor,u_signal, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential = False,include_noise = include_noise,resistance = resistance, temperature = temperature)   
            
            Vdi,udi = gnosim.sim.fpga.digitizeSignal(ui,Vi,1.5,7,noise_rms,3, dc_offset = 0, random_time_offset = 0, plot = False)
            if index == 0:
                V = Vi
                u = ui
                f = fi
                
                Vd = Vdi
                ud = udi
            else:
                V = numpy.vstack((V,Vi))
                u = numpy.vstack((u,ui))
                f = numpy.vstack((f,fi))
                
                Vd = numpy.vstack((Vd,Vdi))
                ud = numpy.vstack((ud,udi))
        
        nrows = len(t_offset) #might want to add + len(tshift) or something to add a summed signal
        ntables = 5
        
        gs_left = gridspec.GridSpec(nrows, 2, width_ratios=[3, 1]) #should only call left plots.  pylab.subplot(gs_left[0]),pylab.subplot(gs_left[2]),...
        gs_right = gridspec.GridSpec(ntables, 2, width_ratios=[3, 1]) #should only call odd tables pylab.subplot(gs_right[1])
        
        pylab.figure(figsize=(16.,11.2)) 
        
        #Plotting signals
        
        first_in_loop = True
        axis2 = []
        max_ax1_range = numpy.array([1e20,-1e20])
        for index_antenna in range(0, len(t_offset)):
            antenna_label_number = event_info['antenna'][index_antenna]
            
            #pylab.subplot(len(t_offset),2,2*index_antenna+1,sharex=ax,sharey=ax)
            if first_in_loop == True:
                first_in_loop = False
                ax = pylab.subplot(gs_left[2*index_antenna])
            ax1 = pylab.subplot(gs_left[2*index_antenna],sharex = ax,sharey = ax)
            ax2 = ax1.twinx() #this is not perfect and can be janky with zooming.        
            axis2.append(ax2)
            if index_antenna == 0:
                pylab.title('Event %i, summed_signals = %s'%(eventid,'False'))
                
            #Plotting
            c1 = 'b'
            c2 = 'r'
            
            ax1.plot(u[index_antenna,:],V[index_antenna,:],c = c1,label='s%ia%i analog'%(0,antenna_label_number),linewidth=0.6)
            ax2.plot(ud[index_antenna,:],Vd[index_antenna,:],c = c2,label='s%ia%i digital'%(0,antenna_label_number),linewidth=0.4)
            
            ax1.tick_params('y', colors=c1)
            ax1.set_ylabel('V', color=c1)
            
            ax2.tick_params('y', colors=c2)
            ax2.set_ylabel('adu', color=c2)
            #print(numpy.array(ax1.get_ylim()) * 3 / noise_rms)
            ax2.set_ylim(numpy.array(ax1.get_ylim()) * 3 / noise_rms)
            #if ( len(t_offset) // 2 == index_antenna):
            #    pylab.ylabel('V$_{%i}$ (V)'%(eventid),fontsize=12)
            ax1.legend(fontsize=8,loc='upper left')
            ax2.legend(fontsize=8,loc='upper right')
            
            ax1_ylim = numpy.array(ax1.get_ylim())
                        
            if ax1_ylim[0] < max_ax1_range[0]:
                max_ax1_range[0] = ax1_ylim[0]
            if ax1_ylim[1] > max_ax1_range[1]:
                max_ax1_range[1] = ax1_ylim[1]
                
        for ax2 in axis2:
            ax2.set_ylim(max_ax1_range * 3 / noise_rms)
        pylab.xlabel('t-t_emit (ns)',fontsize=12)
        
        
        #Making Tables
        #Making position table
        table_fig = pylab.subplot(gs_right[1])
        
        table_ax = pylab.gca()
        table_fig.patch.set_visible(False)
        table_ax.axis('off')
        table_ax.axis('tight')
        x_neutrino = reader['x_0'][eventid]
        y_neutrino = reader['y_0'][eventid]
        z_neutrino = reader['z_0'][eventid]
        r_neutrino = numpy.sqrt(x_neutrino**2 + y_neutrino**2)
        phi_neutrino = reader['phi_0'][eventid]
        df = pandas.DataFrame({'x(m)':[ x_neutrino ] , 'y(m)':[ y_neutrino ] , 'z(m)':[ z_neutrino ] , 'r(m)':[ r_neutrino ] , '$\phi_0$(deg)':[ phi_neutrino ] })
        table = pylab.table(cellText = df.values.round(2), colLabels = df.columns, loc = 'center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        pylab.title('Neutrino Info')
        
        #Making Neutrino Energetics table
        '''
        >>> list(reader.keys())
        ['a_h', 'a_v', 'd', 'electric_field', 'energy_neutrino', 'index_antenna', 
        'index_station', 'inelasticity', 'info', 'observation_angle', 'p_detect', 
        'p_earth', 'p_interact', 'phi_0', 'solution', 't', 'theta_0', 'theta_ant', 
        'theta_ray', 'x_0', 'y_0', 'z_0']
        '''
        table_fig = pylab.subplot(gs_right[3])
        
        table_ax = pylab.gca()
        table_fig.patch.set_visible(False)
        table_ax.axis('off')
        table_ax.axis('tight')
        
        df = pandas.DataFrame({'E$_\\nu$ (GeV)':'%0.4g'%(energy_neutrino) , 'Inelasticity':'%0.4g'%inelasticity , 'p_interact':'%0.4g'%p_interact, 'p_earth':'%0.4g'%p_earth},index=[0])
        #decimals = pandas.Series([3,3,3,3],index = df.columns)
        table = pylab.table(cellText = df.values , colLabels = df.columns, loc = 'center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        #Making observed angles and attenuations table
        table_fig = pylab.subplot(gs_right[5])
        
        table_ax = pylab.gca()
        table_fig.patch.set_visible(False)
        table_ax.axis('off')
        table_ax.axis('tight')
        antenna = event_info['antenna'].astype(int)
        observation_angle = event_info['observation_angle'].astype(float)
        theta_ant = event_info['theta_ant'].astype(float)
        distance = event_info['distance'].astype(float)
        df = pandas.DataFrame({'antenna':antenna , '$\\theta_\mathrm{ant}$ (deg)':theta_ant , '$\\theta_\mathrm{emit}$ (deg)':observation_angle,'d$_\mathrm{path}$ (m)':distance})
        decimals = pandas.Series([0,3,3,3],index = df.columns)
        table = pylab.table(cellText = df.round(decimals).values, colLabels = df.columns, loc = 'center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        pylab.subplots_adjust(left = 0.06, bottom = 0.05, right = 0.99, top = 0.97, wspace = 0.15, hspace = 0.28)
            
        return V, u, Vd, ud
            
            
            
############################################################

if __name__ == "__main__":
    old_testing = True
    interp_testing = False
    new_testing = False
    
    pylab.close('all')
    energy_neutrino = 3.e9 # GeV
    n = 1.78
    R = 1000. #m
    cherenkov_angle = numpy.arccos(1./n)
    cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./n))
    h_fft,sys_fft,freqs = loadSignalResponse()
    input_u, h_fft, sys_fft, freqs = calculateTimes(up_sample_factor=20)
    
    if new_testing == True:
        inelasticity = 0.2#gnosim.interaction.inelasticity.inelasticity(energy_neutrino, mode='cc')
        #Testing the digitizations of the signal.  
        V_noiseless, u, dominant_freq, V_noise,  SNR = quickSignalSingle(numpy.deg2rad(50),R,inelasticity*energy_neutrino,n,2500,0.7,0.7,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)
        noise_rms = numpy.std(quickSignalSingle(0,R,inelasticity*energy_neutrino,1.8,2500,0,0,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)[3])
        sampling_rate = 1.5 #GHz
        bytes = 7
        scale_noise_from = noise_rms
        scale_noise_to = 3
        
        random_time_offset = numpy.random.uniform(-5.0,5.0) #ns
        dc_offset = 0.0 #V
        V_bit, sampled_times = gnosim.sim.fpga.digitizeSignal(u,V_noise,sampling_rate,bytes,scale_noise_from,scale_noise_to, random_time_offset = random_time_offset, dc_offset = dc_offset, plot = False)
        
        
    if interp_testing ==True:
    
        inelasticity = 0.2#gnosim.interaction.inelasticity.inelasticity(energy_neutrino, mode='cc')
        #Testing the digitizations of the signal.  
        V_noiseless, u, dominant_freq, V_noise,  SNR = quickSignalSingle(numpy.deg2rad(50),R,inelasticity*energy_neutrino,n,2500,0.5,0.5,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)
        
        
        
        sampling_rate = 1.5 #GHz
        sampling_period = 1.0 / sampling_rate #ns
        #sample_number = u / sampling_period
        sample_times = numpy.arange(u[1],u[-1],sampling_period)

        @profile
        def f1(sample_times, u, V_noise):        
            snap_index = []
            for i in sample_times:
                index = numpy.argmin(numpy.fabs(u-i))
                snap_index.append(index)
            snap_index = numpy.array(snap_index)
            V_sampled = V_noise[snap_index]
            return V_sampled
            
        V_sampled = f1(sample_times, u, V_noise)
        pylab.figure()
        ax = pylab.subplot(3,1,1)
        pylab.scatter(u,V_noise,label='Signal')
        pylab.ylabel('V (V)')
        pylab.xlabel('t (ns)')
        pylab.stem(sample_times,V_sampled,linefmt='r-', markerfmt='rs', basefmt='r-',label='Snap Sampled at %0.2f GSPS'%sampling_rate)
        pylab.legend()
        
        #linear interpolation #Can probably use interp1d here
        @profile
        def f2(sample_times, u, V_noise): 
            V_interp = []
            for i in sample_times:
                left_cut = numpy.less(u-i , 0)
                xl = u[left_cut][-1]
                xr = u[~left_cut][0]
                yl = V_noise[left_cut][-1]
                yr = V_noise[~left_cut][0] 
                V_interp.append( ((yr - yl)/(xr-xl))*(i-xl)+yl )
            V_interp = numpy.array(V_interp)
            return V_interp
        
        V_interp = f2(sample_times, u, V_noise)
        pylab.subplot(3,1,2,sharex=ax,sharey=ax)
        pylab.ylabel('V (V)')
        pylab.xlabel('t (ns)')
        pylab.scatter(u,V_noise,label='Signal')
        pylab.stem(sample_times,V_interp,linefmt='r-', markerfmt='rs', basefmt='r-',label='Interp Sampled at %0.2f GSPS'%sampling_rate)
        
        #using scipy interp1d
        
        @profile
        def f3(sample_times, u, V_noise): 
            #Clearly the fastest from profiling
            f = scipy.interpolate.interp1d(u,V_noise)
            return f(sample_times)
        
        V_interp2 = f3(sample_times, u, V_noise)
        pylab.subplot(3,1,3,sharex=ax,sharey=ax)
        pylab.ylabel('V (V)')
        pylab.xlabel('t (ns)')
        pylab.scatter(u,V_noise,label='Signal')
        pylab.stem(sample_times,V_interp2,linefmt='r-', markerfmt='rs', basefmt='r-',label='Interp Sampled at %0.2f GSPS'%sampling_rate)
        
        
    if old_testing == True:
        #Testing making a table for an event
        from gnosim.trace.refraction_library_beta import *
        reader = h5py.File('./Output/results_2018_Dec_config_dipole_octo_-200_polar_120_rays_3.00e+09_GeV_100_events_1_seed_1.h5' , 'r')
        
        info = reader['info'][...]
        
        for eventid in [15]:
            #Note noise is kind of jank and will always be the same
            df = signalsFromInfo(eventid,reader,input_u,n,h_fft,sys_fft,freqs,include_noise = True,resistance = 50, temperature = 320)
            sub_info = info[info['eventid'] == eventid]
            origin = [[0,0,-200],[0,0,-201],[0,0,-202],[0,0,-203],[0,0,-204],[0,0,-205],[0,0,-206],[0,0,-207]]
            neutrino_loc = [reader['x_0'][eventid],reader['y_0'][eventid],reader['z_0'][eventid]]
            phi_0 = reader['phi_0'][eventid]
            print(sub_info)

############################################################
