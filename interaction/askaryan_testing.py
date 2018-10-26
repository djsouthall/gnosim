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
from gnosim.trace.refraction_library_beta import *
import gnosim.interaction.askaryan
import gnosim.sim.detector
pylab.ion()

############################################################

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
    
    cherenkov_angle = numpy.arccos(1./n)
    #mu_0 = gnosim.utils.constants.mu_0 # m kg s^-2 A^-2
    prefactor = 4. * numpy.pi /( gnosim.utils.constants.mu_0 * numpy.sin(cherenkov_angle))
    ra = RA(Energy_GeV,t_ns)  
    #print(ra)
    return prefactor * ra / LQ
    
def vectorPotentialTimeDomain(theta_obs_rad,R,Energy_GeV,n,u,plot = False):
    '''
    This should do it all?
    '''
    cherenkov_angle = numpy.arccos(1./n)
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
        fourier_fp = numpy.fft.rfft(fp)
        fourier_q = numpy.fft.rfft(q)

        convo = numpy.multiply(fourier_fp,fourier_q)
        
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

def electricFieldTimeDomainSignal(theta_obs_rad,R,Energy_GeV,n,h_fft=None,sys_fft=None,freqs=None,plot=False,out_dom_freq = False,return_pos = False,mode='v2',up_sample_factor=10,deriv_mode = 'time'):  
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
    if out_dom_freq == True:
        dominant_freq = f[numpy.argmax(numpy.absolute(V_fft))]
        return V, u, dominant_freq
    else:
        return V, u
    
def addSignals(u_in,E_in,plot=False):
    '''
    u_in should be an array of times with dimensions (n_signal , n_timestep )
    E_in should be an array of electric fields with dimensions (n_signal , n_timestep )
    Not that the timing of signals may be shifted by up to a u-step here to align 
    descretized timing values.  There may be a more elegant want do this if this
    added wiggle becomes a problem. 
    '''
    u_out = numpy.concatenate(u_in)
    u_step = abs(u_out[1]-u_out[0])
    u_out = numpy.arange(min(u_out),max(u_out)+u_step,u_step)
    E_out = numpy.zeros_like(u_out)
    if plot == True:
        pylab.figure()    
        ax = pylab.subplot(numpy.shape(E_in)[0]+1,1,numpy.shape(E_in)[0]+1)
        pylab.xlim((min(u_out),max(u_out)))
    for i in range(numpy.shape(E_in)[0]):
        E = E_in[i]
        u = u_in[i]
        
        if len(u) == 0:
            u = u_out
            E = numpy.zeros_like(u_out)   
        #print('Lengths:')
        #print(len(E))
        #print(len(u))
        #print('%i:%i ->%i'%(numpy.argmin(abs(u_out - min(u))),numpy.argmin(abs(u_out - min(u)))+len(u),len(E_out[numpy.argmin(abs(u_out - min(u))):numpy.argmin(abs(u_out - min(u)))+len(u)])))
        left_index = numpy.argmin(abs(u_out - min(u)))
        right_index = left_index + len(E)
        #print('left_index',left_index)
        #print('right_index',right_index)
        cut = numpy.arange(left_index,right_index)
        #print(len(cut))
        E_out[cut] += E
        if plot == True:
            pylab.subplot(numpy.shape(E_in)[0]+1,1,i+1,sharex=ax)
            pylab.plot(u,E,label='Signal %i'%(i))
            pylab.ylabel('E (V/m)',fontsize=16)
            pylab.xlabel('t-t_emit (ns)',fontsize=16)
            pylab.legend(fontsize=14)
    if plot == True:
        pylab.subplot(numpy.shape(E_in)[0]+1,1,numpy.shape(E_in)[0]+1)
        pylab.plot(u_out,E_out,label='Total Signal')
        pylab.ylabel('E (V/m)',fontsize=16)
        pylab.xlabel('t-t_emit (ns)',fontsize=16)
        pylab.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
        pylab.legend(fontsize=14)
    return E_out,u_out


############################################################

if __name__ == "__main__":
    pylab.close('all')
    energy_neutrino = 3.e9 # GeV
    n = 1.78
    R = 1000. #m
    cherenkov_angle = numpy.arccos(1./n)
    cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./n))
    '''
    u_steps = numpy.round(numpy.linspace(1000,100000,40))*2
    sums = []
    angle = 50.
    for us in u_steps:
        A,u = vectorPotentialTimeDomain(numpy.deg2rad(angle),R,energy_neutrino,n,u_steps = us)
        sums.append(numpy.absolute(sum(A)*(u[1] - u[0])))
    pylab.plot(u_steps,sums)
    '''  
    '''
    steps = 2000000
    A1,u = vectorPotentialTimeDomain(cherenkov_angle,R,energy_neutrino,n,u_steps = steps)
    A2,u = vectorPotentialTimeDomain(cherenkov_angle+0.00001,R,energy_neutrino,n,u_steps = steps)
    A3,u = vectorPotentialTimeDomain(cherenkov_angle-0.00001,R,energy_neutrino,n,u_steps = steps)
    
    pylab.figure()
    pylab.plot(u,A1,label='cherenkov')
    pylab.plot(u,A2,label='cherenkov+0.00001')
    pylab.plot(u,A2,label='cherenkov-0.00001')
    print('sum(numpy.absolute(A2))/sum(numpy.absolute(A1))',sum(numpy.absolute(A2))/sum(numpy.absolute(A1)))
    print('sum(numpy.absolute(A3))/sum(numpy.absolute(A1))',sum(numpy.absolute(A3))/sum(numpy.absolute(A1)))
    A,u = vectorPotentialTimeDomain(cherenkov_angle,R,energy_neutrino,n)
    pylab.figure()
    pylab.semilogy(u,R*abs(numpy.absolute(A)))
    pylab.xlim([-2,2])
    pylab.xlabel('$\Delta t$')
    pylab.ylabel('R |A(t)| (V s)')
    
    #angles = numpy.linspace(0.9*numpy.rad2deg(cherenkov_angle),1.1*numpy.rad2deg(cherenkov_angle),4)
    #angles = numpy.append(angles,numpy.rad2deg(cherenkov_angle))
    angles = numpy.linspace(5,85,20) 
    #angles = [numpy.rad2deg(cherenkov_angle)+0.01,numpy.rad2deg(cherenkov_angle)]
    pylab.figure()
    pylab.title('Vector Potential for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R))
    pylab.ylabel('RA (V s)')
    pylab.xlabel('t (ns)')
    for angle in angles:
        A,u = vectorPotentialTimeDomain(numpy.deg2rad(angle),R,energy_neutrino,n)
        pylab.plot(u,R*A,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend() 
    
    pylab.figure()
    pylab.title('Electric Field for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R))
    pylab.ylabel('RE (V)')
    pylab.xlabel('t (ns)')
    for angle in angles:
        E,u = electricFieldTimeDomainRaw(numpy.deg2rad(angle),R,energy_neutrino,n)
        pylab.plot(u,R*E,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend()
    
    A,u = vectorPotentialTimeDomain(cherenkov_angle,R,energy_neutrino,n)
    pylab.figure()
    pylab.semilogy(u,R*abs(numpy.absolute(A)))
    pylab.xlim([-2,2])
    pylab.xlabel('$\Delta t$')
    pylab.ylabel('R |A(t)| (V s)')
    #pylab.semilogy(u,abs(RA(energy_neutrino,u)))
    '''
    '''
    test_angle = 70.
    E,u = electricFieldTimeDomainRaw(numpy.deg2rad(test_angle),R,energy_neutrino,n)
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
      
    angle = 60.
    h_fft,sys_fft,freqs = loadSignalResponse()
    t_step = numpy.absolute(1/(len(freqs)*(freqs[1]-freqs[0])))*1e9 #ns
    E,u = electricFieldTimeDomainRaw(numpy.deg2rad(angle),R,energy_neutrino,n,u_step=t_step/2**7)
    E_fft = numpy.fft.rfft(E.real)
    freqs2 = numpy.fft.rfftfreq(len(u),d=(u[1]-u[0])*1e-9)
    #E_fft = scipy.fftpack.fftshift(scipy.fftpack.fft(E))
    #freqs2 = scipy.fftpack.fftshift((numpy.fft.fftfreq(len(u),d=(u[1]-u[0])*1e-9)))
    pylab.figure()
    
    pylab.scatter(freqs,numpy.absolute(h_fft),label='h_fft')
    pylab.scatter(freqs,numpy.absolute(sys_fft),label='sys_fft')
    pylab.scatter(freqs2[numpy.arange(len(freqs))],numpy.absolute(E_fft)[numpy.arange(len(freqs))],label='E_fft')
    pylab.xlim(0,1e9)
    
    E2 = numpy.fft.irfft(numpy.multiply(numpy.multiply(E_fft[numpy.arange(len(freqs))],sys_fft),h_fft))
    t = numpy.arange(len(E2))*(u[1]-u[0])
    pylab.figure()
    pylab.plot(t,E2)
    pylab.plot(u,E)
    '''
    
    '''
    angles = numpy.linspace(cherenkov_angle_deg-2,cherenkov_angle_deg+2,10) 
    
    
    #angles = [numpy.rad2deg(cherenkov_angle)+0.01,numpy.rad2deg(cherenkov_angle)]
    pylab.figure()
    pylab.title('Electric Field With Response for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f, n = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R,n),fontsize=16)
    pylab.ylabel('E (arb)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    for angle in angles:
        E_sig, u = electricFieldTimeDomainSignal(numpy.deg2rad(angle),R,energy_neutrino,n,plot=False,normalize_plot = True)
        pylab.plot(u,E_sig,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend() 
    pylab.xlim([0,50])
    ''' 
    '''
    angles = numpy.linspace(30,150,10)
    pylab.figure()
    pylab.title('Electric Field Before Response for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f, n = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R,n),fontsize=16)
    pylab.ylabel('R|E| (V)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    for angle in angles:
        E, u = electricFieldTimeDomainRaw(numpy.deg2rad(angle),R,energy_neutrino,n)
        pylab.plot(u,R*E,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend(loc='upper right') 
    pylab.xlim(-15,15)
    
    pylab.figure()
    pylab.title('Electric Field With Response for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f, n = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R,n),fontsize=16)
    pylab.ylabel('E (V/m)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    for angle in angles:
        E_sig, u = electricFieldTimeDomainSignal(numpy.deg2rad(angle),R,energy_neutrino,n,plot=False,normalize_plot = True)
        pylab.plot(u,E_sig,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend(loc='upper right') 
    
    angles = numpy.linspace(cherenkov_angle_deg-1,cherenkov_angle_deg+1,20)
    angles = numpy.sort(numpy.append(angles,cherenkov_angle_deg))
    
    pylab.figure()
    pylab.title('Electric Field Before Response for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f, n = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R,n),fontsize=16)
    pylab.ylabel('R|E| (V)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    for angle in angles:
        E, u = electricFieldTimeDomainRaw(numpy.deg2rad(angle),R,energy_neutrino,n)
        pylab.plot(u,R*E,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend(loc='upper right') 
    pylab.xlim(-15,15)
    
    pylab.figure()
    pylab.title('Electric Field With Response for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f, n = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R,n),fontsize=16)
    pylab.ylabel('E (V/m)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    for angle in angles:
        E_sig, u = electricFieldTimeDomainSignal(numpy.deg2rad(angle),R,energy_neutrino,n,plot=False,normalize_plot = True)
        pylab.plot(u,E_sig,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend(loc='upper right') 
    
    angles = numpy.linspace(cherenkov_angle_deg-.1,cherenkov_angle_deg+.1,20)
    angles = numpy.sort(numpy.append(angles,cherenkov_angle_deg))
    
    pylab.figure()
    pylab.title('Electric Field Before Response for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f, n = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R,n),fontsize=16)
    pylab.ylabel('R|E| (V)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    for angle in angles:
        E, u = electricFieldTimeDomainRaw(numpy.deg2rad(angle),R,energy_neutrino,n)
        pylab.plot(u,R*E,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend(loc='upper right') 
    pylab.xlim(-15,15)
    
    pylab.figure()
    pylab.title('Electric Field With Response for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f, n = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R,n),fontsize=16)
    pylab.ylabel('E (V/m)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    for angle in angles:
        E_sig, u = electricFieldTimeDomainSignal(numpy.deg2rad(angle),R,energy_neutrino,n,plot=False,normalize_plot = True)
        pylab.plot(u,E_sig,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend(loc='upper right') 
    #pylab.xlim([0,50])
    
    
    #pylab.xlim([-10,10])
    #angle = cherenkov_angle_deg - 0.3
    #A,u = vectorPotentialTimeDomain(numpy.deg2rad(angle),R,energy_neutrino,n,plot=True)
    #E_raw, u = electricFieldTimeDomainRaw(numpy.deg2rad(angle),R,energy_neutrino,n,plot = True)
    '''
    '''
    angle = cherenkov_angle_deg+0.3
    pylab.figure()
    for n in numpy.linspace(1.3,1.8,10):
        E_sig, u = electricFieldTimeDomainSignal(numpy.deg2rad(angle),R,energy_neutrino,n,plot=False,normalize_plot = False)
        pylab.plot(u[u>=0],E_sig[u>=0])
        print(sum(E_sig))
    pylab.legend() 
    '''
    #####################
    
    #E_sig1, u1 = electricFieldTimeDomainSignal(numpy.deg2rad(50),R,energy_neutrino,n,plot=True,normalize_plot = False,return_pos=True,mode='v1')
    #E_sig2, u2 = electricFieldTimeDomainSignal(numpy.deg2rad(50),R,energy_neutrino,n,plot=True,normalize_plot = False,return_pos=True,mode='v2')
    
    #reader = h5py.File('/home/dsouthall/Projects/GNOSim/Output/results_2018_Sep_config_duo_-200_polar_120_rays_1.00e+08_GeV_100_events_0_seed_1.h5' , 'r')
    #event = 25
    #info = reader['info'][...]
    #sub_info = info[info['eventid'] == event]
    #origin = [[0,0,-200],[0,0,-207]]
    #neutrino_loc = [reader['x_0'][event],reader['y_0'][event],reader['z_0'][event]]
    #phi_0 = reader['phi_0'][event]
    #fig = plotGeometry(origin,neutrino_loc,phi_0,sub_info)
    
    
    angles = numpy.linspace(cherenkov_angle_deg-2,cherenkov_angle_deg+2,10)
    angles = numpy.append( angles,cherenkov_angle_deg)
    
    
    #angles = [numpy.rad2deg(cherenkov_angle)+0.01,numpy.rad2deg(cherenkov_angle)]
    pylab.figure()
    pylab.title('Electric Field With Response for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f, n = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R,n),fontsize=16)
    pylab.ylabel('E (arb)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    for angle in angles:
        E_sig, u = electricFieldTimeDomainSignal(numpy.deg2rad(angle),R,energy_neutrino,n,plot=False,deriv_mode='freq')
        pylab.plot(u,E_sig,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend() 
    pylab.xlim([0,50])
    
    angles = numpy.linspace(cherenkov_angle_deg-2,cherenkov_angle_deg+2,10)
    angles = numpy.append( angles,cherenkov_angle_deg)
    
    
    #angles = [numpy.rad2deg(cherenkov_angle)+0.01,numpy.rad2deg(cherenkov_angle)]
    pylab.figure()
    pylab.title('Electric Field With Response for Various Angles ($\\theta_{Cherenkov}$ = %f, R = %0.2f, n = %0.2f)'%(numpy.rad2deg(cherenkov_angle),R,n),fontsize=16)
    pylab.ylabel('E (arb)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    for angle in angles:
        E_sig, u = electricFieldTimeDomainSignal(numpy.deg2rad(angle),R,energy_neutrino,n,plot=False,deriv_mode='time')
        pylab.plot(u,E_sig,label='$\\theta$ = %.2f deg'%(angle))
    pylab.legend() 
    pylab.xlim([0,50])
    
    
    #V,u,f_dom = electricFieldTimeDomainSignal(numpy.deg2rad(50),R,energy_neutrino,n,h_fft=None,sys_fft=None,freqs=None,plot=True,out_dom_freq = True,return_pos = True,mode='v2')
    #'''

    #'''
    #######################
    
    test_angle = cherenkov_angle_deg-0.3
    
    #######################
    
    upsamples = numpy.arange(0,15)
    pylab.figure()
    
    pylab.subplot(311)
    pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(energy_neutrino,test_angle,n))
    pylab.ylabel('Max Voltage (V)',fontsize=16)
    pylab.xlabel('time step',fontsize=16)
    pylab.subplot(312)
    pylab.xlabel('time window',fontsize=16)
    pylab.ylabel('Max Voltage (V)',fontsize=16)
    pylab.subplot(313)
    pylab.xlabel('N_points',fontsize=16)
    pylab.ylabel('Max Voltage (V)',fontsize=16)
    #pylab.subplot(414)
    #pylab.xlabel('Upsample')
    #pylab.ylabel('Max Voltage (V)')
    
    timesteps = numpy.array([])
    timewindows = numpy.array([])
    N_points = numpy.array([])
    max_V = numpy.array([])
    for upsample in upsamples:
        V1,u1,f_dom1 = electricFieldTimeDomainSignal(numpy.deg2rad(test_angle),R,energy_neutrino,n,h_fft=None,sys_fft=None,freqs=None,plot=False,out_dom_freq = True,return_pos = False,mode='v2',up_sample_factor = upsample)    
        V1_hilbert = scipy.signal.hilbert(V1)
        V1_envelope = numpy.abs(V1_hilbert)
        timesteps = numpy.append(timesteps,u1[1]-u1[0])
        timewindows = numpy.append(timewindows,u1[-1]-u1[0])
        N_points = numpy.append(N_points,len(u1))
        max_V = numpy.append(max_V,max(V1_envelope))
    
    pylab.subplot(311)
    pylab.scatter(timesteps,max_V)
    pylab.subplot(312)
    pylab.scatter(timewindows,max_V)
    pylab.subplot(313)
    pylab.scatter(N_points ,max_V)
    #pylab.subplot(414)
    #pylab.scatter(upsamples ,max_V)
    #####################
    
    
    upsamples = [0,1,5,10,20]
    pylab.figure()
    
    pylab.subplot(211)
    pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(energy_neutrino,test_angle,n))
    pylab.ylabel('Voltage (V)',fontsize=16)
    pylab.subplot(212)
    pylab.xlabel('t (ns)',fontsize=16)
    pylab.ylabel('Voltage (V)',fontsize=16)
    
    for upsample in upsamples:
        V1,u1,f_dom1 = electricFieldTimeDomainSignal(numpy.deg2rad(test_angle),R,energy_neutrino,n,h_fft=None,sys_fft=None,freqs=None,plot=False,out_dom_freq = True,return_pos = False,mode='v2',up_sample_factor = upsample)    
        V1_hilbert = scipy.signal.hilbert(V1)
        V1_envelope = numpy.abs(V1_hilbert)
        #print('max(V1_envelope)',max(V1_envelope))
        #print('len(u1)',len(u1))
        #print('u1[1]-u1[0]',u1[1]-u1[0])
        pylab.subplot(211)
        pylab.scatter(u1,V1,label='V_signal Upsample = %i'%(upsample))
        pylab.subplot(212)
        pylab.scatter(u1,V1_envelope,label='V_evelope Upsample = %i'%(upsample))
    
    pylab.subplot(211)
    pylab.legend()
    pylab.xlim(0,100)
    pylab.subplot(212)
    pylab.legend()
    pylab.xlim(0,100)
    ##################################################
    
    
    pylab.figure()
    pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(energy_neutrino,test_angle,n))
    pylab.xlabel('f(MHz)',fontsize=16)
    pylab.ylabel('V_fft',fontsize=16)
    for upsample in upsamples:
        V1,u1,f_dom1 = electricFieldTimeDomainSignal(numpy.deg2rad(test_angle),R,energy_neutrino,n,h_fft=None,sys_fft=None,freqs=None,plot=False,out_dom_freq = True,return_pos = False,mode='v2',up_sample_factor = upsample)    
        V1_fft = numpy.fft.rfft(V1)
        time_step = (u1[1]-u1[0]) #ns
        freq_step = 1/(len(u1)*(time_step*1e-9))
        max_freq = 1/(2*(time_step*1e-9))
        freqs = numpy.arange(len(V1_fft))*freq_step
        pylab.semilogy(freqs/1e6,numpy.absolute(V1_fft),label='n_points = %i'%(len(freqs)))
    pylab.xlim(0,1000)
    pylab.legend()
    
    pylab.figure()
    pylab.subplot(311)
    pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(energy_neutrino,test_angle,n))
    pylab.ylabel('R|A| (V s)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    pylab.subplot(312)
    pylab.ylabel('Time Domain Derivative\n R $\cdot$ E$_{raw}$ (V)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    pylab.subplot(313)
    pylab.ylabel('Freq Domain Derivative\nR $\cdot$ E$_{raw}$ (V)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    for upsample in upsamples:
        V1,u1,f_dom1 = electricFieldTimeDomainSignal(numpy.deg2rad(test_angle),R,energy_neutrino,n,h_fft=None,sys_fft=None,freqs=None,plot=False,out_dom_freq = True,return_pos = False,mode='v2',up_sample_factor = upsample)    
        
        A,u = vectorPotentialTimeDomain(numpy.deg2rad(test_angle),R,energy_neutrino,n,u1)
        pylab.subplot(311)
        pylab.plot(u,R*numpy.absolute(A),label='len(u) = %i'%(len(u)))
        
        E,u = electricFieldTimeDomainRaw(numpy.deg2rad(test_angle),R,energy_neutrino,n,u,plot = False,deriv_mode = 'time')
        pylab.subplot(312)
        pylab.plot(u,R*E,label='len(u) = %i'%(len(u)))
        
        E,u = electricFieldTimeDomainRaw(numpy.deg2rad(test_angle),R,energy_neutrino,n,u,plot = False,deriv_mode = 'freq')
        pylab.subplot(313)
        pylab.plot(u,R*E,label='len(u) = %i'%(len(u)))
    
    pylab.subplot(311)
    pylab.xlim(-1,1)
    pylab.legend(loc='upper right')
    pylab.subplot(312)
    pylab.xlim(-1,1)
    pylab.legend(loc='upper right')
    pylab.subplot(313)
    pylab.xlim(-1,1)
    pylab.legend(loc='upper right')
        
    pylab.figure()
    pylab.subplot(211)
    pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(energy_neutrino,test_angle,n))
    pylab.ylabel('fp',fontsize=16)
    pylab.xlabel('u (ns)',fontsize=16)
    pylab.subplot(212)
    pylab.xlabel('f(MHz)',fontsize=16)
    pylab.ylabel('fourier_fp',fontsize=16)
    for upsample in upsamples:
        V1,u1,f_dom1 = electricFieldTimeDomainSignal(numpy.deg2rad(test_angle),R,energy_neutrino,n,h_fft=None,sys_fft=None,freqs=None,plot=False,out_dom_freq = True,return_pos = False,mode='v2',up_sample_factor = upsample)    
        fp = F_p(energy_neutrino,u1,n,1)
        pylab.subplot(211)
        pylab.plot(u1,fp,label='n_points = %i'%(len(u1)))
        fourier_fp = numpy.fft.rfft(fp)
        fourier_fp = fourier_fp/len(fourier_fp)
        
        time_step = (u1[1]-u1[0]) #ns
        freq_step = 1/(len(u1)*(time_step*1e-9))
        max_freq = 1/(2*(time_step*1e-9))
        freqs = numpy.arange(len(fourier_fp))*freq_step #might be off by 1 step somewhere
        pylab.subplot(212)
        pylab.plot(freqs/1e6,numpy.absolute(fourier_fp),label='n_points = %i'%(len(freqs)))
    pylab.subplot(211)
    pylab.xlim(-1.5,1.5)
    pylab.legend()
    pylab.subplot(212)
    pylab.legend()
       
       
    V,u,f_dom = electricFieldTimeDomainSignal(numpy.deg2rad(cherenkov_angle_deg-0.3),R,energy_neutrino,n,h_fft=None,sys_fft=None,freqs=None,plot=True,out_dom_freq = True,return_pos = False,mode='v2',up_sample_factor = 15)
    pylab.figure()
    pylab.ylabel('Voltage (V)',fontsize=16)
    pylab.xlabel('t (ns)',fontsize=16)
    V_hilbert = scipy.signal.hilbert(V)
    V_envelope = numpy.abs(V_hilbert)
    pylab.scatter(u,V,label='V_signal')
    pylab.scatter(u,V_envelope,label='V_envelope')
    pylab.legend(fontsize=16)
    ''' 
    if abs(theta_deg-theta_cherenkov_deg) < 0.5:
        up_sample_factor = 30
    elif abs(theta_deg-theta_cherenkov_deg) < 10:
        up_sample_factor = 20
    else:
        up_sample_factor = 10
    '''
    #reduce_by = 2**7
    #lenu = len(sys)*reduce_by
    #u = numpy.arange(-(lenu*t_step/reduce_by)/2,(lenu*t_step/reduce_by)/2,t_step/reduce_by)
    #freqs_sig = numpy.fft.rfftfreq(len(u),d=(numpy.diff(u)[0])*1e-9)
    #print(freqs)
    #print(freqs_sig)
    #E_sig, u = electricFieldTimeDomainRaw(numpy.deg2rad(50),R,energy_neutrino,n,plot=True,u_step = t_step/reduce_by)
    #freqs_sig = numpy.fft.rfftfreq(len(u),d=(t_step/2**7)*1e-9)
    #E_fft = numpy.fft.rfft(E_sig)
    #V_fft = numpy.multiply(E_fft,response_fft)
    #V = numpy.fft.irfft(V_fft)
    #t = numpy.arange(len(u))*t_step
    #sys = numpy.append(sys,numpy.zeros(len(u)-len(sys)))
    #h = numpy.append(h,numpy.zeros(len(u)-len(h)))
    #h_fft = numpy.fft.rfft(h)
    #sys_fft = numpy.fft.rfft(sys)
    #freqs_sig = numpy.fft.rfftfreq(len(t),d=(t_step)*1e-9)
    #pylab.figure()
    #pylab.subplot(211)
    #pylab.plot(t,sys)
    #pylab.xlim(-10,200)
    
    #pylab.subplot(212)
    #pylab.plot(t,h)
    #pylab.xlim(-10,100)
    
    
    
    
    
    
    
############################################################
