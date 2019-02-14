'''
This contains two physics models:
Older Method:
Askaryan Emission
Source: Lehtinen et al. 2003, arXiv:0309656
info: Calculates E in frequency domain

Newer Method:
Practical and Accurate Calculations of Askaryan Radiation
Source: Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283
info: Calculates E in the time domain for eventual phasing
'''


import numpy
import pylab
import math
import scipy.signal
import gnosim.utils.constants
import gnosim.interaction.inelasticity
import gnosim.sim.fpga
pylab.ion()

############################################################

def electricFieldFrequencyDomainRaw(frequency, d, angle, Energy_GeV, inelasticity, mode, index_of_refraction):
    '''
    This is the old method
    Askaryan Emission
    Source: Lehtinen et al. 2003, arXiv:0309656
    
    
    frequency = frequency (GHz)
    d = observation distance (m)
    angle = observation angle relative to shower axis (deg)
    energy neutrino = neutrino energy (GeV)
    inelasticity = inelasticity
    Return electric field (V m^-1 GHz^-1)
    '''
    omega = 2. * numpy.pi * frequency
    mu = 1.
    length = 1.5 # m
    
    energy_shower = inelasticity * Energy_GeV # GeV
    q = 5.5e-20 * energy_shower # C
    k = index_of_refraction * omega / gnosim.utils.constants.speed_light
    
    return (mu * gnosim.utils.constants.mu_0 * q * length * (gnosim.utils.constants.GHz_to_Hz * omega) / (d * numpy.sqrt(2. * numpy.pi))) \
        * numpy.sin(numpy.radians(angle)) \
        * numpy.exp(-0.5 * (k * length)**2 * (numpy.cos(numpy.radians(angle)) - index_of_refraction**-1)**2) \
        * gnosim.utils.constants.GHz_to_Hz # V m^-1 GHz^-1, dimensionless
    
############################################################

############################################################
'''
Practical and Accurate Calculations of Askaryan Radiation
Source: Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283
'''

def loadSignalResponse(mode='v7'):
    '''
    Returns the fft's of the signals, and their frequencies.  
    Eventually these file locations should be part of the config file such that
    individual response functions could be input per antenna.  
    '''
    if mode == 'v1':
        print('Loading Signal Response V1')
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_elect_response.npy')
        freqs, h_fft = numpy.hsplit(antenna_response, 2)
        freqs, sys_fft = numpy.hsplit(electronic_response, 2)
        h_fft = numpy.ravel(h_fft)
        sys_fft = numpy.ravel(sys_fft)
        freqs = freqs[:,0]
    elif mode == 'v2':
        print('Loading Signal Response V2')
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_v2.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_elect_response_v2.npy')
        freqs, h_fft = numpy.hsplit(antenna_response, 2)
        freqs, sys_fft = numpy.hsplit(electronic_response, 2)
        h_fft = numpy.ravel(h_fft)
        sys_fft = numpy.ravel(sys_fft)
        freqs = freqs[:,0]
    elif mode == 'v3':
        print('Loading Signal Response V3')
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_v3.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_v3.npy')
        freqs, h_fft = numpy.hsplit(antenna_response, 2)
        freqs, sys_fft = numpy.hsplit(electronic_response, 2)
        h_fft = numpy.ravel(h_fft)
        sys_fft = numpy.ravel(sys_fft)
        freqs = freqs[:,0]
    elif mode == 'v4':
        print('Loading Signal Response V4')
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_v4.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_v4.npy')
        freqs, h_fft = numpy.hsplit(antenna_response, 2)
        freqs, sys_fft = numpy.hsplit(electronic_response, 2)
        h_fft = numpy.ravel(h_fft)
        sys_fft = numpy.ravel(sys_fft)
        freqs = freqs[:,0]
    elif mode == 'v5':
        print('Loading Signal Response V5')
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_v5.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_v5.npy')
        freqs, h_fft = numpy.hsplit(antenna_response, 2)
        freqs, sys_fft = numpy.hsplit(electronic_response, 2)
        h_fft = numpy.ravel(h_fft)
        sys_fft = numpy.ravel(sys_fft)
        freqs = freqs[:,0]
    elif mode == 'v6':
        print('Loading Signal Response V6')
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_v6.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_v6.npy')
        freqs, h_fft = numpy.hsplit(antenna_response, 2)
        freqs, sys_fft = numpy.hsplit(electronic_response, 2)
        h_fft = numpy.ravel(h_fft)
        sys_fft = numpy.ravel(sys_fft)
        freqs =  numpy.ravel(freqs)
    elif mode == 'v7':
        print('Loading Signal Response V7')
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_v7.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_v7.npy')
        freqs, h_fft = numpy.hsplit(antenna_response, 2)
        freqs, sys_fft = numpy.hsplit(electronic_response, 2)
        h_fft = numpy.ravel(h_fft)
        sys_fft = numpy.ravel(sys_fft)
        freqs =  numpy.ravel(freqs).astype(float)
    else:
        print('Error, defaulting to loading Signal Response V7')
        antenna_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_antenna_response_v7.npy')
        electronic_response = numpy.load('/home/dsouthall/Projects/GNOSim/gnosim/sim/response/ara_system_response_v7.npy')
        freqs, h_fft = numpy.hsplit(antenna_response, 2)
        freqs, sys_fft = numpy.hsplit(electronic_response, 2)
        h_fft = numpy.ravel(h_fft)
        sys_fft = numpy.ravel(sys_fft)
        freqs =  numpy.ravel(freqs).astype(float)
    
    #sys_fft = numpy.ones_like(sys_fft) #to look at just one
    #h_fft = numpy.ones_like(h_fft) #to look at just one
    return h_fft,sys_fft,freqs

def RA(Energy_GeV,t_ns):
    '''
    Ideally this would be a function of index of refraction as well, right now
    this model is parameterized specifically for n = 1.78
    I beliee this should return units of V s (i.e. it is disgned to output SI,
    not something in terms of ns)
    
    #THIS IS NOT DEPENDANT ON ANYTHING OTHER THAN ENERGY, CAN BE CALCULATED EXTERNALLY AND FED TO QUICKSIGNAL
    '''
    Energy_TeV = 0.001*Energy_GeV
    if numpy.size(t_ns) == 1:
        if t_ns > 0:
            return (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns)/0.057) + (1. + 2.87*numpy.fabs(t_ns))**(-3.0))
        else:
            return (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns)/0.030) + (1. + 3.05*numpy.fabs(t_ns))**(-3.5)) 
    else:
        ra = numpy.zeros_like(t_ns)
        #ra_gt = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns > 0])/0.057) + (1. + 2.87*numpy.fabs(t_ns[t_ns > 0]))**(-3.0))
        #ra[t_ns > 0] = ra_gt
        #ra_lt = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns <= 0])/0.030) + (1. + 3.05*numpy.fabs(t_ns[t_ns <= 0]))**(-3.5)) 
        #ra[t_ns <= 0] = ra_lt
        ra[t_ns > 0] = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns > 0])/0.057) + (1. + 2.87*numpy.fabs(t_ns[t_ns > 0]))**(-3.0))
        ra[t_ns <= 0] = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns[t_ns <= 0])/0.030) + (1. + 3.05*numpy.fabs(t_ns[t_ns <= 0]))**(-3.5)) 
        return ra


def Q(x,a=4.85,b=0.60,loc=0,scale=1.0,random_params=False,random_local = None):
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
            #NOTE THIS MAY THROW OFF SEEDING.  THIS FEATURE WAS ADDED BUT NEVER USED, SO RANDOMNESS NOT REALLY ACCOUNTED FOR.
            #With random_local this should be consistent between trigger/pretrigger events, but likely not
            #between when using this feature and not.  Would ruin all noise etc.  Be careful if you ever implement this
            if random_local == None:
                a = numpy.random.normal(loc=4.85,scale=1.01,size=None)
                b = numpy.random.normal(loc=0.60,scale=0.14,size=None)
            else:
                a = random_local.normal(loc=4.85,scale=1.01,size=None)
                b = random_local.normal(loc=0.60,scale=0.14,size=None)
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
    '''
    cherenkov_angle = numpy.arccos(1./n)
    #mu_0 = gnosim.utils.constants.mu_0 # m kg s^-2 A^-2
    prefactor = 4. * numpy.pi /( gnosim.utils.constants.mu_0 * numpy.sin(cherenkov_angle))
    ra = RA(Energy_GeV,t_ns)  
    #print(ra)
    return prefactor * ra / LQ
    '''
    ra = RA(Energy_GeV,t_ns)  
    fp = (4. * numpy.pi /(LQ * gnosim.utils.constants.mu_0 * math.sqrt(1-1/n**2))) * ra #note that math.sqrt(1-1/1.78**2) is a faster form of numpy.sin(cherenkov_angle) = numpy.sin(numpy.arccos(1/n))
    fp = numpy.multiply(scipy.signal.tukey(len(fp),alpha=0.05),fp)
    return fp
    
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
            pylab.title('E = %g \t$\\theta$=%0.3f \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
            pylab.ylabel('$R|\\vec{E}|$ (V)',fontsize=16)
            pylab.xlabel('t (ns)',fontsize=16)
            pylab.plot(u,R*E,label = '$R|\\vec{E}_{raw}|$ ')
    return  E , u

def electricFieldTimeDomainSignal(theta_obs_rad,R,Energy_GeV,n,h_fft=None,sys_fft=None,freqs=None,plot=False,out_dom_freq = False,return_pos = False,mode='v7',up_sample_factor=10,deriv_mode = 'time'):  
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
        pylab.title('E = %g \t$\\theta$=%0.3f \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
        pylab.plot(u,V)
        pylab.ylabel('Signal (V)',fontsize=16)
        pylab.xlabel('t (ns)',fontsize=16)
        
        pylab.figure()
        pylab.title('E = %g \t$\\theta$=%0.3f \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
        pylab.plot(f/1e6,20.0 * numpy.log10(numpy.absolute(E_fft)),label='Raw Signal (fft)')
        pylab.plot(f/1e6,20.0 * numpy.log10(numpy.absolute(h_fft)),label='Antenna Response')
        pylab.plot(f/1e6,20.0 * numpy.log10(numpy.absolute(sys_fft)),label='System Response')
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

def addSignals(u_in,V_in,plot=False,V_noise_in = [], remove_noise_overlap = False):
    '''
    u_in should be an array of times with dimensions (n_signal , n_timestep )
    u is assumed to be in order, i.e. u[0] is the min of each row and u[-1] is the max.
    Each row of u is also assumed to have the same time step.  
    V_in should be an array of electric fields with dimensions (n_signal , n_timestep )
    Note that the timing of signals may be shifted by up to a u-step here to align 
    descretized timing values.  There may be a more elegant want do this if this
    added wiggle becomes a problem. 
    
    If remove_noise_overlap is true and an array V_noise_in is provided, this will
    remove noise from each signal in regions of overlapping noise (using a ramp) such
    that noise does not add where realistically it should be one continuous stream
    of noise.  
    
    V_noise in should be the same shape as V_in, and should correspond to the same
    times u_in.  
    
    For future Dan:  I need to add an option input for when adding signals with Noise
    to remove noise in overlapping region.  Should have a flag for remove_noise=True
    and then two required inputs: V_in_no_noise, u_in_no_noise. That should be
    the same length as u_in, V_in. In region of overlapping noise/signal I need
    to subtract off real signs, ramp noise in overlapping regions, and then add
    back the signals.  
    '''
    if len(numpy.shape(u_in)) <=1:
        return V_in, u_in
    elif numpy.shape(u_in)[0] == 1:
        return V_in.flatten(),u_in.flatten()
    else:
        if numpy.logical_and(remove_noise_overlap == True, V_noise_in == []):
            print('V_noise_in is empty, ignoring.')
            remove_noise_overlap = False
        u_step = u_in[0,1]-u_in[0,0]
        u_out_min = min(u_in[:,0])
        u_out_max = max(u_in[:,-1])
        
        u_out = numpy.arange(u_out_min,u_out_max+u_step,u_step)
        
        if remove_noise_overlap == True:
            V_just_signal = numpy.zeros_like(u_out)
            V_out = numpy.tile(V_just_signal,(numpy.shape(V_in)[0],1))
        else:
            V_out = numpy.zeros_like(u_out)
        if plot == True:
            pylab.figure(figsize=(16.,11.2))   
            ax = pylab.subplot(numpy.shape(V_in)[0]+1,1,numpy.shape(V_in)[0]+1)
            
            pylab.xlim((u_out_min,u_out_max))
        for i in range(numpy.shape(V_in)[0]):
            V = V_in[i,:]
            u = u_in[i,:]
            
            if len(u) == 0:
                u = u_out
                V = numpy.zeros_like(u_out)   
            left_index = numpy.argmin(abs(u_out - u[0]))
            right_index = left_index + len(V)
            
            if remove_noise_overlap == True:
                V_out[i,left_index:right_index] += V_noise_in[i,:]
                V_just_signal[left_index:right_index] += numpy.add(V,-V_noise_in[i,:])
            else:
                V_out[left_index:right_index] += V
            if plot == True:
                pylab.subplot(numpy.shape(V_in)[0]+1,1,i+1,sharex=ax,sharey=ax)
                if i == 0:
                    if remove_noise_overlap == True:
                        pylab.title('Overlapping Noise Effect Removed') 
                    else:
                        pylab.title('Overlapping Noise Effect NOT Removed') 
                pylab.plot(u,V,label='Signal %i'%(i))
                pylab.ylabel('V (V)',fontsize=14)
                #pylab.xlabel('t-t_emit (ns)',fontsize=14)
                pylab.legend(fontsize=12,loc='upper right')
        
        if remove_noise_overlap == True:
            weights = 1/numpy.sqrt(numpy.maximum(numpy.sum((V_out != 0)*1.0,axis=0),numpy.ones_like(u_out)))
            V_out = numpy.multiply(numpy.sum(V_out,axis=0),weights) + V_just_signal
        
        if plot == True:
            pylab.subplot(numpy.shape(V_in)[0]+1,1,numpy.shape(V_in)[0]+1,sharex=ax,sharey=ax)
            pylab.plot(u_out,V_out,label='Total Signal')
            pylab.ylabel('V (V)',fontsize=14)
            pylab.xlabel('t-t_emit (ns)',fontsize=14)
            pylab.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
            pylab.legend(fontsize=12,loc='upper right')
            
        return V_out,u_out

def calculateTimes(up_sample_factor=20,h_fft=None,sys_fft=None,freqs=None,mode=None):
    '''
    Calculates the times used for signal calculations based on the response functions
    (assumed to have the same frequency step).  up_sample_factor is not exact, as the
    number of points is rounded to a factor of 2 to ensure future ifft's are as fast
    as possible.
    
    This also returns the newly padded responses and freqs.
    '''
    #Loading in response function and setting frequency / time steps
    if any([numpy.size(h_fft) ==1,numpy.size(sys_fft)==1,numpy.size(freqs)==1]):
        if mode == None:
            print('No mode given to pass to loadSignalResponse, that functions default mode will be used')
            h_fft,sys_fft,freqs = loadSignalResponse()
        else:
            print('Passing loadSignalResponse(mode=%s)'%mode)
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
    #response_fft = numpy.multiply(h_fft,sys_fft)
    
    t_step = 1/(2*max(freqs))*1e9 #ns
    u = numpy.arange(-(n_points_freq-1),(n_points_freq-1))*t_step #To increase time duration of signal I should just need to upsample?
    return u, h_fft, sys_fft, freqs

def quickSignalSingle(theta_obs_rad,R,Energy_GeV,n,t_offset,attenuation,beam_pattern_factor,u, h_fft, sys_fft, freqs, fp_fft = None,plot_signals=False,plot_spectrum=False,plot_angles = False,plot_potential = False,include_noise = False, resistance = 50, temperature = 320,random_local = None):  
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
    LQ = 1.0#excessProjectedTrackLength(Q,int_min=min(u),int_max=max(u),n_steps = len(u)) #can probably set to 1 for now as this Q is normalized?
    alpha = (1. - n*numpy.cos(theta_obs_rad))/gnosim.utils.constants.speed_light #scaling factor of u substitution units of ns/m

    #calling the below was slow, so I put it just calculate it in this function
    #fp = F_p(Energy_GeV,u,n,LQ)
    #fp = numpy.multiply(scipy.signal.tukey(len(fp),alpha=0.05),fp)
    if numpy.size(fp_fft) == 1:
        #ra = RA(Energy_GeV,u)  
        #fp = (4. * numpy.pi /(LQ * gnosim.utils.constants.mu_0 * math.sqrt(1-1/n**2))) * ra #note that math.sqrt(1-1/1.78**2) is a faster form of numpy.sin(cherenkov_angle) = numpy.sin(numpy.arccos(1/n))
        #fp = numpy.multiply(scipy.signal.tukey(len(fp),alpha=0.05),fp)
        fp = F_p(Energy_GeV,u,n,LQ)
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
        q_fft = numpy.fft.rfft(q) #should look into an analytica form of fft(gamma function) to see if there is a way to avoid this fft
        A_fft = numpy.multiply(fp_fft,q_fft) * ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) / (4. * numpy.pi * R ) ) * ( t_step/abs(alpha) ) #the t_step already accounts for scaling required with irffting.  Might not want here?  unsure
        
        if numpy.logical_or(plot_potential == True,plot_signals == True):
            A = numpy.fft.irfft(A_fft,n=len(u))
            A = numpy.fft.fftshift(A)
        if plot_potential == True:
            pylab.figure(figsize=(16.,11.2))
            ax = pylab.subplot(311)
            pylab.title('alpha = %0.3f, $\\theta$ = %0.2f deg'%(alpha,numpy.rad2deg(theta_obs_rad)),fontsize=20)
            pylab.plot(u,fp,label='fp')
            pylab.ylabel('$F_p$ ($Amps$)',fontsize=16)
            #pylab.xlim(-10,50)
            pylab.subplot(312,sharex = ax)
            pylab.plot(u,q,label='q')
            pylab.ylabel('$Q (arb)$ ',fontsize=16)
            #pylab.xlim(-10,50)
            pylab.subplot(313,sharex = ax)
            pylab.semilogy(u,numpy.fabs(R*numpy.absolute(A)),label='RA')
            pylab.ylabel('$R|A|$ ',fontsize=16)
            pylab.xlabel('$\Delta t$',fontsize=16)
            #pylab.xlim(-10,50)
    #calculating E_raw_fft    
    E_raw_fft = -1.0j*2.0*numpy.pi*numpy.multiply(A_fft , freqs) #negitive sign because E = -dA/dt
    
    #Accouning for beam pattern
    E_raw_fft *= beam_pattern_factor
    
    #Accounting for attenuation
    E_raw_fft *= attenuation #Want to do before noise is added.  Noise is not attenuated by 
    #Adding antenna response
    E_antenna_fft = numpy.multiply(E_raw_fft, h_fft) 
    V_fft_noiseless = numpy.multiply(E_antenna_fft,sys_fft)
    
    V_noiseless = numpy.fft.irfft(V_fft_noiseless,n=len(u))
    
    if include_noise == True:
        #SNR Peak to Peak calculation and noiseless signal calculation
        
        V_noiseless_sorted = numpy.sort(V_noiseless)
        p2p_half = (V_noiseless_sorted[-1] - V_noiseless_sorted[0]) / 2
        
        #calculating noise
        bandwidth = freqs[-1]/1e9 #Calculating full band noise, response cuts out stuff we don't see
        
        V_rms = numpy.sqrt(gnosim.utils.constants.boltzmann * temperature * resistance * bandwidth * gnosim.utils.constants.GHz_to_Hz)
        #print('Internal V_rms is: %f'%V_rms)
        sigma = V_rms 
        
        #Noise in Polar
        #noise_phase = numpy.random.uniform(-numpy.pi,numpy.pi,size = len(freqs))
        #noise_amp = numpy.random.normal(loc = 0.0, scale = sigma , size = len(freqs)) #these might need the normalization factor of *numpy.sqrt(len(u)/2) if used at some point for some reason
        
        #Noise in Cartesian
        if random_local == None:
            noise_cartesian = numpy.sqrt(len(u)/2)*(numpy.random.normal(loc = 0.0, scale = sigma , size = len(freqs)) + 1.0j*numpy.random.normal(loc = 0.0, scale = sigma , size = len(freqs))) # the *numpy.sqrt(len(u)/2) factor is to handle some normalization issues
        else:
            #Should be used for multithreading
            noise_cartesian = numpy.sqrt(len(u)/2)*(random_local.normal(loc = 0.0, scale = sigma , size = len(freqs)) + 1.0j*random_local.normal(loc = 0.0, scale = sigma , size = len(freqs))) # the *numpy.sqrt(len(u)/2) factor is to handle some normalization issues
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
        V_rms_measured = numpy.sqrt(numpy.mean((V_noise - V_noiseless)**2)) #This is the 'measured' V_rms, rather than the set.  This is after system response
        SNR = (p2p_half/V_rms_measured)**2
        #SNR_dB = 10*numpy.log10( SNR )#dB, using 10log10 because input is power ratio 
        #print('SNR', SNR)
    else:
        SNR = 0
        #print('No noise included.  Cannot perform SNR calculation.')

    if plot_signals==True:
        print('Plotting within quickSignalSingle')
        E_raw = numpy.fft.irfft(E_raw_fft,n=len(u)) #should just be the signal, as A_fft already divided by scaling factor of t_step?
        A = numpy.fft.irfft(A_fft,n=len(u))
        if abs(alpha) >= 0.001:
            E_raw = numpy.fft.fftshift(E_raw) #This centres E so it occurs at t=0. and ensures it is located temporaly in the same place for if the calculation was done using the exception or on cone or not
            A = numpy.fft.fftshift(A) #This centres A so it occurs at t=0. and ensures it is located temporaly in the same place for if the calculation was done using the exception or on cone or not
            
        
        if include_noise == True:
            if plot_potential == True:
                pylab.figure(figsize=(16.,11.2))
                ax = pylab.subplot(211)
                pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f\tt_step = %g ns'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n,t_step),fontsize=20)
                pylab.ylabel('R*|A| (V s)')
                pylab.xlabel('t (ns)')
                #pylab.scatter(u,R*numpy.absolute(A),s=1)
                pylab.plot(u,R*numpy.absolute(A))
                
                pylab.subplot(212,sharex = ax)
                pylab.ylabel('$R \cdot E_{raw}$ (V)')
                pylab.xlabel('t (ns)')
                #pylab.scatter(u,R*E_raw,s=1)
                pylab.plot(u,R*E_raw)
            
            pylab.figure(figsize=(16.,11.2))
            ax = pylab.subplot(211)
            pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f\tt_step = %g ns'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n,t_step),fontsize=20)
            pylab.ylabel('Noiseless Signal Voltage (V)')
            pylab.xlabel('t (ns)')
            #pylab.scatter(u,V,s=1)
            pylab.plot(u,V_noiseless)
            
            pylab.subplot(212,sharex = ax)
            pylab.ylabel('Signal Voltage (V)')
            pylab.xlabel('t (ns)')
            #pylab.scatter(u,V,s=1)
            pylab.plot(u,V_noise)
        else:
            if plot_potential == True:
                pylab.figure(figsize=(16.,11.2))
                ax = pylab.subplot(211)
                pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f\tt_step = %g ns'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n,t_step),fontsize=20)
                pylab.ylabel('R*|A| (V s)')
                pylab.xlabel('t (ns)')
                #pylab.scatter(u,R*numpy.absolute(A),s=1)
                pylab.plot(u,R*numpy.absolute(A))
            
                pylab.subplot(212,sharex = ax)
                pylab.ylabel('$R \cdot E_{raw}$ (V)')
                pylab.xlabel('t (ns)')
                #pylab.scatter(u,R*E_raw,s=1)
                pylab.plot(u,R*E_raw)
            
            pylab.figure(figsize=(16.,11.2))
            pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f\tt_step = %g ns'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n,t_step),fontsize=20)
            #pylab.subplot(313,sharex = ax)
            pylab.ylabel('Noiseless Signal Voltage (V)')
            pylab.xlabel('t (ns)')
            #pylab.scatter(u,V,s=1)
            pylab.plot(u,V_noiseless)
    if plot_spectrum == True:
        pylab.figure(figsize=(16.,11.2))
        pylab.title('MAGNITUDE E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
        scatter = False
        
        '''
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
        pylab.xlabel("Freq. [Hz]")
        pylab.ylabel("Realized Gain [dBi]")
        '''
        
        #These scalings are for v6 onward.  Older scalings were probably incorrect anyways so I am just leaving these for all versions.
        sys_gain = numpy.absolute(sys_fft)**2
        sys_gain *= 2.0 # Because the rfft is missing half the power
        
        h_gain = numpy.absolute(h_fft)
        h_gain *= (1.74 / 3e8) * numpy.array(freqs)
        h_gain *= h_gain 
        h_gain *= 2.0 # Because the rfft is missing half the power
        h_gain *= 4.0 * numpy.pi
        
        #I don't know what to do about the two below to plot them on the same plot.  
        #Dan Smith determined the appropriate calculation for the sys_gain, h_gain
        
        ##
        raw_signal_gain = numpy.absolute(E_raw_fft)**2
        raw_signal_gain *= 2.0 # Because the rfft is missing half the power
        raw_signal_gain *= t_step**2   #To plot in units that are invariant under upsampling, doesn't effect time domain signal
        
        if include_noise == True:
            signal_gain = numpy.absolute(V_fft_noise)**2
            signal_gain *= 2.0 # Because the rfft is missing half the power
            signal_gain *= t_step**2
        else:
            signal_gain = numpy.absolute(V_fft_noiseless)**2
            signal_gain *= 2.0 # Because the rfft is missing half the power
            signal_gain *= t_step**2
        ##
        
        if scatter == True:
            pylab.scatter(freqs/1e6,10.0 * numpy.log10(raw_signal_gain),label='Raw Signal (fft)', color = 'blue')
            pylab.scatter(freqs/1e6,10.0 * numpy.log10(sys_gain),label='System Response (fft)', color = 'darkorange')
            pylab.scatter(freqs/1e6,10.0 * numpy.log10(h_gain),label='Antenna Response (fft)', color = 'limegreen')
            pylab.scatter(freqs/1e6,10.0 * numpy.log10(signal_gain),label='Processed Signal (fft)', color = 'red')
        else:
            pylab.plot(freqs/1e6,10.0 * numpy.log10(raw_signal_gain),label='Raw Signal (fft)', color = 'blue')
            pylab.plot(freqs/1e6,10.0 * numpy.log10(sys_gain),label='System Response (fft)', color = 'darkorange')
            pylab.plot(freqs/1e6,10.0 * numpy.log10(h_gain),label='Antenna Response (fft)', color = 'limegreen')
            pylab.plot(freqs/1e6,10.0 * numpy.log10(signal_gain),label='Processed Signal (fft)', color = 'red')
        
        pylab.ylabel("Realized Gain [dBi] (Signals in arb)",fontsize=16)
        pylab.xlabel('Freq. [MHz]',fontsize=16)
        #pylab.ylabel('dB (20.0 log10(numpy.absolute(V_fft)))',fontsize=16)
        pylab.ylim(-75,75)
        pylab.xlim(0,1000)
        pylab.minorticks_on()
        pylab.grid(which="both")
        pylab.legend(fontsize=14)
    if plot_angles == True:
        pylab.figure(figsize=(16.,11.2))
        ax = pylab.subplot(2,1,1)
        pylab.title('PHASE E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
        pylab.plot(freqs/1e6,numpy.rad2deg(numpy.angle(E_raw_fft)),label='Raw Signal (fft)', color = 'blue')
        pylab.plot(freqs/1e6,numpy.rad2deg(numpy.angle(sys_fft)),label='System Response (fft)', color = 'darkorange')
        pylab.plot(freqs/1e6,numpy.rad2deg(numpy.angle(h_fft)),label='Antenna Response (fft)', color = 'limegreen')
        pylab.ylabel('Angle [deg]',fontsize=16)
        pylab.legend(fontsize=14)
        
        
        pylab.subplot(2,1,2,sharex=ax,sharey=ax)
        if include_noise == True:
            pylab.plot(freqs/1e6,numpy.rad2deg(numpy.angle(V_fft_noise)),label='Processed Signal (fft)', color = 'red')
            pylab.plot(freqs/1e6,numpy.rad2deg(numpy.angle(V_fft_just_noise)),label='Just Noise (fft)', color = 'magenta')
        else:
            pylab.plot(freqs/1e6,numpy.rad2deg(numpy.angle(V_fft_noiseless)),label='Processed Signal (fft)', color = 'red')
            
        
        pylab.xlabel('Freq. [MHz]',fontsize=16)
        pylab.ylabel('Angle [deg]',fontsize=16)
        #pylab.ylim(-100,100)
        pylab.xlim(0,1500)
        pylab.legend(fontsize=14)
    
    if include_noise == True:
        dominant_freq = freqs[numpy.argmax(numpy.absolute(V_fft_noise))]
        return V_noiseless, u + t_offset, dominant_freq, V_noise,  SNR
    else:
        dominant_freq = freqs[numpy.argmax(numpy.absolute(V_fft_noiseless))]
        return V_noiseless, u + t_offset, dominant_freq

############################################################



if __name__ == "__main__":
    '''
    energy_neutrino = 1.e9 # GeV
    mode = 'cc'
    d = 1000. # m
    angle = numpy.linspace(0., 90., 1000)
    index_of_refraction = 1.8
    inelasticity = gnosim.interaction.inelasticity.inelasticity(energy_neutrino, mode)

    #omega = 0.5 # GHz
    '''
    """
    omega_array = numpy.arange(0.1, 1., 0.05)
    electric_field_array = []
    for omega in omega_array:
        electric_field_array.append(electricFieldFrequencyDomainRaw(omega, d, angle, energy_neutrino, mode, index_of_refraction))
    pylab.figure()
    for ii, omega in enumerate(omega_array):
        pylab.plot(angle, electric_field_array[ii])
    """
    #
    """
    angle_array = numpy.arange(45., 65, 2.5)
    omega = numpy.linspace(0.1, 1., 1000)
    electric_field_array = []
    for angle in angle_array:
        electric_field_array.append(electricFieldFrequencyDomainRaw(omega, d, angle, energy_neutrino, mode, index_of_refraction))
    pylab.figure()
    for ii, angle in enumerate(angle_array):
        pylab.plot(omega, electric_field_array[ii])
    """
    '''
    frequency_mesh, angle_mesh = numpy.meshgrid(numpy.linspace(0.01, 1.5, 100), numpy.arange(50., 65. + 1.e-10, 0.1))
    electric_field = electricFieldFrequencyDomainRaw(frequency_mesh, d, angle_mesh, energy_neutrino, inelasticity, mode, index_of_refraction)
    pylab.figure()
    #pylab.scatter(frequency_mesh, angle_mesh, c=electric_field, edgecolors='none')
    pylab.pcolormesh(frequency_mesh, angle_mesh, numpy.roll(electric_field, 0, 0))
    colorbar = pylab.colorbar()
    colorbar.set_label(r'V m$^{-1}$ GHz$^{-1}$')
    pylab.title(r'E$_{\nu}$ = %.1e GeV; d = %.1f m'%(energy_neutrino, d))
    pylab.xlabel('Frequency (GHz)')
    pylab.ylabel('Observation Angle (deg)')
    pylab.xlim([numpy.min(frequency_mesh), numpy.max(frequency_mesh)])
    pylab.ylim([numpy.min(angle_mesh), numpy.max(angle_mesh)])
    
    u,V,f_dom = electricFieldTimeDomainSignal(numpy.deg2rad(50),R,energy_neutrino,n,h_fft=None,sys_fft=None,freqs=None,plot=True,out_dom_freq = True,return_pos = True,mode='v2')
    u,V,f_dom = electricFieldTimeDomainSignal(numpy.deg2rad(50),R,energy_neutrino,n,h_fft=None,sys_fft=None,freqs=None,plot=True,out_dom_freq = True,return_pos = True,mode='v2')    
    '''
    pylab.close('all')
    energy_neutrino = 3.e9 # GeV
    n = 1.78
    c = gnosim.utils.constants.speed_light #m/ns
    
    R = 1000. #m
    cherenkov_angle = numpy.arccos(1./n)
    cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./n))
    h_fft,sys_fft,freqs = loadSignalResponse()
    input_u, h_fft, sys_fft, freqs = calculateTimes(up_sample_factor=20)
    inelasticity = 0.2
    noise_rms = numpy.std(quickSignalSingle(0,R,inelasticity*energy_neutrino,n,R,0,0,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)[3])
    V_noiseless, u, dominant_freq, V_noise,  SNR = quickSignalSingle(numpy.deg2rad(50),R,inelasticity*energy_neutrino,n,2500,0.7,0.7,input_u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential=False,include_noise = True)
    sampling_rate = 1.5 #GHz
    bytes = 7
    scale_noise_from = noise_rms
    scale_noise_to = 3
    
    random_time_offset = numpy.random.uniform(-5.0,5.0) #ns
    dc_offset = 0.0 #V
    sample_times=gnosim.sim.fpga.calculateDigitalTimes(u[0],u[-1],sampling_rate,  random_time_offset = random_time_offset)
    V_bit, sampled_times = gnosim.sim.fpga.digitizeSignal(u,V_noise,sample_times,bytes,scale_noise_from,scale_noise_to, dc_offset = dc_offset, plot = False)
    dt = sampled_times[1] - sampled_times[0]
    #################################################################
    config_file = '/home/dsouthall/Projects/GNOSim/gnosim/sim/ConfigFiles/Config_dsouthall/config_dipole_octo_-200_polar_120_rays.py'
    import yaml
    config = yaml.load(open(config_file))
    
    from gnosim.trace.refraction_library_beta import *
    #reader = h5py.File('./Output/results_2018_Dec_config_dipole_octo_-200_polar_120_rays_3.00e+09_GeV_100_events_1_seed_6.h5' , 'r')
    reader = h5py.File('./Output/results_2019_Jan_config_dipole_octo_-200_polar_120_rays_3.10e+09_GeV_100_events_1_seed_3.h5' , 'r')
    
    
    info = reader['info'][...]
    #info_cut = info[numpy.logical_and(info['SNR'] > 1 , info['SNR'] < 10) ]
    info_cut = info[numpy.logical_and(info['SNR'] > 1 , info['SNR'] < 100) ]
    #events 15, 92
    
    eventids = numpy.unique(info_cut[info_cut['has_solution']==1]['eventid'])
    choose_n = 3
    try:
        do_events = numpy.random.choice(eventids,choose_n,replace=False)
    except:
        do_events = numpy.unique(numpy.random.choice(eventids,choose_n,replace=True))
    
    output_just_noise = True
    for eventid in do_events:
        print('On event %i'%eventid)
        if output_just_noise == True:
            V, u, Vd, ud, V_just_noise = gnosim.interaction.askaryan_testing.signalsFromInfo(eventid,reader,input_u,n,h_fft,sys_fft,freqs,include_noise = True,resistance = 50, temperature = 320,plot = True,output_just_noise = True)
            V_out2,u_out2 = addSignals(u,V,plot=True,V_noise_in = [], remove_noise_overlap = False)
            V_out,u_out = addSignals(u,V,plot=True,V_noise_in = V_just_noise, remove_noise_overlap = True)
            '''
            V_in = V
            u_in = u
            left_indices = []
            right_indices = []
            
            u_step = u_in[0,1]-u_in[0,0]
            u_out_min = min(u_in[:,0])
            u_out_max = max(u_in[:,-1])
            u_out = numpy.arange(u_out_min,u_out_max+u_step,u_step)
            V_just_signal = numpy.zeros_like(u_out)
            V_out = numpy.tile(V_just_signal,(numpy.shape(V_in)[0],1))
            
            
            for i in range(numpy.shape(V_in)[0]):
                V = V_in[i,:]
                u = u_in[i,:]
                
                if len(u) == 0:
                    u = u_out
                    V = numpy.zeros_like(u_out)   
                left_index = numpy.argmin(abs(u_out - u[0]))
                right_index = left_index + len(V)
                
                V_out[i,left_index:right_index] += V_just_noise[i,:]
                
                V_just_signal[left_index:right_index] += numpy.add(V,-V_just_noise[i,:])
            
            
            weights = 1/numpy.sqrt(numpy.maximum(numpy.sum((V_out != 0)*1.0,axis=0),numpy.ones_like(u_out)))
            #V_out = numpy.sum(V_out,axis = 0)
            V_out = numpy.multiply(numpy.sum(V_out,axis=0),weights) + V_just_signal
            '''
            '''
            pylab.figure(figsize=(16.,11.2))
            pylab.plot(1/weights)
            
            pylab.figure(figsize=(16.,11.2))
            
            pylab.plot(u_out2,V_out2,label='Old')
            pylab.plot(u_out,V_out,label='New')
            
            pylab.legend()
            '''
        else:
            V, u, Vd, ud = gnosim.interaction.askaryan_testing.signalsFromInfo(eventid,reader,input_u,n,h_fft,sys_fft,freqs,include_noise = True,resistance = 50, temperature = 320,plot = False,output_just_noise = False)
        sub_info = info[info['eventid'] == eventid]
    
    
    
    
