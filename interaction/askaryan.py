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

pylab.ion()

############################################################

def electricFieldFrequencyDomainRaw(frequency, d, angle, energy_neutrino, inelasticity, mode, index_of_refraction):
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
    
    energy_shower = inelasticity * energy_neutrino # GeV
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
            pylab.title('E = %g \t$\\theta$=%0.3f \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
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
        #print(u_in)
        #print(numpy.shape(u_in))
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

def quickSignalSingle(theta_obs_rad,R,Energy_GeV,n,t_offset,attenuation,u, h_fft, sys_fft, freqs,plot_signals=False,plot_spectrum=False,plot_potential = False,include_noise = False, resistance = 50, temperature = 320):  
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



############################################################



if __name__ == "__main__":

    energy_neutrino = 1.e9 # GeV
    mode = 'cc'
    d = 1000. # m
    angle = numpy.linspace(0., 90., 1000)
    index_of_refraction = 1.8
    inelasticity = gnosim.interaction.inelasticity.inelasticity(energy_neutrino, mode)

    #omega = 0.5 # GHz
    
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
    
############################################################
