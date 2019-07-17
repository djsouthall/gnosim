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

import os
import sys
sys.path.append(os.environ['GNOSIM_DIR'])
import numpy
import pylab
import math
import scipy.signal
import gnosim.utils.constants
import gnosim.interaction.inelasticity
import gnosim.detector.fpga
pylab.ion()

############################################################

def electricFieldFrequencyDomainRaw(frequency, d, angle, Energy_GeV, inelasticity, index_of_refraction):
    '''
    The old method of calculating the electric field.  This is a frequency domain model:
    Askaryan Emission
    Source: Lehtinen et al. 2003, arXiv:0309656

    Note that this is in no way maintained and is a remnant of before the time domain signals (and the significant structural
    changes that came with them) were added.  It is HIGHLY recommended that you use time domain calculation. 
    
    Parameters
    ----------
    frequency : float
        Frequencies for which to calculate the electric field.  Given in GHz.
    d : float
        Observation distance.  Given in m.
    angle : float
        Observation angle relative to shower axis.  Given in degrees.
    Energy_GeV : float
        Neutrino energy given.  Given in GeV.  This should represent the raw energy going into of the neutrino
        before it interacts, and thus should NOT be reduced by the inelasticity factor (this is done internally).
    inelasticity : float
        The inelasticty factor of the interaction of the neutrino in ice.  Represents the portion of energy that
        is actually transferred to the energy of the shower.
    index_of_refraction : float
        The index of refraction where the interaction occurs.

    Returns
    -------
    electric_field : float
        The electric field given in units of  V m^-1 GHz^-1.
    '''
    omega = 2. * numpy.pi * frequency
    mu = 1.
    length = 1.5 # m
    
    energy_shower = inelasticity * Energy_GeV # GeV
    q = 5.5e-20 * energy_shower # C
    k = index_of_refraction * omega / gnosim.utils.constants.speed_light
    electric_field = (mu * gnosim.utils.constants.mu_0 * q * length * (gnosim.utils.constants.GHz_to_Hz * omega) / (d * numpy.sqrt(2. * numpy.pi))) \
                    * numpy.sin(numpy.radians(angle)) \
                    * numpy.exp(-0.5 * (k * length)**2 * (numpy.cos(numpy.radians(angle)) - index_of_refraction**-1)**2) \
                    * gnosim.utils.constants.GHz_to_Hz # V m^-1 GHz^-1, dimensionless
    return electric_field
    
############################################################

def RA(Energy_GeV,t_ns):
    '''
    The corresponds to Equation 16 of Practical and Accurate Calculations of Askaryan Radiation, and is the far field 
    parameterization of the vector potential (multiplied by the observation distance R).  Ideally this would be a 
    function of index of refraction as well, right now this model is parameterized specifically for n = 1.78. 
    Returns in units of V s .

    Source:
    Practical and Accurate Calculations of Askaryan Radiation - Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283

    Parameters
    ----------
    Energy_GeV : float
        The energy for which to calculate the vector potential of the Askaryan radiation.  Should already be reduced
        by the inelasticity factor if applicable.  Given in GeV.
    t_ns : numpy.ndarray of float or float
        The observer times for which to calculate the Askaryan radiation.  Should span both negative and positive times
        to get the full details of the pulse.  Given in ns.

    Returns
    -------
    ra : numpy.ndarray of float or float
        The vector potential (multiplied by the observation distance).  Given in V s.
    '''

    Energy_TeV = 0.001*Energy_GeV
    if numpy.size(t_ns) == 1:
        if t_ns > 0:
            ra = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns)/0.057) + (1. + 2.87*numpy.fabs(t_ns))**(-3.0))
        else:
            ra = (-4.5e-14) * Energy_TeV * ( numpy.exp(- numpy.fabs(t_ns)/0.030) + (1. + 3.05*numpy.fabs(t_ns))**(-3.5)) 
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
    This is the charge profile of the Askaryan calculation.  b is included to make the python variant of the gamma function align with
    that used in the papers definition of the charge profile.  The scale is arbitrary as it is normalized in the use of it by LQ (ideally).
    The scale should be set to 1.0, as it is assumed to be so. 

    This was not specified in the 'Practical and Accurate Calculations 
    of Askaryan Radiation' paper, but a parameterization was seperately found in 'Coherent radio pulses from GEANT generated 
    electromagnetic showers in ice'  -  Phys. Rev. D 65, 103002

    The mean a an b values they found are a = 4.85, b = 0.60 with sig_a = 1.01, sig_b = 0.14
    Perhaps we would want to pull randomly from gaussion distros for each of these params for each event (not currently done). 
    
    Source:
    Coherent radio pulses from GEANT generated electromagnetic showers in ice  -  Phys. Rev. D 65, 103002

    Parameters
    ----------
    x : numpy.ndarray of float
        The input value for Q.  Originally intended to be a position in m, however in practice is a u-substitution term:
        Q(u/alpha).   My explaination for this is as follows:
        The scale factor is added to modify this function by scaling the z' input such that the convolution described 
        in Eq17 of arXiv:1106.6283. Essentially in trying to understand how to implement the convolution described by 
        that equation I decided a u-substitution had to be done where  u = z' * alpha where alpha = (1-n*cos(theta))/c.   
        The convolution is then 1/alpha INT du Q(u/alpha)*Fp(dt - u).  The scale factor is meant to account for the 
        scaling in Q to make it appear more like a convolution: 1/alpha INT du Q'(u)*Fp(dt - u), where Q' scales u.  
        The scaling factor not being one implies the input x is some value of ns that was converted from meters using 
        the scale factor.

        I apologize if this is not clear, I am doing my best :/
    a : float, optional
        Charge profile parameter from Phys. Rev. D 65, 103002 (Default is 4.85, the mean value from the paper).
    b : float, optional
        Charge profile parameter from Phys. Rev. D 65, 103002 (Default is 0.60, the mean value from the paper).
    loc : float, optional
        The location parameter of the gamma function used in the charge distrobution.  (Default is 0).
    scale : float, optional
        The scaling factor from the gamma function used in the charge distrobution.  (Default is 1).
    random_params : float, optional
        Enables randomly generated values for a and b based on the distrobutions found in the source.
    random_local : numpy.random.RandomState, optional
        A prevously seeded random object, to avoid issues with consistency in seeding of events.  Should be set in 
        gnosim.sim.antarcticsim.Sim.event using random_local = numpy.random.RandomState(seed = event_seed).  (Default is None).

    Returns
    -------
    Q : numpy.ndarray of float or float
        The charge profile.
    '''
    if random_params:
        if random_local == None:
            a = numpy.random.normal(loc=4.85,scale=1.01,size=None)
            b = numpy.random.normal(loc=0.60,scale=0.14,size=None)
        else:
            a = random_local.normal(loc=4.85,scale=1.01,size=None)
            b = random_local.normal(loc=0.60,scale=0.14,size=None)
    return b * scipy.stats.gamma.pdf( b * x , a = a , loc = loc , scale = scale )
        
def excessProjectedTrackLength(Q,int_min=-100.,int_max=100.,n_steps = 1000):
    '''
    This is the charge profile of the Askaryan calculation.  b is included to make the python variant of the gamma function align with
    that used in the papers definition of the charge profile.  The scale is arbitrary as it is normalized in the use of it by LQ (ideally).
    The scale should be set to 1.0, as it is assumed to be so. 

    Source:
    Practical and Accurate Calculations of Askaryan Radiation - Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283

    Parameters
    ----------
    Q : function
        The charge profile function to be integrated.
    int_min : float, optional
        The lower integration bound. (Default is -100.0).
    int_max : float, optional
        The upper integration bound. (Default is 100.0).
    n_steps : int, optional
        The number of steps in the integration. (Default is 1000).

    Returns
    -------
    LQ : float
        Integrated Q.
    '''
    LQ,error = scipy.integrate.quad(Q,int_min,int_max)
    return LQ
    
def F_p(Energy_GeV,t_ns,n,LQ):
    '''
    This is the form factor as in Equation 15 of Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283
    This is to be calculated at the cherenkov angle (which in the paper was for n=1.78)
    And then convolved with the charge profile Q to determine the vector potential.  Despite the
    paremeterization in the paper being specifically for n=1.78 we assume that applying it for
    other indices of refractions is approximately correct.

    Source:
    Practical and Accurate Calculations of Askaryan Radiation - Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283

    Parameters
    ----------
    Energy_GeV : float
        The energy for which to calculate the vector potential of the Askaryan radiation.  Should already be reduced
        by the inelasticity factor if applicable.  Given in GeV.
    t_ns : numpy.ndarray of float or float
        The observer times for which to calculate the Askaryan radiation.  Should span both negative and positive times
        to get the full details of the pulse.  Given in ns.
    n : float
        The index of refraction where the interaction occurs. 
    LQ : function
        The integrated charge profile.

    Returns
    -------
    fp : float
        The form factor from Equation 15 of Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283.
        fp represents the average vector potential at the Chernkov angle per unit excess track length (LQ)
        scaled with the factor 4*pi*R/mu.
    '''
    ra = RA(Energy_GeV,t_ns)  
    fp = (4. * numpy.pi /(LQ * gnosim.utils.constants.mu_0 * math.sqrt(1-1/n**2))) * ra #note that math.sqrt(1-1/1.78**2) is a faster form of numpy.sin(cherenkov_angle) = numpy.sin(numpy.arccos(1/n))
    fp = numpy.multiply(scipy.signal.tukey(len(fp),alpha=0.05),fp)
    return fp

def addSignals(u_in,V_in,plot=False,V_noise_in = [], remove_noise_overlap = False):
    '''
    Joins signals from V_in into a continuous signal signal stream.  Should be used to combine
    voltages calculated independently for each solution type, such that they appear in the same
    stream as they would in reality.  If remove_noise_overlap is True and an array V_noise_in 
    is provided, this will remove noise from each signal in regions of overlapping noise 
    (by reducing them by the appropriate factor) such that noise does not add where realistically 
    it should be one continuous stream of noise.  
    
    V_noise in should be the same shape as V_in, and should correspond to the same
    times u_in.  
    
    For future Dan:  I need to add an option input for when adding signals with Noise
    to remove noise in overlapping region.  Should have a flag for remove_noise=True
    and then two required inputs: V_in_no_noise, u_in_no_noise. That should be
    the same length as u_in, V_in. In region of overlapping noise/signal I need
    to subtract off real signs, ramp noise in overlapping regions, and then add
    back the signals.  
    
    Source:
    Practical and Accurate Calculations of Askaryan Radiation - Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283

    Parameters
    ----------
    u_in : numpy.ndarray of float
        The times corresponding to the electric field.  Given in ns. Should be an array of times with 
        dimensions (n_signal , n_timestep ). It is assumed to be in order, i.e. u[0] is the min of 
        each row and u[-1] is the max.  Each row of u is also assumed to have the same time step.  
    V_in : numpy.ndarray of float
        The electric field of the askaryan radiation (including noise if applicable).  Given in V.  
        V_in should be an array of electric fields with dimensions (n_signal , n_timestep ).  Note 
        that the timing of signals may be shifted by up to a u-step here to align descretized timing 
        values.  There may be a more elegant way do this if this if added wiggle becomes a problem. 
    plot : bool, optional
        Enables plotting.
    V_noise_in : , optional
        The electric field of JUST the noise.  Given in V.  V_noise_in should be the same format as V_in.
    remove_noise_overlap : bool, optional
        If remove_noise_overlap is True and an array V_noise_in is provided, this will remove noise 
        from each signal in regions of overlapping noise (by reducing them by the appropriate factor) 
        such that noise does not add where realistically it should be one continuous stream of noise.  

    Returns
    -------
    u_out : numpy.ndarray of float
        The times corresponding to the electric field.  Given in ns.
    V_out : numpy.ndarray of float
        The electric field.  Given in V_out.
    '''
    if len(numpy.shape(u_in)) <=1:
        return u_in , V_in
    elif numpy.shape(u_in)[0] == 1:
        return u_in.flatten(), V_in.flatten()
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
            pylab.figure()   
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
            
        return u_out,V_out


def quickSignalSingle(theta_obs_rad, R, Energy_GeV, n, t_offset, signal_reduction_factor, u, h_fft, sys_fft, freqs, fp_fft=None, plot_signals=False, plot_spectrum=False, plot_signals_spectrum=False, plot_angles=False, plot_potential=False, include_noise=False, resistance=50, noise_temperature=320, random_local=None):  
    '''
    This calculates the electric field in the time domain following the source listed below.  
    Includes the antenna and system responses.  This should do the entire calculation, mostly
    in the frequency domain.  It is intended to be as fast as possible, though as it is a large
    calculaton it is possible I missed possible ways to speed it up.

    Source:
    Practical and Accurate Calculations of Askaryan Radiation - Phys. Rev. D 84, 103003 (2011), arXiv:1106.6283

    Parameters
    ----------
    theta_obs_rad : float
        The observation angle relative to the shower axis.  Given in radians.
    R : float
        The observation distance.  Given in m.
    Energy_GeV : float
        The energy for which to calculate the vector potential of the Askaryan radiation.  Should already be reduced
        by the inelasticity factor if applicable.  Given in GeV.
    n : float
        The index of refraction where the interaction occurs.
    t_offset : float
        An offset in time to apply to the output times.  Given in ns.
    signal_reduction_factor : float
        This is the reduction factor that should be multiplied with the antenna response.  This should be calculated
        using gnosim.detector.detector.Antenna.getAntennaResponseFactor
    u : numpy.ndarray of floats
        The times for which to calculate the vector potential.  Given in ns.
    h_fft : numpy.ndarray of cfloat
        The values for the antenna response. (Should have units of m, i.e. effective height).
        Should be loaded using gnosim.detector.detector.Antenna.addTimingInfo
    sys_fft : numpy.ndarray of cfloat
        The values for the syste response. (Should be unitless).
        Should be loaded using gnosim.detector.detector.Antenna.addTimingInfo
    freqs : numpy.ndarray of float
        The values for the frequencies corresponding to the above responses.
        Should be loaded using gnosim.detector.detector.Antenna.addTimingInfo
    fp_fft : numpy.ndarray of float, optional
        The frequency domain version of the form factor.  This is the same for a particular neutrino event, so
        can be calculated in advance and passed to this function to save computation time.  (Default is None).
    plot_signals : bool, optional
        Enables plotting of signals.  (Default is False).
    plot_spectrum : bool, optional
        Enables plotting of the response spectrum.  (Default is False).
    plot_signals_spectrum : bool, optional
        Emables plotting signal spectrum on top of response spectra.  plot_spectrum must be enabled for this to work.
        (Default is False).
    plot_angles : bool, optional
        Enables plotting of the signal phases.  (Default is False).
    plot_potential : bool, optional
        Enables plotting of the vector potential.  (Default is False).
    include_noise : bool, optional
        Enables the addition of noise (Default is False).
    resistance : float, optional
        The resistance to be used in the noise calculation.  Given in Ohms.  (Default is 50.0)
        Note that the noise is also processed by the system response, which may be scaled to obtain a particular noise
        level for a certain temperature.
    noise_temperature : float, optional
        The temperature to be used in the noise calculation.  Given in K.  (Default is 320.0)
        Note that the noise is also processed by the system response, which may be scaled to obtain a particular noise
        level for a certain temperature.
    random_local : numpy.random.RandomState, optional
        A prevously seeded random object, to avoid issues with consistency in seeding of events.  Should be set in 
        gnosim.sim.antarcticsim.Sim.event using random_local = numpy.random.RandomState(seed = event_seed).  (Default is None).

    Returns
    -------
    u : numpy.ndarray of float
        The times corresponding to the electric field.  Given in ns.
    V_noiseless : numpy.ndarray of float
        The electric field without noise.  Given in V.
    dominant_freq : float, optional
        The frequency corresponding to the max power bin in the signal spectrum.
    V_noise : numpy.ndarray of float, optional
        The electric field with noise.  Given in V.
    SNR : numpy.ndarray of float, optional
        SNR is calculated as the ratio of the peak to peak/2 over rms(noise), squared (ratio of powers)

    See Also
    --------
    gnosim.detector.detector.Antenna
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
        
        q = 0.60 * scipy.stats.gamma.pdf( 0.60 * u/alpha , a = 4.85 ) #This seems to be rate limiting.  If you think of faster way to calculate you can save some time.  Potentially just a precomputed lookup table?
        q = numpy.multiply(scipy.signal.tukey(len(q),alpha=0.05),q)
        q_fft = numpy.fft.rfft(q) #should look into an analytica form of fft(gamma function) to see if there is a way to avoid this fft
        A_fft = numpy.multiply(fp_fft,q_fft) * ( gnosim.utils.constants.mu_0 * numpy.sin(theta_obs_rad) / (4. * numpy.pi * R ) ) * ( t_step/abs(alpha) ) #the t_step already accounts for scaling required with irffting.  Might not want here?  unsure
        
        if numpy.logical_or(plot_potential == True,plot_signals == True):
            A = numpy.fft.irfft(A_fft,n=len(u))
            A = numpy.fft.fftshift(A)
        if plot_potential == True:
            pylab.figure()
            ax = pylab.subplot(311)
            pylab.title('alpha = %0.3f, $\\theta$ = %0.2f deg'%(alpha,numpy.rad2deg(theta_obs_rad)),fontsize=20)
            pylab.plot(u,fp,label='fp')
            pylab.ylabel('$F_p$ ($Amps$)',fontsize=16)
            pylab.minorticks_on()
            pylab.grid(which="both")
            pylab.grid(b=True, which='major', color='k', linestyle='-')
            pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            #pylab.xlim(-10,50)
            pylab.subplot(312,sharex = ax)
            pylab.plot(u,q,label='q')
            pylab.ylabel('$Q (arb)$ ',fontsize=16)
            pylab.minorticks_on()
            pylab.grid(which="both")
            pylab.grid(b=True, which='major', color='k', linestyle='-')
            pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            #pylab.xlim(-10,50)
            pylab.subplot(313,sharex = ax)
            pylab.semilogy(u,numpy.fabs(R*numpy.absolute(A)),label='RA')
            pylab.ylabel('$R|A|$ ',fontsize=16)
            pylab.xlabel('$\Delta t$',fontsize=16)
            pylab.minorticks_on()
            pylab.grid(which="both")
            pylab.grid(b=True, which='major', color='k', linestyle='-')
            pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            #pylab.xlim(-10,50)
    #calculating E_raw_fft    
    E_raw_fft = -1.0j*2.0*numpy.pi*numpy.multiply(A_fft , freqs) #negitive sign because E = -dA/dt
    #TODO: Add a pulser option that would put a different electric pulse here.  
    E_raw_fft *= signal_reduction_factor #This includes beam pattern, signal attenuation, and polarization
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
        
        #V_rms = numpy.sqrt(gnosim.utils.constants.boltzmann * noise_temperature * resistance * bandwidth * gnosim.utils.constants.GHz_to_Hz)
        V_rms = gnosim.utils.rf.thermalNoise(resistance, noise_temperature, bandwidth)
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
        # TODO: I should move this calculation to outside of the function.  Return p2p and then divide by the same rms for each event rather than calculate each time
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
                pylab.figure()
                ax = pylab.subplot(211)
                pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f\tt_step = %g ns'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n,t_step),fontsize=20)
                pylab.ylabel('R*|A| (V s)',fontsize=16)
                pylab.xlabel('t (ns)',fontsize=16)
                pylab.minorticks_on()
                pylab.grid(which="both")
                pylab.grid(b=True, which='major', color='k', linestyle='-')
                pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                #pylab.scatter(u,R*numpy.absolute(A),s=1)
                pylab.plot(u,R*numpy.absolute(A))
                
                pylab.subplot(212,sharex = ax)
                pylab.ylabel('$R \cdot E_{raw}$ (V)',fontsize=16)
                pylab.xlabel('t (ns)',fontsize=16)
                pylab.minorticks_on()
                pylab.grid(which="both")
                pylab.grid(b=True, which='major', color='k', linestyle='-')
                pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                #pylab.scatter(u,R*E_raw,s=1)
                pylab.plot(u,R*E_raw)
            
            pylab.figure()
            ax = pylab.subplot(211)
            pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f\tt_step = %g ns'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n,t_step),fontsize=20)
            pylab.ylabel('Noiseless Signal Voltage (V)',fontsize=16)
            pylab.xlabel('t (ns)',fontsize=16)
            #pylab.scatter(u,V,s=1)
            pylab.minorticks_on()
            pylab.grid(which="both")
            pylab.grid(b=True, which='major', color='k', linestyle='-')
            pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            pylab.plot(u,V_noiseless)
            
            pylab.subplot(212,sharex = ax)
            pylab.ylabel('Signal Voltage (V)',fontsize=16)
            pylab.xlabel('t (ns)',fontsize=16)
            pylab.minorticks_on()
            pylab.grid(which="both")
            pylab.grid(b=True, which='major', color='k', linestyle='-')
            pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            #pylab.scatter(u,V,s=1)
            pylab.plot(u,V_noise)
        else:
            if plot_potential == True:
                pylab.figure()
                ax = pylab.subplot(211)
                pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f\tt_step = %g ns'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n,t_step),fontsize=20)
                pylab.ylabel('R*|A| (V s)',fontsize=16)
                pylab.xlabel('t (ns)',fontsize=16)
                pylab.minorticks_on()
                pylab.grid(which="both")
                pylab.grid(b=True, which='major', color='k', linestyle='-')
                pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                #pylab.scatter(u,R*numpy.absolute(A),s=1)
                pylab.plot(u,R*numpy.absolute(A))
            
                pylab.subplot(212,sharex = ax)
                pylab.ylabel('$R \cdot E_{raw}$ (V)',fontsize=16)
                pylab.xlabel('t (ns)',fontsize=16)
                pylab.minorticks_on()
                pylab.grid(which="both")
                pylab.grid(b=True, which='major', color='k', linestyle='-')
                pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                #pylab.scatter(u,R*E_raw,s=1)
                pylab.plot(u,R*E_raw)
            
            '''
            fig = pylab.figure()
            pylab.title('E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f\tt_step = %g ns'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n,t_step),fontsize=20)
            #pylab.subplot(313,sharex = ax)
            pylab.ylabel('Noiseless Signal Voltage (V)',fontsize=20)
            pylab.xlabel('t (ns)',fontsize=20)
            #pylab.scatter(u,V,s=1)
            #plot_cut = u>0
            #pylab.plot(u[plot_cut],V_noiseless[plot_cut],linewidth=5)
            ax = pylab.gca()
            ax.tick_params(axis = 'both',labelsize = 14)
            pylab.subplots_adjust(left = 0.09, bottom = 0.06, right = 0.97, top = 0.97, wspace = 0.20, hspace = 0.20)
            #fig.patch.set_alpha(0.)
            #ax.patch.set_alpha(0.)
            '''
            
    if plot_spectrum == True:
        fig_spec = pylab.figure()

        if plot_signals_spectrum:
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


        pylab.figure()
        pylab.plot(freqs, 10.0 * numpy.log10(h_gain), label="Antenna", color = 'limegreen')
        pylab.plot(freqs, 10.0 * numpy.log10(sys_gain), label="Elect.", color = 'darkorange')
        pylab.xlabel("Freq. [Hz]")
        pylab.ylabel("Realized Gain [dBi]")
        '''
        
        #These scalings are for v6 onward.  Older scalings were probably incorrect anyways so I am just leaving these for all versions.
        sys_gain = numpy.absolute(sys_fft)**2 #Squaring instead of just plottiong as 20log10
        sys_gain *= 2.0 # Because the rfft is missing half the power
        
        h_gain = numpy.absolute(h_fft)
        h_gain *= (1.74 / 3e8) * numpy.array(freqs) #1/lambda
        h_gain *= h_gain #squaring
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
        linewidth = 3 
        markersize = 4
        if scatter == True:
            pylab.scatter(freqs/1e6,10.0 * numpy.log10(sys_gain),label='System Response (fft)', color = 'darkorange',s = markersize)
            pylab.scatter(freqs/1e6,10.0 * numpy.log10(h_gain),label='Antenna Response (fft)', color = 'limegreen',s = markersize)
            if plot_signals_spectrum:
                pylab.scatter(freqs/1e6,10.0 * numpy.log10(raw_signal_gain),label='Raw Signal (fft)', color = 'blue',s = markersize)
                pylab.scatter(freqs/1e6,10.0 * numpy.log10(signal_gain),label='Processed Signal (fft)', color = 'red',s = markersize)
        else:
            pylab.plot(freqs/1e6,10.0 * numpy.log10(sys_gain),label='System Response (fft)', color = 'darkorange',linewidth = linewidth)
            pylab.plot(freqs/1e6,10.0 * numpy.log10(h_gain),label='Antenna Response (fft)', color = 'limegreen',linewidth = linewidth)
            if plot_signals_spectrum:
                pylab.plot(freqs/1e6,10.0 * numpy.log10(raw_signal_gain),label='Raw Signal (fft)', color = 'blue',s = markersize)
                pylab.plot(freqs/1e6,10.0 * numpy.log10(signal_gain),label='Processed Signal (fft)', color = 'red',s = markersize)
        
        if plot_signals_spectrum:
            pylab.ylabel("Realized Gain [dBi] (Signals in arb)",fontsize=16)
        else:
            pylab.ylabel("Realized Gain [dBi]",fontsize=16)
        pylab.xlabel('Freq. [MHz]',fontsize=16)
        #pylab.ylabel('dB (20.0 log10(numpy.absolute(V_fft)))',fontsize=16)
        pylab.ylim(-75,75)
        pylab.xlim(0,1000)
        pylab.minorticks_on()
        pylab.grid(which="both")
        pylab.grid(b=True, which='major', color='k', linestyle='-')
        pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        pylab.legend(fontsize=14)
        ax = pylab.gca()
        ax.tick_params(axis = 'both',labelsize = 16)
        #pylab.subplots_adjust(left = 0.09, bottom = 0.06, right = 0.97, top = 0.97, wspace = 0.20, hspace = 0.20)
        #fig_spec.patch.set_alpha(0.)
        #ax.patch.set_alpha(0.)
    if plot_angles == True:
        pylab.figure()
        ax = pylab.subplot(2,1,1)
        pylab.title('PHASE E = %g GeV \t$\\theta$=%0.3f deg \tn = %0.2f'%(Energy_GeV,numpy.rad2deg(theta_obs_rad),n))
        pylab.plot(freqs/1e6,numpy.rad2deg(numpy.angle(E_raw_fft)),label='Raw Signal (fft)', color = 'blue')
        pylab.plot(freqs/1e6,numpy.rad2deg(numpy.angle(sys_fft)),label='System Response (fft)', color = 'darkorange')
        pylab.plot(freqs/1e6,numpy.rad2deg(numpy.angle(h_fft)),label='Antenna Response (fft)', color = 'limegreen')
        pylab.ylabel('Angle [deg]',fontsize=16)
        pylab.legend(fontsize=14)
        pylab.minorticks_on()
        pylab.grid(which="both")
        pylab.grid(b=True, which='major', color='k', linestyle='-')
        pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        
        
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
        pylab.minorticks_on()
        pylab.grid(which="both")
        pylab.grid(b=True, which='major', color='k', linestyle='-')
        pylab.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    if include_noise == True:
        dominant_freq = freqs[numpy.argmax(numpy.absolute(V_fft_noise))]
        return u + t_offset , V_noiseless, dominant_freq, V_noise,  SNR
    else:
        dominant_freq = freqs[numpy.argmax(numpy.absolute(V_fft_noiseless))]
        return u + t_offset , V_noiseless, dominant_freq

############################################################



if __name__ == "__main__":
    #pylab.close('all')
    #################################################################
    import yaml
    import gnosim.detector.detector
    solutions = numpy.array(['direct'])
    config_file = os.environ['GNOSIM_DIR'] + '/gnosim/detector/station_config/real_config_full_station.py'
    config = yaml.load(open(config_file))
    station = gnosim.detector.detector.Station('ARA5',config,solutions = solutions)
    station.plotStation()

    ant = station.antennas[0]

    energy_neutrino = 3.e9 # GeV
    n = 1.78
    c = gnosim.utils.constants.speed_light #m/ns
    R = 1000. #m
    cherenkov_angle = numpy.arccos(1./n)
    cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./n))

    inelasticity = 1.00

    u, V_noiseless, dominant_freq, V_noise,  SNR = gnosim.interaction.askaryan.quickSignalSingle(numpy.deg2rad(50),R,inelasticity*energy_neutrino,n,1,0.7,ant.signal_times, ant.h_fft, ant.sys_fft, ant.freqs_response,plot_signals=True,plot_spectrum=True,plot_potential=True,include_noise = True)
    minimum_time = numpy.min(u)
    maximum_time = numpy.max(u)
    random_time_offset = numpy.random.uniform(-1, 1, size=1)
    digital_sample_times = numpy.arange(minimum_time,maximum_time,station.digital_sampling_period) + random_time_offset #these + random_time_offset #these
    sample_times,V_bit = gnosim.detector.fpga.digitizeSignal(u,V_noise,digital_sample_times,7,0.0204,3, dc_offset = 0, plot = True)
    '''
    See gnosim.sim.testing_single_event for how to rerun calculations for an event.
    '''    

