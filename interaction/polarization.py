'''
This file is used for the development of poarization capabilities for interactions. 

'''


import numpy
import pylab
import math
import gnosim.trace.refraction_library
pylab.ion()
import gnosim.earth.ice
from mpl_toolkits.mplot3d import Axes3D
import gnosim.utils.linalg

############################################################

def getWaveVector(phi_ray_from_ant,theta_ray_from_ant):
    '''
    This will get the unit vector the travelling radiation along the ray towards the antenna.

    Parameters
    ----------
    phi_ray_from_ant : float
        Azimuthal theta of vector of ray from antenna along path to neutrino (degrees).
    theta_ray_from_ant : float
        Zenith theta of vector of ray from antenna along path to neutrino (degrees).

    Returns
    -------
    vec_ray_to_ant: numpy.ndarray
        The unit vector for the vector directed towards the antenna along the observation ray. 
        This is returned in ice-frame cartesian coordinates.
    
    See Also
    --------
    gnosim.detector.detector
    '''
    vec_ray_from_ant = gnosim.utils.linalg.angToVec(phi_ray_from_ant,theta_ray_from_ant)#numpy.array([numpy.sin(numpy.deg2rad(theta_ray_from_ant))*numpy.cos(numpy.deg2rad(phi_ray_from_ant)),numpy.sin(numpy.deg2rad(theta_ray_from_ant))*numpy.sin(numpy.deg2rad(phi_ray_from_ant)),numpy.cos(numpy.deg2rad(theta_ray_from_ant))])
    vec_ray_to_ant = - vec_ray_from_ant #Vector from neutrino location to observation along ray
    return vec_ray_to_ant

def getNeutrinoMomentumVector(phi_neutrino_source_dir,theta_neutrino_source_dir):
    '''
    This will get the unit vector for the direction the neutrino is moving/direction of the shower

    Parameters
    ----------
    phi_ray_from_ant : float
        Azimuthal theta of vector of ray from antenna along path to neutrino (degrees).
    theta_ray_from_ant : float
        Zenith theta of vector of ray from antenna along path to neutrino (degrees).

    Returns
    -------
    vec_ray_to_ant: numpy.ndarray
        The unit vector for the vector directed towards the antenna along the observation ray. 
        This is returned in ice-frame cartesian coordinates.
    
    See Also
    --------
    gnosim.detector.detector
    '''
    vec_neutrino_source_dir = gnosim.utils.linalg.angToVec(phi_neutrino_source_dir,theta_neutrino_source_dir)#numpy.array([numpy.sin(numpy.deg2rad(theta_neutrino_source_dir))*numpy.cos(numpy.deg2rad(phi_neutrino_source_dir)),numpy.sin(numpy.deg2rad(theta_neutrino_source_dir))*numpy.sin(numpy.deg2rad(phi_neutrino_source_dir)),numpy.cos(numpy.deg2rad(theta_neutrino_source_dir))])
    vec_neutrino_travel_dir = - vec_neutrino_source_dir
    return vec_neutrino_travel_dir

def getInitialPolarization(vec_neutrino_travel_dir,emission_wave_vector):
    '''
    This will get the unit vector of the electric field direction (polarization) of the travelling Askaryan radiation given the observation
    ray geometry and the neutrino direction geometry.  The polarization is in the plane of the shower and observation ray, and perpendicular
    to the observation ray.

    Parameters
    ----------
    vec_neutrino_travel_dir : numpy.ndarray
        The unit vector for the direction the shower is propogating.
        This is returned in ice-frame cartesian coordinates.
    emission_wave_vector : numpy.ndarray
        The unit vector for the vector directed towards the antenna along the observation ray. 
        This is returned in ice-frame cartesian coordinates.  This should be the wave vector
        as it was emitted from the neutrino.

    Returns
    -------
    polarization_vector: numpy.ndarray
        The unit vector for the polarization. This is returned in ice-frame cartesian coordinates.

    See Also
    --------
    gnosim.detector.detector
    '''
    #vec_neutrino_travel_dir = getNeutrinoMomentumVector(phi_neutrino_source_dir,theta_neutrino_source_dir)
    #emission_wave_vector = getWavceVector(phi_ray_from_ant,theta_ray_from_ant) #Vector from neutrino location to observation along ray
    polarization_vector = gnosim.utils.linalg.normalize(numpy.cross(emission_wave_vector, numpy.cross(vec_neutrino_travel_dir,emission_wave_vector)))
    #polarization_vector = gnosim.utils.linalg.normalize(polarization_vector)

    return polarization_vector

def calculateSPUnitVectors(wave_vector):
    '''
    This will calculate the s-polarization and p-polarization unit vectors from 
    a given wave_vecter (vector along observation ray towards antenna)

    Parameters
    ----------
    wave_vector : numpy.ndarray
        The unit vector for the vector directed towards the antenna along the observation ray. 
        Given in the ice-frame cartesian coordinates.

    Returns
    -------
    s_vector : numpy.ndarray
        The unit vector for the s-polarization. This is returned in ice-frame cartesian coordinates.
    p_vector : numpy.ndarray
        The unit vector for the p-polarization. This is returned in ice-frame cartesian coordinates.
    
    See Also
    --------
    gnosim.detector.detector
    '''
    s_vector   = gnosim.utils.linalg.normalize(numpy.cross(wave_vector,numpy.array([0.0,0.0,1.0])))
    p_vector   = gnosim.utils.linalg.normalize(numpy.cross(s_vector,wave_vector))
    return s_vector, p_vector
############################################################

def getPolarizationAtAntenna( vec_neutrino_travel_dir , emission_wave_vector , detection_wave_vector , a_s , a_p, return_initial_polarization = False):
    '''
    Parameters
    ----------
    vec_neutrino_travel_dir : numpy.ndarray
        The unit vector for the direction the shower is propogating.
        This is returned in ice-frame cartesian coordinates.
    emission_wave_vector : numpy.ndarray
        The unit vector for the vector directed towards the antenna along the observation ray. 
        This is returned in ice-frame cartesian coordinates.  This should be the wave vector
        as it was emitted from the neutrino.
    detection_wave_vector : numpy.ndarray
        The unit vector for the vector directed towards the antenna along the observation ray. 
        This is returned in ice-frame cartesian coordinates.  This should be the wave vector
        as it interacts with the antenna.
    a_s : float
        This is the attenuation factor of the s-polarization.  It should contain both the
        attenuation resulting from attenuation length, as well as the net effect of the
        fresnel amplitudes over the corse of the ray's path to the antenna.
        Currently only numpy.real(a_s) is returned from refraction_library.makeLibrary,
        so a real float is expected here.
    a_p : float
        This is the attenuation factor of the p-polarization.  It should contain both the
        attenuation resulting from attenuation length, as well as the net effect of the
        fresnel amplitudes over the corse of the ray's path to the antenna.
        Currently only numpy.real(a_p) is returned from refraction_library.makeLibrary,
        so a real float is expected here.
    return_initial_polarization : bool
        If True then the initial polarization will also be output.  (Defaul is False).

    Returns
    -------
    polarization_vector_1_unit : numpy.ndarray
        The unit vector for the polarization as it is just before interacting with the antenna.  Returned in
        ice-frame coordinates.
    polarization_vector_1_mag : float
        This is the magnitude of the polarization vector.  This essentially contains information
        about a_s and a_p (the attenuation of the two polarization modes) and how they go into a final
        multiplicative factor.  As the polarizations are not simple geometry the proportionate effect of
        each polarizations attenuation has to be accounted for using vectors.  
    polarization_vector_0 : numpy.ndarray, optional
        The unit vector for the polarization as it is just after emissionat the neutrino.
        This is a unit vector, as the magnitudes have not reduced yet.
        This is returned in ice-frame cartesian coordinates.
    
    See Also
    --------
    gnosim.detector.detector
    '''
    #vec_neutrino_travel_dir = getNeutrinoMomentumVector(phi_neutrino_source_dir,theta_neutrino_source_dir)
    #emission_wave_vector = getWaveVector(phi_ray_from_ant,theta_ray_from_ant) #Vector from neutrino location to observation along ray
    #detection_wave_vector = getWaveVector(phi_ray_from_ant_at_antenna,theta_ray_from_ant_at_antenna)
    polarization_vector_0_unit = getInitialPolarization(vec_neutrino_travel_dir,emission_wave_vector) #Inputs to this are confirmed normalized.  Output is also normalized now.
    s_vector_0_unit, p_vector_0_unit = calculateSPUnitVectors(emission_wave_vector) #Output is now normalized. 
    s_vector_1_unit, p_vector_1_unit = calculateSPUnitVectors(detection_wave_vector) #Output is now normalized.
    #The p polarization is attenuated by a_p, and changes direction with the ray
    polarization_vector_1_p = numpy.dot(polarization_vector_0_unit,p_vector_0_unit)*a_p*p_vector_1_unit
    #The s polarization is attenuated by a_s, but maintains direction.
    polarization_vector_1_s = numpy.dot(polarization_vector_0_unit,s_vector_0_unit)*a_s*s_vector_1_unit
    #The final polarization is the sum of these
    polarization_vector_1 = polarization_vector_1_p + polarization_vector_1_s #Not a unit vector.  The magnitude changes to represent reduction in E field.  a_s and a_p include attenuation in ice.
    
    polarization_vector_1_unit , polarization_vector_1_mag = gnosim.utils.linalg.normalize(polarization_vector_1,return_original_mag=True)
    #The magnitude of the above is essentially the effect of attenuation.  The unit vector will be used for the effect due to polarization.


    if return_initial_polarization == True:
        return polarization_vector_1_unit, polarization_vector_1_mag, polarization_vector_0_unit
    else:
        return polarization_vector_1_unit, polarization_vector_1_mag

def testPolarization():
    ice = gnosim.earth.ice.Ice('antarctica',suppress_fun = True)
    
    #Antenna Positional Information
    antenna_loc = numpy.array([0.,0.,-200.0])

    #Neutrino Directional Information
    theta_neutrino_source_dir = 170.0 #Deg, zenith
    phi_neutrino_source_dir = 0.0 #Deg 

    #Neutrino Positional Information
    theta_ant = 5.0 #deg #will set the depth ultimately (found from ray tracing below)
    r_neutrino_pos = 50.0 #m    
    phi_neutrino_pos = 30.0 #Deg
    x, y, z, t, d, phi, theta, a_p, a_s, index_reflect_air, index_reflect_water = gnosim.trace.refraction_library.rayTrace(antenna_loc, phi_neutrino_pos, theta_ant ,ice, r_limit = 1.0000001*r_neutrino_pos)
    theta_ray_from_ant_at_neutrino = theta[-1] #deg
    phi_ray_from_ant_at_neutrino = phi[-1] #deg
    theta_ray_from_ant_at_antenna = theta[0] #deg
    phi_ray_from_ant_at_antenna = phi[0] #deg

    x_neutrino_pos = x[-1] #m
    y_neutrino_pos = y[-1] #m
    z_neutrino_pos = z[-1] #m
    vec_neutrino_travel_dir = getNeutrinoMomentumVector(phi_neutrino_source_dir,theta_neutrino_source_dir)
    emission_wave_vector = getWaveVector(phi_ray_from_ant_at_neutrino,theta_ray_from_ant_at_neutrino) #Vector from neutrino location to observation along ray
    detection_wave_vector = getWaveVector(phi_ray_from_ant_at_antenna,theta_ray_from_ant_at_antenna) #Vector from neutrino location to observation along ray
    polarization_vector_0 = getInitialPolarization(vec_neutrino_travel_dir,emission_wave_vector)
    polarization_vector_1, polarization_attenuation = getPolarizationAtAntenna(vec_neutrino_travel_dir , emission_wave_vector , detection_wave_vector , a_s[-1], a_p[-1])#(phi_ray_from_ant_at_neutrino,theta_ray_from_ant_at_neutrino,phi_ray_from_ant_at_antenna,theta_ray_from_ant_at_antenna,phi_neutrino_source_dir,theta_neutrino_source_dir, a_s[-1], a_p[-1], return_k_1 = True)
    print(polarization_vector_0)
    print(polarization_vector_1)
    print(polarization_attenuation)

    elev = 1.0 #viewing
    azim = 90.0#-45.0
    make_cube = True

    ###########
    #PLOTTING AT EMISSION
    ###########

    #prepare plotting
    fig = pylab.figure()
    ax = fig.gca(projection='3d')
    #Plot antenna
    ax.scatter(antenna_loc[0],antenna_loc[1],antenna_loc[2],'Antenna')
    #Plot Ray
    ax.plot(x,y,z,color='k',linestyle='--',label='Ray')
    #plot neutrino vector
    ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,vec_neutrino_travel_dir[0],vec_neutrino_travel_dir[1],vec_neutrino_travel_dir[2],color='r',label = 'Neutrino Travel Direction (Scale arb)',linestyle='-')
    ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,emission_wave_vector[0],emission_wave_vector[1],emission_wave_vector[2],color='b',label = 'Observeration Direction (Scale arb)',linestyle='-')
    ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,polarization_vector_0[0],polarization_vector_0[1],polarization_vector_0[2],color='g',label = 'Initial Polarization Vector (Unit)',linestyle='-')
    
    axis_add = 2
    x_range = numpy.array([x_neutrino_pos - axis_add, x_neutrino_pos + axis_add])
    y_range = numpy.array([y_neutrino_pos - axis_add, y_neutrino_pos + axis_add])
    z_range = numpy.array([z_neutrino_pos - axis_add, z_neutrino_pos + axis_add])

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    
    if make_cube == True:
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = numpy.array([x_range.max()-x_range.min(), y_range.max()-y_range.min(), z_range.max()-z_range.min()]).max()
        Xb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x_range.max()+x_range.min())
        Yb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y_range.max()+y_range.min())
        Zb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z_range.max()+z_range.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('Ice x (m)',fontsize=16)
    ax.set_ylabel('Ice y (m)',fontsize=16)
    ax.set_zlabel('Ice z (m)',fontsize=16)
    ax.view_init(elev = elev, azim = azim)
    pylab.legend(fancybox=True, framealpha=0.5,fontsize=12)

    ###########
    #PLOTTING AT ANTENNA
    ###########

    #prepare plotting
    fig = pylab.figure()
    ax = fig.gca(projection='3d')
    #Plot antenna
    ax.scatter(antenna_loc[0],antenna_loc[1],antenna_loc[2],'Antenna')
    #Plot Ray
    ax.plot(x,y,z,color='k',linestyle='--',label='Ray')
    #plot neutrino vector
    ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,vec_neutrino_travel_dir[0],vec_neutrino_travel_dir[1],vec_neutrino_travel_dir[2],color='r',label = 'Neutrino Travel Direction (Scale arb)',linestyle='-')
    ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,detection_wave_vector[0],detection_wave_vector[1],detection_wave_vector[2],color='b',label = 'Observeration Direction (Scale arb)',linestyle='-')
    ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,polarization_vector_1[0],polarization_vector_1[1],polarization_vector_1[2],color='g',label = 'Final Polarization Vector',linestyle='-')
    
    axis_add = 2
    x_range = numpy.array([x_neutrino_pos - axis_add, x_neutrino_pos + axis_add])
    y_range = numpy.array([y_neutrino_pos - axis_add, y_neutrino_pos + axis_add])
    z_range = numpy.array([z_neutrino_pos - axis_add, z_neutrino_pos + axis_add])

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    
    if make_cube == True:
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = numpy.array([x_range.max()-x_range.min(), y_range.max()-y_range.min(), z_range.max()-z_range.min()]).max()
        Xb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x_range.max()+x_range.min())
        Yb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y_range.max()+y_range.min())
        Zb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z_range.max()+z_range.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('Ice x (m)',fontsize=16)
    ax.set_ylabel('Ice y (m)',fontsize=16)
    ax.set_zlabel('Ice z (m)',fontsize=16)
    ax.view_init(elev = elev, azim = azim)
    pylab.legend(fancybox=True, framealpha=0.5,fontsize=12)



if __name__ == "__main__":
    pylab.close('all')
    testPolarization()
