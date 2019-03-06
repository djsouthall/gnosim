'''
This file is used for the development of poarization capabilities for interactions. 

'''


import numpy
import pylab
import math
import gnosim.trace.refraction_library_beta
pylab.ion()
import gnosim.earth.ice
from mpl_toolkits.mplot3d import Axes3D

############################################################

def getPolarizationFactorsWrong(theta_ray_from_ant,phi_ray_from_ant,theta_neutrino_source_dir,phi_neutrino_source_dir,return_additional_vectors = False):
    '''
    My first attempt at calculating the polarization.  This does calculates the electric field component of the Askaryan radiation
    at the observed angle, however is perhaps not correct because it is not perpendicular to the eventual motion of the radiation.
    ------
    Given the theta and phi of ray (from antenna to neutrino), and theta and phi of neutrino
    (directed towards the source of the neutrino) this calculates the p_h and p_v factors.
    if return_additional_vectors == True:
        return p_h, p_v, vec_ray_to_ant, vec_neutrino_travel_dir, polarization_vector #Each should be normalized)
    else:
        return p_h, p_v

    All angles should be given in degrees
    '''
    vec_neutrino_source_dir = numpy.array([numpy.sin(numpy.deg2rad(theta_neutrino_source_dir))*numpy.cos(numpy.deg2rad(phi_neutrino_source_dir)),numpy.sin(numpy.deg2rad(theta_neutrino_source_dir))*numpy.sin(numpy.deg2rad(phi_neutrino_source_dir)),numpy.cos(numpy.deg2rad(theta_neutrino_source_dir))])
    vec_neutrino_travel_dir = - vec_neutrino_source_dir
    vec_ray_from_ant = numpy.array([numpy.sin(numpy.deg2rad(theta_ray_from_ant))*numpy.cos(numpy.deg2rad(phi_ray_from_ant)),numpy.sin(numpy.deg2rad(theta_ray_from_ant))*numpy.sin(numpy.deg2rad(phi_ray_from_ant)),numpy.cos(numpy.deg2rad(theta_ray_from_ant))])
    vec_ray_to_ant = - vec_ray_from_ant #Vector from neutrino location to observation
    polarization_vector = - vec_ray_to_ant + (numpy.dot(vec_ray_to_ant,vec_neutrino_travel_dir))*vec_neutrino_travel_dir
    polarization_vector = polarization_vector/numpy.linalg.norm(polarization_vector)

    #Currently unsure how to handle the signs of these:
    p_h = numpy.sqrt(polarization_vector[0]**2 + polarization_vector[1]**2)
    p_v = polarization_vector[2] 
    if return_additional_vectors == True:
        return polarization_vector, vec_ray_to_ant, vec_neutrino_travel_dir
    else:
        return polarization_vector

def getPolarizationFactors(theta_ray_from_ant,phi_ray_from_ant,theta_neutrino_source_dir,phi_neutrino_source_dir,return_additional_vectors = False):
    '''
    This will get the unit vector of the electric field direction (polarization) of the travelling Askaryan radiation given the observation
    ray geometry and the neutrino direction geometry.  The polarization is in the plane of the shower and observation ray, and perpendicular
    to the observation ray.

    Parameters:
    ----------
    theta_ray_from_ant : float
        Zenith theta of vector of ray from antenna along path to neutrino (degrees).
    phi_ray_from_ant : float
        Azimuthal theta of vector of ray from antenna along path to neutrino (degrees).
    theta_neutrino_source_dir : float
        Zenith theta of vector directed towards the source of the neutrino (degrees).
    phi_neutrino_source_dir : float
        Azimuthal theta of vector directed towards the source of the neutrino (degrees).
    return_additional_vectors : bool
        if return_additional_vectors == True:
            return polarization_vector, vec_ray_to_ant, vec_neutrino_travel_dir  #Each should be normalized)
        else:
            return polarization_vector

    Returns:
    ----------
    polarization_vector: numpy.ndarray
        The unit vector for the polarization.
    vec_ray_to_ant: numpy.ndarray
        The unit vector for the vector direct towards the antenna along the observation ray.
    vec_neutrino_travel_dir: numpy.ndarray
        The unit vector for the direction the shower is propogating.
    See Also:
    gnosim.sim.detector
    '''
    vec_neutrino_source_dir = numpy.array([numpy.sin(numpy.deg2rad(theta_neutrino_source_dir))*numpy.cos(numpy.deg2rad(phi_neutrino_source_dir)),numpy.sin(numpy.deg2rad(theta_neutrino_source_dir))*numpy.sin(numpy.deg2rad(phi_neutrino_source_dir)),numpy.cos(numpy.deg2rad(theta_neutrino_source_dir))])
    vec_neutrino_travel_dir = - vec_neutrino_source_dir
    vec_ray_from_ant = numpy.array([numpy.sin(numpy.deg2rad(theta_ray_from_ant))*numpy.cos(numpy.deg2rad(phi_ray_from_ant)),numpy.sin(numpy.deg2rad(theta_ray_from_ant))*numpy.sin(numpy.deg2rad(phi_ray_from_ant)),numpy.cos(numpy.deg2rad(theta_ray_from_ant))])
    vec_ray_to_ant = - vec_ray_from_ant #Vector from neutrino location to observation
    polarization_vector = numpy.cross(vec_ray_to_ant, numpy.cross(vec_neutrino_travel_dir,vec_ray_to_ant))
    polarization_vector = polarization_vector/numpy.linalg.norm(polarization_vector)

    #Currently unsure how to handle the signs of these:
    p_h = numpy.sqrt(polarization_vector[0]**2 + polarization_vector[1]**2)
    p_v = polarization_vector[2] 
    if return_additional_vectors == True:
        return polarization_vector, vec_ray_to_ant, vec_neutrino_travel_dir
    else:
        return polarization_vector

############################################################



if __name__ == "__main__":
    #Setup
    plot_quiver = True
    pylab.close('all')
    pol1 = True
    pol2 = True
    elev = 1.0 #viewing
    azim = 90.0#-45.0
    make_cube = True
    ice = gnosim.earth.ice.Ice('antarctica',suppress_fun = True)
    
    #Antenna Positional Information
    antenna_loc = numpy.array([0.,0.,-200.0])

    #Neutrino Directional Information
    theta_neutrino_source_dir = 170.0 #Deg, zenith
    phi_neutrino_source_dir = 0.0 #Deg 

    #Neutrino Positional Information
    theta_ant = 30.0 #deg #will set the depth ultimately (found from ray tracing below)
    r_neutrino_pos = 300.0 #m
    phi_neutrino_pos = 30.0 #Deg
    x, y, z, t, d, phi, theta, a_v, a_h, index_reflect_air, index_reflect_water = gnosim.trace.refraction_library_beta.rayTrace(antenna_loc, phi_neutrino_pos, theta_ant ,ice, r_limit = 1.0000001*r_neutrino_pos)
    theta_ray_from_ant_at_neutrino = theta[-1] #deg
    phi_ray_from_ant_at_neutrino = phi[-1] #deg
    theta_ray_from_ant_at_antenna = theta[0] #deg
    phi_ray_from_ant_at_antenna = phi[0] #deg
    x_neutrino_pos = x[-1] #m
    y_neutrino_pos = y[-1] #m
    z_neutrino_pos = z[-1] #m

    if pol1 == True:
        print('Attempting the first polarization calculation')
        polarization_vector, vec_ray_to_ant, vec_neutrino_travel_dir = getPolarizationFactorsWrong(theta_ray_from_ant_at_neutrino,phi_ray_from_ant_at_neutrino,theta_neutrino_source_dir,phi_neutrino_source_dir,return_additional_vectors=True)
        print('polarization_vector = ', polarization_vector)
        if plot_quiver == True:
            #prepare plotting
            fig = pylab.figure()
            #fig = pylab.figure(figsize=(16,11.2))
            ax = fig.gca(projection='3d')
            #Plot antenna
            ax.scatter(antenna_loc[0],antenna_loc[1],antenna_loc[2],'Antenna')
            #Plot Ray
            ax.plot(x,y,z,color='k',linestyle='--',label='Ray')
            #plot neutrino vector
            ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,vec_neutrino_travel_dir[0],vec_neutrino_travel_dir[1],vec_neutrino_travel_dir[2],color='r',label = 'Neutrino Travel Direction (Scale arb)',linestyle='-')
            ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,vec_ray_to_ant[0],vec_ray_to_ant[1],vec_ray_to_ant[2],color='b',label = 'Observeration Direction (Scale arb)',linestyle='-')
            ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,0.5*polarization_vector[0],0.5*polarization_vector[1],0.5*polarization_vector[2],color='g',label = 'Polarization Vector (Scale arb)',linestyle='-')
            ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,-0.5*polarization_vector[0],-0.5*polarization_vector[1],-0.5*polarization_vector[2],color='g',linestyle='-')
            
            axis_add = 2
            ax.set_xlim([x_neutrino_pos - axis_add, x_neutrino_pos + axis_add])
            ax.set_ylim([y_neutrino_pos - axis_add, y_neutrino_pos + axis_add])
            ax.set_zlim([z_neutrino_pos - axis_add, z_neutrino_pos + axis_add])
            
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


    if pol2 == True:
        print('Attempting the second polarization calculation')
        polarization_vector, vec_ray_to_ant, vec_neutrino_travel_dir = getPolarizationFactors(theta_ray_from_ant_at_neutrino,phi_ray_from_ant_at_neutrino,theta_neutrino_source_dir,phi_neutrino_source_dir,return_additional_vectors=True)
        print('polarization_vector = ', polarization_vector)
        if plot_quiver == True:
            #prepare plotting
            fig = pylab.figure()
            #fig = pylab.figure(figsize=(16,11.2))
            ax = fig.gca(projection='3d')
            #Plot antenna
            ax.scatter(antenna_loc[0],antenna_loc[1],antenna_loc[2],'Antenna')
            #Plot Ray
            ax.plot(x,y,z,color='k',linestyle='--',label='Ray')
            #plot neutrino vector
            ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,vec_neutrino_travel_dir[0],vec_neutrino_travel_dir[1],vec_neutrino_travel_dir[2],color='r',label = 'Neutrino Travel Direction (Scale arb)',linestyle='-')
            ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,vec_ray_to_ant[0],vec_ray_to_ant[1],vec_ray_to_ant[2],color='b',label = 'Observeration Direction (Scale arb)',linestyle='-')
            ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,0.5*polarization_vector[0],0.5*polarization_vector[1],0.5*polarization_vector[2],color='g',label = 'Polarization Vector (Scale arb)',linestyle='-')
            ax.quiver(x_neutrino_pos,y_neutrino_pos,z_neutrino_pos,-0.5*polarization_vector[0],-0.5*polarization_vector[1],-0.5*polarization_vector[2],color='g',linestyle='-')
            
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
        


        ################
        #At antenna
        ################

    if pol1 == True:
        print('Attempting the first polarization calculation')
        polarization_vector, vec_ray_to_ant, vec_neutrino_travel_dir = getPolarizationFactorsWrong(theta_ray_from_ant_at_antenna,phi_ray_from_ant_at_antenna,theta_neutrino_source_dir,phi_neutrino_source_dir,return_additional_vectors=True)
        print('polarization_vector = ', polarization_vector)
        if plot_quiver == True:
            #prepare plotting
            fig = pylab.figure()
            #fig = pylab.figure(figsize=(16,11.2))
            ax = fig.gca(projection='3d')
            #Plot antenna
            ax.scatter(antenna_loc[0],antenna_loc[1],antenna_loc[2],'Antenna')
            #Plot Ray
            ax.plot(x,y,z,color='k',linestyle='--',label='Ray')
            #plot neutrino vector
            ax.quiver(antenna_loc[0],antenna_loc[1],antenna_loc[2],vec_neutrino_travel_dir[0],vec_neutrino_travel_dir[1],vec_neutrino_travel_dir[2],color='r',label = 'Neutrino Travel Direction (Scale arb)',linestyle='-')
            ax.quiver(antenna_loc[0],antenna_loc[1],antenna_loc[2],vec_ray_to_ant[0],vec_ray_to_ant[1],vec_ray_to_ant[2],color='b',label = 'Observeration Direction (Scale arb)',linestyle='-')
            ax.quiver(antenna_loc[0],antenna_loc[1],antenna_loc[2],0.5*polarization_vector[0],0.5*polarization_vector[1],0.5*polarization_vector[2],color='g',label = 'Polarization Vector (Scale arb)',linestyle='-')
            ax.quiver(antenna_loc[0],antenna_loc[1],antenna_loc[2],-0.5*polarization_vector[0],-0.5*polarization_vector[1],-0.5*polarization_vector[2],color='g',linestyle='-')

            axis_add = 2
            ax.set_xlim([antenna_loc[0] - axis_add, antenna_loc[0] + axis_add])
            ax.set_ylim([antenna_loc[1] - axis_add, antenna_loc[1] + axis_add])
            ax.set_zlim([antenna_loc[2] - axis_add, antenna_loc[2] + axis_add])
            
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


    if pol2 == True:
        print('Attempting the second polarization calculation')
        polarization_vector, vec_ray_to_ant, vec_neutrino_travel_dir = getPolarizationFactors(theta_ray_from_ant_at_antenna,phi_ray_from_ant_at_antenna,theta_neutrino_source_dir,phi_neutrino_source_dir,return_additional_vectors=True)
        print('polarization_vector = ', polarization_vector)
        if plot_quiver == True:
            #prepare plotting
            fig = pylab.figure()
            #fig = pylab.figure(figsize=(16,11.2))
            ax = fig.gca(projection='3d')
            #Plot antenna
            ax.scatter(antenna_loc[0],antenna_loc[1],antenna_loc[2],'Antenna')
            #Plot Ray
            ax.plot(x,y,z,color='k',linestyle='--',label='Ray')
            #plot neutrino vector
            ax.quiver(antenna_loc[0],antenna_loc[1],antenna_loc[2],vec_neutrino_travel_dir[0],vec_neutrino_travel_dir[1],vec_neutrino_travel_dir[2],color='r',label = 'Neutrino Travel Direction (Scale arb)',linestyle='-')
            ax.quiver(antenna_loc[0],antenna_loc[1],antenna_loc[2],vec_ray_to_ant[0],vec_ray_to_ant[1],vec_ray_to_ant[2],color='b',label = 'Observeration Direction (Scale arb)',linestyle='-')
            ax.quiver(antenna_loc[0],antenna_loc[1],antenna_loc[2],0.5*polarization_vector[0],0.5*polarization_vector[1],0.5*polarization_vector[2],color='g',label = 'Polarization Vector (Scale arb)',linestyle='-')
            ax.quiver(antenna_loc[0],antenna_loc[1],antenna_loc[2],-0.5*polarization_vector[0],-0.5*polarization_vector[1],-0.5*polarization_vector[2],color='g',linestyle='-')


            axis_add = 2
            x_range = numpy.array([antenna_loc[0] - axis_add, antenna_loc[0] + axis_add])
            y_range = numpy.array([antenna_loc[1] - axis_add, antenna_loc[1] + axis_add])
            z_range = numpy.array([antenna_loc[2] - axis_add, antenna_loc[2] + axis_add])

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