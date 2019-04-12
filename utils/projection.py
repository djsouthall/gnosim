'''
This module contains a tool for calculating the projected visible area of a circular patch 
on the Earth for various viewing angles.  I (Dan Southall) did this for Cosmin as an aside
and it might be useful here some day.
'''

import numpy
import pylab
import scipy.spatial
import scipy.interpolate
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import sys
import gnosim.utils.linalg
pylab.ion()

def calculateProjectedArea(theta_cone , theta_obs , n_points , r_curvature = 6.371008e6 , plot = True , verbose = True , equal = True , fig = None , ax = None):
    '''
    This calculates the projected visible area of a portion of the surface of a sphere (with curvature r_curvature).  
    The size of the portion is selected using theta_cone.  This determines the convex hull 
    (or convave if theta_obs > 90.0) of the projected points in the xy plane and obtains the area of it.

    Parameters
    ----------
    theta_cone : float
        The angle of the cone.  The full opening angle is twice this.  Given in degrees.
    theta_obs : float
        The observation angle.  How much to rotate the spot from head on.  Given in degrees.
    n_points : int
        The number of points in the hull.  
    r_curvature : float, optional
        The radius of curvature to plot the points on and perform calculations for.  (Default is 6.371008e6)
    plot : bool, optional
        Enables plotting.  (Default is True).
    verbose : bool, optional
        Adds print statements.  (Default is True).
    equal : bool, optional
        Forces equal scaling of axis.  (Default is True).
    fig : matplotlib.pyplot.figure, optional
        The figure on which to plot on.  (Default is None).
    ax : matplotlib.pyplot.axis, optional
        The axis on which to plot on.  (Default is None).

    Returns
    ------
    area : float
        The calculated projected visible area of the shape.
    '''
    if theta_cone == 0.0:
        return 0.0
    if numpy.logical_or(theta_cone > 360.0, theta_cone < 0.0):
        theta_cone = theta_cone %360.0
    if theta_cone > 180.0:
        theta_cone = theta_cone-360.0
    #Preparing rotation matrix
    R = gnosim.utils.linalg.yRotationMatrix(numpy.deg2rad(theta_obs))

    #Outlining the sphere
    sphere_surface = gnosim.utils.linalg.angToVec(numpy.linspace(0,360.0,n_points), numpy.ones(n_points)*90.0)*r_curvature

    #Outlining the surface of the spot.
    unit_vectors = gnosim.utils.linalg.angToVec(numpy.linspace(0,360.0,n_points), numpy.ones(n_points)*theta_cone)
    vectors = unit_vectors*r_curvature
    points = numpy.array(list((zip(vectors[:,0],vectors[:,1]))))
    #original_hull = scipy.spatial.ConvexHull(points)

    #Rotating vectors
    rot_unit_vectors = numpy.zeros_like(unit_vectors)
    for i in range(len(unit_vectors)):
        rot_unit_vectors[i] = R.dot(unit_vectors[i])
    rot_phi = gnosim.utils.linalg.vecToAng(rot_unit_vectors)[0]
    rot_vectors = rot_unit_vectors*r_curvature
    rot_points = numpy.array(list((zip(rot_vectors[:,0],rot_vectors[:,1]))))

    #Determining visible points
    ang = numpy.rad2deg(numpy.arccos(rot_unit_vectors[:,2])) #Dotting each with z axis takes out z component.  Then arccos gets the angle between the vec and the z axis.
    visible_cut = numpy.logical_or(abs(ang) < 90.0, abs(ang) > 270.0) 

    #Building hull
    visible_hull_points = rot_points[visible_cut]
    #obstructed_hull_points = rot_points[~visible_cut]
    if numpy.size(visible_hull_points) == 0:
        area = 0
        concave_hull_points = numpy.array([])
    else:
        #if theta_obs is more than 90 degrees 
        if sum(~visible_cut) != 0:
            if theta_obs >= 0.0:
                if theta_obs <= 90.0:
                    phi_new_hull_points = rot_phi[~visible_cut]
                    phi_new_hull_points[ phi_new_hull_points >= 180.0 ] -= 360.0
                else:
                    phi_new_hull_points = rot_phi[visible_cut]
                    phi_new_hull_points[ phi_new_hull_points >= 180.0 ] -= 360.0

                phi_new_hull_points = numpy.sort(phi_new_hull_points)
                new_phi = numpy.linspace(phi_new_hull_points[0],phi_new_hull_points[-1],sum(~visible_cut))
                new_phi[new_phi <= 0.0 ] += 360.0
                
                new_surface_unit = gnosim.utils.linalg.angToVec(new_phi, numpy.ones(sum(~visible_cut))*90.0)
                new_surface = new_surface_unit*r_curvature
                new_points = numpy.array(list((zip(new_surface[:,0],new_surface[:,1]))))
                p1 = visible_hull_points
                p2 = new_points
                concave_hull_points = numpy.vstack((p1,p2))
            else:
                if theta_obs >= -90.0:
                    phi_new_hull_points = rot_phi[~visible_cut]
                else:
                    phi_new_hull_points = rot_phi[visible_cut]

                phi_new_hull_points = numpy.sort(phi_new_hull_points)
                new_phi = numpy.linspace(phi_new_hull_points[0],phi_new_hull_points[-1],sum(~visible_cut))
                new_phi[new_phi <= 0.0 ] += 360.0
                
                new_surface_unit = gnosim.utils.linalg.angToVec(new_phi, numpy.ones(sum(~visible_cut))*90.0)
                new_surface = new_surface_unit*r_curvature
                new_points = numpy.array(list((zip(new_surface[:,0],new_surface[:,1]))))
                visible_hull_points = visible_hull_points[numpy.argsort(180.0 - rot_phi[visible_cut])]
                p1 = visible_hull_points
                p2 = new_points
                concave_hull_points = numpy.vstack((p1,p2))

            
        else:
            concave_hull_points = visible_hull_points

        if abs(theta_obs) <= 90.0:
            area = scipy.spatial.ConvexHull(concave_hull_points).volume #because area for 2d data prints perimeter.
        else:
            p1 = numpy.vstack((p1,p1[0]))
            p2 = numpy.vstack((p2,p2[0]))
            try:
                p1_hull = scipy.spatial.ConvexHull(p1)
                a1 = p1_hull.volume
            except:
                #Can fail if not enough points to make hull.
                a1 = 0.0
            try:
                p2_hull = scipy.spatial.ConvexHull(p2)
                a2 = p2_hull.volume
            except:
                a2 = 0.0
            p2_hull = scipy.spatial.ConvexHull(p2)
            area = a2 - a1

    if plot == True:
        if fig == None:
            fig= pylab.figure()
        if ax == None:
            ax = fig.gca()
        ax.plot(sphere_surface[:,0]/1000.0, sphere_surface[:,1]/1000.0, c='g',label = 'Earth')
        ax.scatter(rot_vectors[~visible_cut,0]/1000.0,rot_vectors[~visible_cut,1]/1000.0,label='Obstructed Boundary', c = 'r')
        ax.scatter(rot_vectors[visible_cut,0]/1000.0,rot_vectors[visible_cut,1]/1000.0,label='Visible Boundary', c = 'b')
        if numpy.size(concave_hull_points)!= 0:
            concave_hull_points = numpy.vstack((concave_hull_points,concave_hull_points[0])) #Closes off hull.
            ax.plot(concave_hull_points[:, 0]/1000.0, concave_hull_points[:, 1]/1000.0, c='m',label = 'Surface Boundary')
        pylab.title('%0.3g deg'%theta_obs,fontsize=16)
        pylab.ylabel('y (km)',fontsize=14)
        pylab.xlabel('x (km)',fontsize=14)
        pylab.ylim([-r_curvature/1000.0,r_curvature/1000.0])
        pylab.tick_params(axis='both', which='major', labelsize=12)
        ax.yaxis.offsetText.set_fontsize(12)
        ax.xaxis.offsetText.set_fontsize(12)

        pylab.tick_params(axis='both', which='minor', labelsize=8)
        if equal == True:
            ax.set_aspect('equal', 'box')

    if verbose == True:
        print('Area observed at %0.23g degrees is %0.4g m^2'%(theta_obs,area))
    return area
    
def calculateProjectedAreaScan(theta_cone_vals , theta_obs_vals , n_points , r_curvature = 6.371008e6 , plot = True, verbose = True, fig = None , ax = None):
    '''
    This runs calculateProjectedArea over the list of angles specifed and returns gridded data for those.
    This calculates the projected visible area of a portion of the surface of a sphere (with curvature r_curvature).  
    The size of the portion is selected using theta_cone.  This determines the convex hull 
    (or convave if theta_obs > 90.0) of the projected points in the xy plane and obtains the area of it.

    Parameters
    ----------
    theta_cone_vals : float
        The list of cone angles to run.  The full opening angle is twice these values.  Given in degrees.
    theta_obs_valss : float
        The list of observation angle to run.  How much to rotate the spot from head on.  Given in degrees.
    n_points : int
        The number of points in the hull.  
    r_curvature : float, optional
        The radius of curvature to plot the points on and perform calculations for.  (Default is 6.371008e6)
    plot : bool, optional
        Enables plotting.  (Default is True).
    verbose : bool, optional
        Adds print statements.  (Default is True).
    fig : figure, optional
        The figure on which to plot on.  (Default is None).
    ax : axis, optional
        The axis on which to plot on.  (Default is None).

    Returns
    ------
    theta_cone_out : numpy.ndarray
        The corresponding cone angles to the output areas.  Given in degrees.
    theta_obs_out : numpy.ndarray
        The corresponding observation angles to the output areas.  Given in degrees.
    areas : numpy.ndarray
        The calculated projected visible area of the shape.
    '''
    areas = numpy.zeros((len(theta_obs_vals),len(theta_cone_vals)))
    theta_cone_out, theta_obs_out = numpy.meshgrid(theta_cone_vals,theta_obs_vals)
    n_iter = len(theta_cone_vals) * len(theta_obs_vals)
    
    for i, theta_cone in enumerate(theta_cone_vals):
        for j, theta_obs in enumerate(theta_obs_vals):
            areas[j][i] = calculateProjectedArea(theta_cone , theta_obs , n_points , r_curvature = r_curvature , plot = False , verbose = False , equal = False , fig = None , ax = None)
            if verbose == True:
                sys.stdout.write('%0.3f percent completed\r'%(100*(i*len(theta_cone_vals) + j+1.0)/n_iter))
                sys.stdout.flush()

    if verbose == True:
        print('')

    if plot == True:
        if fig == None:
            fig= pylab.figure()
        if ax == None:
            ax = fig.gca(projection='3d')
            
        # Plot the surface.
        surf = ax.plot_surface(theta_cone_out, theta_obs_out, areas/area_scale_factor, cmap=pylab.cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_xlabel('$\\theta_\\mathrm{cone}$ (Half of Opening Angle) (degrees)',fontsize=14)
        ax.set_ylabel('$\\theta_\\mathrm{obs}$ (degrees)',fontsize=14)
        ax.set_zlabel('Visible Projected Area ($km^2$)',fontsize=14)
    return theta_cone_out, theta_obs_out, areas

if __name__ == "__main__":
    #pylab.close('all')
    r_earth = 6.371008e6 #m
    r_curvature = 1.0#r_earth
    area_scale_factor = 1.0#1e6 #The areas are divide by this.  1e6 to convert from m^2 to km^2
    n_points = 1000 #Number of points in surface polygon

    #plot_single parameters
    plot_single = True
    verbose = True
    theta_cone = 10.0 #deg
    
    theta_obs = numpy.arange(181.0) #deg
    plot_at = numpy.linspace(0.0,180.0,9,dtype=int) #deg, only plots if they are all in theta_obs

    #plot_multi parameters
    plot_multi = False
    verbose_multi = True
    theta_cone_vals = numpy.linspace(1.0,60.0,100) #This angle is half the opening angle
    theta_obs_vals = numpy.linspace(0.0,180.0,100)

    #plotting

    if plot_single == True:
        print('Making Single Plot')
        height_ratios = [6,1]
        plot_at = plot_at[numpy.isin(plot_at,theta_obs)]
        fig = pylab.figure()
        gs_top = gridspec.GridSpec(2, 1, height_ratios=height_ratios)
        gs_bot = gridspec.GridSpec(2, len(plot_at), height_ratios=height_ratios)
        
        areas_single = numpy.zeros_like(theta_obs)
        full_area = calculateProjectedArea(theta_cone,0.0,n_points,r_curvature = r_curvature,plot = False,verbose = False,fig=None,ax=None,equal = True)
        for index,theta in enumerate(theta_obs):
            if numpy.isin(theta,plot_at):
                sub_index = numpy.where(numpy.isin(plot_at,theta))[0][0]
                ax = pylab.subplot(gs_bot[len(plot_at) + sub_index])
                if index != 0:
                    ax.get_yaxis().set_visible(False)
                areas_single[index] = calculateProjectedArea(theta_cone,theta,n_points,r_curvature = r_curvature,plot = True,verbose = verbose,fig=fig,ax=ax,equal = True)
            else:
                areas_single[index] = calculateProjectedArea(theta_cone,theta,n_points,r_curvature = r_curvature,plot = False,verbose = verbose)

        ax = pylab.subplot(gs_top[0])
        pylab.title('Visible Projected Area For Cone Angle of %0.2g degrees (Full Opening Angle of %0.2g degrees)'%(theta_cone,2*theta_cone),fontsize=16)
        pylab.plot(theta_obs,areas_single/(area_scale_factor),label='Projected Area')
        pylab.scatter(theta_obs[numpy.isin(theta_obs,plot_at)],areas_single[numpy.isin(theta_obs,plot_at)]/(area_scale_factor),c='b',label='Projection Plotted Below\ngreen=Earth\nblue=visible\nred=obstructed\nmagenta=hull')
        pylab.ylabel('Visible Projected Area ($km^2$)',fontsize=14)
        pylab.xlabel('Observation Angle (degrees)',fontsize=14)
        pylab.subplots_adjust(left = 0.07, bottom = 0.05, right = 0.97, top = 0.97, wspace = 0.15, hspace = 0.2)
        pylab.minorticks_on()
        pylab.ticklabel_format(axis='y',style='sci',scilimits=(-4,4))
        pylab.tick_params(axis='both', which='major', labelsize=12)
        ax.yaxis.offsetText.set_fontsize(12)
        pylab.tick_params(axis='both', which='minor', labelsize=8)
        pylab.grid(b=True, which='both', color='k', linestyle='-',alpha=0.2)
        pylab.plot(theta_obs,numpy.cos(numpy.deg2rad(theta_obs))*(full_area/area_scale_factor),label='Cosine')
        pylab.legend(fontsize=14)
        pylab.ylim([min(min(areas_single),-0.1*max(areas_single))/area_scale_factor,1.1*max(areas_single)/area_scale_factor])
        print('Single Plot Done')
    if plot_multi == True:
        print('Making Multi Plot')
        theta_cone_out, theta_obs_out, areas_multi = calculateProjectedAreaScan(theta_cone_vals , theta_obs_vals , n_points , r_curvature = r_curvature , plot = True , verbose = verbose_multi, fig = None , ax = None)
        print('Multi Plot Done')
        #coordinates = numpy.vstack((theta_cone_out.ravel(), theta_obs_out.ravel(), areas_multi.ravel()))
        
                               
