"""
This tool is meant to generate the coordinates and orientation angles for neutrinos for specific
test cases such as a circle of neutrinos with on cone events.  

Note that this is not done to be fast one the user side (as it is not likely to be used often
and when it is used it may only be used once), and so was coded moreso to be quick on the developer
side.  i.e. I wrote this quick, acknowledging there are smarter ways to do the same thing.
"""
import os
import sys
sys.path.append(os.environ['GNOSIM_DIR'])
import csv
import numpy
import gnosim.trace.refraction_library
import gnosim.utils.linalg
import gnosim.earth.ice
import pylab
############################################################

if __name__ == "__main__":

    outfile = 'test.py'#'10m_circle_oncone_theta180xphi0.csv'
    radius = 10.0 #m
    ice_model = 'antarctica'
    origin_antenna = numpy.array([[0.0,0.0,-173.0]])
    z_antenna = numpy.array([-173.0])
    ice = gnosim.earth.ice.Ice('antarctica')
    
    theta_ant = numpy.linspace(0.01,179.9,180) #Limited by 180.0 as it is redundent otherwise
    phi_ant = numpy.array([0.0])#numpy.linspace(0.0,360.0,360)

    #Mode decides whether radius is in cylindrical coordinates (resulting in cylinder of points) or spheral.
    mode = 'sphere'#'cylinder' #sphere 

    ###########

    x_0 = []
    y_0 = []
    z_0 = []
    phi_0 = [] #Neutrino source dir
    theta_0 = [] #Neutrino source dir
    header = ["x_0" , "y_0" , "z_0" , "phi_0" , "theta_0"]
    with open(outfile,'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(header)

        for origin in origin_antenna:
            for theta_ant_i in theta_ant:
                for phi_ant_i in phi_ant:
                    print('(theta_ant_i = %0.3g , phi_ant_i = %0.3g)'%(theta_ant_i,phi_ant_i))
                    x, y, z, t, d, phi, theta, a_p, a_s, index_reflect_air, index_reflect_water = gnosim.trace.refraction_library.rayTrace(origin, phi_ant_i, theta_ant_i, ice, t_max=50000., t_step=1., r_limit=radius+0.00001, fresnel_mode='amplitude')
                    try:
                        if mode == 'cylinder':
                            index = -1
                        elif mode == 'sphere':
                            index = numpy.where(d >= radius)[0][0]
                    except:
                        index = -1

                    index_of_refraction_at_neutrino = ice.indexOfRefraction(z[index])
                    cherenkov_angle_deg = numpy.rad2deg(numpy.arccos(1./index_of_refraction_at_neutrino))
                    x_0.append(x[index] + origin[0])
                    y_0.append(y[index] + origin[1])
                    z_0.append(z[index])
                    phi_0.append(phi[index])
                    theta_0.append(theta[index] - cherenkov_angle_deg)
                    writer.writerow([x[index] + origin[0], y[index] + origin[1], z[index], phi[index], theta[index] - cherenkov_angle_deg])

    writeFile.close()

    from mpl_toolkits.mplot3d import Axes3D

    fig = pylab.figure()
    ax = fig.gca(projection='3d')
    #Plot antenna
    for origin in origin_antenna:
        ax.scatter(origin[0],origin[1],origin[2],label='Antenna')
    ax.scatter(x_0,y_0,z_0)


