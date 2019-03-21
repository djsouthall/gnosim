'''
This module contains the class Plane.
'''

import numpy

############################################################

'''
This used to be used in the test below, however is not currently used anywhere in the simulation.

def test2(self, field, n_trials=10):

        r_min, r_max = numpy.min(self.direct['r']), numpy.max(self.direct['r'])
        z_min, z_max = numpy.min(self.direct['z']), numpy.max(self.direct['z'])

        r = numpy.zeros(n_trials)
        z = numpy.zeros(n_trials)
        val = numpy.zeros(n_trials)

        for ii in range(0, n_trials):
            r[ii], z[ii] = numpy.random.uniform(r_min, r_max), numpy.random.uniform(z_min, z_max)
            while True:                                                                                                                                     
                r[ii], z[ii] = numpy.random.uniform(r_min, r_max), numpy.random.uniform(z_min, z_max)                                                       
                if self.query(r[ii], z[ii])[0]:                                                                                                             
                    break
            
            d = numpy.sqrt((r[ii] - self.direct['r'])**2 + (z[ii] - self.direct['z'])**2)
            index = numpy.argsort(d)
            #print (ii, r[ii], z[ii])
            #print (index, d[index[0:3]])
            r_select = self.direct['r'][index[0:3]]
            z_select = self.direct['z'][index[0:3]]
            val_select = self.direct[field][index[0:3]]
            #print (r_select, z_select, val_select)
            p = gnosim.utils.plane.Plane(r_select, z_select, val_select)
            val[ii] = p(r[ii], z[ii])
            #print (val[ii])
        
        pylab.figure()
        pylab.scatter(r, z, c=val, edgecolors='none')
        pylab.colorbar()

        pylab.figure()
        pylab.scatter(self.direct['r'], self.direct['z'], c=self.direct[field], edgecolors='none')
        pylab.colorbar()

        if field == 'z':
            pylab.figure()
            pylab.hist(val - z, bins=50)

        return r, z, val

'''

class Plane:
    '''
    Contains the properties of a plane which follows the equation:
    a*x + b*y + c*z + d = 0.

    Parameters
    ----------
    x : numpy.ndarray of floats
        A length 3 array contains the x, y, z coordinates of a point defining the plane.
    y : numpy.ndarray of floats
        A length 3 array contains the x, y, z coordinates of a point defining the plane.
    z : numpy.ndarray of floats
        A length 3 array contains the x, y, z coordinates of a point defining the plane.
    
    Attributes
    ----------
    x : numpy.ndarray of floats
        A length 3 array contains the x, y, z coordinates of a point defining the plane.
    y : numpy.ndarray of floats
        A length 3 array contains the x, y, z coordinates of a point defining the plane.
    z : numpy.ndarray of floats
        A length 3 array contains the x, y, z coordinates of a point defining the plane.
    a : float
        The a parameter used in describing the equation of the plane.
    b : float
        The b parameter used in describing the equation of the plane.
    c : float
        The c parameter used in describing the equation of the plane.
    d : float
        The d parameter used in describing the equation of the plane.
    '''

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.solveNormalPoint()

    def solveSystem(self):
        '''
        Define plane by solving system of equations.
        (Numerically unstable.)
        '''
        D = numpy.linalg.det([[self.x[0], self.y[0], self.z[0]],
                              [self.x[1], self.y[1], self.z[1]],
                              [self.x[2], self.y[2], self.z[2]]])
        self.d = 1.
        self.a = -1. * (self.d / D) *  numpy.linalg.det([[1., self.y[0], self.z[0]],
                                                         [1., self.y[1], self.z[1]],
                                                         [1., self.y[2], self.z[2]]])
        self.b = -1. * (self.d / D) *  numpy.linalg.det([[self.x[0], 1., self.z[0]],
                                                         [self.x[1], 1., self.z[1]],
                                                         [self.x[2], 1., self.z[2]]])
        self.c = -1. * (self.d / D) *  numpy.linalg.det([[self.x[0], self.y[0], 1.],
                                                         [self.x[1], self.y[1], 1.],
                                                         [self.x[2], self.y[2], 1.]])
    
    def solveNormalPoint(self):
        '''
        Define plane by normal-point method.
        (Numerically stable.)
        '''
        # Vector from point 0 to point 2
        u = [self.x[2] - self.x[0], 
             self.y[2] - self.y[0], 
             self.z[2] - self.z[0]] 
        # Vector from point 0 to point 1
        v = [self.x[1] - self.x[0], 
             self.y[1] - self.y[0], 
             self.z[1] - self.z[0]]
        # Components of normal vector determined via cross product
        self.a = numpy.linalg.det([[u[1], u[2]],
                                   [v[1], v[2]]])
        self.b = -1. * numpy.linalg.det([[u[0], u[2]],
                                         [v[0], v[2]]])
        self.c = numpy.linalg.det([[u[0], u[1]],
                                   [v[0], v[1]]])
        # Substitute point 0 for final constraint
        self.d = -1. * (self.a * self.x[0] + self.b * self.y[0] + self.c * self.z[0])

    def __call__(self, x, y):
        '''
        Calculates the z value of a plane for a particular set of x and y.
        
        Parameters
        ----------
        x : float
            The x coordinate.
        y : float
            The y coordinate.

        Returns
        -------
        z : float
            The z value of the plane for the given x and y inputs.
        '''
        return (self.a * x + self.b * y + self.d) / (-1. * self.c)
                                                        
############################################################
