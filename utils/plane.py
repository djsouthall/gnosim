import numpy

############################################################

class Plane:
    
    def __init__(self, x, y, z):
        """
        x, y, z are each length 3 arrays corresponding to 
        the x, y, z coordinates of 3 points which define a plane
        """
        self.x, self.y, self.z = x, y, z
        self.solveNormalPoint()

    def solveSystem(self):
        """
        Define plane by solving system of equations.
        (Numerically unstable.)
        """
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
        """
        Define plane by normal-point method.
        (Numerically stable.)
        """
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
        return (self.a * x + self.b * y + self.d) / (-1. * self.c)
                                                        
############################################################
