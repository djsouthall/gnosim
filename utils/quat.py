'''
Generally contains tools for vector calculations.  Orignally this was intended to contain quaternion information,
but has been repurposed.
'''

import numpy

############################################################
def vecToAng(v):
    '''
    Converts a cartesian 3-vector into the corresponding spherical angular coordinates (ignores radius).

    Parameters
    ----------
    v : numpy.ndarray of floats
        Input vector, expected to be in cartesian basis.

    Returns
    -------
    phi : float
        The azimuthal spherical coordinate.  Given in degrees.
    theta : float
        The polar spherical coordinate.  Given in degrees.
    '''
    x, y, z = v
    r = numpy.sqrt(x**2 + y**2 + z**2)
    theta = numpy.degrees(numpy.arccos(z / r)) # angle from zenith
    phi = numpy.degrees(numpy.arctan2(y, x)) % 360.
    return phi, theta

############################################################

def angToVec(phi, theta):
    '''
    Converts from spherical angular coordinates (ignores radius) to a unit cartesian 3-vector.
    If phi and thetas are arrays then each row of output corresponds to a set of phi,theta.
    Parameters
    ----------
    phi : float or numpy.ndarray of floats
        The azimuthal spherical coordinate(s).  Given in degrees.
    theta : float or numpy.ndarray of floats
        The polar spherical coordinate(s).  Given in degrees.

    Returns
    -------
    v : numpy.ndarray of floats
        Output vector(s), in cartesian basis. If phi and thetas are arrays then each row of this
        corresponds to a set of phi,theta.

    '''
    phi = numpy.radians(phi)
    theta = numpy.radians(theta)
    x = numpy.cos(phi) * numpy.sin(theta)
    y = numpy.sin(phi) * numpy.sin(theta)
    z = numpy.cos(theta)
    v = numpy.array([x, y, z]).T
    return v

############################################################

def normalize(v):
    '''
    Return the unit vector of the input vector, works for 3-vectors or 4-vectors.
    If v contains multiple vectors each row is expected to be a vector.

    Parameters
    ----------
    in_vector : numpy.ndarray
        Input vector of abitrary length

    Returns
    -------
    v : numpy.ndarray
        out_vector of unit length in same direction as input vector.  If v contains multiple vectors
        (each as a different row) then each oupput row represents an input vector row.
    '''
    if len(numpy.shape(v)) > 1:
        return v / numpy.sqrt(numpy.sum(v**2,axis = 1))[:,None]
    else:
        return v / numpy.sqrt(numpy.sum(v**2))

############################################################

def angTwoVec(v1, v2):
    '''
    Returns the angle between two vectors.  If v1 and v2 contain multiple rows each, then
    each row is expected to represent a different vector, and angles are calculated between
    pairs of row vectors.

    Parameters
    ----------
    in_vector : numpy.ndarray
        Input vector of abitrary length

    Returns
    -------
    angles : numpy.ndarray
        The angles between the vectors.  Given in degrees.  If v1 and v2 contain multiple 
        rows each, then each row is expected to represent a different vector, and angles 
        are calculated between pairs of row vectors.
    '''

    v1 = normalize(v1)
    v2 = normalize(v2)

    if len(numpy.shape(v1)) > 1:
        angles = numpy.degrees(numpy.arccos(numpy.sum(v1 * v2,axis = 1)))
    else:
        angles = numpy.degrees(numpy.arccos(numpy.sum(v1 * v2)))
    return angles
############################################################

def qMultiply(q1, q2):
    '''
    Multiply two quaternions.

    ***
    Note:  Quaternion calculations were added in the past and are currently not used in the code.
    I am unsure what the original intent for adding them was.  - DS
    ***

    Parameters
    ----------
    q1 : numpy.ndarray
        Input quaternion.
    q2 : numpy.ndarray
        Input quaternion.

    Returns
    -------
    q : numpy.ndarray
        Output quaternion.
    '''
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

############################################################

def qConjugate(q):
    '''
    Calculates the conjugate of a quaternion.

    ***
    Note:  Quaternion calculations were added in the past and are currently not used in the code.
    I am unsure what the original intent for adding them was.  - DS
    ***

    Parameters
    ----------
    q : numpy.ndarray
        Input quaternion.

    Returns
    -------
    q : numpy.ndarray
        Output quaternion, conjugate to input.
    '''
    q = normalize(q)
    w, x, y, z = q
    return numpy.array([w, -x, -y, -z])

############################################################

def rotate(q, v):
    '''
    Rotates the input vector by the quaternion.

    ***
    Note:  Quaternion calculations were added in the past and are currently not used in the code.
    I am unsure what the original intent for adding them was.  - DS
    ***

    Parameters
    ----------
    q : numpy.ndarray
        Input quaternion.
    v : numpy.ndarray
        Input vector.

    Returns
    -------
    output : numpy.ndarray
        I think the output is a vector?
    '''
    q1 = normalize(q)
    q2 = numpy.concatenate([[0], v])
    return qMultiply(qMultiply(q1, q2), qConjugate(q1))[1:]

############################################################

def axisAngleToQuat(v, theta):
    '''
    Vector v defines axis, theta is rotation angle [deg].

    ***
    Note:  Quaternion calculations were added in the past and are currently not used in the code.
    I am unsure what the original intent for adding them was.  - DS
    ***

    Parameters
    ----------
    v : numpy.ndarray
        Input vector which defines rotation axis ( I think ).
    theta : float
        Rotation angle.  Given in degrees.

    Returns
    -------
    q : numpy.ndarray
        Output quaternion.
    '''
    v = normalize(v)
    x, y, z = v
    theta = numpy.radians(theta)
    theta /= 2.
    w = numpy.cos(theta)
    x = x * numpy.sin(theta)
    y = y * numpy.sin(theta)
    z = z * numpy.sin(theta)
    return numpy.array([w, x, y, z])

############################################################

def quatToAxisAngle(q):
    '''
    Does something.  Made by previous developer and was never used so I am not bothering to
    figure out the precise purpose of it.

    ***
    Note:  Quaternion calculations were added in the past and are currently not used in the code.
    I am unsure what the original intent for adding them was.  - DS
    ***

    Parameters
    ----------
    q : numpy.ndarray
        Input quaternion.

    Returns
    -------
    v : numpy.ndarray
        Output vector.
    theta : numpy.ndarray
        Output vector.  Given in degrees.
    '''
    w, v = q[0], q[1:]
    theta = numpy.arccos(w) * 2.0
    return normalize(v), numpy.degrees(theta)

############################################################

def makeCone(phi, theta, angle, n):
    '''
    Does something (presumably makes a cone?).  Made by previous developer and was never used so I am not bothering to
    figure out the precise purpose of it.

    ***
    Note:  Quaternion calculations were added in the past and are currently not used in the code.
    I am unsure what the original intent for adding them was.  - DS
    ***

    Parameters
    ----------
    phi : float
        Rotation angle.  Given in degrees.
    theta : float
        Rotation angle.  Given in degrees.
    angle : float
        Some angle?  Give in degrees. 
    n : int
        Seems to be the number of angles in the cone to determine.

    Returns
    -------
    phi_rot_array : float
        Given in degrees.
    theta_rot_array : float
        Given in degrees.
    '''
    v_axis = angToVec(phi, theta)
    v_cone = angToVec(phi, theta + angle)

    phi_rot_array = numpy.zeros(n)
    theta_rot_array = numpy.zeros(n)

    rotation_array = numpy.linspace(0., 360., n)
    for ii in range(0, len(rotation_array)):
        q_rot = axisAngleToQuat(v_axis, rotation_array[ii])
        phi_rot_array[ii], theta_rot_array[ii] = vecToAng(rotate(q_rot, v_cone))

    return phi_rot_array, theta_rot_array

############################################################


# ORIENTATION TOOLS

def xRotationMatrix(theta_rad):
    '''
    Returns a 3x3 rotation matrix for rotating theta radians about the x axis.

    Parameters
    ----------
    theta_rad : float
        The angle by which to apply a rotation about the x axis.  Given in radians.

    Returns
    -------
    R : numpy.ndarray
        The constructed rotation matrix.
    '''
    R = numpy.array([   [1,0,0],
                        [0,numpy.cos(theta_rad),-numpy.sin(theta_rad)],
                        [0,numpy.sin(theta_rad),numpy.cos(theta_rad)]   ])
    return R

def yRotationMatrix(theta_rad):
    '''
    Returns a 3x3 rotation matrix for rotating theta radians about the y axis.

    Parameters
    ----------
    theta_rad : float
        The angle by which to apply a rotation about the y axis.  Given in radians.

    Returns
    -------
    R : numpy.ndarray
        The constructed rotation matrix.
    '''
    R = numpy.array([   [numpy.cos(theta_rad),0,numpy.sin(theta_rad)],
                        [0,1,0],
                        [-numpy.sin(theta_rad),0,numpy.cos(theta_rad)]   ])
    return R
    
def zRotationMatrix(theta_rad):
    '''
    Returns a 3x3 rotation matrix for rotating theta radians about the z axis.

    Parameters
    ----------
    theta_rad : float
        The angle by which to apply a rotation about the z axis.  Given in radians.

    Returns
    -------
    R : numpy.ndarray
        The constructed rotation matrix.
    '''
    R = numpy.array([   [numpy.cos(theta_rad),-numpy.sin(theta_rad),0],
                        [numpy.sin(theta_rad),numpy.cos(theta_rad),0],
                        [0,0,1]   ])
    return R

def eulerRotationMatrix(alpha_rad, beta_rad, gamma_rad):
    '''
    Returns a general 3x3 rotation matrix corresponding to the input angles. The output rotation matrix, 
    R, is created using the given Euler angles and a z-x-z extrinsic rotation.

    Parameters
    ----------
    theta_rad : float
        The alpha euler angle for a z-x-z extrinsic rotation.  Given in radians.
    beta_rad : float
        The beta euler angle for a z-x-z extrinsic rotation.  Given in radians.
    gamma_rad : float
        The gamma euler angle for a z-x-z extrinsic rotation.  Given in radians.

    Returns
    -------
    R : numpy.ndarray
        The constructed rotation matrix.
    '''
    Rz1 = zRotationMatrix(gamma_rad)
    Rx1 = xRotationMatrix(beta_rad)
    Rz2 = zRotationMatrix(alpha_rad)
    R = numpy.dot(Rz2,numpy.dot(Rx1,Rz1))
    return R