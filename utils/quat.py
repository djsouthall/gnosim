"""
Quaternions
"""

import numpy

############################################################

def vecToAng(v):
    """
    Convert from 3-vector to angles [deg].
    """
    x, y, z = v
    r = numpy.sqrt(x**2 + y**2 + z**2)
    theta = numpy.degrees(numpy.arccos(z / r)) # angle from zenith
    phi = numpy.degrees(numpy.arctan2(y, x)) % 360.
    return phi, theta

############################################################

def angToVec(phi, theta):
    """
    Convert from angles [deg] to unit 3-vector.
    If phi and thetas are arrays then each row of output
    corresponds to a set of phi,theta
    """
    phi = numpy.radians(phi)
    theta = numpy.radians(theta)
    x = numpy.cos(phi) * numpy.sin(theta)
    y = numpy.sin(phi) * numpy.sin(theta)
    z = numpy.cos(theta)
    return numpy.array([x, y, z]).T

############################################################

def normalize(v):
    """
    Return normalized vector, works for 3-vectors or 4-vectors.
    If v contains multiple vectors each row is expected to be
    a vector.
    """
    if len(numpy.shape(v)) > 1:
        return v / numpy.sqrt(numpy.sum(v**2,axis = 1))[:,None]
    else:
        return v / numpy.sqrt(numpy.sum(v**2))

############################################################

def angTwoVec(v1, v2):
    """
    Return angle [deg] between two vectors.
    If v1 and v2 contain multiple vectors this assumes
    each row to be a seperate vector
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    if len(numpy.shape(v1)) > 1:
        return numpy.degrees(numpy.arccos(numpy.sum(v1 * v2,axis = 1)))
    else:
        return numpy.degrees(numpy.arccos(numpy.sum(v1 * v2)))

############################################################

def qMultiply(q1, q2):
    """
    Multiply to quaternions.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

############################################################

def qConjugate(q):
    """
    return conjugate of quaternion.
    """
    q = normalize(q)
    w, x, y, z = q
    return numpy.array([w, -x, -y, -z])

############################################################

def rotate(q, v):
    """
    Rotate vector v1 by quanternion q1.
    """
    q1 = normalize(q)
    q2 = numpy.concatenate([[0], v])
    return qMultiply(qMultiply(q1, q2), qConjugate(q1))[1:]

############################################################

def axisAngleToQuat(v, theta):
    """
    Vector v defines axis, theta is rotation angle [deg].
    """
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
    w, v = q[0], q[1:]
    theta = numpy.arccos(w) * 2.0
    return normalize(v), numpy.degrees(theta)

############################################################

def makeCone(phi, theta, angle, n):
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

