import numpy as np
import scipy
from numpy.linalg import inv


def rotate_y(theta):
    s = np.sin(theta);
    c = np.cos(theta);
    return np.array(
        [[c, 0, s, 0],
         [0, 1, 0, 0],
         [-s, 0, c, 0],
         [0, 0, 0, 1]])


def rotate_z(theta):
    s = np.sin(theta);
    c = np.cos(theta);
    return np.array(
        [[c, -s, 0, 0],
         [s, c, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])


def z_offset(offset):
    tsl = np.identity(4)
    tsl[2, 3] = offset
    return tsl


def map_function(t, q):
    mapfunc = np.identity(4)
    y_init = np.array([0, 0., 0., 1.0]).T
    for k in np.arange(4):
        mapfunc = t[4 - k].dot(mapfunc)
        if k < 3:
            mapfunc = rotate_y(q[3 - k]).dot(mapfunc)
        else:
            mapfunc = rotate_z(q[3 - k]).dot(mapfunc)
    mapfunc = t[0].dot(mapfunc)
    return mapfunc


# Parameters
y_init = np.array([0, 0., 0., 1.0]).T
y_target = np.array([0.27, -0.13, 0.89])
# Joint states
q = np.array([1.34, 0.21, 2.0, -0.22])
# q = np.array([np.pi/2, np.pi/2., -np.pi/2., np.pi/2.])
# Translation
T_base_L1 = z_offset(0.2)
T_L1_L2 = z_offset(0.25)
T_L2_L3 = z_offset(0.15)
T_L3_L4 = z_offset(0.25)
T_L4_G = z_offset(0.1)
t = [T_base_L1, T_L1_L2, T_L2_L3, T_L3_L4, T_L4_G]

p_L1 = t[0].dot(y_init)
p_L2 = t[0].dot(rotate_z(q[0]).dot(t[1].dot(y_init)))
p_L3 = t[0].dot(rotate_z(q[0]).dot(t[1].dot(rotate_y(q[1]).dot(t[2].dot(y_init)))))
p_L4 = t[0].dot(rotate_z(q[0]).dot(t[1].dot(rotate_y(q[1]).dot(t[2].dot(rotate_y(q[2]).dot(t[3].dot(y_init)))))))
eff = t[0].dot(rotate_z(q[0]).dot(t[1].dot(rotate_y(q[1]).dot(t[2].dot(rotate_y(q[2]).dot(t[3].dot(rotate_y(q[3]).dot(t[4].dot(y_init)))))))))
a1 = np.array([0, 0, 1, 1])
a2 = rotate_z(q[0]).dot(np.array([0, 1, 0, 1]))
a3 = a2
a4 = a3

# 1.1
mapFunc = map_function(t, q)
y = mapFunc.dot(y_init)
print 'y:\n', np.round(y, 5), '\n'
print 'eff:\n', np.round(eff, 5), '\n'
print 'map:\n', np.round(mapFunc, 5), '\n'


# 1.2
y0 = mapFunc.dot(y_init)
J_1 = np.cross(a1[0:3], (eff - p_L1)[0:3])
J_2 = np.cross(a2[0:3], (eff - p_L2)[0:3])
J_3 = np.cross(a3[0:3], (eff - p_L3)[0:3])
J_4 = np.cross(a4[0:3], (eff - p_L4)[0:3])
J = np.array([J_1, J_2, J_3, J_4]).T
J_inv = J.T.dot(inv(J.dot(J.T) + inv(np.identity(3, dtype=float) * 1000)))

qd = J_inv.dot(y_target - y0[0:3])
q = q + qd

print 'Jacobian:\n', np.round(J, 5),'\n'
print 'Jacobian Inverse:\n', np.round(J_inv, 5), '\n'
print 'Joint difference:\n', np.round(qd, 5)

mapFunc = map_function(t, q)
y = mapFunc.dot(y_init)
print 'y:\n', np.round(y, 5), '\n'
print 'qd:\n', np.round(qd, 5), '\n'
