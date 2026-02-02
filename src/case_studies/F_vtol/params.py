import numpy as np
ts = .01

d = 1
mixer = np.array([[1.0, 1.0],
                 [d,   -d]])
unmixer=np.linalg.inv(mixer)

m_c = 1
J_c = 0.0042
m_r = .25
m_l = .25
d = .3
mu = .1
g = 9.81

z0 = 0
h0 = 0
theta0 = 0
zdot0 = 0
hdot0 = 0
thetadot0 = 0