from sympy.physics.vector import dynamicsymbols

from case_studies.common import sym_utils as su
from case_studies.F_vtol.generate_state_variable_form import *

z = dynamicsymbols("z")
h = dynamicsymbols("h")
theta = dynamicsymbols("theta")

print("EOM, in state variable form: \n")
su.printeq("\ndot{x} ", state_dot)

A = state_dot.jacobian(state)
B = state_dot.jacobian(ctrl_input)

F = f_l + f_r
tau = (f_r - f_l) * d

A_lin = A.subs([(theta.diff(t), 0), (theta, 0), (F, (m_c + 2 * m_r) * g), (tau, 0)])
B_lin = B.subs([(theta.diff(t), 0), (theta, 0), (F, (m_c + 2 * m_r) * g), (tau, 0)])

print("\nLinearized A Matrix is:")
su.printeq("A_lin", A_lin)

print("\nLinearized B Matrix is:")
su.printeq("B_lin", B_lin)

A_lin_lon = A_lin[[1, 4], [1, 4]]
B_lin_lon = B_lin[[1, 4], [0]]

print("\nLinearized A_lon Matrix is:")
su.printeq("A_lin_lon", A_lin_lon)

print("\nLinearized B_lon Matrix is:")
su.printeq("B_lin_lon", B_lin_lon)

A_lin_lat = A_lin[[0, 2, 3, 5], [0, 2, 3, 5]] 
B_lin_lat = B_lin[[0, 2, 3, 5], [1]] 
print("\nLinearized A_lat Matrix is:") 
su.printeq("A_lin_lat", A_lin_lat) 

print("\nLinearized B_lat Matrix is:") 
su.printeq("B_lin_lat", B_lin_lat)