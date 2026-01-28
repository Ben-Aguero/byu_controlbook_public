# %%
import sys
!{sys.executable} -m pip install sympy

#%%
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from sympy.physics.vector.printing import vlatex
from IPython.display import Math, display

####################################################################################
# example from Case Study A to find Kinetic Energy (with vectors/matrices)
####################################################################################
# importing these functions directly to make life a little easier and code a little more readable
from sympy import sin, cos, Matrix, symbols, simplify, diff

# init_printing(use_latex=True)

# defining mathematical variables (called symbols in sp) and time varying functions like z and theta
t, m, b, k = symbols("t, m, b, k")

# defining generalized coord and their derivatives
z = dynamicsymbols("z")
q = Matrix([[z]])
qdot = q.diff(t)

# to find KE, start with the position of the mass in the inertial frame, then find the velocity
p = Matrix([[0], [0], [z]])
v = p.diff(t)

# calculate the kinetic energy and display it
K = simplify(0.5 * m * v.T @ v)
K = K[0, 0]

display(Math(vlatex(K)))

# %%
