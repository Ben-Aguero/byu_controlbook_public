# %%
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from sympy.physics.vector.printing import vlatex
from IPython.display import Math, display
from sympy import sin, cos, Matrix, symbols, simplify, diff, diag

# defining constants
t, m1, m2, ell, g = symbols("t, m1, m2, ell, g")

# defining generalized coordinates
z = dynamicsymbols("z")
theta = dynamicsymbols("theta")

# defining generalized coords and derivatives
q = Matrix([[z], [theta]])
qdot = q.diff(t)

p1 = Matrix([[z * cos(theta)], [z * sin(theta)], [0]])
p2 = Matrix([[ell/2*cos(theta)],[ell/2*sin(theta)],[0]])

v1 = p1.diff(t)   # Velocity of Beam COM
v2 = p2.diff(t) # Velocity of Block

# by inspection, we can find the angular velocity of the beam
omega = Matrix([[0], [0], [theta.diff(t)]])

R = Matrix([[cos(theta), -sin(theta), 0], 
            [sin(theta),  cos(theta), 0], 
            [0,           0,          1]])

# Next we define the intertia tensor for the beam, modeled as a thin rod
J = diag(0, m2 * ell**2 / 12.0, m2 * ell**2 / 12.0)

# Calculate the kinetic energy and display it
K = simplify(
    0.5 * m1 * v1.T @ v1 + 0.5 * m2 * v2.T @ v2 + 0.5 * omega.T @ R @ J @ R.T @ omega
)

# Just grabbing the scalar inside this matrix so that we can do L = K-P since P is a scalar
K = K[0, 0]

display(Math(vlatex(K)))
# %%