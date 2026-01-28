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

# --- POSITION VECTORS ---
# p_beam: Center of Mass of beam (at ell/2)
p_beam = Matrix([[(ell / 2) * cos(theta)], 
                 [(ell / 2) * sin(theta)], 
                 [0]])

# p_block: Block position (at z)
p_block = Matrix([[z * cos(theta)], 
                  [z * sin(theta)], 
                  [0]])

# --- VELOCITIES ---
v_beam = diff(p_beam, t)   # Velocity of Beam COM
v_block = diff(p_block, t) # Velocity of Block

# --- ROTATIONAL ---
omega = Matrix([[0], [0], [theta.diff(t)]])
R = Matrix([[cos(theta), -sin(theta), 0], 
            [sin(theta),  cos(theta), 0], 
            [0,           0,          1]])

# Inertia of beam about COM (1/12)
# We use 1/12 because we are adding the linear KE of the beam separately below
I_beam_com = (m2 * (ell**2)) / 12.0
J_beam = diag(0, I_beam_com, I_beam_com)

# --- KINETIC ENERGY CALCULATION ---
# FIX: Ensure m2 is used with v_beam, and m1 is used with v_block
KE_beam_trans = 0.5 * m2 * (v_beam.T @ v_beam)
KE_block_trans = 0.5 * m1 * (v_block.T @ v_block)
KE_beam_rot   = 0.5 * (omega.T @ R @ J_beam @ R.T @ omega)

K = simplify(KE_beam_trans + KE_block_trans + KE_beam_rot)
K = K[0, 0]

display(Math(vlatex(K)))
# %%