# %%
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from sympy.physics.vector.printing import vlatex
from IPython.display import Math, display
from sympy import sin, cos, Matrix, symbols, simplify, diff, diag

# defining constants
t, m_c, m_l, m_r, d, u, g, Jc = symbols("t, m_c, m_l, m_r, d, u, g, Jc")

# defining generalized coordinates
z_v = dynamicsymbols("z_v")
theta = dynamicsymbols("theta")
h = dynamicsymbols("h")

# --- POSITION VECTORS ---
p_center = Matrix([[z_v], 
                  [h], 
                  [0]])

r_left = Matrix([[-d], [0], [0]])
r_right = Matrix([[d], [0], [0]])

# --- 2. ROTATION MATRIX & TRANSFORMATIONS ---
# We define R here, after setting up the basics
R = Matrix([[cos(theta), -sin(theta), 0], 
            [sin(theta),  cos(theta), 0], 
            [0,           0,          1]])

# Now calculate absolute positions of motors
# Position = Center + (Rotation @ Body_Offset)
p_left  = p_center + (R @ r_left)
p_right = p_center + (R @ r_right)

# --- VELOCITIES ---
v_center = diff(p_center, t)
v_left = diff(p_left, t)
v_right = diff(p_right, t)

# --- ROTATIONAL ---
omega = Matrix([[0], [0], [theta.diff(t)]])

# Inertia
J_center_mat = diag(0, 0, Jc)
J_left = 0
J_right = 0


# --- KINETIC ENERGY CALCULATION ---
KE_center = 0.5 * m_c * (v_center.T @ v_center)
KE_rot_center = 0.5 * (omega.T @ R @ J_center_mat @ R.T @ omega)
KE_left = 0.5 * m_l * (v_left.T @ v_left)
KE_right = 0.5 * m_r * (v_right.T @ v_right)

K = simplify(KE_center + KE_left + KE_right + KE_rot_center)
K = K[0, 0]

display(Math(vlatex(K)))
# %%