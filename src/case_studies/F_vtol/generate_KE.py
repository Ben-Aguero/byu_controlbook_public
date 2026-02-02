# %%
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from sympy.physics.vector.printing import vlatex
from IPython.display import Math, display
from sympy import sin, cos, Matrix, symbols, simplify, diff, diag

# defining constants
t, m_c, m_l, m_r, d, g, J_c = symbols("t, m_c, m_l, m_r, d, g, J_c")

# defining generalized coordinates
z = dynamicsymbols("z")
theta = dynamicsymbols("theta")
h = dynamicsymbols("h")

# Defining generalized coords and derivatives
q = Matrix([[z],[h],[theta]])
qdot = q.diff(t)

p_l = Matrix([[z - d * cos(theta)], [h - d * sin(theta)], [0]])
p_r = Matrix([[z + d * cos(theta)], [h + d * sin(theta)], [0]])
p_c = Matrix([[z], [h], [0]])

v_l = p_l.diff(t)
v_r = p_r.diff(t)
v_c = p_c.diff(t)

omega = Matrix([[0], [0], [theta.diff(t)]])

# --- 2. ROTATION MATRIX & TRANSFORMATIONS ---
# We define R here, after setting up the basics
R = Matrix([[cos(theta), -sin(theta), 0], 
            [sin(theta),  cos(theta), 0], 
            [0,           0,          1]])


J = diag(0, 0, J_c)

K = simplify(
    0.5 * m_l * v_l.T @ v_l
    + 0.5 * m_r * v_r.T @ v_r
    + 0.5 * m_c * v_c.T @ v_c
    + 0.5 * omega.T @ R @ J @ R.T @ omega
)

K = K[0, 0].expand().simplify()

display(Math(vlatex(K)))
# %%