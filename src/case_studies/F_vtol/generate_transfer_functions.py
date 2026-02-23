# %%
from case_studies.F_vtol.generate_linearization import *
from sympy import eye, zeros, simplify, symbols, Matrix
from IPython.display import display, Math
from sympy.physics.vector import vlatex

# ####################################################################################
# Case Study F: VTOL Transfer Functions
# ####################################################################################
# the order for rows in C is just because I had defined my states as [z, h, theta, z_dot, h_dot, theta_dot], but I want
# h first, then z, then theta for my transfer functions
C = Matrix([[0, 1.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0]])
D = Matrix(zeros(3, 2))

# %%
# define 's' as symbolic variable for the Laplace domain
s = symbols("s")

# these are the three transfer functions for h, z, and theta with respect to inputs F and tau
transfer_func = simplify(C @ (s * eye(6) - A_lin).inv() @ B_lin + D)

# %%
# these indices that we select from transfer_func are based on the
# output (row) and input (column)
print("Transfer function H(s)/F(s)")
display(Math(vlatex(transfer_func[0, 0])))

print("\nTransfer function z(s)/Tau(s)")
display(Math(vlatex(transfer_func[1, 1])))

print("\nTransfer function Theta(s)/Tau(s)")
display(Math(vlatex(transfer_func[2, 1])))

# %%