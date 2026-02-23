import sympy as sp
from sympy import symbols, Matrix, sin, cos, diff, simplify, eye, solve

# --- 1. DEFINE VARIABLES ---
t = symbols('t')
# States
z, theta = symbols('z theta')
zdot, thetadot = symbols('zdot thetadot')
# Accelerations (What we solve for)
zddot, thetaddot = symbols('zddot thetaddot')
# Force
F = symbols('F')
# Parameters
m1, m2, ell, g = symbols('m1 m2 ell g')

# Groupings
q = Matrix([z, theta])
qdot = Matrix([zdot, thetadot])
qddot = Matrix([zddot, thetaddot])

# --- 2. KINETIC & POTENTIAL ENERGY ---
# Inertia
J_beam = (m2 * ell**2) / 3 

# Velocity of block squared
# v^2 = r_dot^2 + (r*theta_dot)^2
v_block_sq = zdot**2 + (z * thetadot)**2

K = 0.5 * m1 * v_block_sq + 0.5 * J_beam * thetadot**2

# Potential Energy
h1 = z * sin(theta)
h2 = (ell / 2) * sin(theta)
P = m1 * g * h1 + m2 * g * h2

L = K - P

# --- 3. EULER-LAGRANGE (Manual Chain Rule) ---
# We calculate d/dt( dL/dqdot ) manually to avoid the "diff(symbol, t) = 0" issue.

# Gradient dL/dqdot
dL_dqdot = Matrix([diff(L, zdot), diff(L, thetadot)])

# Gradient dL/dq
dL_dq = Matrix([diff(L, z), diff(L, theta)])

# Time derivative d/dt( dL/dqdot )
# Chain rule: d/dt(f) = (df/dq) * qdot + (df/dqdot) * qddot
# Note: dL_dqdot is a vector, so we apply this to each element.
ddt_dL_dqdot = Matrix([0, 0])

for i in range(2):
    expr = dL_dqdot[i]
    # Jacobian w.r.t q dotted with qdot
    term1 = sum([diff(expr, q[j]) * qdot[j] for j in range(2)])
    # Jacobian w.r.t qdot dotted with qddot
    term2 = sum([diff(expr, qdot[j]) * qddot[j] for j in range(2)])
    ddt_dL_dqdot[i] = term1 + term2

# Generalized Forces
tau = Matrix([0, F * ell * cos(theta)])

# Final EOM: d/dt(dL/dqdot) - dL/dq = tau
EOM = ddt_dL_dqdot - dL_dq - tau

# --- 4. SOLVE FOR ACCELERATIONS ---
# Now we solve strictly for the symbols zddot and thetaddot
sol = solve(EOM, [zddot, thetaddot])

z_dd_expr = sol[zddot]
theta_dd_expr = sol[thetaddot]

# --- 5. LINEARIZATION ---
# State Space Form: x = [z, theta, zdot, thetadot]
f = Matrix([
    zdot,
    thetadot,
    z_dd_expr,
    theta_dd_expr
])

x = Matrix([z, theta, zdot, thetadot])
u = Matrix([F])

# Equilibrium
z_e = ell / 2
x_eq = {z: z_e, theta: 0, zdot: 0, thetadot: 0}
# Force to hold static equilibrium
u_eq = {F: (m1 * g * z_e) / ell + (m2 * g * ell/2) / ell}

# Compute A and B matrices
A_sym = f.jacobian(x).subs(x_eq).subs(u_eq)
B_sym = f.jacobian(u).subs(x_eq).subs(u_eq)

# --- 6. TRANSFER FUNCTIONS & GAINS ---
s = symbols('s')
C_mat = Matrix([[1, 0, 0, 0], [0, 1, 0, 0]]) 
D_mat = Matrix([[0], [0]])

# (A) Simplified Transfer Function (Decoupled for E.8)
# Apply "m1*g = 0" assumption to A matrix only
A_simple = A_sym.subs(m1 * g, 0)
TF_simple = C_mat * (s * eye(4) - A_simple).inv() * B_sym + D_mat

H_z_simple = simplify(TF_simple[0])
H_theta_simple = simplify(TF_simple[1])

print("--- Simplified Transfer Functions ---")
print(f"H_z(s)     = {H_z_simple}")
print(f"H_theta(s) = {H_theta_simple}")

# (B) Calculate Gains
print("\n--- Calculated Gains ---")
# Extract b_in (Coefficient of 1/s^2 in H_theta)
# H_theta = b_in / s^2  => b_in = H_theta * s^2
b_in = simplify(H_theta_simple * s**2)
b_out = -g # from inspection of physics (-g/s^2 relationship)

# Parameters
params = {m1: 0.35, m2: 2.0, ell: 0.5, g: 9.8}

b_in_val = float(b_in.subs(params))
b_out_val = float(b_out.subs(params))

# Specifications
tr_theta = 0.25
zeta = 0.707
tr_z = 2.5 

# Inner Loop (Theta)
wn_th = 2.2 / tr_theta
kp_th = wn_th**2 / b_in_val
kd_th = (2 * zeta * wn_th) / b_in_val

# Outer Loop (Position)
wn_z = 2.2 / tr_z
kp_z = wn_z**2 / b_out_val
kd_z = (2 * zeta * wn_z) / b_out_val

print(f"Inner Loop (Theta): kp = {kp_th:.4f}, kd = {kd_th:.4f}")
print(f"Outer Loop (Z)    : kp = {kp_z:.4f},  kd = {kd_z:.4f}")