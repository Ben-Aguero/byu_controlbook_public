# %%
# local (controlbook)
import dis
from case_studies.common import sym_utils as su

from case_studies.F_vtol.generate_KE import *
su.enable_printing(__name__=="__main__")
# %%[markdown]
# The code imported from above shows how we defined q, q_dot, and necessary system parameters.
# Then we used position, velocity, and angular velocity to calculate kinetic energy.

# %%
# defining potential energy
# k = symbols("k")

P = (
    (m_c * g * h) + 
    (m_r * g * (h + d * sin(theta)) + 
     (m_l * g * (h - d * sin(theta)))) 
)  # this is "mgh", where "h" is a function of generalized coordinate "q"

# can also do the following to get the same answer
#   g_vec = Matrix([[0], [g], [0]])  # defining gravity in the direction that increases potential energy
#   p1 = Matrix([[ell/2*cos(theta)], [ell/2*sin(theta)], [0]])
#   P = m*g_vec.T@p1
#   P = P[0,0]

# calculate the lagrangian, using simplify intermittently can help the equations to be
# simpler, there are also options for factoring and grouping if you look at the sympy
# documentation.
L = simplify(K - P)
display(Math(vlatex(L)))
# %%
# Solution for Euler-Lagrange equations, but this does not include right-hand side (like friction and tau)
qd = q.diff(t)
qdd = qd.diff(t)
EL_case_studyF = simplify(diff(diff(L, qd), t) - diff(L, q))

display(Math(vlatex(EL_case_studyF)))


# %%
############################################################
### Including friction and generalized forces, then solving for highest order derivatives
############################################################

# these are just convenience variables
hd = h.diff(t)
hdd = hd.diff(t)
zd = z.diff(t)
zdd = zd.diff(t)
thetad = theta.diff(t)
thetadd = thetad.diff(t)

# defining symbols for external force and friction
u_F, u_tau, mu = symbols("u_F, u_tau, mu")

# defining the right-hand side of the equation and combining it with E-L part
RHS = Matrix([[-u_F * sin(theta) - mu * zd],
              [u_F * cos(theta)],
              [u_tau] 
              ])
full_eom = EL_case_studyF - RHS

# finding and assigning zdd and thetadd
# if our eom were more complicated, we could rearrange, solve for the mass matrix, and invert it to move it to the other side and find qdd and thetadd
result = simplify(sp.solve(full_eom, qdd))

# TODO - add an example of finding the same thing, but not using sp.solve

# result is a Python dictionary, we get to the entries we are interested in
# by using the name of the variable that we were solving for
zdd_eom = result[zdd]
hdd_eom = result[hdd]
thetadd_eom = result[thetadd]  # EOM for thetadd, as a function of states and inputs

# display(Math(vlatex(zdd_eom)))
# display(Math(vlatex(hdd_eom)))
# display(Math(vlatex(thetadd_eom)))
# M = EL_case_studyF.jacobian(qdd)
# display(Math("M(q) = " + vlatex(simplify(M))))

#%%
#F.4 and F.6
state_sym = sp.Matrix([z, h, theta, zd, hd, thetad])
input_sym = sp.Matrix([u_F, u_tau])

f_sym = sp.Matrix([
    zd,
    hd,
    thetad,
    zdd_eom,
    hdd_eom,
    thetadd_eom
])

eq_subs = {zd: 0, hd: 0, thetad: 0}
eq_zdd = zdd_eom.subs(eq_subs)
eq_hdd = hdd_eom.subs(eq_subs)
eq_thetadd = thetadd_eom.subs(eq_subs)

equilibria = sp.solve([eq_zdd, eq_hdd, eq_thetadd], (u_F, u_tau, theta))

print("Equilibrium conditions (u_F, u_tau, theta) for hover:")
# The result might be a list of solutions, usually the real one is theta=0
if isinstance(equilibria, list):
    for sol in equilibria:
        # Filter for the realistic hover solution (theta = 0)
        # We display the raw solution to be safe
        display(Math(vlatex(sol)))
else:
    display(Math(vlatex(equilibria)))


A_sym = f_sym.jacobian(state_sym)
B_sym = f_sym.jacobian(input_sym)

# Define the operating point for hover
# Theta = 0, Velocities = 0
# Force = Total Weight (sum of all masses * g)
# Note: In your PE equation, you have m_c, m_r, m_l.
m_total = m_c + m_r + m_l 
op_point = {
    theta: 0,
    zd: 0, hd: 0, thetad: 0,
    u_tau: 0,
    u_F: m_total * g  # Force must counteract gravity
}

# Substitute operating point
A_lin = A_sym.subs(op_point)
B_lin = B_sym.subs(op_point)

# Simplify results
A_lin = simplify(A_lin)
B_lin = simplify(B_lin)

display(Math("A_{lin} = " + vlatex(A_lin)))
display(Math("B_{lin} = " + vlatex(B_lin)))


C_sym = sp.Matrix([
    [1, 0, 0, 0, 0, 0], # z
    [0, 1, 0, 0, 0, 0], # h
    [0, 0, 1, 0, 0, 0]  # theta
])

D_sym = sp.Matrix([
    [0, 0],
    [0, 0],
    [0, 0]
])

print("Symbolic State Space Matrices:")
display(Math("A = " + vlatex(A_lin)))
display(Math("B = " + vlatex(B_lin)))
display(Math("C = " + vlatex(C_sym)))
display(Math("D = " + vlatex(D_sym)))

# --- Numerical Evaluation ---
print("\nNumerical Check:")

# Re-import params as PAR to avoid conflict with the 'P' variable used for Potential Energy
import params as PAR

# Calculate the numerical value for total mass
mass_sum_val = PAR.mc + PAR.mr + PAR.mr

params_subs = {
    m_c: PAR.mc,
    m_r: PAR.mr,
    m_l: PAR.mr,
    J_c: PAR.Jc,
    d: PAR.d,
    mu: PAR.mu,
    g: PAR.g,
    m_total: mass_sum_val # Substitute the sum we used in linearization
}

A_num = A_lin.subs(params_subs)
B_num = B_lin.subs(params_subs)

display(Math("A_{num} = " + vlatex(A_num)))
display(Math("B_{num} = " + vlatex(B_num)))

print("\n--- Part (a): Longitudinal State Space (h, h_dot) ---")

A_lon = sp.Matrix([
    [0, 1],
    [0, 0]
])

# From dynamics: h_ddot = u_F / m_total
B_lon = sp.Matrix([
    [0],
    [1/m_total]
])

display(Math("A_{lon} = " + vlatex(A_lon)))
display(Math("B_{lon} = " + vlatex(B_lon)))


# --- Part (b): Lateral System (Position & Orientation) ---
# States: x_lat = [z, theta, z_dot, theta_dot]
# Input: u_lat = u_tau
print("\n--- Part (b): Lateral State Space (z, theta, z_dot, theta_dot) ---")

# We extract these terms from the full symbolic A_lin matrix we calculated earlier.
# Rows/Cols indices in full state [z, h, theta, zd, hd, thetad]:
# z=0, theta=2, zd=3, thetad=5

# A_lat structure:
# [0, 0, 1, 0]  (z_dot = z_dot)
# [0, 0, 0, 1]  (theta_dot = theta_dot)
# [?, ?, ?, ?]  (z_ddot equation)
# [?, ?, ?, ?]  (theta_ddot equation)

# Extracting symbolic terms from our full A_lin
A_lat = sp.Matrix([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [A_lin[3, 0], A_lin[3, 2], A_lin[3, 3], A_lin[3, 5]], # z_ddot row
    [A_lin[5, 0], A_lin[5, 2], A_lin[5, 3], A_lin[5, 5]]  # theta_ddot row
])

# Extracting symbolic terms from our full B_lin
# Input is u_tau (column 1 in the full B matrix)
B_lat = sp.Matrix([
    [0],
    [0],
    [B_lin[3, 1]], # Effect of torque on z_ddot (usually 0)
    [B_lin[5, 1]]  # Effect of torque on theta_ddot (1/J_c)
])

display(Math("A_{lat} = " + vlatex(A_lat)))
display(Math("B_{lat} = " + vlatex(B_lat)))


# %% [markdown]
# OK, now we can get the state variable form of the equations of motion.

# %%

import numpy as np

# defining fixed parameters that are not states or inputs (like g, ell, m, b)
# can be done like follows:
params = [(m_c, PAR.m_c),
          (J_c, PAR.J_c),
          (m_r, PAR.m_r),
          (m_l, PAR.m_l),
          (d, PAR.d),
          (mu, PAR.mu),
          (g, PAR.g)]

# but in this example, I want to keep the masses, length, and damping as variables so
# that I can simulate uncertainty in those parameters in real life.
# params = [(g, P.g)]

# substituting parameters into the equations of motion
zdd_eom = zdd_eom.subs(params)
hdd_eom = hdd_eom.subs(params)
thetadd_eom = thetadd_eom.subs(params)

# now defining the state variables that will be passed into f(x,u)
# state = np.array([theta, thetad])
# ctrl_input = np.array([tau])

state = sp.Matrix([z, h, theta, zd, hd, thetad])
ctrl_input = sp.Matrix([u_F, u_tau])

# defining the function that will be called to get the derivatives of the states
state_dot = sp.Matrix([zd, hd, thetad, zdd_eom, hdd_eom, thetadd_eom])


# %%
import numpy as np

# converting the function to a callable function that uses numpy to evaluate and
# return a list of state derivatives
eom = sp.lambdify([state, ctrl_input, m_c, J_c, m_r, m_l, d, mu, g], state_dot, "numpy")

# calling the function as a test to see if it works:
cur_state = np.array([0, 0, 0, 0, 0, 0])
cur_input = np.array([1, 1])
print("x_dot = ", eom(cur_state, cur_input, PAR.m_c, PAR.J_c, PAR.m_r, PAR.m_l, PAR.d, PAR.mu, PAR.g))


# %% [markdown]
# The next step is to save this function "f" so that we can use it with a numerical integrator, like
# scipy.integrate.ivp.solve_ivp or the rk4 functions in the case studies. To save this function, we can use the following:

# %%
# this code will only run if this file is executed directly,
# not if it is imported as a module.
if __name__ == "__main__":
    from case_studies import F_vtol

    # make sure printing only happens when running this file directly
    su.enable_printing(__name__ == "__main__")

    su.write_eom_to_file(state, ctrl_input, [m_c, J_c, m_l, m_r, d, mu, g], F_vtol, eom=state_dot)

    import numpy as np
    from case_studies.F_vtol import eom_generated
    import importlib

    importlib.reload(eom_generated)  # reload in case it was just generated/modified
    P = F_vtol.params

    param_vals = {
        "m_c": PAR.m_c,
        "m_r": PAR.m_r,
        "m_l": PAR.m_l,
        "J_c": PAR.J_c,
        "mu": PAR.mu,
        "d": PAR.d,
        "g": PAR.g
    }

    x_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_test = np.array([1.0, 1.0])

    x_dot_test = eom_generated.calculate_eom(x_test, u_test, **param_vals)
    print("\nx_dot_test from generated function = ", x_dot_test)
    # should match what was printed earlier when we called eom directly
