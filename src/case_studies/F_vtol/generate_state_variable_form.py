# %%
from case_studies.F_vtol.generate_KE import *
from case_studies.common import sym_utils as su
import sympy as sp
from IPython.display import display, Math
from sympy.physics.vector import vlatex

# Case Study F: VTOL - Find Equations of Motion via Euler-Lagrange
# #####################################################################

# Find Potential Energy
# P = m_c * g * h + m_l * g * (h - d * sin(theta)) + m_r * g * (h + d * sin(theta)) # general case
# but if we assume that m_l = m_r the sin terms cancel:
P_energy = (m_c + m_l + m_r) * g * h

# Lagrangian
L = K - P_energy

# Left-hand side of Euler-Lagrange equation
LHS = sp.simplify(L.diff(qdot).diff(t) - L.diff(q))

# Define symbols for control inputs and friction force
mu, f_r, f_l = sp.symbols("mu, f_r, f_l")

# Right-hand side: generalized forces (control inputs + friction)
tau = sp.Matrix([
    [sp.sin(theta) * (f_r + f_l)],
    [sp.cos(theta) * (f_r + f_l)],
    [d * (f_r - f_l)]
])

friction = sp.Matrix([
    [-mu * qdot[0]],
    [0],
    [0]
])

RHS = tau + friction

# Solve Euler-Lagrange equations for accelerations
# substitute m_l = m_r to simplify
total_eom = (LHS - RHS).subs([(m_l, m_r)])

zd = z.diff(t)
hd = h.diff(t)
thetad = theta.diff(t)

zdd = zd.diff(t)
hdd = hd.diff(t)
thetadd = thetad.diff(t)

result = sp.solve(total_eom, (zdd, hdd, thetadd))

# Display the equations of motion
zdd_eom = sp.simplify(result[zdd])
hdd_eom = sp.simplify(result[hdd])
thetadd_eom = sp.simplify(result[thetadd])

print("z_ddot equation:")
display(Math(vlatex(zdd_eom)))
print("\nh_ddot equation:")
display(Math(vlatex(hdd_eom)))
print("\ntheta_ddot equation:")
display(Math(vlatex(thetadd_eom)))

# [markdown]
# OK, now we can get the state variable form of the equations of motion.

import case_studies.F_vtol.params as P
import numpy as np

# defining fixed parameters that are not states or inputs (Like g, mc, mr, c, d, mu)
# can be done like follows:
# params = [(g, P.g), (mc, P.mc), (mr, P.mr), (Jc, P.Jc), (d, P.d), (mu, P.mu)]
# but in this example, I want to keep these as variables so
# that I can simulate uncertainty in those parameters in real life.
params = [(g, P.g)]

# substituting parameters into the equations of motion
zdd_eom = zdd_eom.subs(params)
hdd_eom = hdd_eom.subs(params)
thetadd_eom = thetadd_eom.subs(params)

# now defining the state variables that will be passed into f(x,u)
state = sp.Matrix([z, h, theta, zd, hd, thetad])
ctrl_input = sp.Matrix([f_r, f_l])

# defining the function that will be called to get the derivatives of the states
state_dot = sp.Matrix([zd, hd, thetad, zdd_eom, hdd_eom, thetadd_eom])

# [markdown]
# The next step is to save this function "f" so that we can use it with a numerical integrator, like
# scipy.integrate.solve_ivp or the rk4 functions in the case studies. To save this function, we do the
# following:

if __name__ == "__main__":
    from case_studies import F_vtol
    # make sure printing only happens when running this file directly
    su.enable_printing(__name__ == "__main__")
    
    su.write_eom_to_file(state, ctrl_input, [m_c, m_r, J_c, d, mu], F_vtol, eom=state_dot)
    
    from case_studies.F_vtol import eom_generated
    import importlib
    importlib.reload(eom_generated) # reload in case it was just generated/modified
    
    param_vals = {
        "m_c": P.mc,
        "m_r": P.mr,
        "J_c": P.Jc,
        "d": P.d,
        "mu": P.mu,
    }
    
    x_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_test = np.array([0.0, 0.0])
    x_dot_test = eom_generated.calculate_eom(x_test, u_test, **param_vals)
    
    print("\nx_dot test from generated function:\n", x_dot_test)
# %%
