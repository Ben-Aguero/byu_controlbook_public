# %%
# local (controlbook)
import dis
from case_studies.common import sym_utils as su
import case_studies.E_blockbeam.params as PAR

from case_studies.E_blockbeam.generate_KE import *
su.enable_printing(__name__=="__main__")
# %%[markdown]
# The code imported from above shows how we defined q, q_dot, and necessary system parameters.
# Then we used position, velocity, and angular velocity to calculate kinetic energy.

# %%
# defining potential energy
# z, theta = symbols("z, theta")

P = (
    (m1 * g * (z * sin(theta))) + (m2 * g * ell / 2 * sin(theta))
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
EL_case_studyE = simplify(diff(diff(L, qdot), t) - diff(L, q))

display(Math(vlatex(EL_case_studyE)))


# %%
############################################################
### Including friction and generalized forces, then solving for highest order derivatives
############################################################

# these are just convenience variables
zd = z.diff(t)
zdd = zd.diff(t)
thetad = theta.diff(t)
thetadd = thetad.diff(t)

# defining symbols for external force and friction
b, F = symbols("b, F")
tau = sp.Matrix([[0], [F*ell*cos(theta)]])
friction = sp.Matrix([[0], [0]])
# defining the right-hand side of the equation and combining it with E-L part
RHS = tau + friction
full_eom = EL_case_studyE - RHS

# finding and assigning zdd and thetadd
# if our eom were more complicated, we could rearrange, solve for the mass matrix, and invert it to move it to the other side and find qdd and thetadd
result = sp.solve(full_eom, (zdd, thetadd))

# TODO - add an example of finding the same thing, but not using sp.solve

# result is a Python dictionary, we get to the entries we are interested in
# by using the name of the variable that we were solving for
zdd_eom = result[zdd]
thetadd_eom = result[thetadd]  # EOM for thetadd, as a function of states and inputs

display(Math(vlatex(zdd_eom)))
display(Math(vlatex(thetadd_eom)))

#%%
# E4

f_sym = sp.Matrix([zd, thetad, zdd_eom, thetadd_eom])
state_sym = sp.Matrix([z, theta, zd, thetad])
input_sym = sp.Matrix([F])

equilibria = equilibria = sp.solve([zdd_eom.subs({zd:0, thetad:0}), 
                       thetadd_eom.subs({zd:0, thetad:0})], (z, theta, F))

display(Math(vlatex(equilibria)))


A_sym = f_sym.jacobian(state_sym)
B_sym = f_sym.jacobian(input_sym)

display(Math(vlatex(A_sym)))
display(Math(vlatex(B_sym)))

v = sp.symbols('v')
desired_dynamics = sp.Eq(thetadd_eom, v)

fl_control_law = sp.solve(desired_dynamics, F)[0]

fl_control_law = simplify(fl_control_law)

display(Math(r"F_{fl} = " + vlatex(fl_control_law)))

C_sym = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

D_sym = sp.Matrix([
    [0],
    [0]
])

op_point = {z: 0, theta: 0, zd: 0, thetad: 0, F: 0}

A_lin = A_sym.subs(op_point)
B_lin = B_sym.subs(op_point)

params_subs = {
    m1: PAR.m1,
    m2: PAR.m2, 
    ell: PAR.ell,
    g: PAR.g,
    b: b
}

A_num = A_lin.subs(params_subs)
B_num = B_lin.subs(params_subs)

C_num = C_sym
D_num = D_sym

display(Math("A_{num} = " + vlatex(A_num)))
display(Math("B_{num} = " + vlatex(B_num)))
display(Math("C_{num} = " + vlatex(C_num)))
display(Math("D_{num} = " + vlatex(D_num)))

# %% [markdown]
# OK, now we can get the state variable form of the equations of motion.

# %%

import numpy as np

# defining fixed parameters that are not states or inputs (like g, ell, m, b)
# can be done like follows:
# params = [(m, P.m), (ell, P.ell), (g, P.g), (b, P.b)]

# but in this example, I want to keep the masses, length, and damping as variables so
# that I can simulate uncertainty in those parameters in real life.
params = [(g, PAR.g)]


# substituting parameters into the equations of motion
zdd_eom = zdd_eom.subs(params)
thetadd_eom = thetadd_eom.subs(params)

# now defining the state variables that will be passed into f(x,u)
# state = np.array([theta, thetad])
# ctrl_input = np.array([tau])

state = sp.Matrix([z, theta, zd, thetad])
ctrl_input = sp.Matrix([F])

# defining the function that will be called to get the derivatives of the states
state_dot = sp.Matrix([zd, thetad, zdd_eom, thetadd_eom])


# %%
import numpy as np

# converting the function to a callable function that uses numpy to evaluate and
# return a list of state derivatives
eom = sp.lambdify([state, ctrl_input, m1, m2, ell], state_dot, "numpy")

# calling the function as a test to see if it works:
cur_state = np.array([0, 0, 0, 0])
cur_input = np.array([1])
print("x_dot = ", eom(cur_state, cur_input, PAR.m1, PAR.m2, PAR.ell))


# %% [markdown]
# The next step is to save this function "f" so that we can use it with a numerical integrator, like
# scipy.integrate.ivp.solve_ivp or the rk4 functions in the case studies. To save this function, we can use the following:

# %%
# this code will only run if this file is executed directly,
# not if it is imported as a module.
if __name__ == "__main__":
    from case_studies import E_blockbeam

    # make sure printing only happens when running this file directly
    su.enable_printing(__name__ == "__main__")

    su.write_eom_to_file(state, ctrl_input, [m1, m2, ell], E_blockbeam, eom=state_dot)

    import numpy as np
    from case_studies.E_blockbeam import eom_generated
    import importlib

    importlib.reload(eom_generated)  # reload in case it was just generated/modified
    P = E_blockbeam.params

    param_vals = {
        "m1": PAR.m1,
        "m2": PAR.m2,
        "ell": PAR.ell
    }

    x_test = np.array([0.25, 0.0, 0.0, 0.0])
    u_test = np.array([1.0])

    x_dot_test = eom_generated.calculate_eom(x_test, u_test, **param_vals)
    print("\nx_dot_test from generated function = ", x_dot_test)
    # should match what was printed earlier when we called eom directly

# %%
