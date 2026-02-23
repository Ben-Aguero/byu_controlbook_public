#%%

from sympy import symbols, simplify, cancel, factor, collect, together, fraction 
from sympy.physics.vector import dynamicsymbols 
from case_studies.common import sym_utils as su 
################################################################################### 
# Case Study E: Block-Beam - Linearization 
# ################################################################################### 
# defining mathematical variables (called symbols in sympy) and time varying functi 
t, m1, m2, ell, g, F, ze = symbols("t, m1, m2, ell, g, F, z_e") 
z = dynamicsymbols("z") 
theta = dynamicsymbols("theta")

from case_studies.E_blockbeam.generate_state_variable_form import *
print("State variable form of the equations of motion (f(x,u)):\n")
su.printeq("\ndot{x}", state_dot)

A = state_dot.jacobian(state)
B = state_dot.jacobian(ctrl_input)

def simplify_rational(expr):   
    """Simplify rational expressions by factoring numerator and denominator separat"""    
    expr = simplify(expr)    
    expr = together(expr)    
    numer, denom = fraction(expr)    
    numer = factor(numer)    
    denom = factor(denom)    
    return numer / denom

A_lin = simplify(A.subs(
    [
        (theta.diff(t), 0),
        (theta, 0),
        (z.diff(t), 0),
        (F, m1 * g / ell * ze + m2 * g / 2),
        (z, ze),
    ]
))

B_lin = simplify(B.subs(
    [
        (theta.diff(t), 0),
        (theta, 0),
        (z.diff(t), 0),
        (F, m1 * g / ell * ze + m2 * g / 2),
        (z, ze),
    ]
))

print("\nLinearized A Matrix is:") 
su.printeq("A_lin", A_lin) 
print("\nLinearized B Matrix is:") 
su.printeq("B_lin", B_lin)
# %%
