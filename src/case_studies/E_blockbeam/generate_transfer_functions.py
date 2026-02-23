#%%
import sympy as sp
from case_studies.E_blockbeam.generate_linearization import *
from sympy import eye, zeros, simplify

C = sp.Matrix([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
D = sp.Matrix([[0], [0]]) 

s = sp.symbols("s")

#%%
transfer_func = simplify(C @ (s * eye(4) - A_lin).inv() @ B_lin + D)
print("Transfer functions (without simplifying assumption):")
display(Math(vlatex(transfer_func)))

#%%
A_E5 = A_lin.subs([(m1 * g, 0)])
B_E5 = B_lin.subs([(m1 * g, 0)])

transfer_func_partC = simplify(C * (s * eye(4) - A_E5).inv() * B_E5 + D)
print("\nTransfer functions (with simplifying assumption):")
display(Math(vlatex(transfer_func_partC)))
# %%
