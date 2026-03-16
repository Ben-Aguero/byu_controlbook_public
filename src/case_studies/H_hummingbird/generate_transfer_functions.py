# %%
################################################################################
# This file is meant to be run interactively with VSCode's Jupyter extension.
# It works as a regular Python script as well, but the printed results will not
# display as nicely.
################################################################################

# %% [markdown]
# # Labs H.5 and H.6: Transfer Functions and State Space Models
# ### Load H.4 (Linearization)
# %%
from case_studies.H_hummingbird.generate_linearization import *
import sympy as sp

# This makes it so printing from su only happens when running this file directly
su.enable_printing(__name__ == "__main__")

# %% [markdown]
# # Lab H.5: Transfer Function Models
# We will take the Laplace transform of the linearized equations to find 
# the transfer functions for the longitudinal and lateral dynamics.
# Because the equations are linear, we can extract the coefficients directly.
# %%
s = sp.symbols("s")

# %% [markdown]
# #### Longitudinal Transfer Function
# Find $\frac{\tilde{\theta}(s)}{\tilde{F}(s)}$
# %%
# Thetaddot = b_theta * F_ctrl. 
# We can extract the constant b_theta by differentiating with respect to F_ctrl.
b_theta = thetaddot_eom.diff(F_ctrl)

# Apply Laplace transform (dividing by s^2 for double integrator)
G_theta_F = b_theta / s**2
su.printeq("H_theta_F", G_theta_F)

# %% [markdown]
# #### Lateral Transfer Functions
# Find $\frac{\tilde{\phi}(s)}{\tilde{\tau}(s)}$ and $\frac{\tilde{\psi}(s)}{\tilde{\phi}(s)}$
# %%
# Extract the roll coefficient with respect to torque (tau / T)
b_phi = phiddot_expr.diff(T)
G_phi_tau = b_phi / s**2
su.printeq("H_phi_tau", G_phi_tau)

# Extract the yaw coefficient with respect to roll (phi)
# Extract the yaw coefficient, then substitute the equilibrium values!
b_psi = psiddot_expr.diff(phi).subs(lat_eq_vals)
G_psi_phi = sp.simplify(b_psi) / s**2
su.printeq("H_psi_phi", G_psi_phi)


# %% [markdown]
# # Lab H.6: State Space Models
# Now we organize the equations into state-space form: 
# $\dot{x} = A x + B u$ and $y = C x + D u$

# %% [markdown]
# #### Longitudinal State Space
# %%
# State: x_lon = [θ, θ̇]ᵀ, Input: u_lon = [F_ctrl]ᵀ
# Output: y_lon = [θ]ᵀ
x_lon = sp.Matrix([theta, qdot[1]])
u_lon = sp.Matrix([F_ctrl])

# Define the state derivative vector f_lon = [θ̇, θ̈]ᵀ
f_lon = sp.Matrix([
    qdot[1],
    thetaddot_eom
])

# Compute Jacobians to get A_lon and B_lon
A_lon = f_lon.jacobian(x_lon)
B_lon = f_lon.jacobian(u_lon)

# Define Output matrices (we measure pitch angle θ)
C_lon = sp.Matrix([[1, 0]])
D_lon = sp.Matrix([[0]])

su.printeq("A_lon", A_lon)
su.printeq("B_lon", B_lon)
su.printeq("C_lon", C_lon)
su.printeq("D_lon", D_lon)

# %% [markdown]
# #### Lateral State Space
# %%
# State: x_lat = [φ, ψ, φ̇, ψ̇]ᵀ, Input: u_lat = [τ]ᵀ
# Output: y_lat = [φ, ψ]ᵀ (we measure roll and yaw angles)
# Note: A_lat and B_lat were already calculated in generate_linearization.py!

C_lat = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

D_lat = sp.Matrix([
    [0],
    [0]
])

su.printeq("A_lat", A_lat)
su.printeq("B_lat", B_lat)
su.printeq("C_lat", C_lat)
su.printeq("D_lat", D_lat)

# %%