# 3rd-party
import numpy as np
import matplotlib.pyplot as plt

# local (controlbook)
from case_studies import common, H_hummingbird

# Load parameters and initialize dynamics
P = H_hummingbird.params
vtol = H_hummingbird.Dynamics()

# Calculate Equilibrium Force (F_e)
# Derived in Lab H.4 Part 2: F_e = (m1*l1 + m2*l2)*g / lT
# This is the force required to balance gravity at theta=0.
F_e = (P.m1 * P.ell1 + P.m2 * P.ell2) * P.g / P.ellT
print(f"Equilibrium Force (F_e): {F_e:.4f} N")

# Initialize Signal Generators
# Force Input: Center it at Equilibrium (F_e) + small square wave doublet
# This tests the "Longitudinal" (Pitch) dynamics
force_gen = common.SignalGenerator(amplitude=0.5, frequency=0.5, y_offset=F_e)

# Torque Input: Center at 0 + small square wave doublet
# This tests the "Lateral" (Roll/Yaw) dynamics
torque_gen = common.SignalGenerator(amplitude=0.5, frequency=0.5, y_offset=0.0)
vtol.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Simulation Loop
x_hist = [vtol.state]
u_hist = []
time = np.arange(start=0.0, stop=10.0, step=P.ts, dtype=np.float64)

for t in time[1:]:
    # Get High-Level Inputs (F, tau)
    F_val = F_e  # Hold force steady
    tau_val = 0.0

    # The Mixer: Convert (F, tau) -> (f_l, f_r)
    # Derived in H.4 Part 1:
    u_l = (1.0 / (2.0 * vtol.km)) * (F_val + (tau_val / P.d))
    u_r = (1.0 / (2.0 * vtol.km)) * (F_val - (tau_val / P.d))

    u_pwm = np.array([u_l, u_r])
    
    u_pwm = np.clip(u_pwm, 0.0, 1.0)

    # Update Dynamics
    y = vtol.update(u_pwm)

    # Store Data
    u_hist.append(u_pwm)
    x_hist.append(vtol.state)

# Visualization
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

viz = H_hummingbird.Visualizer(time, x_hist, u_hist)
viz.animate()