# 3rd-party
import numpy as np
import matplotlib.pyplot as plt

# local (controlbook)
from case_studies import common, H_hummingbird

# Load parameters
P = H_hummingbird.params

# Initialize dynamics and controller
vtol = H_hummingbird.Dynamics()
ctrl = H_hummingbird.ControllerLonPD()

print(f"Calculated Proportional Gain (kp): {ctrl.kp:.4f}")
print(f"Calculated Derivative Gain (kd): {ctrl.kd:.4f}")

# Using a square wave (SignalGenerator defaults to square/sinusoid based on the textbook's implementation)
# 10 degrees = ~0.1745 radians. Frequency = 0.1 Hz (10-second period).
reference_gen = common.SignalGenerator(amplitude=np.deg2rad(15.0), frequency=0.1, y_offset=0.0)

# Ensure initial state is zeroed out
vtol.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Simulation Setup
time = np.arange(start=0.0, stop=20.0, step=P.ts, dtype=np.float64)
x_hist = [vtol.state]
u_hist = []
r_hist = []

# Get initial reference to match history lengths
theta_r_init = reference_gen.square(0.0)
r_init = np.zeros(6)
r_init[1] = theta_r_init
r_hist.append(r_init)

# Simulation Loop
for t in time[1:]:
    # 1. Get Reference Signal (desired pitch angle)
    theta_r = reference_gen.square(t) 
    
    # 2. Calculate Control Input (u_pwm)
    u_pwm = ctrl.update(theta_r, vtol.state)
    
    # 3. Update Dynamics
    vtol.update(u_pwm)
    
    # 4. Store Data
    u_hist.append(u_pwm)
    x_hist.append(vtol.state)
    
    # Store reference vector (placing theta_r in index 1 to match the state vector)
    r_state = np.zeros(6)
    r_state[1] = theta_r
    r_hist.append(r_state)

# Convert lists to numpy arrays for the visualizer
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)
r_hist = np.array(r_hist)

# Visualization
# Passing r_hist allows the visualizer to plot the reference command alongside the actual pitch
viz = H_hummingbird.Visualizer(time, x_hist, u_hist, r_hist=r_hist)
viz.animate()