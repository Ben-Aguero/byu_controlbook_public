# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, H_hummingbird

P = H_hummingbird.params
# initialize system and input generator
hummingbird = H_hummingbird.Dynamics()

# Calculate Equilibrium Force for the signal generator offset
# F_eq = (m1*l1 + m2*l2) * g / lT
F_eq = (P.m1 * P.ell1 + P.m2 * P.ell2) * P.g / P.ellT

# Force centered at Equilibrium, Torque centered at 0
force_gen = common.SignalGenerator(amplitude=0.0, frequency=1.0, y_offset=F_eq)
torque_gen = common.SignalGenerator(amplitude=0.00, frequency=1.0, y_offset=0.0)

# initialize data storage
x_hist = [hummingbird.state]
u_hist = []

# loop over time
time = np.arange(start=0.0, stop=15.0, step=P.ts, dtype=np.float64)
for t in time[1:]:
    # Mix forces: [F, tau] -> [f_l, f_r]
    u_force = P.mixer @ np.array([force_gen.sin(t), torque_gen.sin(t)])

    # Convert motor forces to PWM for the dynamics update
    u_pwm = u_force / hummingbird.km

    # simulate system response
    y = hummingbird.update(u_pwm)

    # store data for visualization
    u_hist.append(u_pwm)
    x_hist.append(hummingbird.state)

# convert data to numpy arrays
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

# visualize
# Note: Ensure hummingbirdAnimation.py is accessible as H_hummingbird.Visualizer 
# or import it directly if your package structure differs.
viz = H_hummingbird.Visualizer(time, x_hist, u_hist)
viz.animate()  # could also just call viz.plot()