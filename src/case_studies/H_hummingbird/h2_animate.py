# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, H_hummingbird

P=H_hummingbird.params

# initialize signals for generating data
tau_gen = common.SignalGenerator(amplitude=.05, frequency=0.5)
F_gen = common.SignalGenerator(amplitude=5, frequency=.5)
phi_gen = common.SignalGenerator(amplitude=np.pi, frequency=0.1)
theta_gen = common.SignalGenerator(amplitude=np.pi, frequency=0.1)
psi_gen = common.SignalGenerator(amplitude=np.pi, frequency=0.1)

# initialize data storage
x0 = np.zeros(6)
x_hist = [x0]
u_hist = []

# loop over time
time = np.arange(start=0, stop=20, step=.01, dtype=np.float64)
for t in time[1:]:
    # generate fake state and input data
    x = np.zeros_like(x0)
    x[0] = phi_gen.sin(t)
    x[1] = theta_gen.sin(t)  # velocity info can be anything for animation
    x[2] = psi_gen.sin(t)
    x[3:] = [
        theta_gen.sawtooth(t),
        phi_gen.sawtooth(t),
        psi_gen.sawtooth(t),
    ]

    # inputs can be anything for animation
    u = np.array([F_gen.square(t), tau_gen.square(t)]) # Shape (2,)

    # store data for visualization
    x_hist.append(x)
    u_hist.append(u)

# convert data to numpy arrays
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

# visualize generated data
viz = H_hummingbird.Visualizer(time, x_hist, u_hist)
viz.animate()  # may need to play with arguments to achieve desired animation speed
