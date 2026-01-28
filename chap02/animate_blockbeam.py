# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, E_blockbeam


# initialize signals for generating data
z_gen = common.SignalGenerator(amplitude=np.pi, frequency=0.1)
theta_gen = common.SignalGenerator(amplitude=np.pi, frequency=0.1)
tau_gen = common.SignalGenerator(amplitude=5, frequency=0.5)

# initialize data storage
x0 = np.zeros(4)
x_hist = [x0]
u_hist = []

# loop over time
time = np.arange(start=0, stop=20, step=E_blockbeam.params.ts, dtype=np.float64)
for t in time[1:]:
    # generate fake state and input data
    x = np.empty(4)
    x[0] = z_gen.sin(t)
    x[1] = theta_gen.sin(t)  # velocity info can be anything for animation
    x[2] = 0
    x[3] = 0

    # inputs can be anything for animation
    u = np.array([tau_gen.sawtooth(t)])

    # store data for visualization
    x_hist.append(x)
    u_hist.append(u)

# convert data to numpy arrays
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

# visualize generated data
viz = E_blockbeam.Visualizer(time, x_hist, u_hist)
viz.animate()  # may need to play with arguments to achieve desired animation speed
