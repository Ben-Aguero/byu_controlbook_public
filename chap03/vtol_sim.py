# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, F_vtol


# initialize system and input generator
vtol = F_vtol.dynamics()
force_gen = common.SignalGenerator(amplitude=2.0, frequency=0.2)
torque_gen = common.SignalGenerator(amplitude=0.1, frequency=0.1)

# initialize data storage
x_hist = [vtol.state]
u_hist = []

# loop over time
time = np.arange(start=0.0, stop=50.0, step=F_vtol.params.ts, dtype=np.float64)
for t in time[1:]:
    u_F = force_gen.sin(t)
    u_tau = torque_gen.sin(t)
    # generate input signal
    fl = 0.5 * (u_F - u_tau / F_vtol.params.d)
    fr = 0.5 * (u_F + u_tau / F_vtol.params.d)
    
    u = np.array([fl, fr])

    # simulate system response
    y = vtol.update(u)

    # store data for visualization
    u_hist.append(u)
    x_hist.append(vtol.state)

# convert data to numpy arrays
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

# visualize
viz = F_vtol.Visualizer(time, x_hist, u_hist)
viz.animate()  # could also just call viz.plot()
