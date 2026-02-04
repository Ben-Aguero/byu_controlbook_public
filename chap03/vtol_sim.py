# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, F_vtol

P = F_vtol.params
# initialize system and input generator
vtol = F_vtol.dynamics()
force_gen = common.SignalGenerator(amplitude=.5, frequency=1.0, y_offset=14.715)
torque_gen = common.SignalGenerator(amplitude=0.001, frequency=1.0, y_offset=.01)

# initialize data storage
x_hist = [vtol.state]
u_hist = []

# loop over time
time = np.arange(start=0.0, stop=15.0, step=P.ts, dtype=np.float64)
for t in time[1:]:
    u = P.mixer @ np.array([force_gen.sin(t), torque_gen.sin(t)])

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
