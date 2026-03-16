# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, E_blockbeam


mass = E_blockbeam.Dynamics()
controller = E_blockbeam.ControllerSS()
z_ref = common.SignalGenerator(amplitude=np.radians(30), frequency=0.01)
theta_ref = common.SignalGenerator(amplitude=np.radians(30), frequency=.01)

time, x_hist, u_hist, r_hist, xhat_hist, *_ = common.run_simulation(
    mass,
    [z_ref],
    controller,
    controller_input="state",
    t_final=20,
    dt=E_blockbeam.params.ts,
)

viz = E_blockbeam.Visualizer(time, x_hist, u_hist, r_hist, xhat_hist)
viz.plot()
viz.animate()
