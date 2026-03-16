# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, F_vtol


mass = F_vtol.Dynamics()
controller = F_vtol.ControllerSS()
z_ref = common.SignalGenerator(amplitude=np.radians(50), frequency=0.05)

time, x_hist, u_hist, r_hist, xhat_hist, *_ = common.run_simulation(
    mass,
    [z_ref],
    controller,
    controller_input="state",
    t_final=20,
    dt=F_vtol.params.ts,
)

viz = F_vtol.Visualizer(time, x_hist, u_hist, r_hist, xhat_hist)
viz.plot()
viz.animate()
