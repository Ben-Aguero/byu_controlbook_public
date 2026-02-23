# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, E_blockbeam


mass = E_blockbeam.Dynamics()
controller = E_blockbeam.ControllerPD(use_feedback_linearization=True)
z_ref = common.SignalGenerator(amplitude=np.radians(30), frequency=0.01)
theta_ref = common.SignalGenerator(amplitude=np.radians(30), frequency=.01)

time, x_hist, u_hist, r_hist, *_ = common.run_simulation(
    mass,
    [z_ref],
    controller,
    controller_input="state",
    t_final=5,
    dt=E_blockbeam.params.ts,
)

viz = E_blockbeam.Visualizer(time, x_hist, u_hist, r_hist)
viz.animate()