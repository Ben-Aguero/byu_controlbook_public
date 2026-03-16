# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common, D_mass

# 1. Add alpha=0.2 to turn on 20% parameter uncertainty
mass = D_mass.Dynamics(alpha=0.2)

# 2. Swap to your new PID Controller 
controller = D_mass.ControllerPID() 

z_ref = common.SignalGenerator(amplitude=2.0, frequency=0.05)

time, x_hist, u_hist, r_hist, *_ = common.run_simulation(
    mass,
    [z_ref],
    controller,
    # 3. Change this from "state" to "y" so the controller only gets the measured position
    controller_input="measurement", 
    t_final=50,
    dt=D_mass.params.ts,
)

viz = D_mass.Visualizer(time, x_hist, u_hist, r_hist)
viz.animate()