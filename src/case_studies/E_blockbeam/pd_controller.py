# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common
import case_studies.E_blockbeam.params as P


class BlockbeamControllerPD(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = True):
        # tuning parameters
        # tr_theta = 0.25  # from part (b)
        zeta_theta = 0.95
        M = 10  # time separation factor between inner and outer loop
        
        # Toggle tr_z for part (d) vs part (f)
        # tr_z = tr_theta * M  # part (d) standard
        tr_z = 0.98           # part (f) tuned to saturate for a 0.25m step
        tr_theta = tr_z / M
        
        zeta_z = 0.95

        # system parameters
        m1 = P.m1
        m2 = P.m2
        ell = P.length
        g = P.g

        # plant gains (derived from E.8 transfer functions)
        # P_inner(s) = b_in / s^2
        # P_outer(s) = b_out / s^2
        b_in = 6.0 / (ell * (1.5 * m1 + 2.0 * m2))
        b_out = -g

        # Inner loop (theta)
        wn_theta = 2.2 / tr_theta
        self.kp_theta = wn_theta**2 / b_in
        self.kd_theta = (2 * zeta_theta * wn_theta) / b_in
        print(f"Inner loop (theta): {self.kp_theta = :.3f}, {self.kd_theta = :.3f}")

        # DC gain of inner loop
        # Since we use a PD controller on a 1/s^2 plant, DC gain is exactly 1
        DC_gain = 1.0
        print(f"{DC_gain = :.3f}")

        # Outer loop (z)
        wn_z = 2.2 / tr_z
        self.kp_z = wn_z**2 / b_out
        self.kd_z = (2 * zeta_z * wn_z) / b_out
        print(f"Outer loop (z): {self.kp_z = :.3f}, {self.kd_z = :.3f}")

        self.F_max = 15.0  # From part (f)
        self.use_feedback_linearization = use_feedback_linearization

    def update_with_state(self, r, x):
        z_ref = r[0]
        z, theta, zdot, thetadot = x

        # outer loop control
        error_z = z_ref - z
        theta_ref = self.kp_z * error_z - self.kd_z * zdot
        
        # Optional: store theta_ref if your visualizer expects it in the reference array
        if len(r) > 1:
            r[1] = theta_ref  

        # inner loop control
        error_theta = theta_ref - theta
        F_tilde = self.kp_theta * error_theta - self.kd_theta * thetadot

        # feedback linearization (equilibrium force)
        if self.use_feedback_linearization:
            # Balances the block's mass dynamically at position z, plus the beam's mass
            F_fl = (P.m1 * P.g * z) / P.length + (P.m2 * P.g) / 2.0
            F_unsat = F_tilde + F_fl
        else: 
            # use nominal equilibrium force
            F_unsat = F_tilde + P.u_eq

        # saturation
        u_unsat = np.array([F_unsat])
        u = self.saturate(u_unsat, u_max=np.array([self.F_max]))
        
        return u