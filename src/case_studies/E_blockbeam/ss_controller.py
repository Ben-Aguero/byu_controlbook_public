# 3rd-party
import numpy as np
import control as cnt

# local (controlbook)
from . import params as P
from ..common import ControllerBase


class BlockbeamSSController(ControllerBase):
    def __init__(self):
        # tuning parameters (replace with actual tuned values)
        tr_z = 2.0     
        zeta_z = 0.707
        tr_theta = 0.5 
        zeta_theta = 0.707

        # check controllability
        if np.linalg.matrix_rank(cnt.ctrb(P.A, P.B)) != 4:
            raise ValueError("System not controllable")

        # compute gains
        wn_z = 2.2 / tr_z
        des_poles_z = np.roots([1, 2 * zeta_z * wn_z, wn_z**2])
        wn_theta = 2.2 / tr_theta
        des_poles_theta = np.roots([1, 2 * zeta_theta * wn_theta, wn_theta**2])
        des_poles = np.concatenate((des_poles_z, des_poles_theta))
        
        self.K = cnt.place(P.A, P.B, des_poles)
        self.kr = -1.0 / (P.Cr @ np.linalg.inv(P.A - P.B @ self.K) @ P.B)
        print("des_poles:", des_poles)
        print("K:", self.K)
        print("kr:", self.kr)

        # linearization point
        self.x_eq = P.x_eq
        self.r_eq = P.Cr @ self.x_eq

        # dirty derivative variables
        sigma = 0.05
        self.beta = (2 * sigma - P.ts) / (2 * sigma + P.ts)
        self.zdot_hat = P.zdot0
        self.z_prev = P.z0
        self.thetadot_hat = P.thetadot0
        self.theta_prev = P.theta0

    def update_with_state(self, r, x):
        # convert to linearization (tilde) variables
        x_tilde = x - self.x_eq
        r_tilde = r - self.r_eq

        # compute state feedback control
        u_tilde = -self.K @ x_tilde + self.kr @ r_tilde

        # convert back to original variables (feedback linearization setup)
        z = x[0]
        u_fl = P.m1 * P.g * (z / P.length) + P.m2 * P.g / 2.0  # approximate equilibrium force 
        u_unsat = u_tilde + u_fl
        u = self.saturate(u_unsat, u_max=P.force_max)
        return u