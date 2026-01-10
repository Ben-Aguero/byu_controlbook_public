# 3rd-party
import numpy as np

# local (controlbook)
from . import params as P
from ..common import ControllerBase
from ..control import utils_design
from ..control.lqr_integral_control import LQRIntegralController
from ..control.disturbance_observer import DisturbanceObserver


class BlockbeamSSIDOController(ControllerBase):
    def __init__(self):
        # controller design parameters
        Q = np.diag([1.0, 1.0, 1.0, 1.0, 10.0])
        R = np.array([[0.01]])

        # create controller
        self.ssi_ctrl = LQRIntegralController(P.A, P.B, P.Cr, Q, R, P.ts)

        # observer design parameters (same as HW 14)
        tr_theta = 0.25
        zeta_theta = 0.707
        tr_z = 1.2
        zeta_z = 0.707
        tr_z_obs = tr_z / 10
        tr_theta_obs = tr_theta / 10
        disturbance_poles = [-1.0]

        # create observer
        obs_z_poles = utils_design.get_2nd_order_poles(tr_z_obs, zeta_z)
        obs_theta_poles = utils_design.get_2nd_order_poles(tr_theta_obs, zeta_theta)
        obs_poles = np.hstack([obs_theta_poles, obs_z_poles, disturbance_poles])
        self.d_observer = DisturbanceObserver(P.A, P.B, P.Cm, obs_poles, P.ts)

        # save linearized system information
        self.x_eq = P.x_eq
        self.r_eq = P.Cr @ self.x_eq
        self.y_eq = P.Cm @ self.x_eq
        self.u_eq = P.u_eq

        self.u_tilde_prev = np.zeros(1)

    def update_with_state(self, r, x):
        x_tilde = x - self.x_eq
        r_tilde = r - self.r_eq
        u_tilde = self.ssi_ctrl.update(r_tilde, x_tilde)
        u_unsat = u_tilde + self.u_eq
        u = self.saturate(u_unsat, u_max=P.force_max)
        return u

    def update_with_measurement(self, r, y):
        # update the observer with the measurement
        y_tilde = y - self.y_eq
        x2hat_tilde = self.d_observer.update(y_tilde, self.u_tilde_prev)

        # unpack the state and disturbance estimates
        dim_state = len(self.x_eq)
        xhat = x2hat_tilde[:dim_state] + self.x_eq
        dhat = x2hat_tilde[dim_state:]  # d_eq is zero

        # compute control input
        u = self.update_with_state(r, xhat) - dhat

        # save the previous input for the observer
        self.u_tilde_prev = u - self.u_eq

        return u, xhat, dhat
