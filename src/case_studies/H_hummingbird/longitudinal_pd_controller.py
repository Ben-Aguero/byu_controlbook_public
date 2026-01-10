# 3rd-party
import numpy as np

# local (controlbook)
from .. import common
from . import params as P
from ..control.pd import PD
from ..control import utils_design


class HummingbirdControllerLonPD(common.ControllerBase):
    def __init__(self):
        # tuning parameters
        tr_theta = 1.0  # starting point
        tr_theta = 0.4  # tuned
        zeta_theta = 0.707
        # zeta_theta = 0.9  # tried higher damping

        # system parameters
        b0 = P.tf_lon_num[-1]
        a1, a0 = P.tf_lon_den[-2:]

        # desired characteristic equation parameters
        alpha1, alpha0 = utils_design.get_des_CE(tr_theta, zeta_theta)

        # find gains
        kp = (alpha0 - a0) / b0
        kd = (alpha1 - a1) / b0
        print(f"{kp = :.2f}, {kd = :.2f}")

        self.pitch_pd = PD(kp, kd)

        # Dirty derivative filter for update_with_measurement
        from ..control.dirty_derivative_filter import DirtyDerivativeFilter

        self.thetadot_filter = DirtyDerivativeFilter(P.ts, sigma=0.05)

    def update_with_state(self, r, x):
        # unpack references and states
        theta_ref = r[1]
        theta = x[1]
        thetadot = x[4]

        # pitch PD
        F_ctrl = self.pitch_pd.update_modified(theta_ref, theta, thetadot)
        F_fl = P.g * (P.m1 * P.ell1 + P.m2 * P.ell2) * np.cos(theta) / P.ellT
        F = F_ctrl + F_fl

        # lateral control not used in this example
        tau = 0.0

        # convert force-torque to pwm
        u_FT = np.array([F, tau])
        u = P.mixer @ u_FT / P.km
        return u

    def update_with_measurement(self, r, y):
        phi, theta, psi = y
        theta_ref = r[1]

        # Calculate derivative using dirty derivative filter
        thetadot = self.thetadot_filter.update(theta)

        # form xhat from what is known
        xhat = np.array([phi, theta, psi, 0.0, thetadot, 0.0])
        u = self.update_with_state(r, xhat)

        return u, xhat.astype(np.float64)
