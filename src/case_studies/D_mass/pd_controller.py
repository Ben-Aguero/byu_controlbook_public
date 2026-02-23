# 3rd-party
import numpy as np

# local (controlbook)
from case_studies import common
import case_studies.D_mass.params as P


class MassControllerPD(common.ControllerBase):
    def __init__(self, use_feedback_linearization: bool = True):
        # tuning parameters
        p1 = -1
        p2 = -1.5

        # For part D.8a-----------------------
        # tr = 2.0
        # zeta = .707
        # wn = 2.2/tr
        # alpha1 = wn**2
        # alpha0 = 2*zeta*wn
        #--------------------------------------

        # For part D.8b-----------------------
        # F_max = P.force_max
        # m = P.m
        # k = P.k
        # zeta = 0.707
        # wn_saturated = np.sqrt((F_max + k) / m)
        # wn = wn_saturated
        # alpha1 = wn**2
        # alpha0 = 2 * zeta * wn
        #-------------------------------------

        # system parameters
        b0 = P.tf_num[-1]
        a1, a0 = P.tf_den[-2:]

        # desired characteristic equation parameters
        # des_CE: s^2 + alpha1*s + alpha0 = s^2 + (a1 + b0*kd)s + (a0 + b0*kp)
        # FOR D.7 ----------------------------------------------
        des_CE = np.poly([p1, p2])
        alpha1, alpha0 = des_CE[-2:]
        #-----------------------------------------------------

        # find gains
        self.kp = (alpha0 - a0) / b0
        self.kd = (alpha1 - a1) / b0
        # self.kp = 3.05
        # self.kd = 7.275
        print(f"{self.kp = :.2f}, {self.kd = :.3f}")

        # For part D.8b ------------------------------------------
        # self.limit = F_max
        # -------------------------------------------------------

        self.F_eq = P.u_eq
        self.use_feedback_linearization = use_feedback_linearization

    def update_with_state(self, r, x):
        # unpack references and states
        z_ref = r[0]
        z, zdot = x

        # theta (modified) PD
        error = z_ref - z
        F_tilde = self.kp * error - self.kd * zdot

        if self.use_feedback_linearization:
            F_fl = 0
            F = F_tilde + F_fl
        else:  # use equilibrium (Jacobian linearization)
            # NOTE: likely not work well (or at all) for large angles
            F = F_tilde + self.F_eq

        # --- SATURATION (Required for Part B) ---
        # u_unsat = np.array([F])
        # u = self.saturate(u_unsat, self.limit)

        u = u = np.array([F])
        return u
