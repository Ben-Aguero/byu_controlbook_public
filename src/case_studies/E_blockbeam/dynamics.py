# 3rd-party
import numpy as np

# local (controlbook)
from case_studies.E_blockbeam import eom_generated
from . import params as P
from ..common.dynamics_base import DynamicsBase


class BlockbeamDynamics(DynamicsBase):
    def __init__(self, alpha=0.0):
        super().__init__(
            # Initial state conditions
            state0=np.array([P.z0, P.zdot0, P.theta0, P.thetadot0]),
            u_max=P.force_max,
            u_min=-P.force_max,
            # Time step for integration
            dt=P.ts,
        )
        # see params.py/textbook for details on these parameters
        self.m1 = self.randomize_parameter(P.m1, alpha)
        self.m2 = self.randomize_parameter(P.m2, alpha)
        self.length = self.randomize_parameter(P.length, alpha)
        self.g = P.g  # gravity constant is well known, so not randomized

    def f(self, x, u):
       xdot = eom_generated.calculate_eom(
           x, u, m1=self.m1, m2=self.m2, ell=self.length)
       return xdot

    def h(self):
        # return the output equations
        # could also use input u if needed
        y = self.state[:2]
        return y
