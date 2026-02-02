# 3rd-party
import numpy as np
from case_studies.D_mass import eom_generated

# local (controlbook)
from . import params as P
from ..common.dynamics_base import DynamicsBase


class MassDynamics(DynamicsBase):
    def __init__(self, alpha=0.0):
        super().__init__(
            # Initial state conditions
            state0=np.array([P.z0, P.zdot0]),
            u_max=P.z_max,
            u_min=-P.z_max,
            # Time step for integration
            dt=P.ts,
        )
        # see params.py/textbook for details on these parameters
        self.m = P.m
        # self.z = self.randomize_parameter(P.z, alpha)
        self.b = P.b
        self.k = P.k  # gravity constant is well known, so not randomized

    def f(self, x, u):
       return eom_generated.calculate_eom(x, u, self.m, self.k, self.b)

    def h(self):
        # return the output equations
        # could also use input u if needed
        z = self.state[0]
        y = np.array([z])
        return y
