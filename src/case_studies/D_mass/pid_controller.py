import numpy as np
from . import params as P
from ..control.pid import PID  # Import your new helper class

class ControllerPID:
    def __init__(self):
        # 1. Tuning Parameters
        tr = 2.0        # Rise time (adjust as needed for D.10)
        zeta = 0.707    # Damping ratio
        
        # Calculate natural frequency based on textbook approximation
        wn = (np.pi / 2.0) / (tr * np.sqrt(1 - zeta**2))
        
        # Select integrator pole location (typically slightly slower than wn)
        p0 = wn / 2.0   

        # 2. Calculate Gains via Pole Placement
        self.kp = P.m * (wn**2 + 2.0 * zeta * wn * p0) - P.k
        self.kd = P.m * (2.0 * zeta * wn + p0) - P.b
        self.ki = P.m * (wn**2 * p0)

        # 3. Instantiate the PID helper class with the calculated gains
        self.limit = P.force_max
        self.Ts = P.ts
        
        self.pid = PID(
            kp=self.kp, 
            ki=self.ki, 
            kd=self.kd, 
            limit=self.limit, 
            Ts=self.Ts
        )
        
        # 4. Dirty derivative variables for the measured output (z)
        self.sigma = 0.05
        self.beta = (2.0 * self.sigma - P.ts) / (2.0 * self.sigma + P.ts)
        self.z_dot = 0.0
        self.z_d1 = 0.0

    def update_with_measurement(self, r, y):
        """
        r: reference command [z_ref]
        y: measured output [z]
        """
        z_ref = r[0]
        z = y[0]

        # 1. Calculate dirty derivative of the measured position
        self.z_dot = self.beta * self.z_dot + (1.0 - self.beta) * ((z - self.z_d1) / P.ts)
        self.z_d1 = z

        # 2. Calculate the control force using the PREFERRED modified PID 
        force = self.pid.update_modified(y_ref=z_ref, y=z, ydot=self.z_dot)

        # 3. Create the estimated state vector (x_hat) required by the simulator
        x_hat = np.array([z, self.z_dot])

        # Return BOTH the control effort and the estimated states
        return np.array([force]), x_hat