import numpy as np
import case_studies.H_hummingbird.params as P

class HummingbirdControllerLonPD:
    def __init__(self):
        # Known system parameters from params.py
        self.m1 = P.m1
        self.ell1 = P.ell1
        self.m2 = P.m2
        self.ell2 = P.ell2
        self.J1y = P.J1y
        self.J2y = P.J2y
        self.ellT = P.ellT
        self.d = P.d
        self.g = P.g
        
        # Identified motor constant (Estimated from GUI defaults. Tune this on hardware!)
        self.km = 0.354 
        
        # Calculate b_theta dynamically based on params to prevent hardcoding errors
        self.b_theta = self.ellT / (
            self.m1 * self.ell1**2 + 
            self.m2 * self.ell2**2 + 
            self.J1y + self.J2y
        )
        
        # Design requirements for rise time and damping ratio
        self.tr = 1.0
        self.zeta = 0.707
        self.wn = 2.2 / self.tr
        
        # Calculate PD Gains using pole placement matching
        self.kp = (self.wn**2) / self.b_theta
        self.kd = (2.0 * self.zeta * self.wn) / self.b_theta

    def update(self, theta_r, x):
        """
        Calculates the PWM commands for the left and right motors.
        
        Args:
            theta_r (float): Reference pitch angle (radians)
            x (numpy array): System state vector 
                             [phi, theta, psi, phidot, thetadot, psidot]
        Returns:
            numpy array: PWM commands [u_l, u_r] bounded between [0, 1]
        """
        # Extract relevant states for longitudinal control
        theta = x[1]
        thetadot = x[4]
        
        # 1. Feedback linearized force (cancels gravity)
        F_fl = (self.m1 * self.ell1 + self.m2 * self.ell2) * (self.g / self.ellT) * np.cos(theta)
        
        # 2. PD control force for error correction
        F_tilde = self.kp * (theta_r - theta) - self.kd * thetadot
        
        # 3. Total longitudinal force
        F = F_fl + F_tilde
        
        # 4. Constrain torque to zero for longitudinal-only control (Lab H.7 requirement)
        tau = 0.0
        
        # 5. Convert F and tau to individual motor forces
        f_l = 0.5 * (F + tau / self.d)
        f_r = 0.5 * (F - tau / self.d)
        
        # 6. Convert forces to PWM commands
        u_l = f_l / self.km
        u_r = f_r / self.km
        
        # 7. Saturate outputs to strictly be between 0.0 and 1.0 (0% to 100% duty cycle)
        u_l = np.clip(u_l, 0.0, 1.0)
        u_r = np.clip(u_r, 0.0, 1.0)
        
        return np.array([u_l, u_r])