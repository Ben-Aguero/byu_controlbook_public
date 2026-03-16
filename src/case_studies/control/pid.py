class PID:
    """
    PID (Proportional-Integral-Derivative) controller helper class.
    
    NOTE: This is a HELPER class used inside full controller implementations.
    This class implements the core PID control law:
    - Regular PID: u = kp*error + ki*integral + kd*error_dot
    - Modified PID: u = kp*error + ki*integral - kd*ydot (more common, avoids derivative kick)
    
    The derivative of y (or e) must be calculated outside of this class and passed
    in to the update methods. The integration and anti-windup are handled internally.
    """
    def __init__(self, kp: float, ki: float, kd: float, limit: float, Ts: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # System parameters needed for integration and anti-windup
        self.limit = limit
        self.Ts = Ts
        
        # Integrator state variables
        self.integrator = 0.0
        self.error_d1 = 0.0

    def update(self, y_ref: float, y: float, error_dot: float):
        """
        Regular PID control: Returns u = kp*error + ki*integral + kd*error_dot
        Less common than modified PID (see update_modified).
        
        Args:
        y_ref (float): reference value for signal y.
        y (float): current value of signal y.
        error_dot (float): derivative of error.
        
        Returns:
        u_sat (float): saturated control output
        """
        error = y_ref - y
        
        # Integrate error using the trapezoidal rule
        integrator_next = self.integrator + (self.Ts / 2.0) * (error + self.error_d1)
        
        # Calculate unsaturated control effort
        u_unsat = self.kp * error + self.ki * integrator_next + self.kd * error_dot
        
        # Saturate control effort
        u_sat = self.saturate(u_unsat)
        
        # Integrator Anti-windup (Back-calculation method)
        if self.ki != 0.0:
            self.integrator = integrator_next + (self.Ts / self.ki) * (u_sat - u_unsat)
        else:
            self.integrator = 0.0
            
        self.error_d1 = error
        return u_sat
    
    def update_modified(self, y_ref: float, y: float, ydot: float):
        """
        Modified PID control: Returns u = kp*error + ki*integral - kd*ydot
        This is the PREFERRED form of PID control because it does not add an extra zero
        to the numerator of the closed-loop transfer function, avoiding "derivative kick" 
        when the reference changes instantly.
        
        Args:
        y_ref (float): reference value for signal y.
        y (float): current value of signal y.
        ydot (float): current derivative of signal y.
        
        Returns:
        u_sat (float): saturated control output
        """
        error = y_ref - y
        
        # Integrate error using the trapezoidal rule
        integrator_next = self.integrator + (self.Ts / 2.0) * (error + self.error_d1)
        
        # Calculate unsaturated control effort
        u_unsat = self.kp * error + self.ki * integrator_next - self.kd * ydot
        
        # Saturate control effort
        u_sat = self.saturate(u_unsat)
        
        # Integrator Anti-windup (Back-calculation method)
        if self.ki != 0.0:
            self.integrator = integrator_next + (self.Ts / self.ki) * (u_sat - u_unsat)
        else:
            self.integrator = 0.0
            
        self.error_d1 = error
        return u_sat

    def saturate(self, u: float):
        """
        Clamps the control effort to the physical limits of the actuator.
        """
        if u > self.limit:
            return self.limit
        elif u < -self.limit:
            return -self.limit
        else:
            return u