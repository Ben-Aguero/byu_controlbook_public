import numpy as np

# Physical parameters
m = 5.0 # mass (kg)
k = 3.0 # spring constant (kg/s^2)
b = 0.5 # damping coefficient (kg/s)

# Initial conditions
z0 = 0.0 # initial position (m)
zdot0 = 0.0 # initial velocity (m/s)

t0 = 0.0 # start time
tf = 50.0 # end time
ts = 0.01 # integration time step

# Linearization/equilibrium point
x_eq = np.zeros(2)
u_eq = np.zeros(1)

# Transfer function numerator and denominator
tf_num = [1 / m]
tf_den = [1, b / m, k / m]

# State space
A = np.array([[0, 1], [-k / m, -b / m]])
B = np.array([[0], [1 / m]])
Cm = np.eye(1, 2) # measure z
Cr = Cm
D = np.zeros((1, 1))

force_max = 6.0 # max force (N)