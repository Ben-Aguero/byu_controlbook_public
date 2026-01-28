import numpy as np
ts = .01

d = 1
mixer = np.array([[1.0, 1.0],
                 [d,   -d]])
unmixer=np.linalg.inv(mixer)