import os
import sys
sys.path.insert(0, r'd:\Imran\METU\Coursework\CE586 - Earthquake\Assignment2')

# Disable plotting
import matplotlib
matplotlib.use('Agg')

from newmark_sdof_inelastic import *

# Problem parameters.
m = 3600.0
k = 1.4e6
c = 7.0e4
Fy = 24000.0

dt = 0.1
t_end = 2.0
beta = 0.25
gamma = 0.5

# Given force history.
times_load = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
forces_kN = np.array([0, 8, 18, 36, 39, 40, 39, 31, 19, 10, 0], dtype=float)

print("Running linear and elasto-plastic Newmark-beta analyses...")

# Linear solution for comparison.
t, F, u_lin, v_lin, a_lin, f_lin = newmark_sdof_linear(
    m, c, k, times_load, forces_kN, dt, t_end, beta=beta, gamma=gamma
)

# Elasto-plastic solution.
t2, F2, u_pl, v_pl, a_pl, f_pl, u_p = newmark_sdof_elasto_plastic(
    m, c, k, Fy, times_load, forces_kN, dt, t_end, beta=beta, gamma=gamma
)

print("Analysis completed successfully!")
print_summary(u_lin, u_pl, f_pl)
