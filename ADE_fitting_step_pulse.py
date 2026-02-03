"""
Technische Universitat Hamburg                                                                                                                             Geo-Hydroinformatics Institute                          



                                              Estimate retardation factor R and dispersion coefficient D from STEP and PULSE column experiments.
                                            



Zakaria Boukertouta (Group03)
Lecturer : Dr. Milad Aminzadeh
Date of the Experiment : November 25, 2025
Winter Semester 2025/2026












import numpy as np
from scipy.special import erfc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

Q_L_per_min = 0.25
d_m = 0.065
porosity = 0.36
L = 0.32

Q = Q_L_per_min / 1000.0 / 60.0
A = np.pi * d_m**2 / 4.0
q = Q / A
v = q / porosity

t_step = np.array([80,150,180,247,311,373,436,497,562,624,688,751,814,877,938,1003,1071,1133,1198,1261,1325,1388,1452,1516,1580,1644,1712], float)
EC_step = np.array([297,291,281,272,267,268,262,258,257,249,234,243,245,251,300,455,704,968,1231,1478,1670,1736,1831,1843,1829,1828,1827], float)

EC_background = 250.0
EC_inlet = 1850.0

t_pulse = np.array([83,121,162,201,243,284,326,368,408,450,492,532,573,615,658,700,743,785,827,870,911,955,998,1041,1083,1126,1167,1209,1253,1298,1340,1382,1425,1467,1510,1551,1594,1639,1682,1725,1769,1812], float)
EC_pulse = np.array([268,261,255,251,248,242,244,242,240,238,236,234,234,227,231,227,229,235,265,391,658,882,898,890,665,545,465,419,374,345,321,305,291,279,270,262,258,253,250,245,244,241], float)

EC_peak_pulse = EC_pulse.max()

def step_solution(t, R, D):
    t = np.asarray(t)
    C = np.zeros_like(t)
    m = t > 0
    tt = t[m]
    arg = (R*L - v*tt) / np.sqrt(4*R*D*tt)
    C[m] = 0.5 * erfc(arg)
    return C

def pulse_solution(t, R, D, t0):
    return step_solution(t, R, D) - step_solution(t - t0, R, D)

Crel_step = np.clip((EC_step - EC_background) / (EC_inlet - EC_background), 0, 1)
Crel_pulse = np.clip((EC_pulse - EC_background) / (EC_peak_pulse - EC_background), 0, 1)

popt_step, _ = curve_fit(step_solution, t_step, Crel_step, p0=[10,1e-6], bounds=([0.01,1e-12],[1e3,1e-2]), maxfev=20000)
R_step, D_step = popt_step

popt_pulse, _ = curve_fit(pulse_solution, t_pulse, Crel_pulse, p0=[10,1e-6,200], bounds=([0.01,1e-12,1],[1e3,1e-2,5000]), maxfev=40000)
R_pulse, D_pulse, t0_pulse = popt_pulse

out_dir = "ADE_plots"
os.makedirs(out_dir, exist_ok=True)

t_plot = np.linspace(t_step.min(), t_step.max(), 400)
T_step = v * t_step / L
T_plot = v * t_plot / L

plt.figure(figsize=(7,5))
plt.plot(T_step, Crel_step, 'o', label='Measured')
plt.plot(T_plot, step_solution(t_plot, R_step, D_step), '-', label='Modeled')
plt.xlabel('Dimensionless time $T^*$')
plt.ylabel('$C/C_0$')
plt.title('STEP experiment')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(out_dir, 'STEP_measured_modeled.png'), dpi=300)
plt.show()

t_plot_p = np.linspace(t_pulse.min(), t_pulse.max(), 400)
T_pulse = v * t_pulse / L
T_plot_p = v * t_plot_p / L

plt.figure(figsize=(7,5))
plt.plot(T_pulse, Crel_pulse, 'o', label='Measured')
plt.plot(T_plot_p, pulse_solution(t_plot_p, R_pulse, D_pulse, t0_pulse), '-', label='Modeled')
plt.xlabel('Dimensionless time $T^*$')
plt.ylabel('$C/C_0$')
plt.title('PULSE experiment')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(out_dir, 'PULSE_measured_modeled.png'), dpi=300)
plt.show()

print("STEP: R =", R_step, "D =", D_step)
print("PULSE: R =", R_pulse, "D =", D_pulse, "t0 =", t0_pulse)
