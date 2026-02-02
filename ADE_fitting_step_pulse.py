"""
Zakaria Boukertouta (Group03)
Estimate retardation factor R and dispersion coefficient D from STEP and PULSE column experiments.
Includes the experimental datasets (time [s], EC [µS/cm]) embedded in the script.

import numpy as np
from scipy.special import erfc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# ---------------------------
# User / experiment inputs
# ---------------------------

# Hydraulic / geometry inputs (used to compute pore velocity automatically)
Q_L_per_min = 0.25          # volumetric flow rate [L/min]
d_m = 0.065                 # column inner diameter [m]
porosity = 0.36             # porosity (dimensionless)
L = 0.32                    # column length [m]

# Convert Q to m^3/s
Q = Q_L_per_min / 1000.0 / 60.0   # m^3/s

# Compute cross-sectional area, Darcy velocity q, and pore-water velocity v
A = np.pi * (d_m ** 2) / 4.0
q = Q / A
v = q / porosity

# Print velocity summary
print("=== Hydraulic / Geometry Summary ===")
print(f"Flow rate Q = {Q_L_per_min} L/min = {Q:.3e} m^3/s")
print(f"Column diameter d = {d_m:.3f} m, area A = {A:.3e} m^2")
print(f"Porosity = {porosity}")
print(f"Darcy flux q = {q:.3e} m/s")
print(f"Pore-water velocity v = {v:.3e} m/s")
print("====================================\n")

# ---------------------------
# STEP experiment data (time [s], EC [µS/cm])
# This is the STEP dataset you provided earlier
# ---------------------------
t_step = np.array([80,150,180,247,311,373,436,497,562,624,688,751,814,877,938,1003,1071,1133,1198,1261,1325,1388,1452,1516,1580,1644,1712], dtype=float)
EC_step = np.array([297,291,281,272,267,268,262,258,257,249,234,243,245,251,300,455,704,968,1231,1478,1670,1736,1831,1843,1829,1828,1827], dtype=float)

# Background and inlet EC for normalization (based on your input/observations)
EC_background = 250.0    # background EC (µS/cm)
EC_inlet_step = 1850.0   # inlet plateau EC for STEP (µS/cm)

# ---------------------------
# PULSE experiment data (time [s], EC [µS/cm])
# This is the PULSE dataset you later provided (used to fit pulse)
# ---------------------------
t_pulse = np.array([83,121,162,201,243,284,326,368,408,450,492,532,573,615,658,700,743,785,827,870,911,955,998,1041,1083,1126,1167,1209,1253,1298,1340,1382,1425,1467,1510,1551,1594,1639,1682,1725,1769,1812], dtype=float)
EC_pulse = np.array([268,261,255,251,248,242,244,242,240,238,236,234,234,227,231,227,229,235,265,391,658,882,898,890,665,545,465,419,374,345,321,305,291,279,270,262,258,253,250,245,244,241], dtype=float)

# For pulse normalization choose a plausible peak EC (user can adjust if desired)
EC_background_pulse = 250.0   # background EC (µS/cm)
# If you have measured pulse peak, use that; otherwise use approximate value near the peak from data:
EC_peak_pulse = max(EC_pulse)  # best to use measured peak; here we use max observed

# ---------------------------
# Helper functions: ADE analytical solutions (with safety for t<=0)
# ---------------------------

def step_solution(t, R, D, L, v):
    """
    Analytical solution for a step input (normalized concentration at x=L, time t)
    C_rel = 0.5 * erfc((R*L - v*t) / sqrt(4*R*D*t))
    This function returns 0 for t<=0 safely.
    """
    t = np.asarray(t, dtype=float)
    C = np.zeros_like(t)
    mask = t > 0
    tt = t[mask]
    denom = np.sqrt(4.0 * R * D * tt)
    arg = (R * L - v * tt) / denom
    C[mask] = 0.5 * erfc(arg)
    return C

def pulse_solution(t, R, D, t0, L, v):
    """
    Pulse = step(t) - step(t - t0)
    t0 = pulse duration (s)
    """
    t = np.asarray(t, dtype=float)
    s1 = step_solution(t, R, D, L, v)
    # For second term, handle t-t0 <= 0 safely
    t_minus = t - t0
    s2 = step_solution(t_minus, R, D, L, v)
    return s1 - s2

# ---------------------------
# Normalize measured EC to relative concentration C_rel = (EC - EC_bg)/(EC_in - EC_bg)
# ---------------------------
Crel_step = (EC_step - EC_background) / (EC_inlet_step - EC_background)
# Clip to [0,1] to avoid small numerical overshoots
Crel_step = np.clip(Crel_step, 0.0, 1.0)

Crel_pulse = (EC_pulse - EC_background_pulse) / (EC_peak_pulse - EC_background_pulse)
Crel_pulse = np.clip(Crel_pulse, 0.0, 1.0)

# ---------------------------
# Fit STEP data (fit R and D)
# ---------------------------

# Provide initial guesses and bounds
p0_step = [5.0, 1e-5]          # initial guess for [R, D]
bounds_step = ([0.01, 1e-10], [1e3, 1e-1])  # R between small positive and 1000, D between tiny and 0.1 m2/s

# Because step_solution expects L,v fixed we create lambda wrapper
def fit_step(t, R, D):
    return step_solution(t, R, D, L, v)

# perform the fit; try/except to catch possible fitting failures
try:
    popt_step, pcov_step = curve_fit(fit_step, t_step, Crel_step, p0=p0_step, bounds=bounds_step, maxfev=20000)
    R_step, D_step = popt_step
    perr_step = np.sqrt(np.diag(pcov_step))
except Exception as e:
    print("STEP fit failed:", e)
    R_step, D_step = np.nan, np.nan
    perr_step = [np.nan, np.nan]

# ---------------------------
# Fit PULSE data (fit R, D, t0)
# ---------------------------

p0_pulse = [5.0, 1e-6, 200.0]   # initial guesses: R,D,t0
bounds_pulse = ([0.01, 1e-12, 1.0], [1e3, 1e-1, 5e4])

def fit_pulse(t, R, D, t0):
    return pulse_solution(t, R, D, t0, L, v)

try:
    popt_pulse, pcov_pulse = curve_fit(fit_pulse, t_pulse, Crel_pulse, p0=p0_pulse, bounds=bounds_pulse, maxfev=40000)
    R_pulse, D_pulse, t0_pulse = popt_pulse
    perr_pulse = np.sqrt(np.diag(pcov_pulse))
except Exception as e:
    print("PULSE fit failed:", e)
    R_pulse, D_pulse, t0_pulse = np.nan, np.nan, np.nan
    perr_pulse = [np.nan, np.nan, np.nan]

# ---------------------------
# Print results
# ---------------------------
print("\n=== STEP fit results ===")
print(f"R_step = {R_step:.3f}  (± {perr_step[0]:.3f} if available)")
print(f"D_step = {D_step:.3e} m^2/s  (± {perr_step[1]:.3e} if available)")

print("\n=== PULSE fit results ===")
print(f"R_pulse = {R_pulse:.3f}  (± {perr_pulse[0]:.3f} if available)")
print(f"D_pulse = {D_pulse:.3e} m^2/s  (± {perr_pulse[1]:.3e} if available)")
print(f"t0_pulse = {t0_pulse:.1f} s  (± {perr_pulse[2]:.1f} if available)")

# ---------------------------
# Generate plots: measured vs fitted (time and dimensionless time)
# ---------------------------

# Prepare figure directory
out_dir = "ADE_plots"
os.makedirs(out_dir, exist_ok=True)

# STEP plot
t_plot = np.linspace(min(t_step)*0.9, max(t_step)*1.1, 400)
C_fit_step = step_solution(t_plot, R_step, D_step, L, v)
T_star_step = v * t_step / L
T_star_plot = v * t_plot / L

plt.figure(figsize=(7,5))
plt.plot(T_star_step, Crel_step, 'o', label='STEP measured (normalized)', markersize=6)
plt.plot(T_star_plot, C_fit_step, '-', label=f'STEP fit (R={R_step:.2f}, D={D_step:.2e})')
plt.xlabel('Dimensionless time $T^* = v t / L$')
plt.ylabel('Relative concentration $C/C_0$')
plt.title('STEP experiment: measured vs ADE fit')
plt.legend()
plt.grid(linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'STEP_fit.png'), dpi=300)
plt.show()

# PULSE plot
t_plot_p = np.linspace(min(t_pulse)*0.9, max(t_pulse)*1.1, 400)
C_fit_pulse = pulse_solution(t_plot_p, R_pulse, D_pulse, t0_pulse, L, v)
T_star_pulse = v * t_pulse / L
T_star_plot_p = v * t_plot_p / L

plt.figure(figsize=(7,5))
plt.plot(T_star_pulse, Crel_pulse, 'o', label='PULSE measured (normalized)', markersize=6)
plt.plot(T_star_plot_p, C_fit_pulse, '-', label=f'PULSE fit (R={R_pulse:.2f}, D={D_pulse:.2e}, t0={t0_pulse:.0f}s)')
plt.xlabel('Dimensionless time $T^* = v t / L$')
plt.ylabel('Relative concentration $C/C_0$')
plt.title('PULSE experiment: measured vs ADE fit')
plt.legend()
plt.grid(linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'PULSE_fit.png'), dpi=300)
plt.show()

# ---------------------------
# Save numeric outputs to a small text file for professor convenience
# ---------------------------
results_txt = os.path.join(out_dir, "ADE_fit_results.txt")
with open(results_txt, "w") as f:
    f.write("ADE fitting results\n")
    f.write("===================\n")
    f.write(f"Hydraulic/geometry: Q={Q_L_per_min} L/min, d={d_m} m, porosity={porosity}, L={L} m\n")
    f.write(f"Computed pore-water velocity v = {v:.6e} m/s\n\n")
    f.write("STEP fit results:\n")
    f.write(f"R_step = {R_step:.6f}\n")
    f.write(f"D_step = {D_step:.6e} m^2/s\n\n")
    f.write("PULSE fit results:\n")
    f.write(f"R_pulse = {R_pulse:.6f}\n")
    f.write(f"D_pulse = {D_pulse:.6e} m^2/s\n")
    f.write(f"t0_pulse = {t0_pulse:.2f} s\n")

print(f"\nPlots and results saved in folder: {out_dir}")
print(f" - STEP_fit.png, PULSE_fit.png, ADE_fit_results.txt")
