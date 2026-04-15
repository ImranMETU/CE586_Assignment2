"""
SDOF Dynamic Response Analysis - Newmark-Beta Method (Constant Average Acceleration)

This module solves the equation of motion for a linear Single-Degree-of-Freedom (SDOF) system:
    m*u'' + c*u' + k*u = F(t)

Using the Newmark-beta method with constant average acceleration (beta=1/4, gamma=1/2).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from datetime import datetime


def newmark_sdof(m, c, k, force_time, force_values_kN, dt, t_end, beta=0.25, gamma=0.5):
    """
    Solve the dynamic response of a linear SDOF system using Newmark-beta method.
    
    The Newmark-beta method is an implicit time integration scheme that solves:
        m*u'' + c*u' + k*u = F(t)
    
    with constant average acceleration when beta=1/4 and gamma=1/2.
    
    Parameters:
    -----------
    m : float
        Mass (kg)
    c : float
        Damping coefficient (N*s/m)
    k : float
        Stiffness (N/m)
    force_time : array-like
        Time points where forces are specified (seconds)
    force_values_kN : array-like
        Force values at specified times (kilonewtons)
    dt : float
        Time step for integration (seconds)
    t_end : float
        End time for analysis (seconds)
    beta : float
        Newmark-beta parameter (default 0.25 for constant average acceleration)
    gamma : float
        Newmark-gamma parameter (default 0.5 for constant average acceleration)
    
    Returns:
    --------
    t : ndarray
        Time array (seconds)
    F : ndarray
        Force array (Newtons)
    u : ndarray
        Displacement array (meters)
    v : ndarray
        Velocity array (m/s)
    a : ndarray
        Acceleration array (m/s^2)
    F_effective : ndarray
        Effective force array used in Newmark step update (N)
    """
    
    # Convert force from kN to N
    force_values_N = np.array(force_values_kN) * 1000.0
    
    # Create time array
    t = np.arange(0, t_end + dt/2, dt)
    n_steps = len(t)
    
    # Create force array by linear interpolation.
    # For t > max(force_time), force is explicitly set to zero.
    F = np.interp(t, force_time, force_values_N, left=force_values_N[0], right=0.0)
    
    # Initialize response arrays with zero initial conditions
    u = np.zeros(n_steps)  # displacement
    v = np.zeros(n_steps)  # velocity
    a = np.zeros(n_steps)  # acceleration
    F_effective = np.zeros(n_steps)  # effective force in recurrence
    
    # Initial acceleration from equilibrium: a[0] = F[0] / m
    # (with zero displacement and velocity)
    a[0] = F[0] / m
    F_effective[0] = F[0]
    
    # Newmark-beta coefficients
    # These are pre-computed constants used in the recurrence relations
    a0c = 1.0 / (beta * dt**2)
    a1c = gamma / (beta * dt)
    a2c = 1.0 / (beta * dt)
    a3c = 1.0 / (2.0 * beta) - 1.0
    a4c = gamma / beta - 1.0
    a5c = dt * (gamma / (2.0 * beta) - 1.0)
    
    # Effective stiffness (assembled once for efficiency)
    # k_eff = k + a0c*m + a1c*c
    k_eff = k + a0c * m + a1c * c
    
    # Time-stepping loop
    # At each step i, compute response at step i+1
    for i in range(n_steps - 1):
        
        # Effective load at next step
        # Combines applied load with inertial and damping forces from current step
        p_eff = F[i+1] + m * (a0c * u[i] + a2c * v[i] + a3c * a[i]) + \
                c * (a1c * u[i] + a4c * v[i] + a5c * a[i])
        F_effective[i+1] = p_eff
        
        # Solve for displacement at next step
        # u[i+1] = p_eff / k_eff
        u[i+1] = p_eff / k_eff
        
        # Compute acceleration at next step
        # Using kinematic relation from Newmark-beta scheme
        a[i+1] = a0c * (u[i+1] - u[i]) - a2c * v[i] - a3c * a[i]
        
        # Compute velocity at next step
        # Using kinematic relation with weighted average acceleration
        v[i+1] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a[i+1])
    
    return t, F, u, v, a, F_effective


def print_results_table(t, F, u, v, a, F_effective, max_rows=None):
    """
    Print a formatted table of analysis results.
    
    Parameters:
    -----------
    t : ndarray
        Time array
    F : ndarray
        Force array (N)
    u : ndarray
        Displacement array (m)
    v : ndarray
        Velocity array (m/s)
    a : ndarray
        Acceleration array (m/s²)
    F_effective : ndarray
        Effective force array (N)
    max_rows : int, optional
        Maximum number of rows to print. If None, print all.
    """
    print("\n" + "="*122)
    print(f"{'Time (s)':>12} {'Force (N)':>16} {'Disp (m)':>16} {'Vel (m/s)':>16} {'Accel (m/s²)':>16} {'Force Effective (N)':>24}")
    print("="*122)
    
    # Determine which rows to display
    if max_rows is None or len(t) <= max_rows:
        indices = range(len(t))
    else:
        # Print first, last, and evenly spaced rows to stay under max_rows
        step = max(1, (len(t) - 1) // (max_rows - 1))
        indices = list(range(0, len(t), step))
        if indices[-1] != len(t) - 1:
            indices[-1] = len(t) - 1
    
    for i in indices:
        print(f"{t[i]:12.2f} {F[i]:16.2f} {u[i]:16.6e} {v[i]:16.6e} {a[i]:16.6e} {F_effective[i]:24.2f}")
    
    print("="*122)


def plot_dt01_responses(t, u, v, a, show=False, save_dir=None):
    """
    Create 3 standalone plots for dt=0.1 s response: displacement, velocity, acceleration.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Displacement (dt=0.1 s)
    plt.figure(figsize=(11, 6))
    plt.plot(t, u, 'o-', color='tab:blue', linewidth=2.0, markersize=4)
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Displacement (m)', fontsize=12, fontweight='bold')
    plt.title('SDOF Response (dt = 0.10 s): Displacement u(t)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'dt01_displacement.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'dt01_displacement.png')}")

    # Velocity (dt=0.1 s)
    plt.figure(figsize=(11, 6))
    plt.plot(t, v, 'o-', color='tab:green', linewidth=2.0, markersize=4)
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
    plt.title('SDOF Response (dt = 0.10 s): Velocity v(t)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'dt01_velocity.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'dt01_velocity.png')}")

    # Acceleration (dt=0.1 s)
    plt.figure(figsize=(11, 6))
    plt.plot(t, a, 'o-', color='tab:red', linewidth=2.0, markersize=4)
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Acceleration (m/s²)', fontsize=12, fontweight='bold')
    plt.title('SDOF Response (dt = 0.10 s): Acceleration a(t)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'dt01_acceleration.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'dt01_acceleration.png')}")

    if show:
        plt.show()
    else:
        plt.close('all')


def plot_comparison(results_dt_01, results_dt_001, show=False, save_dir=None):
    """
    Plot comparison of responses for dt=0.1 s and dt=0.01 s.

    Parameters:
    -----------
    results_dt_01 : tuple
        (t, u, v, a) for dt=0.1 s
    results_dt_001 : tuple
        (t, u, v, a) for dt=0.01 s
    show : bool
        If True, display plots interactively. Default False.
    save_dir : str, optional
        Directory to save plot images. If provided, plots are saved as PNG.
    """
    import os

    t1, u1, v1, a1 = results_dt_01
    t2, u2, v2, a2 = results_dt_001

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Displacement comparison
    plt.figure(figsize=(11, 6))
    plt.plot(t1, u1, 'o-', color='tab:blue', linewidth=1.8, markersize=4, label='dt = 0.10 s')
    plt.plot(t2, u2, '-', color='tab:orange', linewidth=2.0, label='dt = 0.01 s')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Displacement (m)', fontsize=12, fontweight='bold')
    plt.title('SDOF Response Comparison: Displacement u(t)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparison_displacement.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'comparison_displacement.png')}")

    # Velocity comparison
    plt.figure(figsize=(11, 6))
    plt.plot(t1, v1, 'o-', color='tab:green', linewidth=1.8, markersize=4, label='dt = 0.10 s')
    plt.plot(t2, v2, '-', color='tab:red', linewidth=2.0, label='dt = 0.01 s')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
    plt.title('SDOF Response Comparison: Velocity v(t)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparison_velocity.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'comparison_velocity.png')}")

    # Acceleration comparison
    plt.figure(figsize=(11, 6))
    plt.plot(t1, a1, 'o-', color='tab:purple', linewidth=1.8, markersize=4, label='dt = 0.10 s')
    plt.plot(t2, a2, '-', color='tab:brown', linewidth=2.0, label='dt = 0.01 s')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Acceleration (m/s²)', fontsize=12, fontweight='bold')
    plt.title('SDOF Response Comparison: Acceleration a(t)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparison_acceleration.png'), dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.join(save_dir, 'comparison_acceleration.png')}")

    if show:
        plt.show()
    else:
        plt.close('all')


def save_dynamic_csv(
    save_dir,
    t_01,
    F_01,
    u_01,
    v_01,
    a_01,
    t_001,
    F_001,
    u_001,
    v_001,
    a_001,
):
    """
    Save comparison data to a timestamped CSV file.

    The fine time grid (dt=0.01 s) is used as the base, and dt=0.1 s
    responses are linearly interpolated onto this grid for direct comparison.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Interpolate coarse-grid results onto fine-grid time for side-by-side CSV export.
    F_01_on_001 = np.interp(t_001, t_01, F_01)
    u_01_on_001 = np.interp(t_001, t_01, u_01)
    v_01_on_001 = np.interp(t_001, t_01, v_01)
    a_01_on_001 = np.interp(t_001, t_01, a_01)

    # Differences: coarse(interpolated) - fine.
    du = u_01_on_001 - u_001
    dv = v_01_on_001 - v_001
    da = a_01_on_001 - a_001

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"newmark_comparison_{timestamp}.csv")

    data = np.column_stack(
        [
            t_001,
            F_001,
            u_001,
            v_001,
            a_001,
            F_01_on_001,
            u_01_on_001,
            v_01_on_001,
            a_01_on_001,
            du,
            dv,
            da,
        ]
    )

    header = (
        "time_s,force_dt001_N,u_dt001_m,v_dt001_mps,a_dt001_mps2,"
        "force_dt01_interp_N,u_dt01_interp_m,v_dt01_interp_mps,a_dt01_interp_mps2,"
        "du_dt01minusdt001_m,dv_dt01minusdt001_mps,da_dt01minusdt001_mps2"
    )

    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    return csv_path


if __name__ == "__main__":
    import sys
    
    # ========== System Properties ==========
    m = 3600.0           # Mass (kg)
    k = 1.4e6            # Stiffness (N/m)
    c = 7.0e4            # Damping coefficient (N*s/m)
    
    # ========== Loading: Force History ==========
    # Time points where forces are specified (seconds)
    times_load = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Force values at specified times (kilonewtons)
    # After 1.0 s, force = 0 (free vibration phase)
    forces_kN = np.array([0, 8, 18, 36, 39, 40, 39, 31, 19, 10, 0])
    
    # ========== Analysis Parameters ==========
    dt_01 = 0.1          # Time step (seconds)
    dt_001 = 0.01        # Time step (seconds)
    t_end = 2.0          # End time (seconds)
    beta = 0.25          # Newmark-beta parameter (constant average acceleration)
    gamma = 0.5          # Newmark-gamma parameter (constant average acceleration)
    
    # ========== Print Header ==========
    print("\n" + "="*70)
    print("SDOF DYNAMIC RESPONSE ANALYSIS - NEWMARK-BETA METHOD")
    print("="*70)
    print(f"Mass (m):              {m:.1f} kg")
    print(f"Stiffness (k):         {k:.2e} N/m")
    print(f"Damping (c):           {c:.2e} N·s/m")
    print(f"Time step #1:          {dt_01} s")
    print(f"Time step #2:          {dt_001} s")
    print(f"End time:              {t_end} s")
    print(f"Beta (constant avg):   {beta}")
    print(f"Gamma (constant avg):  {gamma}")
    
    # Natural frequency and period
    omega_n = np.sqrt(k / m)
    T_n = 2 * np.pi / omega_n
    zeta = c / (2 * np.sqrt(k * m))
    print(f"Natural frequency:     {omega_n:.3f} rad/s ({omega_n/(2*np.pi):.3f} Hz)")
    print(f"Natural period:        {T_n:.3f} s")
    print(f"Damping ratio:         {zeta:.4f}")
    print("="*70)
    
    # ========== Solve for dt = 0.1 s ==========
    t_01, F_01, u_01, v_01, a_01, Feff_01 = newmark_sdof(
        m, c, k, times_load, forces_kN, dt_01, t_end, beta, gamma
    )

    # ========== Solve for dt = 0.01 s ==========
    t_001, F_001, u_001, v_001, a_001, Feff_001 = newmark_sdof(
        m, c, k, times_load, forces_kN, dt_001, t_end, beta, gamma
    )

    # ========== Print Results Tables ==========
    print("\nResults Table for dt = 0.1 s")
    print_results_table(t_01, F_01, u_01, v_01, a_01, Feff_01, max_rows=25)

    print("\nResults Table for dt = 0.01 s")
    print_results_table(t_001, F_001, u_001, v_001, a_001, Feff_001, max_rows=30)

    # ========== Peak Values ==========
    max_u_01 = np.max(np.abs(u_01))
    max_v_01 = np.max(np.abs(v_01))
    max_a_01 = np.max(np.abs(a_01))

    max_u_001 = np.max(np.abs(u_001))
    max_v_001 = np.max(np.abs(v_001))
    max_a_001 = np.max(np.abs(a_001))

    print("\nMaximum Absolute Responses")
    print("=" * 70)
    print(f"{'Quantity':<20}{'dt = 0.10 s':>20}{'dt = 0.01 s':>20}")
    print("=" * 70)
    print(f"{'|u|max (m)':<20}{max_u_01:>20.6e}{max_u_001:>20.6e}")
    print(f"{'|v|max (m/s)':<20}{max_v_01:>20.6e}{max_v_001:>20.6e}")
    print(f"{'|a|max (m/s^2)':<20}{max_a_01:>20.6e}{max_a_001:>20.6e}")
    print("=" * 70)

    # ========== Comparison Summary ==========
    def percent_diff(ref, val):
        if np.isclose(ref, 0.0):
            return np.nan
        return (val - ref) / ref * 100.0

    du = percent_diff(max_u_001, max_u_01)
    dv = percent_diff(max_v_001, max_v_01)
    da = percent_diff(max_a_001, max_a_01)

    print("\nShort Comparison Summary")
    print("-" * 70)
    print(f"Using dt = 0.01 s as the finer reference:")
    print(f"  Displacement peak difference (dt=0.10 vs 0.01): {du:+.2f}%")
    print(f"  Velocity peak difference (dt=0.10 vs 0.01):     {dv:+.2f}%")
    print(f"  Acceleration peak difference (dt=0.10 vs 0.01): {da:+.2f}%")

    # ========== Save Data to Dynamic CSV ==========
    csv_file = save_dynamic_csv(
        save_dir="newmark_output",
        t_01=t_01,
        F_01=F_01,
        u_01=u_01,
        v_01=v_01,
        a_01=a_01,
        t_001=t_001,
        F_001=F_001,
        u_001=u_001,
        v_001=v_001,
        a_001=a_001,
    )
    print(f"\nSaved dynamic CSV: {csv_file}")

    # ========== Plot dt=0.1 standalone responses (3 plots) ==========
    save_plots = True
    show_plots = "--show" in sys.argv
    save_dir = "figures"

    print("\nGenerating dt = 0.10 s standalone plots (3 plots)...")
    plot_dt01_responses(
        t_01,
        u_01,
        v_01,
        a_01,
        show=show_plots,
        save_dir=save_dir if save_plots else None,
    )

    # ========== Plot Comparison for dt=0.01 vs dt=0.1 (3 dimensions) ==========
    save_plots = True
    show_plots = "--show" in sys.argv
    save_dir = "figures"

    print("\nGenerating comparison plots for dt = 0.01 s and dt = 0.10 s...")
    plot_comparison(
        (t_01, u_01, v_01, a_01),
        (t_001, u_001, v_001, a_001),
        show=show_plots,
        save_dir=save_dir,
    )
    print(f"\nPlots saved to: {save_dir}/")

    print("\nAnalysis complete!")
