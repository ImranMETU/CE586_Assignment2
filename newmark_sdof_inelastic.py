import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_force_history(force_time, force_values_kN, dt, t_end):
    """Build analysis time grid and force vector in N, with zero force after last input point."""
    force_values_N = np.array(force_values_kN, dtype=float) * 1000.0
    t = np.arange(0.0, t_end + dt / 2.0, dt)
    F = np.interp(t, force_time, force_values_N, left=force_values_N[0], right=0.0)
    return t, F


def newmark_sdof_linear(m, c, k, force_time, force_values_kN, dt, t_end):
    """
    Linear SDOF response using the CE586 incremental direct integration form.

    For the linear case, tangent stiffness is always k.
    Returns: t, F, u, v, a, F_restoring
    """
    t, F = build_force_history(force_time, force_values_kN, dt, t_end)
    n = len(t)

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    F_restoring = np.zeros(n)

    # Initial acceleration from equilibrium at committed step 0.
    F_restoring[0] = k * u[0]
    a[0] = (F[0] - c * v[0] - F_restoring[0]) / m

    for i in range(n - 1):
        # Step 1-2: tangent and effective stiffness (linear branch only).
        k_t = k
        k_eff = k_t + 2.0 * c / dt + 4.0 * m / dt ** 2

        # Step 3: effective force increment.
        dF_eff = (F[i + 1] - F[i]) + (4.0 * m / dt + 2.0 * c) * v[i] + 2.0 * m * a[i]

        # Step 4-6: solve increments and update total states.
        du = dF_eff / k_eff
        dv = 2.0 * du / dt - 2.0 * v[i]

        u[i + 1] = u[i] + du
        v[i + 1] = v[i] + dv

        # Linear restoring force and end-of-step acceleration.
        F_restoring[i + 1] = k * u[i + 1]
        a[i + 1] = (F[i + 1] - c * v[i + 1] - F_restoring[i + 1]) / m

    return t, F, u, v, a, F_restoring


def perfectly_elasto_plastic_update(u_new, u_p_prev, k, Fy):
    """
    Perfectly elasto-plastic constitutive update.

    Returns committed restoring force Fs_new, updated plastic displacement u_p_new,
    and the tangent stiffness to be used at the next step.
    """
    F_trial = k * (u_new - u_p_prev)

    if abs(F_trial) <= Fy:
        return F_trial, u_p_prev, k

    Fs_new = np.sign(F_trial) * Fy
    u_p_new = u_new - Fs_new / k
    return Fs_new, u_p_new, 0.0


def newmark_sdof_elasto_plastic(m, c, k, Fy, force_time, force_values_kN, dt, t_end):
    """
    Nonlinear SDOF response using CE586 lecture-note incremental direct integration.

    Returns: t, F, u, v, a, F_restoring, u_p
    """
    t, F = build_force_history(force_time, force_values_kN, dt, t_end)
    n = len(t)

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    F_restoring = np.zeros(n)
    u_p = np.zeros(n)
    k_t_hist = np.zeros(n)

    # Initial committed state.
    F_restoring[0], u_p[0], k_t_hist[0] = perfectly_elasto_plastic_update(u[0], 0.0, k, Fy)
    a[0] = (F[0] - c * v[0] - F_restoring[0]) / m

    # Tangent stiffness used in step i->i+1 comes from committed state at i.
    k_t_current = k_t_hist[0]

    for i in range(n - 1):
        # Step 2: effective stiffness from committed tangent at step i.
        k_eff = k_t_current + 2.0 * c / dt + 4.0 * m / dt ** 2

        # Step 3: effective force increment.
        dF_eff = (F[i + 1] - F[i]) + (4.0 * m / dt + 2.0 * c) * v[i] + 2.0 * m * a[i]

        # Step 4-6: update kinematic states using incremental formulas.
        du = dF_eff / k_eff
        dv = 2.0 * du / dt - 2.0 * v[i]

        u[i + 1] = u[i] + du
        v[i + 1] = v[i] + dv

        # Step 7: constitutive update and next-step tangent.
        Fs_new, u_p_new, k_t_next = perfectly_elasto_plastic_update(u[i + 1], u_p[i], k, Fy)
        F_restoring[i + 1] = Fs_new
        u_p[i + 1] = u_p_new
        k_t_hist[i + 1] = k_t_next

        # Step 8: end-of-step acceleration from equilibrium.
        a[i + 1] = (F[i + 1] - c * v[i + 1] - F_restoring[i + 1]) / m

        # Step 9: use next tangent for the following increment.
        k_t_current = k_t_next

    # Final equilibrium audit at all time points.
    residual_hist = F - (m * a + c * v + F_restoring)
    eq_max = np.max(np.abs(residual_hist))

    return t, F, u, v, a, F_restoring, u_p, k_t_hist, residual_hist, eq_max


def plot_comparison(t, u_lin, v_lin, a_lin, f_lin, u_pl, v_pl, a_pl, f_pl, k, Fy, save_dir=None, show_plots=False):
    """Create requested comparison plots and hysteresis plot."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Displacement vs time.
    plt.figure(figsize=(10, 5.5))
    plt.plot(t, u_lin, "-", lw=2.0, label="Elastic")
    plt.plot(t, u_pl, "--", lw=2.0, label="Elasto-plastic")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("Displacement Response")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "plastic_displacement_inelastic.png"), dpi=150, bbox_inches="tight")
        plt.savefig(os.path.join(save_dir, "comparison_displacement.png"), dpi=150, bbox_inches="tight")
    if not show_plots:
        plt.close()

    # Velocity vs time.
    plt.figure(figsize=(10, 5.5))
    plt.plot(t, v_lin, "-", lw=2.0, label="Elastic")
    plt.plot(t, v_pl, "--", lw=2.0, label="Elasto-plastic")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity Response")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "plastic_velocity_inelastic.png"), dpi=150, bbox_inches="tight")
        plt.savefig(os.path.join(save_dir, "comparison_velocity.png"), dpi=150, bbox_inches="tight")
    if not show_plots:
        plt.close()

    # Acceleration vs time.
    plt.figure(figsize=(10, 5.5))
    plt.plot(t, a_lin, "-", lw=2.0, label="Elastic")
    plt.plot(t, a_pl, "--", lw=2.0, label="Elasto-plastic")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.title("Acceleration Response")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "plastic_acceleration_inelastic.png"), dpi=150, bbox_inches="tight")
        plt.savefig(os.path.join(save_dir, "comparison_acceleration.png"), dpi=150, bbox_inches="tight")
    if not show_plots:
        plt.close()

    # Hysteresis loop: elasto-plastic restoring force and elastic reference line.
    plt.figure(figsize=(10, 5.5))
    plt.plot(u_pl, f_pl / 1000.0, "-", lw=2.0, label="Elasto-plastic hysteresis")

    # Limit the elastic reference to the yield force level for consistent comparison.
    uy = Fy / k
    u_ref = np.linspace(-uy, uy, 200)
    f_ref = k * u_ref
    plt.plot(u_ref, f_ref / 1000.0, "--", lw=1.8, label="Elastic line up to yield (|F|<=Fy)")

    plt.xlabel("Displacement (m)")
    plt.ylabel("Restoring Force (kN)")
    plt.title("Force-Displacement (Hysteresis)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "force_displacement_hysteresis.png"), dpi=150, bbox_inches="tight")
    
    if show_plots:
        plt.show()


def print_summary(Fy, k, u_lin, u_pl, f_pl, residual_hist):
    """Print requested scalar summaries."""
    uy = Fy / k
    max_u_lin = np.max(np.abs(u_lin))
    max_u_pl = np.max(np.abs(u_pl))
    residual_u_pl = u_pl[-1]
    max_f_pl = np.max(np.abs(f_pl))
    eq_max = np.max(np.abs(residual_hist))

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Yield displacement uy = Fy/k       : {uy:.6e} m")
    print(f"Max |displacement| (elastic)       : {max_u_lin:.6e} m")
    print(f"Max |displacement| (elasto-plastic): {max_u_pl:.6e} m")
    print(f"Elastic vs inelastic peak ratio    : {max_u_pl / max_u_lin:.4f}" if max_u_lin > 0 else "Elastic vs inelastic peak ratio    : undefined")
    print(f"Residual displacement at t=2.0 s   : {residual_u_pl:.6e} m")
    print(f"Max |restoring force| (plastic)    : {max_f_pl:.2f} N")
    print(f"Max |F - m*a - c*v - Fs|           : {eq_max:.3e} N")
    print("=" * 72)


def main():
    # Problem parameters.
    m = 3600.0
    k = 1.4e6
    c = 7.0e4
    Fy = 24000.0

    dt = 0.1
    t_end = 2.0
    # Given force history.
    times_load = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    forces_kN = np.array([0, 8, 18, 36, 39, 40, 39, 31, 19, 10, 0], dtype=float)

    print("Running CE586 incremental direct integration (linear + nonlinear)...")

    # Linear solution for comparison.
    t, F, u_lin, v_lin, a_lin, f_lin = newmark_sdof_linear(m, c, k, times_load, forces_kN, dt, t_end)

    # Elasto-plastic solution.
    t2, F2, u_pl, v_pl, a_pl, f_pl, u_p, k_t_hist, residual_hist, max_residual = newmark_sdof_elasto_plastic(
        m, c, k, Fy, times_load, forces_kN, dt, t_end
    )

    # Sanity checks on time/load consistency.
    if not np.allclose(t, t2):
        raise RuntimeError("Time grids for linear and plastic solutions do not match.")
    if not np.allclose(F, F2):
        raise RuntimeError("Load vectors for linear and plastic solutions do not match.")

    # Validation prints required by assignment.
    uy = Fy / k
    tolerance = 1e-6
    max_f = np.max(np.abs(f_pl))
    max_u_lin = np.max(np.abs(u_lin))
    max_u_pl = np.max(np.abs(u_pl))
    max_du = np.max(np.abs(u_pl - u_lin))

    cap_pass = max_f <= Fy + tolerance
    assert cap_pass, f"Restoring force cap violated: max|Fs|={max_f:.6f} N, Fy={Fy:.6f} N"

    response_diff_pass = max_du > 1e-8
    if not response_diff_pass:
        raise RuntimeError("Nonlinear response is not distinguishable from linear response.")

    print(f"Yield displacement uy = Fy/k       : {uy:.6e} m")
    print(f"Restoring force cap check          : {max_f:.6f} <= {Fy + tolerance:.6f} N -> {cap_pass}")
    print(f"Peak displacement comparison       : elastic={max_u_lin:.6e} m, inelastic={max_u_pl:.6e} m")
    print(f"Max |u_inelastic - u_elastic|      : {max_du:.6e} m -> distinct={response_diff_pass}")
    print(f"Max dynamic equilibrium residual   : {max_residual:.3e} N")
    print(f"Residual at final time             : {residual_hist[-1]:.3e} N")

    # Print requested scalar outputs.
    print_summary(Fy, k, u_lin, u_pl, f_pl, residual_hist)

    # Plots (saved to 'figures' directory, display disabled for non-interactive environments).
    plot_comparison(t, u_lin, v_lin, a_lin, f_lin, u_pl, v_pl, a_pl, f_pl, k, Fy, save_dir="figures", show_plots=False)


if __name__ == "__main__":
    main()
