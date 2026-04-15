import numpy as np

from newmark_sdof import newmark_sdof

# Prefer importing inelastic solvers from newmark_sdof.py if present.
# Fall back to newmark_sdof_inelastic.py to keep this test script usable
# with the current project structure.
try:
    from newmark_sdof import newmark_sdof_linear, newmark_sdof_elasto_plastic
except ImportError:
    from newmark_sdof_inelastic import newmark_sdof_linear, newmark_sdof_elasto_plastic


BETA = 0.25
GAMMA = 0.5
TOL = 1e-6


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def pass_fail(ok):
    return "PASS" if ok else "FAIL"


def force_series_constant(FN, t_end, n_points=2):
    t = np.linspace(0.0, t_end, n_points)
    f_kN = np.full_like(t, FN / 1000.0, dtype=float)
    return t, f_kN


def get_problem_force_history():
    times_load = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    forces_kN = np.array([0, 8, 18, 36, 39, 40, 39, 31, 19, 10, 0], dtype=float)
    return times_load, forces_kN


def newmark_linear_with_initial_conditions(
    m,
    c,
    k,
    force_time,
    force_values_kN,
    dt,
    t_end,
    u0=0.0,
    v0=0.0,
    beta=BETA,
    gamma=GAMMA,
):
    """
    Minimal wrapper for linear Newmark-beta with nonzero initial conditions.
    Mirrors the solver scheme used in newmark_sdof.py.
    """
    force_values_N = np.array(force_values_kN, dtype=float) * 1000.0
    t = np.arange(0.0, t_end + dt / 2.0, dt)
    n = len(t)
    F = np.interp(t, force_time, force_values_N, left=force_values_N[0], right=0.0)

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    u[0] = u0
    v[0] = v0
    a[0] = (F[0] - c * v0 - k * u0) / m

    a0c = 1.0 / (beta * dt ** 2)
    a1c = gamma / (beta * dt)
    a2c = 1.0 / (beta * dt)
    a3c = 1.0 / (2.0 * beta) - 1.0
    a4c = gamma / beta - 1.0
    a5c = dt * (gamma / (2.0 * beta) - 1.0)

    k_eff = k + a0c * m + a1c * c

    for i in range(n - 1):
        p_eff = (
            F[i + 1]
            + m * (a0c * u[i] + a2c * v[i] + a3c * a[i])
            + c * (a1c * u[i] + a4c * v[i] + a5c * a[i])
        )

        u[i + 1] = p_eff / k_eff
        a[i + 1] = a0c * (u[i + 1] - u[i]) - a2c * v[i] - a3c * a[i]
        v[i + 1] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a[i + 1])

    return t, F, u, v, a


def find_extrema_simple(t, y):
    """Simple local-extrema detector (both maxima and minima)."""
    dy = np.diff(y)
    extrema = []
    for i in range(1, len(dy)):
        # Max: + to -, Min: - to +
        if (dy[i - 1] > 0.0 and dy[i] <= 0.0) or (dy[i - 1] < 0.0 and dy[i] >= 0.0):
            extrema.append(i)

    if not extrema:
        return np.array([]), np.array([])

    ext_idx = np.array(extrema, dtype=int)
    return t[ext_idx], y[ext_idx]


def estimate_period_from_zero_crossings(t, y):
    """Estimate period from sign-change crossings using linear interpolation."""
    crossings = []
    for i in range(len(y) - 1):
        y0 = y[i]
        y1 = y[i + 1]

        if y0 == 0.0:
            crossings.append(t[i])
            continue

        if y0 * y1 < 0.0:
            # Linear interpolation for crossing time.
            tc = t[i] - y0 * (t[i + 1] - t[i]) / (y1 - y0)
            crossings.append(tc)

    crossings = np.array(crossings, dtype=float)
    if len(crossings) < 3:
        return None

    half_periods = np.diff(crossings)
    return 2.0 * np.mean(half_periods)


def test_1_zero_input_zero_response():
    print_header("TEST 1: Zero Input Gives Zero Response")

    m, c, k = 3600.0, 7.0e4, 1.4e6
    dt, t_end = 0.01, 2.0

    force_time, force_kN = force_series_constant(0.0, t_end)

    t, F, u, v, a = newmark_sdof(m, c, k, force_time, force_kN, dt, t_end, beta=BETA, gamma=GAMMA)

    ok_u = np.allclose(u, 0.0, atol=TOL)
    ok_v = np.allclose(v, 0.0, atol=TOL)
    ok_a = np.allclose(a, 0.0, atol=TOL)
    ok = ok_u and ok_v and ok_a

    print(f"max|u| = {np.max(np.abs(u)):.3e}, max|v| = {np.max(np.abs(v)):.3e}, max|a| = {np.max(np.abs(a)):.3e}")
    print(f"Result: {pass_fail(ok)}")
    return ok


def test_2_static_limit_constant_load():
    print_header("TEST 2: Static Limit Under Constant Load")

    m, c, k = 3600.0, 7.0e4, 1.4e6
    dt, t_end = 0.01, 20.0
    F_const = 10000.0  # N

    force_time, force_kN = force_series_constant(F_const, t_end)

    t, F, u, v, a = newmark_sdof(m, c, k, force_time, force_kN, dt, t_end, beta=BETA, gamma=GAMMA)

    u_static = F_const / k
    u_final = u[-1]
    abs_err = abs(u_final - u_static)
    pct_err = abs_err / abs(u_static) * 100.0 if not np.isclose(u_static, 0.0) else np.nan

    ok = abs_err < 1e-5

    print(f"u_static = {u_static:.6e} m")
    print(f"u_final  = {u_final:.6e} m")
    print(f"|error|  = {abs_err:.6e} m")
    print(f"% error  = {pct_err:.6f} %")
    print(f"Result: {pass_fail(ok)}")
    return ok


def test_3_free_vibration_frequency_check():
    print_header("TEST 3: Free Vibration Frequency Check")

    m, c, k = 3600.0, 7.0e4, 1.4e6
    dt, t_end = 0.001, 3.0

    force_time, force_kN = force_series_constant(0.0, t_end)

    # Use wrapper to impose nonzero initial displacement.
    u0 = 0.01
    v0 = 0.0
    t, F, u, v, a = newmark_linear_with_initial_conditions(
        m,
        c,
        k,
        force_time,
        force_kN,
        dt,
        t_end,
        u0=u0,
        v0=v0,
        beta=BETA,
        gamma=GAMMA,
    )

    omega_n = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(k * m))
    omega_d = omega_n * np.sqrt(max(0.0, 1.0 - zeta ** 2))
    T_d_theory = 2.0 * np.pi / omega_d

    t_ext, y_ext = find_extrema_simple(t, u)

    # Ignore tiny numerical extrema near zero due to strong decay.
    valid = np.abs(y_ext) > (0.005 * abs(u0))
    t_ext = t_ext[valid]

    if len(t_ext) >= 3:
        half_periods = np.diff(t_ext)
        T_d_num = 2.0 * np.mean(half_periods)
        abs_err = abs(T_d_num - T_d_theory)
        pct_err = abs_err / T_d_theory * 100.0
        ok = pct_err < 2.0

        print(f"omega_n  = {omega_n:.6f} rad/s")
        print(f"zeta     = {zeta:.6f}")
        print(f"omega_d  = {omega_d:.6f} rad/s")
        print(f"T_theory = {T_d_theory:.6f} s")
        print(f"T_num    = {T_d_num:.6f} s")
        print(f"|error|  = {abs_err:.6e} s ({pct_err:.3f}%)")
        print(f"extrema used = {len(t_ext)}")
        print(f"Result: {pass_fail(ok)}")
        return ok

    # Fallback: zero-crossing-based period estimate.
    T_d_num = estimate_period_from_zero_crossings(t, u)
    if T_d_num is not None:
        abs_err = abs(T_d_num - T_d_theory)
        pct_err = abs_err / T_d_theory * 100.0
        ok = pct_err < 2.0

        print(f"omega_n  = {omega_n:.6f} rad/s")
        print(f"zeta     = {zeta:.6f}")
        print(f"omega_d  = {omega_d:.6f} rad/s")
        print(f"T_theory = {T_d_theory:.6f} s")
        print(f"T_num    = {T_d_num:.6f} s")
        print(f"|error|  = {abs_err:.6e} s ({pct_err:.3f}%)")
        print("method   = zero-crossing fallback")
        print(f"Result: {pass_fail(ok)}")
        return ok

    print("Could not detect enough extrema or zero-crossings for period estimation.")
    print("Result: FAIL")
    return False


def test_4_time_step_convergence():
    print_header("TEST 4: Time-Step Convergence")

    m, c, k = 3600.0, 7.0e4, 1.4e6
    t_end = 2.0
    force_time, force_kN = get_problem_force_history()

    dts = [0.1, 0.01, 0.005]
    results = {}

    for dt in dts:
        results[dt] = newmark_sdof(m, c, k, force_time, force_kN, dt, t_end, beta=BETA, gamma=GAMMA)

    t_f, F_f, u_f, v_f, a_f = results[0.005]

    def max_abs_diff_on_fine(t_c, x_c, t_ref, x_ref):
        x_interp = np.interp(t_ref, t_c, x_c)
        return np.max(np.abs(x_interp - x_ref))

    diffs = {}
    for dt in [0.1, 0.01]:
        t_c, F_c, u_c, v_c, a_c = results[dt]
        diffs[dt] = {
            "u": max_abs_diff_on_fine(t_c, u_c, t_f, u_f),
            "v": max_abs_diff_on_fine(t_c, v_c, t_f, v_f),
            "a": max_abs_diff_on_fine(t_c, a_c, t_f, a_f),
        }

    # Convergence expectation: dt=0.01 should be closer to dt=0.005 than dt=0.1.
    ok_u = diffs[0.01]["u"] < diffs[0.1]["u"]
    ok_v = diffs[0.01]["v"] < diffs[0.1]["v"]
    ok_a = diffs[0.01]["a"] < diffs[0.1]["a"]
    ok = ok_u and ok_v and ok_a

    print("Max absolute differences vs dt=0.005 reference:")
    print(f"dt=0.1  -> du={diffs[0.1]['u']:.6e}, dv={diffs[0.1]['v']:.6e}, da={diffs[0.1]['a']:.6e}")
    print(f"dt=0.01 -> du={diffs[0.01]['u']:.6e}, dv={diffs[0.01]['v']:.6e}, da={diffs[0.01]['a']:.6e}")
    print(f"Converges in u/v/a: {ok_u}/{ok_v}/{ok_a}")
    print(f"Result: {pass_fail(ok)}")
    return ok


def test_5_energy_decay_damped_linear():
    print_header("TEST 5: Energy Decay for Damped Linear System")

    m, c, k = 3600.0, 7.0e4, 1.4e6
    dt, t_end = 0.001, 3.0

    force_time, force_kN = force_series_constant(0.0, t_end)
    t, F, u, v, a = newmark_linear_with_initial_conditions(
        m,
        c,
        k,
        force_time,
        force_kN,
        dt,
        t_end,
        u0=0.01,
        v0=0.0,
        beta=BETA,
        gamma=GAMMA,
    )

    E = 0.5 * m * v ** 2 + 0.5 * k * u ** 2

    E0 = E[0]
    Ef = E[-1]
    decay_ratio = Ef / E0 if E0 > 0.0 else np.nan

    # Overall decay criterion.
    ok = Ef < 0.1 * E0

    print(f"Initial energy E0 = {E0:.6e} J")
    print(f"Final energy   Ef = {Ef:.6e} J")
    print(f"Ef / E0          = {decay_ratio:.6f}")
    print(f"Result: {pass_fail(ok)}")
    return ok


def test_6_large_yield_matches_linear():
    print_header("TEST 6: Large Yield Force -> Plastic Matches Linear")

    m, c, k = 3600.0, 7.0e4, 1.4e6
    Fy = 1.0e9
    dt, t_end = 0.01, 2.0
    force_time, force_kN = get_problem_force_history()

    t_lin, F_lin, u_lin, v_lin, a_lin = newmark_sdof(m, c, k, force_time, force_kN, dt, t_end, beta=BETA, gamma=GAMMA)

    t_pl, F_pl, u_pl, v_pl, a_pl, f_pl, u_p = newmark_sdof_elasto_plastic(
        m, c, k, Fy, force_time, force_kN, dt, t_end, beta=BETA, gamma=GAMMA
    )

    du = np.max(np.abs(u_pl - u_lin))
    dv = np.max(np.abs(v_pl - v_lin))
    da = np.max(np.abs(a_pl - a_lin))

    ok = du < 1e-5 and dv < 1e-4 and da < 1e-3

    print(f"max|u_pl - u_lin| = {du:.6e}")
    print(f"max|v_pl - v_lin| = {dv:.6e}")
    print(f"max|a_pl - a_lin| = {da:.6e}")
    print(f"Result: {pass_fail(ok)}")
    return ok


def test_7_plastic_force_capped():
    print_header("TEST 7: Plastic Restoring Force Capped by Yield")

    m = 3600.0
    k = 1.4e6
    c = 7.0e4
    Fy = 24000.0

    dt, t_end = 0.1, 2.0
    force_time, force_kN = get_problem_force_history()

    t, F, u, v, a, F_restoring, u_p = newmark_sdof_elasto_plastic(
        m, c, k, Fy, force_time, force_kN, dt, t_end, beta=BETA, gamma=GAMMA
    )

    max_restoring = np.max(np.abs(F_restoring))
    tol = 1e-6
    ok = max_restoring <= Fy + tol

    print(f"max|F_restoring| = {max_restoring:.6f} N")
    print(f"Fy + tol         = {Fy + tol:.6f} N")
    print(f"Result: {pass_fail(ok)}")
    return ok


def test_8_residual_displacement_plastic_case():
    print_header("TEST 8: Residual Displacement Exists in Plastic Case")

    m = 3600.0
    k = 1.4e6
    c = 7.0e4
    Fy = 24000.0

    dt, t_end = 0.01, 8.0
    force_time, force_kN = get_problem_force_history()

    t_lin, F_lin, u_lin, v_lin, a_lin = newmark_sdof(
        m, c, k, force_time, force_kN, dt, t_end, beta=BETA, gamma=GAMMA
    )

    t_pl, F_pl, u_pl, v_pl, a_pl, F_restoring, u_p = newmark_sdof_elasto_plastic(
        m, c, k, Fy, force_time, force_kN, dt, t_end, beta=BETA, gamma=GAMMA
    )

    u_lin_final = u_lin[-1]
    u_pl_final = u_pl[-1]

    ok_linear_near_zero = abs(u_lin_final) < 1e-4
    ok_plastic_nonzero = abs(u_pl_final) > 1e-3
    ok = ok_linear_near_zero and ok_plastic_nonzero

    print(f"linear final displacement        = {u_lin_final:.6e} m")
    print(f"elasto-plastic final displacement = {u_pl_final:.6e} m")
    print(f"linear near zero: {ok_linear_near_zero}")
    print(f"plastic nonzero : {ok_plastic_nonzero}")
    print(f"Result: {pass_fail(ok)}")
    return ok


def run_all_tests():
    test_functions = [
        test_1_zero_input_zero_response,
        test_2_static_limit_constant_load,
        test_3_free_vibration_frequency_check,
        test_4_time_step_convergence,
        test_5_energy_decay_damped_linear,
        test_6_large_yield_matches_linear,
        test_7_plastic_force_capped,
        test_8_residual_displacement_plastic_case,
    ]

    results = []
    for fn in test_functions:
        try:
            ok = fn()
        except Exception as exc:
            ok = False
            print(f"Unhandled exception in {fn.__name__}: {exc}")
            print("Result: FAIL")
        results.append((fn.__name__, ok))

    print_header("TEST SUMMARY")
    n_pass = 0
    for name, ok in results:
        print(f"{name}: {pass_fail(ok)}")
        if ok:
            n_pass += 1

    print(f"\nPassed {n_pass}/{len(results)} tests.")


def main():
    run_all_tests()


if __name__ == "__main__":
    main()
