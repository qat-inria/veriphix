from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class DesignResult:
    detection_rate: float
    epsilon_target: float
    alpha: float
    delta_tilde: float
    lambda_phi: float
    d: int
    s: int
    w: int
    w_over_s: float
    w_over_s_target: float
    eps_test_bound: float
    eps_comp_bound: float
    total_bound: float
    objective_value: float


def alpha_from_c(c: float) -> float:
    if not (0.0 <= c < 0.5):
        raise ValueError("Need 0 <= c < 1/2.")
    return (1.0 - 2.0 * c) / (2.0 - 2.0 * c)


def safe_exp(x: float) -> float:
    if x < -745:
        return 0.0
    if x > 709:
        return float("inf")
    return math.exp(x)


def bounds_corrected_delta(
    c: float,
    detection_rate: float,
    delta_tilde: float,
    d: int,
    s: int,
    lambda_phi: float = 0.5,
) -> dict[str, float | int]:
    alpha = alpha_from_c(c)

    if not (0.0 < detection_rate <= 1.0):
        raise ValueError("Need 0 < detection_rate <= 1.")
    if d <= 0 or s <= 0:
        raise ValueError("Need d,s >= 1.")
    if not (0.0 < delta_tilde < alpha):
        raise ValueError("Need 0 < delta_tilde < alpha.")
    if not (0.0 < lambda_phi < 1.0):
        raise ValueError("Need 0 < lambda_phi < 1.")

    # w/s target = (alpha - delta_tilde) * detection_rate
    w_over_s_target = (alpha - delta_tilde) * detection_rate
    if not (0.0 <= w_over_s_target < alpha * detection_rate + 1e-15):
        raise ValueError("w/s target outside admissible range.")

    # floor preserves w/s <= target
    w = math.floor(s * w_over_s_target)
    w_over_s_actual = w / s

    phi = lambda_phi * delta_tilde

    chi_test = (delta_tilde - phi) / 2.0  # = (1-lambda_phi)*delta_tilde/2
    chi_comp = phi / 2.0  # = lambda_phi*delta_tilde/2

    # ---- epsilon bound (test side) ----
    term1_test = safe_exp(-2.0 * (chi_test**2) * s)

    denom_test = alpha - phi - chi_test
    if denom_test <= 0:
        raise ValueError("Need alpha - phi - chi_test > 0.")

    gap_test = denom_test * detection_rate - w_over_s_actual
    term2_test = safe_exp(-2.0 * (gap_test**2) / denom_test * s)

    eps_test = term1_test + term2_test

    # ---- nu bound (computation side) ----
    term1_comp = safe_exp(-2.0 * (chi_comp**2) * d)

    denom_comp = (1.0 - alpha) + phi / 2.0
    if denom_comp <= 0:
        raise ValueError("Need (1-alpha) + phi/2 > 0.")

    gap_comp = denom_comp * (1.0 - c) - 0.5
    term2_comp = safe_exp(-2.0 * (gap_comp**2) / denom_comp * d)

    eps_comp = term1_comp + term2_comp

    return {
        "alpha": alpha,
        "phi": phi,
        "chi_test": chi_test,
        "chi_comp": chi_comp,
        "w": w,
        "w_over_s_actual": w_over_s_actual,
        "w_over_s_target": w_over_s_target,
        "eps_test": eps_test,
        "eps_comp": eps_comp,
        "total_bound": max(eps_test, eps_comp),
    }


def _binary_search_min_int(ok_fn: Callable[[int], bool], hi_max: int) -> int:
    lo, hi = 1, 1
    while hi <= hi_max and not ok_fn(hi):
        hi *= 2
    if hi > hi_max:
        raise RuntimeError(f"Could not find feasible value up to {hi_max}.")
    lo = hi // 2 if hi > 1 else 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ok_fn(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def min_rounds_for_delta_tilde(
    c: float,
    detection_rate: float,
    epsilon_target: float,
    delta_tilde: float,
    lambda_phi: float = 0.5,
    d_max: int = 10**7,
    s_max: int = 10**7,
) -> DesignResult:
    alpha = alpha_from_c(c)

    if not (0.0 < epsilon_target < 1.0):
        raise ValueError("Need 0 < epsilon_target < 1.")
    if not (0.0 < delta_tilde < alpha):
        raise ValueError("Need 0 < delta_tilde < alpha.")
    if not (0.0 < lambda_phi < 1.0):
        raise ValueError("Need 0 < lambda_phi < 1.")

    def test_ok(s: int) -> bool:
        vals = bounds_corrected_delta(
            c=c,
            detection_rate=detection_rate,
            delta_tilde=delta_tilde,
            d=1,
            s=s,
            lambda_phi=lambda_phi,
        )
        return vals["eps_test"] <= epsilon_target

    def comp_ok(d: int) -> bool:
        vals = bounds_corrected_delta(
            c=c,
            detection_rate=detection_rate,
            delta_tilde=delta_tilde,
            d=d,
            s=1,
            lambda_phi=lambda_phi,
        )
        return vals["eps_comp"] <= epsilon_target

    s_star = _binary_search_min_int(test_ok, s_max)
    d_star = _binary_search_min_int(comp_ok, d_max)

    vals = bounds_corrected_delta(
        c=c,
        detection_rate=detection_rate,
        delta_tilde=delta_tilde,
        d=d_star,
        s=s_star,
        lambda_phi=lambda_phi,
    )

    return DesignResult(
        detection_rate=detection_rate,
        epsilon_target=epsilon_target,
        alpha=alpha,
        delta_tilde=delta_tilde,
        lambda_phi=lambda_phi,
        d=d_star,
        s=s_star,
        w=vals["w"],
        w_over_s=vals["w_over_s_actual"],
        w_over_s_target=vals["w_over_s_target"],
        eps_test_bound=vals["eps_test"],
        eps_comp_bound=vals["eps_comp"],
        total_bound=vals["total_bound"],
        objective_value=float(d_star + s_star),
    )


def _default_objective(res: DesignResult) -> float:
    return res.d + res.s


def optimize_with_robustness_constraint(
    c: float,
    detection_rate: float,
    epsilon_target: float,
    rho_min: float,
    lambda_phi: float = 0.5,
    n_grid: int = 3000,
    objective: Callable[[DesignResult], float] | None = None,
) -> DesignResult:
    """Minimize objective over delta_tilde, with fixed lambda_phi, subject to:
    - total security bound <= epsilon_target
    - w/s >= rho_min
    """
    alpha = alpha_from_c(c)

    if not (0.0 <= rho_min < alpha * detection_rate):
        raise ValueError(f"Need 0 <= rho_min < alpha * detection_rate = {alpha * detection_rate:.6g}")
    if not (0.0 < lambda_phi < 1.0):
        raise ValueError("Need 0 < lambda_phi < 1.")

    delta_min = 1e-12
    delta_max = alpha - rho_min / detection_rate

    if delta_max <= delta_min:
        raise ValueError("No feasible delta interval for this rho_min.")

    if objective is None:
        objective = _default_objective

    best: DesignResult | None = None

    for i in range(n_grid):
        delta_tilde = delta_min + (delta_max - delta_min) * i / (n_grid - 1)
        try:
            res = min_rounds_for_delta_tilde(
                c=c,
                detection_rate=detection_rate,
                epsilon_target=epsilon_target,
                delta_tilde=delta_tilde,
                lambda_phi=lambda_phi,
            )
            if res.w_over_s < rho_min:
                continue

            res.objective_value = float(objective(res))
            if best is None or res.objective_value < best.objective_value:
                best = res
        except (ValueError, RuntimeError):
            continue

    if best is None:
        raise RuntimeError("No feasible design found.")

    return best


def optimize_with_robustness_constraint_over_lambda(
    c: float,
    detection_rate: float,
    epsilon_target: float,
    rho_min: float,
    n_grid_delta: int = 3000,
    n_grid_lambda: int = 101,
    lambda_min: float = 1e-6,
    lambda_max: float = 1.0 - 1e-6,
    objective: Callable[[DesignResult], float] | None = None,
) -> DesignResult:
    """Minimize objective jointly over delta_tilde and lambda_phi, subject to:
      - total security bound <= epsilon_target
      - w/s >= rho_min

    lambda_phi is searched on a grid in (0,1).
    """
    alpha = alpha_from_c(c)

    if not (0.0 <= rho_min < alpha * detection_rate):
        raise ValueError(f"Need 0 <= rho_min < alpha * detection_rate = {alpha * detection_rate:.6g}")
    if not (0.0 < lambda_min < lambda_max < 1.0):
        raise ValueError("Need 0 < lambda_min < lambda_max < 1.")
    if n_grid_delta < 2 or n_grid_lambda < 2:
        raise ValueError("Need n_grid_delta >= 2 and n_grid_lambda >= 2.")

    delta_min = 1e-12
    delta_max = alpha - rho_min / detection_rate

    if delta_max <= delta_min:
        raise ValueError("No feasible delta interval for this rho_min.")

    if objective is None:
        objective = _default_objective

    best: DesignResult | None = None

    for j in range(n_grid_lambda):
        lambda_phi = lambda_min + (lambda_max - lambda_min) * j / (n_grid_lambda - 1)

        for i in range(n_grid_delta):
            delta_tilde = delta_min + (delta_max - delta_min) * i / (n_grid_delta - 1)
            try:
                res = min_rounds_for_delta_tilde(
                    c=c,
                    detection_rate=detection_rate,
                    epsilon_target=epsilon_target,
                    delta_tilde=delta_tilde,
                    lambda_phi=lambda_phi,
                )
                if res.w_over_s < rho_min:
                    continue

                res.objective_value = float(objective(res))
                if best is None or res.objective_value < best.objective_value:
                    best = res
            except (ValueError, RuntimeError):
                continue

    if best is None:
        raise RuntimeError("No feasible design found.")

    return best


def maximize_robustness_under_budget(
    c: float,
    detection_rate: float,
    epsilon_target: float,
    budget: int,
    lambda_phi: float = 0.5,
    n_grid: int = 3000,
) -> DesignResult:
    """Among all designs with d+s <= budget and security <= epsilon_target,
    maximize actual w/s for fixed lambda_phi. Ties broken by smaller d+s.
    """
    alpha = alpha_from_c(c)

    if not (0.0 < lambda_phi < 1.0):
        raise ValueError("Need 0 < lambda_phi < 1.")

    best: DesignResult | None = None

    for i in range(n_grid):
        delta_tilde = 1e-12 + (alpha - 2e-12) * i / (n_grid - 1)
        try:
            res = min_rounds_for_delta_tilde(
                c=c,
                detection_rate=detection_rate,
                epsilon_target=epsilon_target,
                delta_tilde=delta_tilde,
                lambda_phi=lambda_phi,
            )
            if res.d + res.s > budget:
                continue

            if best is None:
                best = res
            else:
                if res.w_over_s > best.w_over_s:
                    best = res
                elif res.w_over_s == best.w_over_s and (res.d + res.s) < (best.d + best.s):
                    best = res
        except (ValueError, RuntimeError):
            continue

    if best is None:
        raise RuntimeError("No feasible design found under the given budget.")

    best.objective_value = float(best.d + best.s)
    return best


def maximize_robustness_under_budget_over_lambda(
    c: float,
    detection_rate: float,
    epsilon_target: float,
    budget: int,
    n_grid_delta: int = 3000,
    n_grid_lambda: int = 101,
    lambda_min: float = 1e-6,
    lambda_max: float = 1.0 - 1e-6,
) -> DesignResult:
    """Among all designs with d+s <= budget and security <= epsilon_target,
    maximize actual w/s, jointly over delta_tilde and lambda_phi.
    Ties broken by smaller d+s.
    """
    alpha = alpha_from_c(c)

    if not (0.0 < lambda_min < lambda_max < 1.0):
        raise ValueError("Need 0 < lambda_min < lambda_max < 1.")
    if n_grid_delta < 2 or n_grid_lambda < 2:
        raise ValueError("Need n_grid_delta >= 2 and n_grid_lambda >= 2.")

    best: DesignResult | None = None

    for j in range(n_grid_lambda):
        lambda_phi = lambda_min + (lambda_max - lambda_min) * j / (n_grid_lambda - 1)

        for i in range(n_grid_delta):
            delta_tilde = 1e-12 + (alpha - 2e-12) * i / (n_grid_delta - 1)
            try:
                res = min_rounds_for_delta_tilde(
                    c=c,
                    detection_rate=detection_rate,
                    epsilon_target=epsilon_target,
                    delta_tilde=delta_tilde,
                    lambda_phi=lambda_phi,
                )
                if res.d + res.s > budget:
                    continue

                if best is None:
                    best = res
                else:
                    if res.w_over_s > best.w_over_s:
                        best = res
                    elif res.w_over_s == best.w_over_s and (res.d + res.s) < (best.d + best.s):
                        best = res
            except (ValueError, RuntimeError):
                continue

    if best is None:
        raise RuntimeError("No feasible design found under the given budget.")

    best.objective_value = float(best.d + best.s)
    return best


if __name__ == "__main__":
    c = 0
    detection_rate = 0.5  # FK12 / RandomTraps: detection_rate = 1/2
    epsilon_target = 1e-6
    rho_min = 0.1

    best = optimize_with_robustness_constraint_over_lambda(
        c=c,
        detection_rate=detection_rate,
        epsilon_target=epsilon_target,
        rho_min=rho_min,
        n_grid_delta=4000,
        n_grid_lambda=11,
    )

    print("Best design with robustness constraint (optimized over lambda)")
    print("-------------------------------------------------------------")
    print(f"detection_rate       = {best.detection_rate:.10f}")
    print(f"alpha                = {best.alpha:.10f}")
    print(f"delta_tilde          = {best.delta_tilde:.10f}")
    print(f"lambda_phi           = {best.lambda_phi:.10f}")
    print(f"d                    = {best.d}")
    print(f"s                    = {best.s}")
    print(f"w                    = {best.w}")
    print(f"w/s (actual)         = {best.w_over_s:.10f}")
    print(f"w/s (target)         = {best.w_over_s_target:.10f}")
    print(f"test bound           = {best.eps_test_bound:.3e}")
    print(f"comp bound           = {best.eps_comp_bound:.3e}")
    print(f"total bound          = {best.total_bound:.3e}")
    print(f"objective (d+s)      = {best.objective_value:.1f}")

    budget = 1000
    best2 = maximize_robustness_under_budget_over_lambda(
        c=c,
        detection_rate=detection_rate,
        epsilon_target=epsilon_target,
        budget=budget,
        n_grid_delta=4000,
        n_grid_lambda=11,
    )

    print()
    print("Best design under budget (optimized over lambda)")
    print("-----------------------------------------------")
    print(f"budget               = {budget}")
    print(f"detection_rate       = {best2.detection_rate:.10f}")
    print(f"delta_tilde          = {best2.delta_tilde:.10f}")
    print(f"lambda_phi           = {best2.lambda_phi:.10f}")
    print(f"d                    = {best2.d}")
    print(f"s                    = {best2.s}")
    print(f"w                    = {best2.w}")
    print(f"w/s (actual)         = {best2.w_over_s:.10f}")
    print(f"w/s (target)         = {best2.w_over_s_target:.10f}")
    print(f"test bound           = {best2.eps_test_bound:.3e}")
    print(f"comp bound           = {best2.eps_comp_bound:.3e}")
    print(f"total bound          = {best2.total_bound:.3e}")
    print(f"objective (d+s)      = {best2.objective_value:.1f}")
