"""LP for a forecast-time adjustment of a previously planned purchase."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from data_io import DT_HOURS
from dispatch_lp import LPInfeasible, StorageParams


@dataclass
class Adjustment:
    grid_power: np.ndarray
    charge_power: np.ndarray
    discharge_power: np.ndarray
    curtail_power: np.ndarray
    emergency_power: np.ndarray
    soc: np.ndarray
    settlement_cost: float
    expected_emergency_cost: float
    reference_grid_power: np.ndarray
    predicted_load: np.ndarray
    predicted_pv: np.ndarray
    price: np.ndarray

    @property
    def n(self) -> int:
        return len(self.grid_power)


def solve_adjustment(
    predicted_load: np.ndarray,
    predicted_pv: np.ndarray,
    price: np.ndarray,
    reference_grid_power: np.ndarray,
    *,
    soc_initial: float,
    soc_terminal: float | None,
    params: StorageParams = StorageParams(),
    emergency_multiplier: float = 5.0,
) -> Adjustment:
    """Choose adjusted purchases under piecewise deviation settlement.

    If q is the adjusted purchase and q0 is the original plan, the settlement
    convention is

        p*q + 0.5*p*|q-q0|,

    which equals p*q+0.5p(q0-q) when q<=q0 and
    p*q0+1.5p(q-q0) when q>=q0.  This is the literal piecewise reading of
    the 0.5/1.5 multipliers and is kept explicit in the paper.
    """
    load = np.asarray(predicted_load, dtype=float)
    pv = np.asarray(predicted_pv, dtype=float)
    p = np.asarray(price, dtype=float)
    q0 = np.asarray(reference_grid_power, dtype=float)
    n = len(load)
    if not (pv.shape == p.shape == q0.shape == load.shape):
        raise ValueError("adjustment arrays must have the same shape")
    if (load < 0).any() or (pv < 0).any() or (p < 0).any() or (q0 < 0).any():
        raise ValueError("adjustment inputs must be nonnegative")

    i_q = 0
    i_c = n
    i_d = 2 * n
    i_w = 3 * n
    i_e = 4 * n
    i_under = 5 * n
    i_over = 6 * n
    i_soc = 7 * n
    nv = 8 * n + 1
    c = np.zeros(nv)
    c[i_q : i_q + n] = p * DT_HOURS
    c[i_under : i_under + n] = 0.5 * p * DT_HOURS
    c[i_over : i_over + n] = 0.5 * p * DT_HOURS
    c[i_e : i_e + n] = emergency_multiplier * p * DT_HOURS
    c[i_c : i_c + n] = 1e-9
    c[i_d : i_d + n] = 1e-9
    c[i_w : i_w + n] = 1e-12

    Aeq: list[np.ndarray] = []
    beq: list[float] = []
    for k in range(n):
        # q + PV + discharge + emergency = load + charge + curtailment
        row = np.zeros(nv)
        row[i_q + k] = 1.0
        row[i_c + k] = -1.0
        row[i_d + k] = 1.0
        row[i_w + k] = -1.0
        row[i_e + k] = 1.0
        Aeq.append(row)
        beq.append(load[k] - pv[k])

        # q = q0 - under + over
        row = np.zeros(nv)
        row[i_q + k] = 1.0
        row[i_under + k] = 1.0
        row[i_over + k] = -1.0
        Aeq.append(row)
        beq.append(q0[k])

        row = np.zeros(nv)
        row[i_soc + k + 1] = 1.0
        row[i_soc + k] = -1.0
        row[i_c + k] = -params.eta * DT_HOURS
        row[i_d + k] = DT_HOURS / params.eta
        Aeq.append(row)
        beq.append(0.0)

    row = np.zeros(nv)
    row[i_soc] = 1.0
    Aeq.append(row)
    beq.append(float(soc_initial))
    if soc_terminal is not None:
        row = np.zeros(nv)
        row[i_soc + n] = 1.0
        Aeq.append(row)
        beq.append(float(soc_terminal))

    bounds = [(0.0, None)] * n  # q
    bounds += [(0.0, params.power_max)] * (2 * n)  # c,d
    bounds += [(0.0, None)] * (3 * n)  # curtailment, emergency, under, over? corrected below
    # The previous line covers w/e/under only; over is added separately.
    bounds += [(0.0, None)] * n
    bounds += [(params.soc_min, params.soc_max)] * (n + 1)
    bounds[i_soc] = (float(soc_initial), float(soc_initial))
    if soc_terminal is not None:
        bounds[i_soc + n] = (float(soc_terminal), float(soc_terminal))

    if len(bounds) != nv:
        raise AssertionError(f"internal bound length {len(bounds)} != {nv}")
    result = linprog(c, A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=bounds, method="highs")
    if not result.success:
        raise LPInfeasible(result.message)
    x = result.x
    q = x[i_q : i_q + n]
    under = x[i_under : i_under + n]
    over = x[i_over : i_over + n]
    adjustment = Adjustment(
        grid_power=q,
        charge_power=x[i_c : i_c + n],
        discharge_power=x[i_d : i_d + n],
        curtail_power=x[i_w : i_w + n],
        emergency_power=x[i_e : i_e + n],
        soc=x[i_soc : i_soc + n + 1],
        settlement_cost=float(np.sum((p * q + 0.5 * p * (under + over)) * DT_HOURS)),
        expected_emergency_cost=float(np.sum(x[i_e : i_e + n] * p * DT_HOURS * emergency_multiplier)),
        reference_grid_power=q0,
        predicted_load=load,
        predicted_pv=pv,
        price=p,
    )
    _check_adjustment(adjustment, under, over, params=params)
    return adjustment


def _check_adjustment(adjustment: Adjustment, under: np.ndarray, over: np.ndarray, *, params: StorageParams, tol: float = 1e-5) -> None:
    n = adjustment.n
    balance = (
        adjustment.grid_power
        + adjustment.predicted_pv
        + adjustment.discharge_power
        + adjustment.emergency_power
        - adjustment.predicted_load
        - adjustment.charge_power
        - adjustment.curtail_power
    )
    if np.max(np.abs(balance)) > tol:
        raise AssertionError(f"adjustment balance residual {np.max(np.abs(balance))}")
    residual = adjustment.grid_power - adjustment.reference_grid_power + under - over
    if np.max(np.abs(residual)) > tol:
        raise AssertionError("piecewise settlement variables inconsistent")
    if np.any((under > tol) & (over > tol)):
        raise AssertionError("both under- and over-adjustment positive")
    if adjustment.soc.min() < params.soc_min - tol or adjustment.soc.max() > params.soc_max + tol:
        raise AssertionError("adjustment SOC outside bounds")
    if adjustment.charge_power.max() > params.power_max + tol or adjustment.discharge_power.max() > params.power_max + tol:
        raise AssertionError("adjustment power limit violated")
    if np.any((adjustment.charge_power > tol) & (adjustment.discharge_power > tol)):
        raise AssertionError("adjustment simultaneous charge/discharge")
