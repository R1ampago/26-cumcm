"""Real-time recourse LP with a fixed grid-purchase schedule."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from data_io import DT_HOURS
from dispatch_lp import Dispatch, LPInfeasible, StorageParams


@dataclass
class Recourse:
    grid_power: np.ndarray
    charge_power: np.ndarray
    discharge_power: np.ndarray
    curtail_power: np.ndarray
    emergency_power: np.ndarray
    soc: np.ndarray
    emergency_cost: float
    start_soc: float
    terminal_soc: float
    actual_load: np.ndarray
    actual_pv: np.ndarray
    price: np.ndarray

    @property
    def n(self) -> int:
        return len(self.grid_power)

    @property
    def charge_energy(self) -> np.ndarray:
        return self.charge_power * DT_HOURS

    @property
    def discharge_energy(self) -> np.ndarray:
        return self.discharge_power * DT_HOURS

    @property
    def emergency_energy(self) -> np.ndarray:
        return self.emergency_power * DT_HOURS


def solve_fixed_grid_recourse(
    fixed_grid_power: np.ndarray,
    actual_load: np.ndarray,
    actual_pv: np.ndarray,
    price: np.ndarray,
    *,
    soc_initial: float,
    soc_terminal: float | None,
    params: StorageParams = StorageParams(),
    emergency_multiplier: float = 5.0,
) -> Recourse:
    """Dispatch storage around fixed purchases, minimizing emergency energy cost.

    ``fixed_grid_power`` is a power schedule decided before actual PV is known.
    The battery may adapt its charge/discharge commands after realization.  A
    terminal SOC is required by the caller for a daily-cycle convention; if it
    is None the LP keeps the final SOC free within the physical bounds.
    """
    q = np.asarray(fixed_grid_power, dtype=float)
    load = np.asarray(actual_load, dtype=float)
    pv = np.asarray(actual_pv, dtype=float)
    price = np.asarray(price, dtype=float)
    n = len(q)
    if not (load.shape == pv.shape == price.shape == q.shape):
        raise ValueError("all recourse arrays must have the same shape")
    if (q < 0).any() or (load < 0).any() or (pv < 0).any() or (price < 0).any():
        raise ValueError("recourse inputs must be nonnegative")

    i_charge = 0
    i_discharge = n
    i_curtail = 2 * n
    i_emergency = 3 * n
    i_soc = 4 * n
    nv = 5 * n + 1
    c = np.zeros(nv)
    c[i_emergency : i_emergency + n] = emergency_multiplier * price * DT_HOURS
    # A tiny secondary penalty selects a low-action solution if emergency is 0.
    c[i_charge : i_charge + n] = 1e-9
    c[i_discharge : i_discharge + n] = 1e-9
    c[i_curtail : i_curtail + n] = 1e-12

    Aeq: list[np.ndarray] = []
    beq: list[float] = []
    for k in range(n):
        # fixed grid + PV + discharge + emergency = load + charge + curtail
        row = np.zeros(nv)
        row[i_charge + k] = -1.0
        row[i_discharge + k] = 1.0
        row[i_curtail + k] = -1.0
        row[i_emergency + k] = 1.0
        Aeq.append(row)
        beq.append(load[k] - q[k] - pv[k])

        row = np.zeros(nv)
        row[i_soc + k + 1] = 1.0
        row[i_soc + k] = -1.0
        row[i_charge + k] = -params.eta * DT_HOURS
        row[i_discharge + k] = DT_HOURS / params.eta
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

    bounds: list[tuple[float | None, float | None]] = [(0.0, params.power_max)] * (2 * n)
    bounds += [(0.0, None)] * (2 * n)
    bounds += [(params.soc_min, params.soc_max)] * (n + 1)
    bounds[i_soc] = (float(soc_initial), float(soc_initial))
    if soc_terminal is not None:
        bounds[i_soc + n] = (float(soc_terminal), float(soc_terminal))

    result = linprog(c, A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=bounds, method="highs")
    if not result.success:
        raise LPInfeasible(result.message)
    x = result.x
    rec = Recourse(
        grid_power=q.copy(),
        charge_power=x[i_charge : i_charge + n],
        discharge_power=x[i_discharge : i_discharge + n],
        curtail_power=x[i_curtail : i_curtail + n],
        emergency_power=x[i_emergency : i_emergency + n],
        soc=x[i_soc : i_soc + n + 1],
        emergency_cost=float(np.sum(x[i_emergency : i_emergency + n] * price * DT_HOURS * emergency_multiplier)),
        start_soc=float(soc_initial),
        terminal_soc=float(x[i_soc + n]),
        actual_load=load,
        actual_pv=pv,
        price=price,
    )
    _check_recourse(rec, params=params)
    return rec


def _check_recourse(rec: Recourse, *, params: StorageParams, tol: float = 1e-5) -> None:
    balance = (
        rec.grid_power
        + rec.actual_pv
        + rec.discharge_power
        + rec.emergency_power
        - rec.actual_load
        - rec.charge_power
        - rec.curtail_power
    )
    if np.max(np.abs(balance)) > tol:
        raise AssertionError(f"recourse balance residual {np.max(np.abs(balance))}")
    if rec.soc.min() < params.soc_min - tol or rec.soc.max() > params.soc_max + tol:
        raise AssertionError("recourse SOC outside bounds")
    if rec.charge_power.max() > params.power_max + tol or rec.discharge_power.max() > params.power_max + tol:
        raise AssertionError("recourse charge/discharge power limit violated")
    if np.any((rec.charge_power > tol) & (rec.discharge_power > tol)):
        raise AssertionError("recourse simultaneous charge/discharge")
