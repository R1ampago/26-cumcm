"""Linear-programming dispatch models used by the C-problem solution.

Internal power variables are in kW.  The published result workbooks use kWh,
obtained by multiplying each ten-minute interval by DT_HOURS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog

from data_io import DT_HOURS


@dataclass(frozen=True)
class StorageParams:
    capacity: float = 12000.0
    soc_min: float = 1200.0
    soc_max: float = 10800.0
    power_max: float = 5000.0
    eta: float = 0.90
    initial_soc: float = 6000.0


@dataclass
class Dispatch:
    grid_power: np.ndarray
    charge_power: np.ndarray
    discharge_power: np.ndarray
    curtail_power: np.ndarray
    soc: np.ndarray
    objective: float
    predicted_load: np.ndarray
    predicted_pv: np.ndarray
    price: np.ndarray

    @property
    def n(self) -> int:
        return len(self.grid_power)

    def energy(self, power: np.ndarray) -> np.ndarray:
        return np.asarray(power) * DT_HOURS

    @property
    def grid_energy(self) -> np.ndarray:
        return self.energy(self.grid_power)

    @property
    def charge_energy(self) -> np.ndarray:
        return self.energy(self.charge_power)

    @property
    def discharge_energy(self) -> np.ndarray:
        return self.energy(self.discharge_power)


class LPInfeasible(RuntimeError):
    pass


def _build_dispatch_lp(
    load: np.ndarray,
    pv: np.ndarray,
    price: np.ndarray,
    *,
    soc_initial: float,
    soc_terminal: Optional[float],
    params: StorageParams,
    grid_cost: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float | None, float | None]]]:
    """Build variables [grid, charge, discharge, curtail, soc_0..soc_n]."""
    load = np.asarray(load, dtype=float)
    pv = np.asarray(pv, dtype=float)
    price = np.asarray(price, dtype=float)
    n = len(load)
    if pv.shape != (n,) or price.shape != (n,):
        raise ValueError("load, pv and price must have equal one-dimensional length")
    if not (np.isfinite(load).all() and np.isfinite(pv).all() and np.isfinite(price).all()):
        raise ValueError("dispatch inputs must be finite")
    if (load < 0).any() or (pv < 0).any() or (price < 0).any():
        raise ValueError("load, PV and price must be nonnegative")

    i_grid = 0
    i_charge = n
    i_discharge = 2 * n
    i_curtail = 3 * n
    i_soc = 4 * n
    nv = 5 * n + 1
    c = np.zeros(nv)
    c[i_grid : i_grid + n] = price * DT_HOURS if grid_cost is None else np.asarray(grid_cost)

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []
    for k in range(n):
        # Grid + PV + battery discharge = load + battery charge + curtailment.
        row = np.zeros(nv)
        row[i_grid + k] = 1.0
        row[i_charge + k] = -1.0
        row[i_discharge + k] = 1.0
        row[i_curtail + k] = -1.0
        eq_rows.append(row)
        eq_rhs.append(load[k] - pv[k])

        # SOC is measured on the storage side; charge/discharge are AC-side powers.
        row = np.zeros(nv)
        row[i_soc + k + 1] = 1.0
        row[i_soc + k] = -1.0
        row[i_charge + k] = -params.eta * DT_HOURS
        row[i_discharge + k] = DT_HOURS / params.eta
        eq_rows.append(row)
        eq_rhs.append(0.0)

    row = np.zeros(nv)
    row[i_soc] = 1.0
    eq_rows.append(row)
    eq_rhs.append(float(soc_initial))
    if soc_terminal is not None:
        row = np.zeros(nv)
        row[i_soc + n] = 1.0
        eq_rows.append(row)
        eq_rhs.append(float(soc_terminal))

    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * (4 * n)
    bounds += [(params.soc_min, params.soc_max)] * (n + 1)
    for k in range(n):
        bounds[i_charge + k] = (0.0, params.power_max)
        bounds[i_discharge + k] = (0.0, params.power_max)
    bounds[i_soc] = (float(soc_initial), float(soc_initial))
    if soc_terminal is not None:
        bounds[i_soc + n] = (float(soc_terminal), float(soc_terminal))
    return c, np.asarray(eq_rows), np.asarray(eq_rhs), bounds


def solve_dispatch(
    load: np.ndarray,
    pv: np.ndarray,
    price: np.ndarray,
    *,
    soc_initial: float,
    soc_terminal: Optional[float] = None,
    params: StorageParams = StorageParams(),
) -> Dispatch:
    """Minimize ordinary grid purchase cost for a known profile.

    The model allows PV curtailment and disallows export.  Simultaneous charging
    and discharging is not explicitly binary-constrained: with nonnegative grid
    costs and eta<1 it is dominated by a non-simultaneous solution, and the
    result is checked after solving.
    """
    c, Aeq, beq, bounds = _build_dispatch_lp(
        load,
        pv,
        price,
        soc_initial=soc_initial,
        soc_terminal=soc_terminal,
        params=params,
    )
    result = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not result.success:
        raise LPInfeasible(result.message)
    n = len(load)
    x = result.x
    dispatch = Dispatch(
        grid_power=x[0:n],
        charge_power=x[n : 2 * n],
        discharge_power=x[2 * n : 3 * n],
        curtail_power=x[3 * n : 4 * n],
        soc=x[4 * n : 5 * n + 1],
        objective=float(result.fun),
        predicted_load=np.asarray(load, dtype=float),
        predicted_pv=np.asarray(pv, dtype=float),
        price=np.asarray(price, dtype=float),
    )
    check_dispatch(dispatch, params=params)
    return dispatch


def check_dispatch(dispatch: Dispatch, *, params: StorageParams = StorageParams(), tol: float = 1e-6) -> None:
    """Validate energy balance, SOC bounds and power bounds."""
    n = dispatch.n
    if any(len(x) != n for x in [dispatch.charge_power, dispatch.discharge_power, dispatch.curtail_power]):
        raise AssertionError("inconsistent dispatch lengths")
    balance = (
        dispatch.grid_power
        + dispatch.predicted_pv
        + dispatch.discharge_power
        - dispatch.predicted_load
        - dispatch.charge_power
        - dispatch.curtail_power
    )
    if np.max(np.abs(balance)) > tol:
        raise AssertionError(f"energy balance residual {np.max(np.abs(balance))}")
    if dispatch.soc.min() < params.soc_min - tol or dispatch.soc.max() > params.soc_max + tol:
        raise AssertionError("SOC outside permitted range")
    if dispatch.charge_power.min() < -tol or dispatch.discharge_power.min() < -tol:
        raise AssertionError("negative charge/discharge power")
    if max(dispatch.charge_power.max(), dispatch.discharge_power.max()) > params.power_max + tol:
        raise AssertionError("charge/discharge power limit violated")
    if np.any((dispatch.charge_power > tol) & (dispatch.discharge_power > tol)):
        raise AssertionError("simultaneous charge and discharge")


def evaluate_fixed_plan(
    dispatch: Dispatch,
    actual_load: np.ndarray,
    actual_pv: np.ndarray,
    price: np.ndarray,
    *,
    emergency_multiplier: float = 5.0,
) -> dict[str, np.ndarray | float]:
    """Evaluate a fixed plan under realized load/PV.

    The dispatch commands and planned grid purchase are held fixed.  A positive
    residual is supplied by emergency purchases; a negative residual is
    curtailed.  This is the explicit day-ahead interpretation used for the
    uncertainty audit, not an implicit claim that all real systems operate this
    way.
    """
    actual_load = np.asarray(actual_load, dtype=float)
    actual_pv = np.asarray(actual_pv, dtype=float)
    price = np.asarray(price, dtype=float)
    if actual_load.shape != actual_pv.shape or actual_load.shape != dispatch.grid_power.shape:
        raise ValueError("realized arrays must match the plan length")
    residual = actual_load + dispatch.charge_power - dispatch.grid_power - dispatch.discharge_power - actual_pv
    emergency = np.maximum(residual, 0.0)
    surplus = np.maximum(-residual, 0.0)
    planned_cost = float(np.sum(dispatch.grid_energy * price))
    emergency_cost = float(np.sum(emergency * DT_HOURS * price * emergency_multiplier))
    return {
        "emergency_power": emergency,
        "surplus_power": surplus,
        "emergency_energy": emergency * DT_HOURS,
        "planned_cost": planned_cost,
        "emergency_cost": emergency_cost,
        "total_cost": planned_cost + emergency_cost,
        "emergency_energy_total": float(np.sum(emergency) * DT_HOURS),
    }
