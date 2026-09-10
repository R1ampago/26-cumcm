"""Question 3 rolling PV-forecast adjustment policy.

The implementation is deliberately causal: q^P is fixed at 00:00; at 06:00,
12:00 and 18:00 only future purchases may be changed.  Actual PV is used for
real-time battery recourse and evaluation, never to create a future forecast.
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adjustment_lp import Adjustment, solve_adjustment  # noqa: E402
from data_io import DT_HOURS, interval_labels, load_attachments, typical_arrays  # noqa: E402
from dispatch_lp import Dispatch, StorageParams, solve_dispatch  # noqa: E402
from recourse_lp import Recourse, solve_fixed_grid_recourse  # noqa: E402


@dataclass
class Operation:
    grid_power: np.ndarray
    charge_power: np.ndarray
    discharge_power: np.ndarray
    curtail_power: np.ndarray
    emergency_power: np.ndarray
    soc: np.ndarray
    emergency_cost: float
    actual_load: np.ndarray
    actual_pv: np.ndarray
    price: np.ndarray

    @property
    def charge_energy(self) -> np.ndarray:
        return self.charge_power * DT_HOURS

    @property
    def discharge_energy(self) -> np.ndarray:
        return self.discharge_power * DT_HOURS

    @property
    def emergency_energy(self) -> np.ndarray:
        return self.emergency_power * DT_HOURS


@dataclass
class Q3Day:
    day: date
    plan: Dispatch
    adjusted_grid_power: np.ndarray
    operation: Operation
    settlement_cost: float
    planned_cost: float
    adjustment_fee: float
    emergency_cost: float
    updates_used: bool
    forecast_method: str

    @property
    def total_cost(self) -> float:
        return self.settlement_cost + self.emergency_cost

    @property
    def emergency_energy(self) -> float:
        return float(self.operation.emergency_energy.sum())


def _minutes(text: object) -> int:
    parts = str(text).strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])


def forecast_to_ten_minute(data, day: date, release_minute: int, *, method: str = "linear") -> np.ndarray:
    """Interpolate the release's hourly PV forecasts onto interval endpoints."""
    if method not in {"linear", "zoh"}:
        raise ValueError("method must be linear or zoh")
    rows = data.forecasts.loc[
        data.forecasts["日期"].eq(np.datetime64(day))
        & data.forecasts["预报时刻"].astype(str).map(_minutes).eq(release_minute)
    ]
    if len(rows) != 1:
        raise ValueError(f"No unique {release_minute}:00 forecast for {day}")
    row = rows.iloc[0]
    target_minutes = np.arange(release_minute + 60, release_minute + 60 * 25, 60, dtype=float)
    target_values = row.iloc[2:].to_numpy(dtype=float)
    endpoints = np.arange(10, 1441, 10, dtype=float)
    result = np.zeros(144, dtype=float)
    # Use the latest measured PV at the release as a physically available anchor
    # where it exists; at 00:00 the source has no 00:00 column, so use zero.
    if release_minute == 0:
        anchor = 0.0
    else:
        actual_row = data.pv_actual.loc[pd_timestamp(day)]
        anchor = float(actual_row.get(release_minute, 0.0))
    for k, endpoint in enumerate(endpoints):
        if endpoint <= release_minute:
            result[k] = anchor
        elif method == "linear":
            result[k] = float(np.interp(endpoint, np.r_[release_minute, target_minutes], np.r_[anchor, target_values]))
        else:
            h = int(np.ceil((endpoint - release_minute) / 60.0))
            h = min(max(h, 1), 24)
            result[k] = float(target_values[h - 1])
    return result


def pd_timestamp(day: date):
    # Kept as a tiny local helper to avoid exposing pandas in the public policy API.
    import pandas as pd

    return pd.Timestamp(day)


def _combine_recourse(segments: list[Recourse], q: np.ndarray, load: np.ndarray, pv: np.ndarray, price: np.ndarray) -> Operation:
    if not segments:
        raise ValueError("at least one recourse segment is required")
    charge = np.concatenate([x.charge_power for x in segments])
    discharge = np.concatenate([x.discharge_power for x in segments])
    curtail = np.concatenate([x.curtail_power for x in segments])
    emergency = np.concatenate([x.emergency_power for x in segments])
    soc = np.concatenate([segments[0].soc, *[x.soc[1:] for x in segments[1:]]])
    op = Operation(
        grid_power=np.asarray(q, dtype=float),
        charge_power=charge,
        discharge_power=discharge,
        curtail_power=curtail,
        emergency_power=emergency,
        soc=soc,
        emergency_cost=float(sum(x.emergency_cost for x in segments)),
        actual_load=np.asarray(load, dtype=float),
        actual_pv=np.asarray(pv, dtype=float),
        price=np.asarray(price, dtype=float),
    )
    if len(op.grid_power) != len(charge) or len(op.soc) != len(charge) + 1:
        raise AssertionError("recourse segments do not cover one day")
    return op


def _run_day(
    data,
    day: date,
    *,
    price: np.ndarray,
    params: StorageParams,
    updates_used: bool,
    forecast_method: str,
) -> Q3Day:
    load = data.load.loc[str(day)].to_numpy(float)
    actual_pv = data.pv_actual.loc[str(day)].to_numpy(float)
    # 00:00 day-ahead plan: actual load schedule + first release's PV forecast.
    pv0 = forecast_to_ten_minute(data, day, 0, method=forecast_method)
    plan = solve_dispatch(
        load,
        pv0,
        price,
        soc_initial=params.initial_soc,
        soc_terminal=params.initial_soc,
        params=params,
    )
    q_plan = plan.grid_power.copy()
    q_final = q_plan.copy()
    segments: list[Recourse] = []
    current_soc = params.initial_soc
    boundaries = [0, 36, 72, 108, 144]

    if not updates_used:
        seg = solve_fixed_grid_recourse(
            q_plan,
            load,
            actual_pv,
            price,
            soc_initial=current_soc,
            soc_terminal=params.initial_soc,
            params=params,
            emergency_multiplier=5.0,
        )
        segments.append(seg)
    else:
        # First run until 06:00 under the original plan.
        first_target = plan.soc[boundaries[1]]
        seg = solve_fixed_grid_recourse(
            q_plan[: boundaries[1]],
            load[: boundaries[1]],
            actual_pv[: boundaries[1]],
            price[: boundaries[1]],
            soc_initial=current_soc,
            soc_terminal=float(first_target),
            params=params,
            emergency_multiplier=5.0,
        )
        segments.append(seg)
        current_soc = float(seg.soc[-1])

        for left, right, release in zip(boundaries[1:-1], boundaries[2:], [360, 720, 1080]):
            predicted_pv_full = forecast_to_ten_minute(data, day, release, method=forecast_method)
            adjustment = solve_adjustment(
                load[left:],
                predicted_pv_full[left:],
                price[left:],
                q_plan[left:],
                soc_initial=current_soc,
                soc_terminal=params.initial_soc,
                params=params,
                emergency_multiplier=5.0,
            )
            q_final[left:right] = adjustment.grid_power[: right - left]
            target = adjustment.soc[right - left]
            seg = solve_fixed_grid_recourse(
                adjustment.grid_power[: right - left],
                load[left:right],
                actual_pv[left:right],
                price[left:right],
                soc_initial=current_soc,
                soc_terminal=float(target),
                params=params,
                emergency_multiplier=5.0,
            )
            segments.append(seg)
            current_soc = float(seg.soc[-1])
            # At the last release, the loop above ends at 24:00.

    operation = _combine_recourse(segments, q_final, load, actual_pv, price)
    planned_cost = float(np.sum(price * q_plan * DT_HOURS))
    settlement_cost = float(np.sum((price * q_final + 0.5 * price * np.abs(q_final - q_plan)) * DT_HOURS))
    return Q3Day(
        day=day,
        plan=plan,
        adjusted_grid_power=q_final,
        operation=operation,
        settlement_cost=settlement_cost,
        planned_cost=planned_cost,
        adjustment_fee=settlement_cost - planned_cost,
        emergency_cost=operation.emergency_cost,
        updates_used=updates_used,
        forecast_method=forecast_method,
    )


def simulate_q3(
    *,
    root: Path = ROOT,
    report_start: str = "2025-02-01",
    report_end: str = "2025-12-31",
    volatile_price: bool = False,
    updates_used: bool = True,
    forecast_method: str = "linear",
    params: StorageParams = StorageParams(),
) -> list[Q3Day]:
    data = load_attachments(root)
    typical_price, _, _ = typical_arrays(data)
    d = date(2025, 1, 1)
    report_start_date = date.fromisoformat(report_start)
    end_date = date.fromisoformat(report_end)
    result: list[Q3Day] = []
    while d <= end_date:
        price = data.price_volatile.loc[str(d)].to_numpy(float) if volatile_price else typical_price
        day_result = _run_day(
            data,
            d,
            price=price,
            params=params,
            updates_used=updates_used,
            forecast_method=forecast_method,
        )
        if d >= report_start_date:
            result.append(day_result)
        d += timedelta(days=1)
    if len(result) != 334 or result[0].day != report_start_date or result[-1].day != end_date:
        raise AssertionError("Q3 report date range is not complete")
    return result


def _clear_and_set_rows(ws, rows: list[list[object]]) -> None:
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for row in rows:
        ws.append(row)


def _emergency_segments(power: np.ndarray) -> list[tuple[int, int, float]]:
    result: list[tuple[int, int, float]] = []
    k = 0
    while k < len(power):
        if power[k] <= 1e-7:
            k += 1
            continue
        a = k
        energy = 0.0
        while k < len(power) and power[k] > 1e-7:
            energy += float(power[k]) * DT_HOURS
            k += 1
        result.append((a, k, energy))
    return result


def fill_result_workbook(results: list[Q3Day], output_path: Path, *, volatile_price: bool) -> None:
    source_name = "result4-3.xlsx" if volatile_price else "result3.xlsx"
    source = ROOT / "CUMCM2026Problems" / "C题" / "附件" / "附件5" / source_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    wb = load_workbook(output_path)
    labels = interval_labels()

    plan_ws = wb["计划购电量"]
    adjust_ws = wb["调整购电量"]
    for ws in (plan_ws, adjust_ws):
        for j, label in enumerate(labels, start=2):
            ws.cell(row=1, column=j).value = label
        if ws.max_row > len(results) + 1:
            ws.delete_rows(len(results) + 2, ws.max_row - len(results) - 1)
    for i, result in enumerate(results, start=2):
        plan_ws.cell(row=i, column=1).value = result.day
        adjust_ws.cell(row=i, column=1).value = result.day
        for k in range(144):
            plan_ws.cell(row=i, column=k + 2).value = round(float(result.plan.grid_energy[k]), 6)
            adjust_ws.cell(row=i, column=k + 2).value = round(float(result.adjusted_grid_power[k] * DT_HOURS), 6)

    storage_rows: list[list[object]] = []
    for result in results:
        for block in range(6):
            a, b = block * 24, (block + 1) * 24
            storage_rows.append([
                result.day if block == 0 else None,
                f"{block * 4:02d}:00-{(block + 1) * 4:02d}:00",
                round(float(result.operation.charge_energy[a:b].sum()), 6),
                round(float(result.operation.discharge_energy[a:b].sum()), 6),
                "0:00" if block == 0 else ("24:00" if block == 1 else None),
                round(float(result.operation.soc[0] if block == 0 else result.operation.soc[-1]), 6)
                if block in (0, 1)
                else None,
            ])
    _clear_and_set_rows(wb["充放电量"], storage_rows)

    emergency_rows: list[list[object]] = []
    for result in results:
        segments = _emergency_segments(result.operation.emergency_power)
        if not segments:
            emergency_rows.append([result.day, None, None])
        else:
            for j, (a, b, energy) in enumerate(segments):
                start = labels[a].split("-")[0]
                end = labels[b - 1].split("-")[1]
                emergency_rows.append([result.day if j == 0 else None, f"{start}-{end}", round(energy, 6)])
    _clear_and_set_rows(wb["紧急购电量"], emergency_rows)
    wb.save(output_path)


def main() -> None:
    volatile = "--volatile" in sys.argv[1:]
    no_updates = "--no-updates" in sys.argv[1:]
    method = "zoh" if "--zoh" in sys.argv[1:] else "linear"
    results = simulate_q3(volatile_price=volatile, updates_used=not no_updates, forecast_method=method)
    out_dir = ROOT / "c2026_solution" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / ("result4-3.xlsx" if volatile else "result3.xlsx")
    fill_result_workbook(results, output, volatile_price=volatile)
    summary = {
        "model": "causal rolling PV-forecast adjustment with fixed original plan and real-time storage recourse",
        "volatile_price": volatile,
        "updates_used": not no_updates,
        "forecast_method": method,
        "n_report_days": len(results),
        "report_start": str(results[0].day),
        "report_end": str(results[-1].day),
        "planned_cost_yuan": float(sum(x.planned_cost for x in results)),
        "settlement_cost_yuan": float(sum(x.settlement_cost for x in results)),
        "adjustment_fee_yuan": float(sum(x.adjustment_fee for x in results)),
        "emergency_cost_yuan": float(sum(x.emergency_cost for x in results)),
        "total_cost_yuan": float(sum(x.total_cost for x in results)),
        "emergency_energy_kwh": float(sum(x.emergency_energy for x in results)),
        "days_with_emergency": int(sum(x.emergency_energy > 1e-7 for x in results)),
        "max_daily_emergency_kwh": float(max(x.emergency_energy for x in results)),
        "terminal_soc_convention": "daily terminal SOC equals the start SOC; January is simulated as warm-up but not reported",
        "information_set": "date-specific load known; PV forecasts are used causally at 00:00/06:00/12:00/18:00; actual PV only enters real-time recourse",
        "settlement_definition": "p*q_adjusted + 0.5*p*abs(q_adjusted-q_plan), interval by interval",
    }
    suffix = "_volatile" if volatile else ""
    if no_updates:
        suffix += "_no_updates"
    if method == "zoh":
        suffix += "_zoh"
    (out_dir / f"q3_summary{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
