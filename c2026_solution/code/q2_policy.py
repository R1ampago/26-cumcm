"""Question 2 policy under an explicit day-ahead information set.

Main information convention:
- the date-specific load profile is available at 00:00;
- the only PV forecast supplied before question 3 is the representative profile
  in attachment 1;
- the actual PV profile in attachment 2 is used only for real-time evaluation;
- the planned grid purchases are fixed, while storage can dispatch in real time;
- a daily terminal SOC equal to the start SOC avoids consuming inventory merely
  because the finite horizon ends at 24:00.

This convention is configurable and recorded in the generated metadata because
question 2 does not explicitly spell out the forecast information set.
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

from data_io import DT_HOURS, interval_labels, load_attachments, typical_arrays  # noqa: E402
from dispatch_lp import Dispatch, StorageParams, solve_dispatch  # noqa: E402
from recourse_lp import Recourse, solve_fixed_grid_recourse  # noqa: E402


@dataclass
class Q2Day:
    day: date
    plan: Dispatch
    recourse: Recourse
    planned_cost: float
    emergency_cost: float

    @property
    def emergency_energy(self) -> float:
        return float(self.recourse.emergency_energy.sum())

    @property
    def total_cost(self) -> float:
        return self.planned_cost + self.emergency_cost


def simulate_q2(
    *,
    root: Path = ROOT,
    start: str = "2025-01-01",
    report_start: str = "2025-02-01",
    report_end: str = "2025-12-31",
    volatile_price: bool = False,
    params: StorageParams = StorageParams(),
) -> list[Q2Day]:
    data = load_attachments(root)
    typical_price, _, typical_pv = typical_arrays(data)
    d = date.fromisoformat(start)
    report_start_date = date.fromisoformat(report_start)
    end_date = date.fromisoformat(report_end)
    answer: list[Q2Day] = []
    while d <= end_date:
        load, actual_pv, volatile = (
            data.load.loc[str(d)].to_numpy(float),
            data.pv_actual.loc[str(d)].to_numpy(float),
            data.price_volatile.loc[str(d)].to_numpy(float),
        )
        price = volatile if volatile_price else typical_price
        # Load is the date-specific known schedule; PV is represented by the
        # only generic day-ahead forecast provided before Q3.
        plan = solve_dispatch(
            load,
            typical_pv,
            price,
            soc_initial=params.initial_soc,
            soc_terminal=params.initial_soc,
            params=params,
        )
        recourse = solve_fixed_grid_recourse(
            plan.grid_power,
            load,
            actual_pv,
            price,
            soc_initial=params.initial_soc,
            soc_terminal=params.initial_soc,
            params=params,
            emergency_multiplier=5.0,
        )
        planned_cost = float(np.sum(plan.grid_energy * price))
        result = Q2Day(d, plan, recourse, planned_cost, recourse.emergency_cost)
        if d >= report_start_date:
            answer.append(result)
        d += timedelta(days=1)
    if not answer or answer[0].day != report_start_date or answer[-1].day != end_date:
        raise AssertionError("Q2 report date range is not complete")
    return answer


def _clear_and_set_rows(ws, rows: list[list[object]], widths: dict[str, float] | None = None) -> None:
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for r in rows:
        ws.append(r)
    if widths:
        for col, width in widths.items():
            ws.column_dimensions[col].width = width


def _emergency_segments(power: np.ndarray) -> list[tuple[int, int, float]]:
    segments: list[tuple[int, int, float]] = []
    k = 0
    while k < len(power):
        if power[k] <= 1e-7:
            k += 1
            continue
        a = k
        total = 0.0
        while k < len(power) and power[k] > 1e-7:
            total += float(power[k]) * DT_HOURS
            k += 1
        segments.append((a, k, total))
    return segments


def fill_result_workbook(results: list[Q2Day], output_path: Path, *, volatile_price: bool) -> None:
    source_name = "result4-2.xlsx" if volatile_price else "result2.xlsx"
    source = ROOT / "CUMCM2026Problems" / "C题" / "附件" / "附件5" / source_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    wb = load_workbook(output_path)

    plan_ws = wb["计划购电量"]
    headers = interval_labels()
    if plan_ws.max_column < 145:
        raise ValueError("plan template has fewer than 144 interval columns")
    for j, label in enumerate(headers, start=2):
        plan_ws.cell(row=1, column=j).value = label
    for i, result in enumerate(results, start=2):
        plan_ws.cell(row=i, column=1).value = result.day
        for k, value in enumerate(result.plan.grid_energy, start=2):
            plan_ws.cell(row=i, column=k).value = round(float(value), 6)
    # Remove any preformatted placeholder rows after the report range.
    if plan_ws.max_row > len(results) + 1:
        plan_ws.delete_rows(len(results) + 2, plan_ws.max_row - len(results) - 1)

    storage_ws = wb["充放电量"]
    storage_rows: list[list[object]] = []
    for result in results:
        charge = result.recourse.charge_energy
        discharge = result.recourse.discharge_energy
        for block in range(6):
            a, b = block * 24, (block + 1) * 24
            storage_rows.append([
                result.day if block == 0 else None,
                f"{block * 4:02d}:00-{(block + 1) * 4:02d}:00",
                round(float(charge[a:b].sum()), 6),
                round(float(discharge[a:b].sum()), 6),
                "0:00" if block == 0 else ("24:00" if block == 1 else None),
                round(float(result.recourse.soc[0] if block == 0 else result.recourse.soc[-1]), 6)
                if block in (0, 1)
                else None,
            ])
    _clear_and_set_rows(storage_ws, storage_rows)

    emergency_ws = wb["紧急购电量"]
    emergency_rows: list[list[object]] = []
    labels = headers
    for result in results:
        segments = _emergency_segments(result.recourse.emergency_power)
        if not segments:
            emergency_rows.append([result.day, None, None])
            continue
        for j, (a, b, energy) in enumerate(segments):
            end_label = labels[b - 1].split("-")[1]
            start_label = labels[a].split("-")[0]
            emergency_rows.append([result.day if j == 0 else None, f"{start_label}-{end_label}", round(energy, 6)])
    _clear_and_set_rows(emergency_ws, emergency_rows)
    wb.save(output_path)


def main() -> None:
    volatile = "--volatile" in sys.argv[1:]
    results = simulate_q2(volatile_price=volatile)
    out_dir = ROOT / "c2026_solution" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / ("result4-2.xlsx" if volatile else "result2.xlsx")
    fill_result_workbook(results, output, volatile_price=volatile)
    summary = {
        "model": "Q2 day-ahead typical-PV forecast with fixed planned grid purchases and real-time storage recourse",
        "volatile_price": volatile,
        "n_report_days": len(results),
        "report_start": str(results[0].day),
        "report_end": str(results[-1].day),
        "planned_cost_yuan": float(sum(x.planned_cost for x in results)),
        "emergency_cost_yuan": float(sum(x.emergency_cost for x in results)),
        "total_cost_yuan": float(sum(x.total_cost for x in results)),
        "emergency_energy_kwh": float(sum(x.emergency_energy for x in results)),
        "days_with_emergency": int(sum(x.emergency_energy > 1e-7 for x in results)),
        "max_daily_emergency_kwh": float(max(x.emergency_energy for x in results)),
        "dt_hours": DT_HOURS,
        "terminal_soc_convention": "daily terminal SOC equals the start SOC; January is simulated as warm-up but not reported",
        "information_set": "date-specific load known; generic PV forecast from attachment 1; actual PV used only for recourse evaluation",
    }
    name = "q2_volatile_summary.json" if volatile else "q2_summary.json"
    (out_dir / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
