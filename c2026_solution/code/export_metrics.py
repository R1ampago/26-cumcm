"""Export compact metrics used by the paper and figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from data_io import DT_HOURS, load_attachments, typical_arrays  # noqa: E402
from dispatch_lp import StorageParams, solve_dispatch  # noqa: E402
from q2_policy import simulate_q2  # noqa: E402
from q3_policy import _emergency_segments as q3_emergency_segments  # noqa: E402
from q3_policy import simulate_q3  # noqa: E402

TARGET_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def q1_metrics(data):
    p, l, pv = typical_arrays(data)
    d = solve_dispatch(l, pv, p, soc_initial=6000, soc_terminal=6000, params=StorageParams())
    selected = []
    for label, endpoint in zip(
        ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"],
        [610, 730, 850, 970, 1090, 1210],
    ):
        k = endpoint // 10 - 1
        selected.append({"time": label, "purchase_kwh": float(d.grid_power[k] * DT_HOURS)})
    blocks = []
    for b in range(6):
        a, z = 24 * b, 24 * (b + 1)
        blocks.append({
            "period": f"{b * 4:02d}:00-{(b + 1) * 4:02d}:00",
            "charge_kwh": float(d.charge_energy[a:z].sum()),
            "discharge_kwh": float(d.discharge_energy[a:z].sum()),
        })
    return {
        "selected": selected,
        "blocks": blocks,
        "grid_energy_kwh": float(d.grid_energy.sum()),
        "cost_yuan": float(d.objective),
        "soc_min_kwh": float(d.soc.min()),
        "soc_max_kwh": float(d.soc.max()),
        "charge_kwh": float(d.charge_energy.sum()),
        "discharge_kwh": float(d.discharge_energy.sum()),
        "dispatch": d,
    }


def write_day_metrics_q2(results, path: Path):
    rows = []
    for r in results:
        rows.append({
            "date": str(r.day),
            "planned_cost_yuan": r.planned_cost,
            "emergency_cost_yuan": r.emergency_cost,
            "total_cost_yuan": r.total_cost,
            "emergency_kwh": r.emergency_energy,
            "soc_min_kwh": float(r.recourse.soc.min()),
            "soc_max_kwh": float(r.recourse.soc.max()),
            "grid_kwh": float(r.plan.grid_energy.sum()),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def write_day_metrics_q3(results, path: Path):
    rows = []
    for r in results:
        rows.append({
            "date": str(r.day),
            "planned_cost_yuan": r.planned_cost,
            "settlement_cost_yuan": r.settlement_cost,
            "adjustment_fee_yuan": r.adjustment_fee,
            "emergency_cost_yuan": r.emergency_cost,
            "total_cost_yuan": r.total_cost,
            "emergency_kwh": r.emergency_energy,
            "planned_grid_kwh": float(r.plan.grid_energy.sum()),
            "adjusted_grid_kwh": float(r.adjusted_grid_power.sum() * DT_HOURS),
            "soc_min_kwh": float(r.operation.soc.min()),
            "soc_max_kwh": float(r.operation.soc.max()),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def target_q2(results, path: Path):
    lookup = {str(r.day): r for r in results}
    rows = []
    for day in TARGET_DATES:
        r = lookup[day]
        for a, b, energy in q3_emergency_segments(r.recourse.emergency_power):
            rows.append({"date": day, "time": f"{a * 10 // 60:02d}:{a * 10 % 60:02d}-{b * 10 // 60:02d}:{b * 10 % 60:02d}", "emergency_kwh": energy})
        if not q3_emergency_segments(r.recourse.emergency_power):
            rows.append({"date": day, "time": "none", "emergency_kwh": 0.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def target_q3(results, path: Path):
    lookup = {str(r.day): r for r in results}
    rows = []
    endpoints = [610, 730, 850, 970, 1090, 1210]
    labels = ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"]
    for day in TARGET_DATES:
        r = lookup[day]
        for label, endpoint in zip(labels, endpoints):
            k = endpoint // 10 - 1
            rows.append({
                "date": day,
                "time": label,
                "planned_kwh": float(r.plan.grid_power[k] * DT_HOURS),
                "adjusted_kwh": float(r.adjusted_grid_power[k] * DT_HOURS),
            })
        rows.append({
            "date": day,
            "time": "daily",
            "planned_kwh": float(r.plan.grid_energy.sum()),
            "adjusted_kwh": float(r.adjusted_grid_power.sum() * DT_HOURS),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    data = load_attachments(ROOT)
    out = ROOT / "c2026_solution" / "results"
    out.mkdir(parents=True, exist_ok=True)
    q1 = q1_metrics(data)
    q1_public = {k: v for k, v in q1.items() if k != "dispatch"}
    (out / "key_q1.json").write_text(json.dumps(q1_public, ensure_ascii=False, indent=2), encoding="utf-8")

    q2 = simulate_q2()
    q2v = simulate_q2(volatile_price=True)
    q3 = simulate_q3()
    q3v = simulate_q3(volatile_price=True)
    write_day_metrics_q2(q2, out / "day_metrics_q2.csv")
    write_day_metrics_q2(q2v, out / "day_metrics_q2_volatile.csv")
    write_day_metrics_q3(q3, out / "day_metrics_q3.csv")
    write_day_metrics_q3(q3v, out / "day_metrics_q3_volatile.csv")
    target_q2(q2, out / "target_q2_emergency.csv")
    target_q3(q3, out / "target_q3.csv")
    target_q2(q2v, out / "target_q4_q2_emergency.csv")
    target_q3(q3v, out / "target_q4_q3.csv")

    summary = {
        "q1": q1_public,
        "q2": {
            "planned_cost_yuan": sum(x.planned_cost for x in q2),
            "emergency_cost_yuan": sum(x.emergency_cost for x in q2),
            "total_cost_yuan": sum(x.total_cost for x in q2),
            "emergency_kwh": sum(x.emergency_energy for x in q2),
        },
        "q2_volatile": {
            "planned_cost_yuan": sum(x.planned_cost for x in q2v),
            "emergency_cost_yuan": sum(x.emergency_cost for x in q2v),
            "total_cost_yuan": sum(x.total_cost for x in q2v),
            "emergency_kwh": sum(x.emergency_energy for x in q2v),
        },
        "q3": {
            "planned_cost_yuan": sum(x.planned_cost for x in q3),
            "settlement_cost_yuan": sum(x.settlement_cost for x in q3),
            "adjustment_fee_yuan": sum(x.adjustment_fee for x in q3),
            "emergency_cost_yuan": sum(x.emergency_cost for x in q3),
            "total_cost_yuan": sum(x.total_cost for x in q3),
            "emergency_kwh": sum(x.emergency_energy for x in q3),
        },
        "q3_volatile": {
            "planned_cost_yuan": sum(x.planned_cost for x in q3v),
            "settlement_cost_yuan": sum(x.settlement_cost for x in q3v),
            "adjustment_fee_yuan": sum(x.adjustment_fee for x in q3v),
            "emergency_cost_yuan": sum(x.emergency_cost for x in q3v),
            "total_cost_yuan": sum(x.total_cost for x in q3v),
            "emergency_kwh": sum(x.emergency_energy for x in q3v),
        },
    }
    (out / "paper_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
