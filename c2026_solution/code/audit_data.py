"""Reproducible first-pass data audit for the 2026 C problem."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from data_io import load_attachments, typical_arrays  # noqa: E402


def main() -> None:
    data = load_attachments(ROOT)
    price1, load1, pv1 = typical_arrays(data)
    summaries: dict[str, object] = {
        "typical": {
            "shape": list(data.typical.shape),
            "columns": [str(x) for x in data.typical.columns],
            "price_min": float(price1.min()),
            "price_max": float(price1.max()),
            "load_min": float(load1.min()),
            "load_max": float(load1.max()),
            "pv_min": float(pv1.min()),
            "pv_max": float(pv1.max()),
        },
        "load_actual": {
            "shape": list(data.load.shape),
            "date_min": str(data.load.index.min().date()),
            "date_max": str(data.load.index.max().date()),
            "missing": int(data.load.isna().sum().sum()),
            "duplicate_dates": int(data.load.index.duplicated().sum()),
        },
        "pv_actual": {
            "shape": list(data.pv_actual.shape),
            "missing": int(data.pv_actual.isna().sum().sum()),
            "negative": int((data.pv_actual.to_numpy() < 0).sum()),
            "zero_count": int((data.pv_actual.to_numpy() == 0).sum()),
        },
        "price_volatile": {
            "shape": list(data.price_volatile.shape),
            "missing": int(data.price_volatile.isna().sum().sum()),
            "min": float(data.price_volatile.to_numpy().min()),
            "max": float(data.price_volatile.to_numpy().max()),
        },
        "forecasts": {
            "shape": list(data.forecasts.shape),
            "dates": int(data.forecasts["日期"].nunique()),
            "release_times": sorted(data.forecasts["预报时刻"].astype(str).unique().tolist()),
            "missing": int(data.forecasts.isna().sum().sum()),
            "negative": int((data.forecasts.iloc[:, 2:].to_numpy() < 0).sum()),
        },
    }
    # quantify the representative-day relation without replacing the supplied values
    load_mean = data.load.to_numpy().mean(axis=0)
    pv_mean = data.pv_actual.to_numpy().mean(axis=0)
    price_mean = data.price_volatile.to_numpy().mean(axis=0)
    summaries["typical_vs_annual_mean"] = {
        "load_max_abs": float(np.max(np.abs(load1 - load_mean))),
        "pv_max_abs": float(np.max(np.abs(pv1 - pv_mean))),
        "price_max_abs": float(np.max(np.abs(price1 - price_mean))),
    }
    out = ROOT / "c2026_solution" / "audit" / "data_audit.json"
    out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
