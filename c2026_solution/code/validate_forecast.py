"""Out-of-sample-style audit of hourly PV forecasts and 10-minute mappings."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from data_io import load_attachments  # noqa: E402
from q3_policy import forecast_to_ten_minute  # noqa: E402


def main() -> None:
    data = load_attachments(ROOT)
    point_rows: list[dict[str, float | int | str]] = []
    for _, row in data.forecasts.iterrows():
        d = pd.Timestamp(row["日期"]).date()
        release = int(str(row["预报时刻"]).split(":")[0]) * 60
        for h in range(1, 25):
            target = pd.Timestamp(d) + pd.Timedelta(minutes=release + 60 * h)
            actual_day = target.normalize()
            if actual_day not in data.pv_actual.index:
                continue
            minute = int((target - actual_day).total_seconds() // 60)
            if minute == 0:
                # The source grid starts at 00:10 rather than 00:00.
                continue
            if minute not in data.pv_actual.columns:
                continue
            actual = float(data.pv_actual.loc[actual_day, minute])
            forecast = float(row[f"预报{h}小时"])
            point_rows.append({"release": release, "horizon": h, "forecast": forecast, "actual": actual, "error": forecast - actual})
    point = pd.DataFrame(point_rows)
    result: dict[str, object] = {"hourly_endpoint": {}}
    for release, group in point.groupby("release"):
        err = group["error"].to_numpy()
        result["hourly_endpoint"][str(release)] = {
            "n": int(len(group)),
            "bias_kw": float(err.mean()),
            "mae_kw": float(np.abs(err).mean()),
            "rmse_kw": float(np.sqrt(np.mean(err**2))),
        }

    ten_minute: dict[str, object] = {}
    for method in ["linear", "zoh"]:
        method_rows: list[float] = []
        by_release: dict[int, list[float]] = {}
        for d in pd.date_range("2025-01-01", "2025-12-31", freq="D"):
            day = d.date()
            actual = data.pv_actual.loc[d].to_numpy(float)
            for release in [0, 360, 720, 1080]:
                predicted = forecast_to_ten_minute(data, day, release, method=method)
                start = release // 10
                errors = (predicted[start:] - actual[start:]).tolist()
                method_rows.extend(errors)
                by_release.setdefault(release, []).extend(errors)
        all_error = np.asarray(method_rows)
        ten_minute[method] = {
            "n": int(len(all_error)),
            "mae_kw": float(np.abs(all_error).mean()),
            "rmse_kw": float(np.sqrt(np.mean(all_error**2))),
            "by_release": {
                str(r): {
                    "mae_kw": float(np.abs(np.asarray(e)).mean()),
                    "rmse_kw": float(np.sqrt(np.mean(np.asarray(e) ** 2))),
                }
                for r, e in by_release.items()
            },
        }
    result["ten_minute_mapping"] = ten_minute
    out = ROOT / "c2026_solution" / "audit" / "forecast_validation.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
