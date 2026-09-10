"""Data loading and time-index normalization for 2026 C problem.

The source workbooks are deliberately read without modifying them.  All internal
arrays use 144 ten-minute intervals whose index k denotes [(k-1)*10, k*10].
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DT_HOURS = 1.0 / 6.0
N_INTERVALS = 144


@dataclass(frozen=True)
class Attachments:
    typical: pd.DataFrame
    load: pd.DataFrame
    pv_actual: pd.DataFrame
    price_volatile: pd.DataFrame
    forecasts: pd.DataFrame


def _source_dir(repo_root: Path | str) -> Path:
    return Path(repo_root) / "CUMCM2026Problems" / "C题" / "附件"


def _minutes(label: object) -> int:
    """Convert source time labels to minutes after the day's 00:00.

    Source labels include datetime.time values, strings such as ``23:50`` and
    the special ``0:00+1`` label for the interval ending at 24:00.
    """
    text = str(label).strip()
    if "0:00+1" in text:
        return 1440
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"Unrecognised time label: {label!r}")
    return int(parts[0]) * 60 + int(parts[1])


def interval_minutes(labels: Iterable[object]) -> np.ndarray:
    result = np.asarray([_minutes(x) for x in labels], dtype=int)
    if len(result) != N_INTERVALS or not np.all(np.diff(result) == 10):
        raise ValueError("Expected 144 consecutive ten-minute interval-end labels")
    return result


def interval_labels() -> list[str]:
    labels: list[str] = []
    for end in range(10, 1441, 10):
        start = end - 10
        start_text = f"{start // 60:02d}:{start % 60:02d}"
        end_text = "24:00" if end == 1440 else f"{end // 60:02d}:{end % 60:02d}"
        labels.append(f"{start_text}-{end_text}")
    return labels


def _read_wide(path: Path, sheet_name: str) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frame = pd.read_excel(path, sheet_name=sheet_name)
    if frame.shape[1] != N_INTERVALS + 1:
        raise ValueError(f"{path.name}/{sheet_name}: expected 145 columns, got {frame.shape[1]}")
    dates = pd.to_datetime(frame.iloc[:, 0], errors="raise")
    minutes = interval_minutes(frame.columns[1:])
    values = frame.iloc[:, 1:].to_numpy(dtype=float)
    if values.shape[1] != N_INTERVALS or not np.isfinite(values).all():
        raise ValueError(f"{path.name}/{sheet_name}: non-finite or wrong-shape values")
    return dates, minutes, values


def load_attachments(repo_root: Path | str) -> Attachments:
    source = _source_dir(repo_root)
    typical = pd.read_excel(source / "附件1.xlsx", sheet_name="Sheet1")
    if typical.shape != (N_INTERVALS, 4):
        raise ValueError(f"附件1 expected (144, 4), got {typical.shape}")
    interval_minutes(typical.iloc[:, 0])
    if typical.iloc[:, 1:].isna().any().any():
        raise ValueError("附件1 contains missing values")

    load_dates, minutes, load = _read_wide(source / "附件2.xlsx", "小区负载")
    pv_dates, minutes_pv, pv_actual = _read_wide(source / "附件2.xlsx", "光伏发电实际功率")
    price_dates, minutes_price, price = _read_wide(source / "附件4.xlsx", "Sheet1")
    if not (minutes.tolist() == minutes_pv.tolist() == minutes_price.tolist()):
        raise ValueError("Attachment 2/4 time columns are not aligned")
    if not (load_dates.equals(pv_dates) and load_dates.equals(price_dates)):
        raise ValueError("Attachment 2/4 date rows are not aligned")

    forecasts = pd.read_excel(source / "附件3.xlsx", sheet_name="Sheet1")
    if forecasts.shape[1] != 26 or forecasts.shape[0] != 1460:
        raise ValueError(f"附件3 expected (1460, 26), got {forecasts.shape}")
    # The date is written only once per group of four release records.
    forecasts = forecasts.copy()
    forecasts["日期"] = forecasts["日期"].replace("", np.nan).ffill()
    forecasts["日期"] = pd.to_datetime(forecasts["日期"], errors="raise")
    if forecasts["日期"].nunique() != 365:
        raise ValueError("附件3 does not contain exactly 365 forecast dates")
    if set(forecasts["预报时刻"].astype(str)) != {"0:00", "6:00", "12:00", "18:00"}:
        raise ValueError("Unexpected forecast release times")
    fvalues = forecasts.iloc[:, 2:].to_numpy(dtype=float)
    if fvalues.shape != (1460, 24) or not np.isfinite(fvalues).all() or (fvalues < 0).any():
        raise ValueError("附件3 contains invalid forecast values")

    return Attachments(
        typical=typical,
        load=pd.DataFrame(load, index=load_dates, columns=minutes),
        pv_actual=pd.DataFrame(pv_actual, index=pv_dates, columns=minutes),
        price_volatile=pd.DataFrame(price, index=price_dates, columns=minutes),
        forecasts=forecasts,
    )


def typical_arrays(data: Attachments) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Q1 representative-day price, load and PV forecast arrays."""
    return tuple(data.typical.iloc[:, j].to_numpy(dtype=float) for j in (1, 2, 3))


def day_arrays(data: Attachments, day: str | date | datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stamp = pd.Timestamp(day)
    try:
        load = data.load.loc[stamp].to_numpy(dtype=float)
        pv = data.pv_actual.loc[stamp].to_numpy(dtype=float)
        price = data.price_volatile.loc[stamp].to_numpy(dtype=float)
    except KeyError as exc:
        raise KeyError(f"Date {stamp.date()} is absent from attachment 2/4") from exc
    return load, pv, price


def forecast_rows(data: Attachments, day: str | date | datetime) -> pd.DataFrame:
    stamp = pd.Timestamp(day).normalize()
    rows = data.forecasts.loc[data.forecasts["日期"].eq(stamp)].copy()
    if len(rows) != 4:
        raise ValueError(f"Expected 4 forecast releases for {stamp.date()}, got {len(rows)}")
    return rows.reset_index(drop=True)


def forecast_targets(data: Attachments, day: str | date | datetime) -> dict[tuple[int, int], float]:
    """Map (release minute, target absolute minute) to the forecasted PV power.

    A target minute may be greater than 1440; it then belongs to the next day.
    The function preserves the forecast's hourly time scale; conversion to the
    ten-minute control grid is a model choice and is deliberately not hidden here.
    """
    rows = forecast_rows(data, day)
    result: dict[tuple[int, int], float] = {}
    for _, row in rows.iterrows():
        release = _minutes(row["预报时刻"])
        for h in range(1, 25):
            result[(release, release + 60 * h)] = float(row[f"预报{h}小时"])
    return result


def actual_pv_as_hourly_endpoint(data: Attachments, day: str | date | datetime) -> dict[int, float]:
    """Return actual PV at the 10-minute grid points as a minute->power map."""
    stamp = pd.Timestamp(day)
    row = data.pv_actual.loc[stamp]
    return {int(m): float(v) for m, v in row.items()}
