"""Generate the high-information figures used in the paper."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

from data_io import load_attachments, typical_arrays  # noqa: E402
from dispatch_lp import StorageParams, solve_dispatch  # noqa: E402
from q3_policy import simulate_q3  # noqa: E402

OUT = ROOT / "c2026_solution" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"blue": "#2F5D8C", "teal": "#4C9F9A", "orange": "#C77C4B", "red": "#A94A4A", "purple": "#756A9A", "gray": "#6B7280", "ink": "#263238"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300})


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 4.2); ax.axis("off")
    boxes = [
        (0.2, 2.45, 1.7, 1.0, "Raw inputs", "A1–A4\n10-min + hourly" , COLORS["blue"]),
        (2.55, 2.45, 1.8, 1.0, "Audit", "time / units /\ninformation set", COLORS["purple"]),
        (5.0, 3.0, 2.0, 1.0, "Q1", "representative-day\nLP benchmark", COLORS["teal"]),
        (5.0, 1.75, 2.0, 1.0, "Q2", "day-ahead plan +\nreal-time recourse", COLORS["orange"]),
        (7.75, 2.75, 2.0, 1.0, "Q3", "causal forecast\nupdates", COLORS["teal"]),
        (7.75, 1.15, 2.0, 1.0, "Q4", "volatile-price\nrecalculation", COLORS["red"]),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor="#F7F9FB", edgecolor=color, linewidth=2))
        ax.text(x + .12, y + .64, title, color=color, weight="bold", va="center")
        ax.text(x + .12, y + .31, subtitle, color=COLORS["ink"], va="center", linespacing=1.25)
    arrows = [((1.9, 2.95), (2.55, 2.95)), ((4.35, 3.0), (5.0, 3.45)), ((4.35, 2.8), (5.0, 2.25)), ((7.0, 3.45), (7.75, 3.25)), ((7.0, 2.15), (7.75, 1.65)), ((8.75, 2.75), (8.75, 2.15))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=COLORS["gray"]))
    ax.text(0.25, .45, "Validation loop: physical balance · SOC/power limits · forecast error · emergency cost · settlement cost", color=COLORS["gray"])
    save(fig, "overview_model.pdf")


def q1_figures(data):
    p, l, pv = typical_arrays(data)
    t = np.arange(144) / 6
    fig, ax1 = plt.subplots(figsize=(8.6, 3.4))
    ax1.plot(t, l, color=COLORS["blue"], lw=1.8, label="Load (kW)")
    ax1.plot(t, pv, color=COLORS["teal"], lw=1.8, label="PV forecast (kW)")
    ax1.set_xlabel("Time (h)"); ax1.set_ylabel("Power (kW)"); ax1.set_xlim(0, 24); ax1.grid(axis="y", alpha=.2)
    ax2 = ax1.twinx(); ax2.plot(t, p, color=COLORS["orange"], lw=1.5, label="Price (yuan/kWh)"); ax2.set_ylabel("Price (yuan/kWh)")
    h1, lab1 = ax1.get_legend_handles_labels(); h2, lab2 = ax2.get_legend_handles_labels(); ax1.legend(h1+h2, lab1+lab2, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.17), frameon=False)
    save(fig, "q1_representative_profile.pdf")

    d = solve_dispatch(l, pv, p, soc_initial=6000, soc_terminal=6000, params=StorageParams())
    fig, ax1 = plt.subplots(figsize=(8.6, 3.8))
    ax1.step(t, d.grid_power, where="post", color=COLORS["blue"], lw=1.2, label="Grid purchase")
    ax1.step(t, d.charge_power, where="post", color=COLORS["teal"], lw=1.1, label="Charge")
    ax1.step(t, d.discharge_power, where="post", color=COLORS["orange"], lw=1.1, label="Discharge")
    ax1.set_xlabel("Time (h)"); ax1.set_ylabel("Power (kW)"); ax1.set_xlim(0, 24); ax1.grid(axis="y", alpha=.2)
    ax2 = ax1.twinx(); ax2.plot(np.r_[t, 24], d.soc, color=COLORS["purple"], lw=1.8, label="SOC")
    ax2.set_ylabel("SOC (kWh)"); ax2.set_ylim(0, 12000)
    h1, lab1 = ax1.get_legend_handles_labels(); h2, lab2 = ax2.get_legend_handles_labels(); ax1.legend(h1+h2, lab1+lab2, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.17), frameon=False)
    save(fig, "q1_optimal_dispatch.pdf")


def forecast_heatmap(data):
    rows = []
    for _, row in data.forecasts.iterrows():
        d = pd.Timestamp(row["日期"]).date(); release = int(str(row["预报时刻"]).split(":")[0])
        for h in range(1, 25):
            target = pd.Timestamp(d) + pd.Timedelta(hours=release // 1 + h)
            # release is in hours; normalize target robustly
            target = pd.Timestamp(d) + pd.Timedelta(hours=release + h)
            actual_day = target.normalize(); minute = int((target - actual_day).total_seconds() // 60)
            if minute == 0 or actual_day not in data.pv_actual.index or minute not in data.pv_actual.columns:
                continue
            actual = float(data.pv_actual.loc[actual_day, minute]); forecast = float(row[f"预报{h}小时"])
            rows.append((release, h, forecast - actual))
    df = pd.DataFrame(rows, columns=["release", "h", "error"])
    piv = df.pivot_table(index="release", columns="h", values="error", aggfunc="mean").reindex(index=[0, 6, 12, 18], columns=range(1, 25))
    fig, ax = plt.subplots(figsize=(9.2, 2.7))
    vmax = np.nanmax(np.abs(piv.to_numpy()))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=LinearSegmentedColormap.from_list("err", ["#4C9F9A", "#F7F9FB", "#C77C4B"]), vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(24)); ax.set_xticklabels(range(1, 25)); ax.set_yticks(range(4)); ax.set_yticklabels(["00:00", "06:00", "12:00", "18:00"])
    ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("Release time"); cb = fig.colorbar(im, ax=ax, pad=.02); cb.set_label("Forecast error (kW)")
    save(fig, "forecast_error_heatmap.pdf")


def seasonal_and_updates():
    q2 = pd.read_csv(ROOT / "c2026_solution" / "results" / "day_metrics_q2.csv", parse_dates=["date"])
    q3 = pd.read_csv(ROOT / "c2026_solution" / "results" / "day_metrics_q3.csv", parse_dates=["date"])
    q3no = json.loads((ROOT / "c2026_solution" / "results" / "q3_summary_no_updates.json").read_text())
    q3yes = json.loads((ROOT / "c2026_solution" / "results" / "q3_summary.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.6), sharex=True)
    axes[0].plot(q2["date"], q2["emergency_kwh"], color=COLORS["red"], lw=.65, alpha=.75, label="Q2: typical PV forecast")
    axes[0].plot(q3["date"], q3["emergency_kwh"], color=COLORS["teal"], lw=.75, alpha=.8, label="Q3: rolling PV forecast")
    axes[0].set_ylabel("Emergency energy (kWh/day)"); axes[0].legend(frameon=False, ncol=2); axes[0].grid(axis="y", alpha=.2)
    axes[1].plot(q2["date"], q2["total_cost_yuan"] / 1000, color=COLORS["red"], lw=.65, alpha=.75, label="Q2 total cost")
    axes[1].plot(q3["date"], q3["total_cost_yuan"] / 1000, color=COLORS["teal"], lw=.75, alpha=.8, label="Q3 total cost")
    axes[1].set_ylabel("Total cost (10³ yuan)"); axes[1].set_xlabel("Date"); axes[1].grid(axis="y", alpha=.2)
    save(fig, "q2_q3_daily_comparison.pdf")

    labels = ["No updates", "Updates"]
    emergency = [q3no["emergency_energy_kwh"], q3yes["emergency_energy_kwh"]]
    total = [q3no["total_cost_yuan"] / 1e6, q3yes["total_cost_yuan"] / 1e6]
    x = np.arange(2)
    fig, ax1 = plt.subplots(figsize=(6.8, 3.7))
    b1 = ax1.bar(x - .18, np.asarray(emergency) / 1e6, .36, color=COLORS["orange"], label="Emergency energy (10⁶ kWh)")
    ax1.set_ylabel("Emergency energy (10⁶ kWh)"); ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.grid(axis="y", alpha=.2)
    ax2 = ax1.twinx(); b2 = ax2.bar(x + .18, total, .36, color=COLORS["blue"], label="Total cost (10⁶ yuan)"); ax2.set_ylabel("Total cost (10⁶ yuan)")
    for bar in b1:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax1.legend([b1, b2], ["Emergency energy", "Total cost"], frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=2)
    save(fig, "q3_update_value.pdf")


def example_day():
    results = simulate_q3()
    result = {str(x.day): x for x in results}["2025-03-20"]
    data = load_attachments(ROOT); load = data.load.loc["2025-03-20"].to_numpy(float); pv = data.pv_actual.loc["2025-03-20"].to_numpy(float)
    t = np.arange(144) / 6
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.0), sharex=True)
    axes[0].plot(t, load, color=COLORS["ink"], lw=1.0, label="Load")
    axes[0].plot(t, pv, color=COLORS["teal"], lw=1.0, label="Actual PV")
    axes[0].set_ylabel("Power (kW)"); axes[0].legend(frameon=False, ncol=2); axes[0].grid(axis="y", alpha=.2)
    axes[1].step(t, result.plan.grid_power, where="post", color=COLORS["blue"], lw=1.1, label="Original plan")
    axes[1].step(t, result.adjusted_grid_power, where="post", color=COLORS["orange"], lw=1.1, label="Final adjustment")
    axes[1].set_ylabel("Grid power (kW)"); axes[1].legend(frameon=False, ncol=2); axes[1].grid(axis="y", alpha=.2)
    axes[2].plot(np.r_[t, 24], result.operation.soc, color=COLORS["purple"], lw=1.5, label="SOC")
    axes[2].fill_between(t, 0, result.operation.emergency_power, color=COLORS["red"], alpha=.32, step="post", label="Emergency power")
    axes[2].set_ylabel("SOC / emergency (kWh / kW)"); axes[2].set_xlabel("Time (h)"); axes[2].set_ylim(bottom=0); axes[2].legend(frameon=False, ncol=2); axes[2].grid(axis="y", alpha=.2)
    save(fig, "q3_example_day.pdf")


def q4_bar():
    files = ["q2_summary.json", "q2_volatile_summary.json", "q3_summary.json", "q3_summary_volatile.json"]
    vals = [json.loads((ROOT / "c2026_solution" / "results" / f).read_text())["total_cost_yuan"] / 1e6 for f in files]
    labels = ["Q2 fixed", "Q2 volatile", "Q3 fixed", "Q3 volatile"]
    colors = [COLORS["blue"], COLORS["red"], COLORS["teal"], COLORS["orange"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Total cost (10⁶ yuan)"); ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, vals): ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    save(fig, "q4_price_comparison.pdf")


def main():
    data = load_attachments(ROOT)
    overview(); q1_figures(data); forecast_heatmap(data); seasonal_and_updates(); example_day(); q4_bar()
    print("generated", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
