"""Matplotlib figures for reports (static PNG/PDF). The UI renders the same data interactively.

Colour roles follow a fixed categorical order (blue = baseline, orange =
optimised/validated, aqua = predicted, yellow = further series); predicted
curves are always dashed and labeled PREDICTED.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from metasurface_optimizer.electromagnetics.rcs import to_dbsm  # noqa: E402
from metasurface_optimizer.geometry.geometry_generator import CellGeometry  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Applied per figure (not globally) so importing this module never changes other matplotlib users.
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "font.size": 9, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
}


def _styled(fn):
    import functools

    @functools.wraps(fn)
    def wrap(*a, **k):
        with plt.rc_context(STYLE):
            return fn(*a, **k)
    return wrap


@_styled
def plot_geometry(geom: CellGeometry, ax=None, title: str = "", color: str = SERIES[0]):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
    px, py = geom.period_x, geom.period_y
    ax.add_patch(plt.Rectangle((-px / 2, -py / 2), px, py, fill=True, facecolor="#f1efe8", edgecolor=INK2, lw=1))
    for poly in getattr(geom.top_metal(), "geoms", []):
        ax.add_patch(MplPolygon(np.asarray(poly.exterior.coords), closed=True, facecolor=color, edgecolor=INK, lw=0.8))
        for hole in poly.interiors:
            ax.add_patch(MplPolygon(np.asarray(hole.coords), closed=True, facecolor="#f1efe8", edgecolor=INK, lw=0.8))
    ax.set_xlim(-px / 2 * 1.05, px / 2 * 1.05); ax.set_ylim(-py / 2 * 1.05, py / 2 * 1.05)
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if own:
        return fig


@_styled
def plot_baseline_vs_optimized(baseline_resp, optimized_resp, predicted: Optional[Dict] = None, band=None,
                               labels=("Baseline", "Optimized"), pol_inc=None):
    """Four panels: |S11| (dB), S11 phase, specular RCS reduction, bistatic cut at the band centre."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    pols = baseline_resp.sparams.polarizations if baseline_resp.sparams else []
    key_pol = next((p for p in pols if not p.endswith("_B")), pols[0] if pols else None)
    for resp, lab, col in ((baseline_resp, labels[0], SERIES[0]), (optimized_resp, labels[1], SERIES[1])):
        if resp is None:
            continue
        tag = lab
        if resp.sparams is not None and key_pol:
            f = resp.sparams.frequency / 1e9
            for suffix, ls in (("", "-"), ("_B", ":")):
                k = key_pol + suffix
                if resp.sparams.has(1, 1, k):
                    axes[0, 0].plot(f, resp.sparams.db(1, 1, k), color=col, ls=ls, label=f"{tag} {k}")
                    axes[0, 1].plot(f, resp.sparams.phase(1, 1, k), color=col, ls=ls, label=f"{tag} {k}")
        if resp.rcs is not None and resp.rcs.reference is not None:
            r = resp.rcs
            pi, ii = (0, 0) if pol_inc is None else (r.pol_index(pol_inc[0]), r.inc_index(pol_inc[1]))
            red = to_dbsm(r.specular()[pi, ii]) - to_dbsm(r.reference[pi, ii])
            axes[1, 0].plot(r.frequency / 1e9, red, color=col, label=tag)
            if r.bistatic is not None:
                fc = 0.5 * (band[0] + band[1]) if band else r.frequency.mean()
                k = int(np.argmin(np.abs(r.frequency - fc)))
                j0 = int(np.argmin(np.abs(r.phi_s - r.phi_inc)))
                j1 = int(np.argmin(np.abs(np.mod(r.phi_s - r.phi_inc - 180, 360))))
                ang = np.concatenate([-r.theta_s[::-1], r.theta_s])
                cut = np.concatenate([r.bistatic[pi, ii, k, ::-1, j1], r.bistatic[pi, ii, k, :, j0]])
                axes[1, 1].plot(ang, to_dbsm(cut), color=col, label=f"{tag} @ {r.frequency[k] / 1e9:.1f} GHz")
    if predicted:
        f = np.asarray(predicted["frequency"]) / 1e9
        for name, vals in predicted.get("rcs_reduction_db", {}).items():
            sd = predicted.get("rcs_reduction_db_std", {}).get(name)
            axes[1, 0].plot(f, vals, color=SERIES[2], ls="--", label=f"PREDICTED {name.replace('RCSred_', '')}")
            if sd is not None:
                axes[1, 0].fill_between(f, vals - 2 * sd, vals + 2 * sd, color=SERIES[2], alpha=0.15, lw=0)
            break
    for ax in axes.ravel():
        if band:
            ax_is_freq = ax is not axes[1, 1]
            if ax_is_freq:
                ax.axvspan(band[0] / 1e9, band[1] / 1e9, color=GRID, alpha=0.5, lw=0)
    axes[0, 0].set(xlabel="Frequency (GHz)", ylabel="|S11| (dB)", title="Reflection magnitude (solid: state A, dotted: state B)")
    axes[0, 1].set(xlabel="Frequency (GHz)", ylabel="Phase S11 (deg)", title="Reflection phase")
    axes[1, 0].set(xlabel="Frequency (GHz)", ylabel="sigma_spec / sigma_ref (dB)", title="Specular RCS reduction vs. reference plate")
    axes[1, 1].set(xlabel="theta_s (deg), phi-cut through incidence plane", ylabel="Bistatic RCS (dBsm)", title="Bistatic RCS cut")
    for ax in axes.ravel():
        ax.legend(fontsize=6.5, loc="best")
    fids = [f"{lab}: {r.fidelity.display}" for r, lab in ((baseline_resp, labels[0]), (optimized_resp, labels[1])) if r is not None]
    fig.suptitle("; ".join(fids) + ("; dashed = PREDICTED (+/- 2 sigma)" if predicted else ""), fontsize=8, color=INK2)
    fig.tight_layout()
    return fig


@_styled
def plot_sensitivity(rankings: Dict[str, List[Dict]]):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    names = [r["parameter"] for r in next(iter(rankings.values()))]
    y = np.arange(len(names))
    h = 0.8 / max(len(rankings), 1)
    for k, (label, rows) in enumerate(rankings.items()):
        d = {r["parameter"]: r["sensitivity"] for r in rows}
        ax.barh(y + k * h, [d.get(n, 0) for n in names], height=h * 0.9, color=SERIES[k], label=label)
    ax.set_yticks(y + h * (len(rankings) - 1) / 2, names)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised local response sensitivity S_hat_i")
    ax.set_title("Local response sensitivities (model-/data-associated, not causal)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


@_styled
def plot_convergence(curves: Dict[str, Dict[str, list]], title: str = "Best validated composite vs. high-fidelity evaluations"):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, (name, c) in enumerate(curves.items()):
        if "mean" not in c:
            continue
        x, m, s = np.asarray(c["n_hf"]), np.asarray(c["mean"]), np.asarray(c["std"])
        ax.plot(x, m, color=SERIES[k % len(SERIES)], label=name, lw=2.5 if "PGUCO" in name else 1.5)
        ax.fill_between(x, m - s, m + s, color=SERIES[k % len(SERIES)], alpha=0.12, lw=0)
    ax.set(xlabel="Number of high-fidelity evaluations N_HF", ylabel="Best composite J (lower is better)", title=title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


@_styled
def plot_pareto(F_valid: np.ndarray, names: Sequence[str], F_pred: Optional[np.ndarray] = None, baseline: Optional[np.ndarray] = None,
                pareto_idx: Optional[np.ndarray] = None):
    m = F_valid.shape[1]
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)][:6]
    n = len(pairs)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows), squeeze=False)
    for ax, (i, j) in zip(axes.ravel(), pairs):
        if F_pred is not None:
            ax.scatter(F_pred[:, i], F_pred[:, j], s=10, color=SERIES[2], alpha=0.35, label="candidate pool (PREDICTED)", edgecolors="none")
        ax.scatter(F_valid[:, i], F_valid[:, j], s=22, color=SERIES[0], label="evaluated (validated/oracle)", edgecolors=SURFACE, linewidths=0.8)
        if pareto_idx is not None and len(pareto_idx):
            P = F_valid[pareto_idx]
            ax.scatter(P[:, i], P[:, j], s=40, color=SERIES[1], label="validated Pareto set", edgecolors=SURFACE, linewidths=1)
        if baseline is not None:
            ax.scatter([baseline[i]], [baseline[j]], marker="*", s=140, color=INK, label="baseline x0")
        ax.set(xlabel=f"J~ {names[i]}", ylabel=f"J~ {names[j]}")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle("Normalised objectives (lower is better)")
    fig.tight_layout()
    return fig


@_styled
def plot_uncertainty(pred_log: List[Dict]):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    if pred_log:
        p = np.array([d["predicted_composite"] for d in pred_log]); v = np.array([d["validated_composite"] for d in pred_log])
        s = np.array([np.sqrt(np.sum(np.square(d.get("sigma_composite", 0.0)))) if "sigma_composite" in d else np.nan for d in pred_log])
        it = np.array([d["iteration"] for d in pred_log])
        sc = ax.scatter(v, p, c=it, cmap="Blues", s=26, edgecolors=INK2, linewidths=0.5, vmin=-1)
        if np.isfinite(s).any():
            ax.errorbar(v, p, yerr=2 * s, fmt="none", ecolor=INK2, alpha=0.4, lw=0.8)
        lo, hi = min(p.min(), v.min()), max(p.max(), v.max())
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
        fig.colorbar(sc, ax=ax, label="iteration")
    ax.set(xlabel="Validated composite (oracle / full-wave)", ylabel="PREDICTED composite (+/- 2 sigma)",
           title="Out-of-sample prediction vs. validation")
    fig.tight_layout()
    return fig
