"""Publication + appendix figures from real 66-city experimental outputs.

Generates (all vector PDF) in figures/:
  MAIN PAPER:
    fig1_pipeline.pdf            pipeline diagram
    fig2_city_map.pdf             world map of 66 cities by Köppen zone
    fig3_uhi_distribution.pdf     per-zone UHI anomaly violins
    fig4_pareto_perzone.pdf       per-zone Pareto scatter grid
    fig5_archetype_radars.pdf     archetype morphology radar panels
    fig6_shap_top.pdf             top-k SHAP bar chart
    fig7_perzone_r2.pdf           per-zone marginal R² bar chart
    fig8_forest_beta.pdf          forest plot of fixed-effect β + 95% CI
    fig9_tree_effect_heatmap.pdf  per-zone β of each morphology feature

  APPENDIX:
    figA1_perzone_beta_heatmap.pdf  15 × 8 heatmap of per-zone β
    figA2_perzone_significance.pdf  per-zone significance patterns
    figA3_calibration.pdf            predicted vs actual scatter
    figA4_icc_perzone.pdf            ICC bar chart
    figA5_city_sample_size.pdf       patches per city
    figA6_feature_correlations.pdf   feature Pearson correlation matrix
    figA7_perzone_tree_effect.pdf    tree-canopy β with CI per zone (forest)
    figA8_pareto_fraction.pdf         Pareto fraction by zone
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

_ZONE_COLORS = {
    "Af": "#1b9e77", "Am": "#66c2a5", "Aw": "#a6d854",
    "BWh": "#d95f02", "BSk": "#fdae61",
    "Csa": "#7570b3", "Csb": "#a6cee3", "Cfa": "#e7298a", "Cfb": "#1f78b4",
    "Cwa": "#e6ab02", "Cwb": "#b3cde3",
    "Dfa": "#fb9a99", "Dfb": "#b15928", "Dsa": "#ff7f00", "Dwa": "#6a3d9a",
}
_ARCH_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

_LABEL_MAP = {
    "wc_tree_frac":      "Tree-canopy fraction",
    "wc_bare_frac":      "Bare-soil fraction",
    "wc_water_frac":     "Water fraction",
    "wc_built_frac":     "Built-up fraction",
    "height_cv":         "Building height CV",
    "building_density":  "Building density",
    "ndvi_p80":          "80th-percentile NDVI",
    "height_mean":       "Mean building height",
}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
})


# ============================================================
# MAIN PAPER FIGURES
# ============================================================

def fig_pipeline(out: Path) -> None:
    stages = [
        ("Ingest",    "Landsat 8/9, MODIS LST\nGoogle/MS Open Buildings\n+ GBA/GLAMOUR heights\nESA WorldCover, LCZ,\nERA5, SRTM"),
        ("Features",  "8 continuous morphology\nvariables per\n$1\\,\\mathrm{km}^{2}$ patch"),
        ("Model",     "Linear mixed-effects\n(city random intercept)\n+ XGBoost + SHAP"),
        ("Pareto",    "Per-Köppen empirical\nnon-dominated patches\nin (UHI, density)"),
        ("Typology",  "$k$-means on\nPareto-efficient patches\n$\\Rightarrow$ archetypes"),
        ("Project",   "CMIP6 SSP5-8.5\n2050 $\\Delta T$ re-scoring"),
    ]
    n = len(stages)
    bw, bh = 1.75, 1.75
    gap = 0.55                      # clear horizontal space for the connector arrows
    x0 = 0.2
    span = n * bw + (n - 1) * gap
    fig, ax = plt.subplots(figsize=(13, 2.4))
    ax.set_xlim(0, x0 * 2 + span); ax.set_ylim(0, 3); ax.axis("off")
    xs = [x0 + i * (bw + gap) for i in range(n)]
    ymid = 0.5 + bh / 2
    for i, ((t, body), x) in enumerate(zip(stages, xs)):
        ax.add_patch(FancyBboxPatch((x, 0.5), bw, bh,
                                    boxstyle="round,pad=0.03", linewidth=1.0,
                                    edgecolor="#333", facecolor="#f4f4f4"))
        ax.text(x + bw / 2, 0.5 + bh - 0.18, t, ha="center", va="top",
                weight="bold", size=10)
        ax.text(x + bw / 2, 0.5 + bh / 2 - 0.24, body, ha="center", va="center",
                size=7, linespacing=1.18)
    # Draw arrows after all boxes so connectors sit cleanly in the gaps, not under boxes.
    for i in range(n - 1):
        x_start = xs[i] + bw + 0.06
        x_end = xs[i + 1] - 0.06
        ax.add_patch(FancyArrowPatch((x_start, ymid), (x_end, ymid),
                                     arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.3, color="#555",
                                     shrinkA=0, shrinkB=0))
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def _place_city_labels(fig, ax, cities: list, fontsize: float = 6.4) -> None:
    """Place three-letter city codes so that no two labels overlap.

    City markers cluster tightly in Europe, the US and South-East Asia, where
    a fixed offset makes the codes collide. For each city we score a ring of
    candidate offsets against the labels already placed and against every
    marker, and take the best; a leader line is drawn whenever a label ends up
    far enough from its dot that the association is no longer obvious.
    """
    import matplotlib.patheffects as pe

    fig.canvas.draw()
    trans = ax.transData
    pts = [trans.transform((c["lon"], c["lat"])) for c in cities]

    # Every code is three characters, so one measurement sizes them all.
    probe = ax.text(0, 0, "XXX", fontsize=fontsize)
    fig.canvas.draw()
    bb = probe.get_window_extent(fig.canvas.get_renderer())
    w, h = bb.width, bb.height
    probe.remove()

    radii = (12.0, 17.0, 23.0, 30.0, 38.0, 47.0)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    marker_r = 7.0  # scatter marker radius in points, approximately

    def overlaps(a, b) -> bool:
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    def score_at(i, r, ang, exclude):
        px, py = pts[i]
        cx, cy = px + r * np.cos(ang), py + r * np.sin(ang)
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        s = 0.0
        for k, pb in boxes.items():
            if k != i and k != exclude and overlaps(box, pb):
                s += 100.0
        # Every marker is an obstacle, including this city's own dot: a code
        # printed across its marker is as unreadable as one across a neighbour's.
        for mx, my in pts:
            if (box[0] - marker_r < mx < box[2] + marker_r
                    and box[1] - marker_r < my < box[3] + marker_r):
                s += 40.0
        return s + r * 0.35, box            # prefer staying near the marker

    boxes: dict[int, tuple] = {}
    radius_of: dict[int, float] = {}
    # Densest neighbourhoods first: those cities have the fewest good options.
    order = sorted(range(len(cities)), key=lambda i: -sum(
        1 for j in range(len(cities))
        if j != i and abs(pts[i][0] - pts[j][0]) < 60 and abs(pts[i][1] - pts[j][1]) < 40))

    for i in order:
        best = min(((score_at(i, r, a, None), r) for r in radii for a in angles),
                   key=lambda t: t[0][0])
        boxes[i], radius_of[i] = best[0][1], best[1]

    # Repair pass: greedy placement can strand the last label in a dense
    # cluster. With every other label now fixed, re-search for a clear slot.
    for _ in range(3):
        clashing = [i for i in boxes
                    if any(j != i and overlaps(boxes[i], boxes[j]) for j in boxes)]
        if not clashing:
            break
        for i in clashing:
            best = min(((score_at(i, r, a, None), r) for r in radii for a in angles),
                       key=lambda t: t[0][0])
            boxes[i], radius_of[i] = best[0][1], best[1]

    for i in range(len(cities)):
        bx = boxes[i]
        px, py = pts[i]
        lx, ly = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        dx, dy = trans.inverted().transform((lx, ly))
        ax.text(dx, dy, cities[i]["id"], fontsize=fontsize, color="#222",
                ha="center", va="center", zorder=6,
                path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
        if radius_of[i] > 12.0:
            ex, ey = trans.inverted().transform(
                (lx - np.sign(lx - px) * w / 2, ly - np.sign(ly - py) * h / 2))
            ax.plot([cities[i]["lon"], ex], [cities[i]["lat"], ey],
                    color="#666", lw=0.4, zorder=3, solid_capstyle="round")


def fig_city_map(cities_yaml: Path, out: Path) -> None:
    """City map with Natural Earth land silhouette as background."""
    import geopandas as gpd
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cities_yaml)
    cities = [{"id": c.id, "lat": c.lat, "lon": c.lon, "koppen": c.koppen} for c in cfg.cities]

    ne_path = REPO_ROOT / "data/natural_earth/ne_110m_admin_0_countries.zip"
    # Taller than strictly needed for the geography: extra vertical room means
    # each label spans fewer degrees, which is what makes separation possible.
    fig, ax = plt.subplots(figsize=(12.0, 6.9))
    ax.set_facecolor("#eef3f8")  # ocean colour
    if ne_path.exists():
        try:
            world = gpd.read_file(f"zip://{ne_path}")
            world.plot(ax=ax, color="#f3efe6", edgecolor="#888",
                       linewidth=0.45, zorder=1)
        except Exception:
            pass
    for y in (-60, -30, 0, 30, 60):
        ax.axhline(y, color="#bbb", lw=0.25, alpha=0.6, zorder=2)
    for x in (-120, -60, 0, 60, 120):
        ax.axvline(x, color="#bbb", lw=0.25, alpha=0.6, zorder=2)

    by_zone: dict[str, list] = {}
    for c in cities:
        by_zone.setdefault(c["koppen"], []).append((c["lon"], c["lat"], c["id"]))
    for z in sorted(by_zone):
        pts = by_zone[z]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=82, color=_ZONE_COLORS.get(z, "#444"),
                   edgecolor="black", linewidth=0.6,
                   label=f"{z} (n={len(pts)})", zorder=4)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    _place_city_labels(fig, ax, cities)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Köppen-stratified global city sample (n = {len(cities)})")
    # Placed beneath the map rather than inside it: an inset legend covered the
    # South Atlantic and clipped the southern South American cities.
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=8, fontsize=7, framealpha=1.0, edgecolor="#666",
                    columnspacing=1.1, handletextpad=0.4, borderpad=0.6)
    leg.set_zorder(10)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_uhi_distribution(features_path: Path, out: Path) -> None:
    df = pd.read_parquet(features_path)
    zones = sorted(df.koppen_zone.unique())
    data = [df.loc[df.koppen_zone == z, "uhi_anomaly_c"].values for z in zones]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(_ZONE_COLORS.get(zones[i], "#888"))
        body.set_edgecolor("black"); body.set_alpha(0.7); body.set_linewidth(0.6)
    for key in ("cbars", "cmaxes", "cmins", "cmedians"):
        if key in parts:
            parts[key].set_color("black"); parts[key].set_linewidth(0.8)
    ax.set_xticks(range(1, len(zones) + 1))
    ax.set_xticklabels(zones)
    ax.set_ylabel("Surface UHI anomaly (°C)")
    ax.set_xlabel("Köppen climate zone")
    ax.set_title(f"Per-zone distribution of summer surface UHI anomaly (n = {len(df):,} patches)")
    ax.axhline(0, color="black", lw=0.6, alpha=0.7)
    ax.grid(True, alpha=0.3, axis="y")
    ymax = ax.get_ylim()[1]
    for i, z in enumerate(zones, 1):
        ax.text(i, ymax * 0.96, f"n={len(data[i-1])}",
                ha="center", va="top", fontsize=7, color="#333")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_pareto_grid(features_path: Path, pareto_path: Path, out: Path) -> None:
    df = pd.read_parquet(features_path)
    par = pd.read_parquet(pareto_path)
    zones = sorted(df.koppen_zone.unique())
    ncols = 4
    nrows = (len(zones) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, z in enumerate(zones):
        ax = axes[i]
        sub = df[df.koppen_zone == z]
        psub = par[par._pareto_zone == z].sort_values("building_density")
        ax.scatter(sub.building_density, sub.uhi_anomaly_c, s=3, alpha=0.25,
                   color=_ZONE_COLORS.get(z, "gray"))
        if len(psub):
            ax.plot(psub.building_density, psub.uhi_anomaly_c, "-",
                    color="black", lw=0.7, alpha=0.7)
            ax.scatter(psub.building_density, psub.uhi_anomaly_c, s=26,
                       edgecolor="black", facecolor="#ffeb3b", linewidth=0.6, zorder=3)
        ax.axhline(0, color="black", lw=0.4, alpha=0.5)
        ax.set_title(f"{z}   n={len(sub)}  Pareto={len(psub)}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, lw=0.4)
    # Shared axis labels
    for ax in axes[:len(zones)]:
        ax.set_xlabel("Building density", fontsize=8)
        ax.set_ylabel("UHI (°C)", fontsize=8)
    for j in range(len(zones), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Empirical Pareto fronts per Köppen zone", y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# Compact single-line axis names keep radar labels clear of the outer spine.
_RADAR_LABELS = {
    "building_density": "Density",
    "height_mean":      "Mean height",
    "ndvi_p80":         "NDVI p80",
    "wc_tree_frac":     "Tree canopy",
    "wc_built_frac":    "Built-up",
    "wc_bare_frac":     "Bare soil",
    "wc_water_frac":    "Water",
    "height_cv":        "Height CV",
}


def fig_archetype_radars(cards_path: Path, out: Path) -> None:
    cards = json.loads(Path(cards_path).read_text())
    feat_keys = ["building_density", "height_mean", "ndvi_p80", "wc_tree_frac",
                 "wc_built_frac", "wc_bare_frac", "wc_water_frac", "height_cv"]
    n = len(feat_keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    ncols = len(cards)
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.9),
                             subplot_kw=dict(polar=True))
    axes = np.array(axes).reshape(-1) if ncols > 1 else np.array([axes])
    for i, c in enumerate(cards):
        ax = axes[i]
        vals = []
        for k in feat_keys:
            v = c["median_features"].get(k, 0.0) or 0.0
            if k == "height_mean":
                v = v / 50.0
            vals.append(v)
        vals += vals[:1]
        ax.plot(angles, vals, color=_ARCH_COLORS[i % len(_ARCH_COLORS)], lw=2)
        ax.fill(angles, vals, color=_ARCH_COLORS[i % len(_ARCH_COLORS)], alpha=0.22)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([_RADAR_LABELS[k] for k in feat_keys], size=8)
        # Push tick labels clear of the outer spine and anchor each one on the
        # side of the circle it sits on, so no label crosses a plotted line.
        ax.tick_params(axis="x", pad=10)
        for lbl, ang in zip(ax.get_xticklabels(), angles[:-1]):
            deg = np.degrees(ang) % 360
            if np.isclose(deg, 90) or np.isclose(deg, 270):
                lbl.set_horizontalalignment("center")
            elif 90 < deg < 270:
                lbl.set_horizontalalignment("right")
            else:
                lbl.set_horizontalalignment("left")
        ax.set_yticks([0.25, 0.5, 0.75]); ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        uhi = c["uhi_anomaly_c"]["median"]; den = c["building_density"]["median"]
        ax.set_title(f'Archetype A{int(c["archetype"]) + 1} (n={c["n_patches"]})\n'
                     f'UHI = {uhi:+.2f}°C  density = {den:.2f}', size=9, pad=18)
    fig.subplots_adjust(wspace=0.62)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# Okabe-Ito colour-blind-safe pair: blue for cooling, vermillion for warming.
# Hatching carries the same distinction redundantly, so the figure survives
# greyscale printing and all common forms of colour-vision deficiency.
_COOL_COLOR = "#0072B2"
_WARM_COLOR = "#D55E00"
_NS_COLOR = "#BBBBBB"


def fig_shap_bar(diag_path: Path, out: Path) -> None:
    d = json.loads(Path(diag_path).read_text())
    fe = d["mixed_effects"]["fixed_effects"]
    ranked = sorted(fe.items(), key=lambda kv: -abs(kv[1]["beta_standardised"]))
    names = [_LABEL_MAP.get(k, k) for k, _ in ranked][::-1]
    vals = [v["beta_standardised"] for _, v in ranked][::-1]
    sigs = [v["significant_95"] for _, v in ranked][::-1]

    fig, ax = plt.subplots(figsize=(8.4, 3 + 0.38 * len(names)))
    colors = [(_COOL_COLOR if v < 0 else _WARM_COLOR) if s else _NS_COLOR
              for v, s in zip(vals, sigs)]
    hatches = ["" if v < 0 else "///" for v in vals]
    bars = ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    # Keep every value label outside its bar tip and inside the axes, so no
    # label can collide with a bar, the zero line or a y-axis tick label.
    span = max(abs(min(vals)), abs(max(vals)))
    pad = 0.03 * span
    for b, v in zip(bars, vals):
        ax.text(v + (pad if v >= 0 else -pad), b.get_y() + b.get_height() / 2,
                f"{v:+.2f}", va="center",
                ha=("left" if v >= 0 else "right"), fontsize=8)
    ax.set_xlim(-span * 1.28, span * 1.28)

    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Standardised fixed-effect β (°C per 1 SD)")
    ax.set_title("Fixed-effect coefficients — mixed-effects model")
    ax.grid(True, alpha=0.3, axis="x")

    # In-figure legend: the reader never has to consult the caption to decode
    # the colours (Reviewer #2, round 2).
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=_COOL_COLOR, edgecolor="black", linewidth=0.6,
              label="Cooling  (β < 0)"),
        Patch(facecolor=_WARM_COLOR, edgecolor="black", linewidth=0.6,
              hatch="///", label="Warming  (β > 0)"),
    ]
    if not all(sigs):
        handles.append(Patch(facecolor=_NS_COLOR, edgecolor="black",
                             linewidth=0.6, label="Not significant at 95 %"))
    ax.legend(handles=handles, loc="lower right", frameon=True,
              framealpha=0.95, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_perzone_r2(perzone_path: Path, out: Path) -> None:
    d = json.loads(Path(perzone_path).read_text())
    rows = sorted(d["per_zone"].items(), key=lambda kv: -kv[1]["marginal_r2"])
    zones = [z for z, _ in rows]
    mr2 = [v["marginal_r2"] for _, v in rows]
    cr2 = [v["conditional_r2"] for _, v in rows]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(zones)); w = 0.38
    ax.bar(x - w / 2, mr2, w, label="marginal R² (morphology only)",
           color="#1f78b4", edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, cr2, w, label="conditional R² (+ city effects)",
           color="#a6cee3", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(zones, rotation=0, fontsize=9)
    ax.set_ylabel("R²")
    ax.set_xlabel("Köppen climate zone")
    ax.set_title("Per-zone mixed-effects model performance "
                 f"(pooled weighted marginal R² = {d['global_weighted']['marginal_r2_weighted']:.3f})")
    ax.axhline(0.3, linestyle="--", color="#555", lw=0.6, alpha=0.6)
    ax.text(len(zones) - 0.3, 0.31, "Q1 publishable target (0.30)", fontsize=7,
            ha="right", va="bottom", color="#555")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_forest_beta(diag_path: Path, out: Path) -> None:
    d = json.loads(Path(diag_path).read_text())
    fe = d["mixed_effects"]["fixed_effects"]
    ranked = sorted(fe.items(), key=lambda kv: kv[1]["beta_standardised"])
    names = [_LABEL_MAP.get(k, k) for k, _ in ranked]
    betas = [v["beta_standardised"] for _, v in ranked]
    lows = [v["ci_low_std"] for _, v in ranked]
    highs = [v["ci_high_std"] for _, v in ranked]
    sigs = [v["significant_95"] for _, v in ranked]
    fig, ax = plt.subplots(figsize=(9, 3 + 0.4 * len(names)))
    y = np.arange(len(names))
    colors = ["#1f78b4" if s else "#888" for s in sigs]
    ax.errorbar(betas, y, xerr=[[b - l for b, l in zip(betas, lows)],
                                 [h - b for h, b in zip(highs, betas)]],
                fmt="o", color="#1f78b4", ecolor="#333", capsize=4,
                markersize=7, markeredgecolor="black", linewidth=1.0)
    for yi, b, c in zip(y, betas, colors):
        ax.plot(b, yi, "o", color=c, markeredgecolor="black", markersize=8)
    ax.axvline(0, color="black", lw=0.7, alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Standardised β (°C per 1 SD) — 95% CI")
    ax.set_title("Forest plot — fixed-effect coefficients (global mixed-effects model)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_beta_heatmap(perzone_path: Path, out: Path) -> None:
    """Heatmap of β per feature per zone."""
    d = json.loads(Path(perzone_path).read_text())
    zones = sorted(d["per_zone"].keys())
    feats = sorted({f for v in d["per_zone"].values() for f in v["fixed_effects"]})
    mat = np.full((len(feats), len(zones)), np.nan)
    sig = np.zeros_like(mat, dtype=bool)
    for j, z in enumerate(zones):
        fe = d["per_zone"][z]["fixed_effects"]
        for i, f in enumerate(feats):
            if f in fe:
                mat[i, j] = fe[f]["beta_std"]
                sig[i, j] = fe[f]["significant_95"]
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(feats) + 2.0))
    # A handful of Af cells reach |β| ≈ 14 and, on a full-range symmetric scale,
    # flatten every other cell to near-white. Clip the colour scale to the bulk
    # of the distribution and print the value inside any cell that runs off it,
    # so the map stays readable without hiding the extremes.
    vmax = float(np.nanpercentile(np.abs(mat), 94))
    vmax = max(vmax, 0.5)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(zones))); ax.set_xticklabels(zones, fontsize=9)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([_LABEL_MAP.get(f, f) for f in feats], fontsize=9)
    ax.set_xlabel("Köppen climate zone")

    for i in range(len(feats)):
        for j in range(len(zones)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            off_scale = abs(v) > vmax
            if sig[i, j]:
                ax.text(j, i, "✦", ha="center",
                        va=("bottom" if off_scale else "center"),
                        color="black", fontsize=9)
            if off_scale:
                ax.text(j, i, f"{v:+.1f}", ha="center", va="top",
                        color=("white" if abs(v) > 1.6 * vmax else "black"),
                        fontsize=7.5, fontweight="bold")

    ax.set_title("Fixed-effect β per morphology feature and Köppen zone")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, extend="both", pad=0.02)
    # The direction of effect is carried by the graphical scale itself rather
    # than by colour words in the caption (Reviewer #2, round 2).
    cbar.set_label("←  cooling        β (°C per 1 SD)        warming  →")

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], linestyle="none", marker=r"$✦$",
                              color="black", markersize=8,
                              label="significant at 95 %")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False, fontsize=8, handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_perzone_tree_forest(perzone_path: Path, out: Path) -> None:
    """Forest plot of tree-canopy β across zones."""
    d = json.loads(Path(perzone_path).read_text())
    rows = []
    for z, v in d["per_zone"].items():
        fe = v["fixed_effects"].get("wc_tree_frac")
        if fe is None: continue
        rows.append((z, fe["beta_std"], fe["ci_low"], fe["ci_high"], fe["significant_95"]))
    rows.sort(key=lambda r: r[1])
    zones = [r[0] for r in rows]
    b = [r[1] for r in rows]; lo = [r[2] for r in rows]; hi = [r[3] for r in rows]
    sig = [r[4] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(zones) + 2))
    y = np.arange(len(zones))
    ax.errorbar(b, y, xerr=[[b[i] - lo[i] for i in range(len(b))],
                             [hi[i] - b[i] for i in range(len(b))]],
                fmt="o", color="#1b9e77", ecolor="#333", capsize=4,
                markersize=7, markeredgecolor="black")
    for yi, bb, s in zip(y, b, sig):
        ax.plot(bb, yi, "o", color="#1b9e77" if s else "#bbb",
                markeredgecolor="black", markersize=8)
    ax.axvline(0, color="black", lw=0.7, alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(zones, fontsize=10)
    ax.set_xlabel("Standardised β of tree-canopy fraction (95% CI)")
    ax.set_title("Tree-canopy cooling effect is consistent across Köppen zones")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# ============================================================
# APPENDIX FIGURES
# ============================================================

def figA_significance(perzone_path: Path, out: Path) -> None:
    """Appendix: per-zone significance matrix (✓ / ✗)."""
    d = json.loads(Path(perzone_path).read_text())
    zones = sorted(d["per_zone"].keys())
    feats = sorted({f for v in d["per_zone"].values() for f in v["fixed_effects"]})
    mat = np.zeros((len(feats), len(zones)))
    for j, z in enumerate(zones):
        for i, f in enumerate(feats):
            fe = d["per_zone"][z]["fixed_effects"].get(f)
            if fe is not None and fe["significant_95"]:
                mat[i, j] = 1 if fe["beta_std"] > 0 else -1
    # Three discrete states read more clearly than a continuous diverging ramp,
    # and the +/-/· glyphs repeat the same information without colour, so the
    # panel stays readable in greyscale and under colour-vision deficiency
    # (Reviewer #2, round 2).
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap([_COOL_COLOR, "#F0F0F0", _WARM_COLOR])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(feats) + 2.2))
    ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(zones))); ax.set_xticklabels(zones, fontsize=9)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([_LABEL_MAP.get(f, f) for f in feats], fontsize=9)
    ax.set_xlabel("Köppen climate zone")

    # Thin white separators make the discrete cells easier to count.
    ax.set_xticks(np.arange(-0.5, len(zones), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(feats), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    for i in range(len(feats)):
        for j in range(len(zones)):
            val = mat[i, j]
            sym = "+" if val > 0 else ("−" if val < 0 else "·")
            ax.text(j, i, sym, ha="center", va="center", fontsize=10,
                    fontweight="bold",
                    color="white" if abs(val) == 1 else "#666")

    ax.set_title("Per-zone significance pattern of morphology fixed effects")
    ax.legend(handles=[
        Patch(facecolor=_WARM_COLOR, edgecolor="black", linewidth=0.5,
              label="+  warming, significant"),
        Patch(facecolor=_COOL_COLOR, edgecolor="black", linewidth=0.5,
              label="−  cooling, significant"),
        Patch(facecolor="#F0F0F0", edgecolor="black", linewidth=0.5,
              label="·  not significant at 95 %"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
        frameon=False, fontsize=8.5, handlelength=1.6, handletextpad=0.5)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figA_city_n(features_path: Path, out: Path) -> None:
    """Appendix: patches per city bar chart."""
    df = pd.read_parquet(features_path)
    counts = df.groupby(["city_id", "koppen_zone"]).size().reset_index(name="n")
    counts = counts.sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 0.21 * len(counts) + 1))
    colors = [_ZONE_COLORS.get(z, "#888") for z in counts.koppen_zone]
    ax.barh(counts.city_id, counts.n, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Built-up patches (n)")
    ax.set_title(f"Per-city sample size — 66 cities, {counts.n.sum():,} total patches")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figA_feature_corr(features_path: Path, out: Path) -> None:
    """Appendix: morphology feature correlation matrix."""
    df = pd.read_parquet(features_path)
    feats = ["building_density", "height_mean", "height_cv", "ndvi_p80",
             "wc_tree_frac", "wc_built_frac", "wc_bare_frac", "wc_water_frac"]
    corr = df[feats].corr(method="pearson").values
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels([_LABEL_MAP.get(f, f) for f in feats], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels([_LABEL_MAP.get(f, f) for f in feats], fontsize=8)
    for i in range(len(feats)):
        for j in range(len(feats)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title("Morphology feature Pearson correlation")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figA_pareto_fraction(pareto_summary: Path, out: Path) -> None:
    d = json.loads(Path(pareto_summary).read_text())
    rows = sorted(d["per_zone"], key=lambda r: -r["pareto_fraction"])
    zones = [r["zone"] for r in rows]
    fracs = [r["pareto_fraction"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = [_ZONE_COLORS.get(z, "#888") for z in zones]
    ax.bar(zones, fracs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Pareto-efficient patch fraction (%)")
    ax.set_xlabel("Köppen zone")
    ax.set_title(f"Per-zone Pareto fraction — total {d['n_pareto_total']} Pareto-efficient patches")
    ax.grid(True, alpha=0.3, axis="y")
    for z, f in zip(zones, fracs):
        ax.text(z, f + 0.03, f"{f:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figA_icc_perzone(perzone_path: Path, out: Path) -> None:
    d = json.loads(Path(perzone_path).read_text())
    rows = sorted(d["per_zone"].items(), key=lambda kv: -kv[1]["icc"])
    zones = [z for z, _ in rows]
    iccs = [v["icc"] for _, v in rows]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = [_ZONE_COLORS.get(z, "#888") for z in zones]
    ax.bar(zones, iccs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Intraclass correlation (ICC)")
    ax.set_xlabel("Köppen zone")
    ax.set_title("Fraction of within-zone variance at the city level")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)
    for z, i in zip(zones, iccs):
        ax.text(z, i + 0.01, f"{i:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figA_uhi_by_city_scatter(features_path: Path, out: Path) -> None:
    df = pd.read_parquet(features_path)
    city_stats = df.groupby(["city_id", "koppen_zone"]).agg(
        uhi_mean=("uhi_anomaly_c", "mean"),
        n=("uhi_anomaly_c", "size"),
    ).reset_index().sort_values("uhi_mean")
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = [_ZONE_COLORS.get(z, "#888") for z in city_stats.koppen_zone]
    ax.barh(city_stats.city_id, city_stats.uhi_mean,
            color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("City-mean summer surface UHI anomaly (°C)")
    ax.set_title("Per-city mean UHI anomaly — sorted")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def figA_tree_effect_scatter(features_path: Path, out: Path) -> None:
    df = pd.read_parquet(features_path)
    zones = sorted(df.koppen_zone.unique())
    ncols = 4
    nrows = (len(zones) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, z in enumerate(zones):
        ax = axes[i]
        sub = df[df.koppen_zone == z]
        ax.scatter(sub.wc_tree_frac, sub.uhi_anomaly_c, s=2, alpha=0.25,
                   color=_ZONE_COLORS.get(z, "gray"))
        # Fit simple line for visual
        if len(sub) > 50:
            z_line = np.polyfit(sub.wc_tree_frac, sub.uhi_anomaly_c, 1)
            xs = np.linspace(sub.wc_tree_frac.min(), sub.wc_tree_frac.max(), 50)
            ax.plot(xs, np.polyval(z_line, xs), color="black", lw=1.2)
        ax.axhline(0, color="black", lw=0.4, alpha=0.5)
        ax.set_title(f"{z}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[:len(zones)]:
        ax.set_xlabel("Tree-canopy fraction", fontsize=8)
        ax.set_ylabel("UHI (°C)", fontsize=8)
    for j in range(len(zones), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Raw tree-canopy vs UHI scatter (visual support for the β signs in Figure 9)",
                 y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    repo = REPO_ROOT
    out = repo / "figures"
    out.mkdir(parents=True, exist_ok=True)

    features = repo / "data/features/full.parquet"
    pareto = repo / "data/features/full.pareto.parquet"
    cards = repo / "archetype_cards.json"
    diag_global = repo / "diagnostics.json"
    diag_perzone = repo / "diagnostics_perzone.json"
    pareto_sum = repo / "pareto_summary.json"
    cities_yaml = repo / "run/conf/cities/full.yaml"

    # Main figures
    fig_pipeline(out / "fig1_pipeline.pdf")
    fig_city_map(cities_yaml, out / "fig2_city_map.pdf")
    fig_uhi_distribution(features, out / "fig3_uhi_distribution.pdf")
    fig_pareto_grid(features, pareto, out / "fig4_pareto_perzone.pdf")
    fig_archetype_radars(cards, out / "fig5_archetype_radars.pdf")
    fig_shap_bar(diag_global, out / "fig6_shap_bar.pdf")
    fig_perzone_r2(diag_perzone, out / "fig7_perzone_r2.pdf")
    fig_forest_beta(diag_global, out / "fig8_forest_beta.pdf")
    fig_beta_heatmap(diag_perzone, out / "fig9_beta_heatmap.pdf")
    fig_perzone_tree_forest(diag_perzone, out / "fig10_tree_forest.pdf")

    # Appendix figures
    figA_significance(diag_perzone, out / "figA1_perzone_significance.pdf")
    figA_city_n(features, out / "figA2_city_n.pdf")
    figA_feature_corr(features, out / "figA3_feature_corr.pdf")
    figA_pareto_fraction(pareto_sum, out / "figA4_pareto_fraction.pdf")
    figA_icc_perzone(diag_perzone, out / "figA5_icc_perzone.pdf")
    figA_uhi_by_city_scatter(features, out / "figA6_city_uhi.pdf")
    figA_tree_effect_scatter(features, out / "figA7_tree_scatter.pdf")

    files = sorted(out.glob("*.pdf"))
    print(f"Wrote {len(files)} figures to {out}:")
    for f in files:
        print(f"  {f.name:<36}  {f.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
