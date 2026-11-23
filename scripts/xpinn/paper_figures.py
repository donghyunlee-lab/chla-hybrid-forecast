"""Figures of the manuscript (hybrid as the main model). No model is run and no number is computed that the
result tables do not hold: every panel reads results/input_design_20261114/ (and the station coordinates of the WEIS map service,
the HydroBASINS level-6 basins and the national river and province shapefiles in src/gis for the map).

Style: Arial, 183 mm wide, black and grays with navy, teal and red, no gridlines, legends written out.
Writes output/figures/ and output/graphical_abstract/. Run with a Python that has fiona and shapely for the map.
Usage: paper_figures.py [all | fig1 | fig_framework | fig2 | fig3 | fig4 | fig5 | fig6 | fig7 | ga]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import _mathtext
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "input_design_20261114"
OUT = ROOT / "output" / "figures"
GIS = ROOT / "src" / "gis"
MM = 1 / 25.4
WIDTH = 183 * MM
INK, GREY, MID, LIGHT, PALE = "#222222", "#7a7a7a", "#a6a6a6", "#d9d9d9", "#f1f1f1"
NAVY, TEAL, RED = "#3B5BA5", "#00A087", "#DC0000"
WATER = "#8fa9c4"


def _measured_xheight(self, fontname, fontsize, dpi):
    # Arial's PCLT table gives matplotlib an x-height about twice too large, which lifts every superscript.
    return self.get_metrics(fontname, plt.rcParams["mathtext.default"], "x", fontsize, dpi).iceberg


_mathtext.TruetypeFonts.get_xheight = _measured_xheight
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.handlelength": 1.6, "pdf.fonttype": 42,
    "axes.unicode_minus": True, "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

BASINS = [("Han", "Han"), ("Nakdong", "Nakdong"), ("Geum", "Geum"), ("Yeongsan", "Yeongsan–Seomjin")]
NAME = {"S01001": "Gapyeong", "S01010": "Seosang", "S01012": "Uiam", "S01020": "Neungseo", "S01025": "Yangpyeong",
        "S02005": "Jeokpo", "S02006": "Cheongam", "S02009": "Dasan", "S02015": "Chilgok", "S02017": "Sangdong", "S02024": "Namgang",
        "S03003": "Daecheong", "S03004": "Gongju", "S03005": "Buyeo", "S03007": "Janggye", "S03008": "Yongdam", "S03009": "Hyeondo",
        "S03013": "Nammyeon", "S03014": "Yugucheon",
        "S04001": "Juam", "S04002": "Naju", "S04004": "Okjeong", "S04005": "Yongbong", "S04007": "Tamjin", "S04008": "Dongbok",
        "S04009": "Uchi", "S04012": "Jiseokcheon"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", dpi=600); fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig); print("wrote", OUT / f"{name}.pdf")


def letter(ax, s, text="", dx=0.0, dy=4.0):
    """Panel letter (bold, lower case) and a short panel title, left-aligned above the axes."""
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", ha="left", va="bottom", fontsize=8, fontweight="bold")
    if text:
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(dx + 9, dy + 0.4), textcoords="offset points", ha="left", va="bottom", fontsize=7)


def tm_korea(lat, lon):
    """Korea 2000 unified TM (GRS80, lat0 38, lon0 127.5, k0 0.9996, FE 1e6, FN 2e6), Snyder (1987) series; the national
    shapefiles are in this system."""
    a, f = 6378137.0, 1 / 298.257222101; e2 = 2 * f - f * f; ep2 = e2 / (1 - e2); k0, lat0, lon0 = 0.9996, np.radians(38.0), np.radians(127.5)

    def M(p):
        return a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * p - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * p)
                    + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * p) - (35 * e2 ** 3 / 3072) * np.sin(6 * p))
    p, l = np.radians(lat), np.radians(lon); N = a / np.sqrt(1 - e2 * np.sin(p) ** 2); T = np.tan(p) ** 2; C = ep2 * np.cos(p) ** 2; A = (l - lon0) * np.cos(p)
    x = 1e6 + k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120)
    y = 2e6 + k0 * (M(p) - M(lat0) + N * np.tan(p) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24 + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


def read_geoms(path, bbox=None):
    import fiona
    from shapely.geometry import shape
    shp = path if path.suffix == ".shp" else next(path.glob("*.shp"))
    with fiona.open(shp) as src:
        it = src.filter(bbox=bbox) if bbox else src
        return [(dict(f["properties"]), shape(f["geometry"])) for f in it if f["geometry"] is not None]


def fill(ax, geoms, **kw):
    """(Multi)polygons, holes included, as one collection."""
    from matplotlib.collections import PathCollection
    from matplotlib.path import Path as MPath
    paths = []
    for g in geoms:
        for q in getattr(g, "geoms", [g]):
            if q.geom_type != "Polygon" or q.is_empty:
                continue
            rings = [np.asarray(r.coords)[:, :2] for r in [q.exterior, *q.interiors] if len(r.coords) >= 4]
            if rings:
                codes = np.concatenate([[MPath.MOVETO] + [MPath.LINETO] * (len(r) - 2) + [MPath.CLOSEPOLY] for r in rings])
                paths.append(MPath(np.concatenate(rings), codes))
    ax.add_collection(PathCollection(paths, **kw))


def stations():
    T = pd.read_csv(RES / "station_table.csv", dtype={"station": str})
    W = pd.read_csv(ROOT / "data" / "national" / "weis_auto_stations.csv", dtype={"code": str}, comment="#").set_index("code")
    T["lat"], T["lon"] = T.station.map(W.lat).astype(float), T.station.map(W.lon).astype(float)    # official coordinates (three decimals)
    T["order"] = T.basin.map({k: i for i, (k, _) in enumerate(BASINS)})
    return T


# ------------------------------------------------------------------------------------------------ Fig. 1: stations and data
HYBAS = {"Han": [4060005050], "Nakdong": [4060003940], "Geum": [4060004680], "Yeongsan": [4060004500, 4060004100]}


def fig1():
    from shapely.ops import transform, unary_union
    T = stations()
    to_tm = lambda g: transform(lambda lon, lat, z=None: tm_korea(np.asarray(lat), np.asarray(lon)), g)
    hb = read_geoms(GIS / "HydroBASINS" / "hybas_as_lev06_v1c.shp", bbox=(124.5, 33.0, 130.5, 39.5))
    basins = {k: to_tm(unary_union([g for p, g in hb if p["HYBAS_ID"] in ids])).simplify(300) for k, ids in HYBAS.items()}
    land = unary_union([g.buffer(0) for _, g in read_geoms(GIS / "시도")]).simplify(400); near = land.buffer(1.5e3)
    rivers = [g.intersection(near) for _, g in read_geoms(GIS / "Rivers_National")]
    x0, y0, x1, y1 = 850e3, 1.612e6, 1.215e6, 2.075e6
    map_w = 80 * MM; map_h = map_w * (y1 - y0) / (x1 - x0); fig_h = map_h + 14 * MM
    fig = plt.figure(figsize=(WIDTH, fig_h))
    ax = fig.add_axes([9 * MM / WIDTH, 7 * MM / fig_h, map_w / WIDTH, map_h / fig_h])
    fill(ax, [land], facecolor=PALE, edgecolor=MID, linewidths=0.3, zorder=1, rasterized=True)
    for k, g in basins.items():
        fill(ax, [g.intersection(land)], facecolor="#dedede", edgecolor="white", linewidths=0.9, zorder=2, rasterized=True)
    fill(ax, rivers, facecolor=WATER, edgecolor=WATER, linewidths=0.35, zorder=3, rasterized=True)
    lab = {"Han": (1.085e6, 1.972e6, "Han"), "Nakdong": (1.142e6, 1.79e6, "Nakdong"), "Geum": (0.972e6, 1.797e6, "Geum"), "Yeongsan": (0.897e6, 1.742e6, "Yeongsan–\nSeomjin")}
    for k, (x, y, s) in lab.items():
        ax.text(x, y, s, fontsize=7, color="#555555", ha="center", va="center", style="italic", zorder=4, linespacing=1.28)
    for r in T.itertuples():
        x, y = tm_korea(r.lat, r.lon); res = r.type == "lake"
        ax.plot(x, y, "^" if res else "o", ms=4.6 if res else 3.6, mfc=RED if res else NAVY, mec="white", mew=0.5, zorder=6)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    lons, lats = (126, 127, 128, 129), (35, 36, 37, 38)                              # frame ticks at the bottom and left edges
    ax.set_xticks([np.interp(y0, tm_korea(np.linspace(33.5, 35.5, 50), np.full(50, float(l)))[1], tm_korea(np.linspace(33.5, 35.5, 50), np.full(50, float(l)))[0]) for l in lons])
    ax.set_xticklabels([f"{l}°E" for l in lons])
    ax.set_yticks([np.interp(x0, tm_korea(np.full(50, float(l)), np.linspace(125.0, 127.0, 50))[0], tm_korea(np.full(50, float(l)), np.linspace(125.0, 127.0, 50))[1]) for l in lats])
    ax.set_yticklabels([f"{l}°N" for l in lats]); ax.tick_params(labelsize=6, length=2, pad=2)
    ax.plot([1.095e6, 1.195e6], [1.632e6, 1.632e6], color=INK, lw=1.0, zorder=7); ax.text(1.145e6, 1.638e6, "100 km", fontsize=6.5, ha="center", va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.6, mfc=NAVY, mec="white", mew=0.5, label="River station (20)"),
                       Line2D([], [], marker="^", ls="", ms=4.6, mfc=RED, mec="white", mew=0.5, label="Reservoir station (7)"),
                       Rectangle((0, 0), 1, 1, facecolor="#dedede", edgecolor="none", label="River basin"),
                       Line2D([], [], color=WATER, lw=1.2, label="National river")],
              loc="upper left", bbox_to_anchor=(0.015, 0.985), handletextpad=0.6, labelspacing=0.45, borderaxespad=0.0)
    letter(ax, "a", dx=-22, dy=4)

    # b: Chl-a level of every station (median and 90th percentile), grouped by basin
    left = (map_w + 43 * MM) / WIDTH; bw = 1 - left - 4 * MM / WIDTH
    bx = fig.add_axes([left, 42 * MM / fig_h, bw, 1 - 42 * MM / fig_h - 7 * MM / fig_h])
    T = T.sort_values(["order", "type", "chl_median"], ascending=[True, False, False]).reset_index(drop=True); y, ticks, labels = 0.0, [], []
    for k, full in BASINS:
        g = T[T.basin == k]
        bx.annotate(full, xy=(0, y - 0.1), xycoords=("axes fraction", "data"), xytext=(-70, 0), textcoords="offset points", fontsize=6.5, color="#555555", style="italic", ha="left", va="center")
        y += 1.0
        for r in g.itertuples():
            res = r.type == "lake"; c = RED if res else NAVY
            bx.plot([r.chl_median, r.chl_p90], [y, y], color=LIGHT, lw=1.6, solid_capstyle="butt", zorder=1)
            bx.plot(r.chl_p90, y, "|", ms=4, mew=0.8, color=GREY, zorder=2)
            bx.plot(r.chl_median, y, "^" if res else "o", ms=3.8 if res else 3.0, mfc=c, mec="white", mew=0.4, zorder=3)
            ticks.append(y); labels.append(NAME[r.station]); y += 1.0
        y += 0.5
    bx.set_yticks(ticks); bx.set_yticklabels(labels, fontsize=6); bx.set_ylim(y - 0.9, -0.9); bx.set_xscale("log"); bx.set_xlim(1.8, 160)
    bx.set_xticks([2, 5, 10, 15, 25, 50, 100]); bx.set_xticklabels(["2", "5", "10", "15", "25", "50", "100"]); bx.xaxis.set_minor_locator(plt.NullLocator())
    bx.tick_params(axis="y", length=0, pad=2); bx.spines["left"].set_visible(False)
    bx.axvline(15, color=MID, lw=0.5, ls=(0, (3, 2)), zorder=0); bx.axvline(25, color=MID, lw=0.5, ls=(0, (3, 2)), zorder=0)
    bx.set_xlabel("Chl-a (mg m$^{-3}$)")
    bx.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.0, mfc=INK, mec="white", mew=0.4, label="Median"),
                       Line2D([], [], marker="|", ls="", ms=4, mew=0.8, color=GREY, label="90th percentile")],
              loc="upper right", bbox_to_anchor=(1.02, 1.045), ncol=2, columnspacing=1.0, handletextpad=0.3, borderaxespad=0.0)
    letter(bx, "b", dx=-70, dy=4)

    # c: records and test years
    cx = fig.add_axes([left, 11 * MM / fig_h, bw, 17 * MM / fig_h])

    def _striped_box(rx0, ry0, rw, rh, step=0.11):
        cx.add_patch(Rectangle((rx0, ry0), rw, rh, facecolor="white", edgecolor="none", zorder=2.5))
        ry1 = ry0 + rh
        n_steps = int(np.ceil(rw / step)) + 2
        for k_s in range(-1, n_steps):
            xa = max(rx0, rx0 + k_s * step)
            xb = min(rx0 + rw, rx0 + (k_s + 1.0) * step)
            if xb - xa > 1e-4:
                ya = ry1 - rh * ((xa - rx0) / step - k_s)
                yb = ry1 - rh * ((xb - rx0) / step - k_s)
                cx.plot([xa, xb], [ya, yb], color="#555555", lw=0.45, solid_capstyle="butt", zorder=2.8)
        cx.add_patch(Rectangle((rx0, ry0), rw, rh, facecolor="none", edgecolor=INK, lw=0.45, joinstyle="miter", zorder=3.0))

    rows = [("36-month record", 2021.0, 2024.0, "#555555"), ("12-month record", 2023.0, 2024.0, "#9a9a9a"), ("Test years", 2024.0, 2026.0, TEAL)]
    for i, (lab_, a, b, c) in enumerate(rows):
        cx.add_patch(Rectangle((a, i - 0.3), b - a, 0.6, facecolor=c, edgecolor="none", zorder=2))
        if i < 2:                                                               # last fifth of the record: validation
            _striped_box(b - 0.2 * (b - a), i - 0.3, 0.2 * (b - a), 0.6, step=0.11)
    cx.set_yticks(range(3)); cx.set_yticklabels([r[0] for r in rows], fontsize=6.5); cx.set_ylim(2.6, -0.6); cx.set_xlim(2021, 2026)
    cx.set_xticks(range(2021, 2027)); cx.set_xticklabels([str(v) for v in range(2021, 2026)] + [""]); cx.tick_params(axis="y", length=0, pad=2); cx.spines["left"].set_visible(False)
    for lab_ in cx.get_xticklabels():
        lab_.set_horizontalalignment("left")
    _striped_box(2024.08, 0.2, 0.16, 0.6, step=0.08)
    cx.text(2024.29, 0.5, "validation split\n(last fifth)", fontsize=6.05, color=INK, ha="left", va="center", linespacing=1.25)
    letter(cx, "c", dx=-70, dy=3)
    save(fig, "fig1_study_area")


# ------------------------------------------------------------------------------------------------ Fig. 2: study framework
def fig_framework():
    n_all = len(pd.read_csv(ROOT / "data" / "national" / "weis_auto_stations.csv", comment="#"))
    n_cov = len(pd.read_csv(RES / "station_coverage.csv"))
    st_df = pd.read_csv(RES / "station_table.csv")
    n_inc = len(st_df)
    n_riv = int((st_df.type == "river").sum())
    n_res = int((st_df.type == "lake").sum())
    b_cnt = {b: int((st_df.basin == b).sum()) for b, _ in BASINS}
    e9_df = pd.read_csv(ROOT / "results" / "clean_revision_v4" / "FREEZE_20261010" / "results" / "e9_algae_proxy" / "summary.csv")
    n_alg = int(e9_df.iloc[0]["n"])
    n_sen = int(e9_df.iloc[2]["n"])

    old_shrink = _mathtext.SHRINK_FACTOR
    _mathtext.SHRINK_FACTOR = 0.79
    try:
        W, H = 183.0, 126.0
        fig = plt.figure(figsize=(WIDTH, H * MM))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

        F_INK, F_DARK, F_GREY, F_MID, F_LIGHT, F_PALE = "#222222", "#555555", "#7A7A7A", "#A6A6A6", "#D9D9D9", "#F1F1F1"
        F_NAVY, F_TEAL, F_RED, F_WHITE = "#3B5BA5", "#00A087", "#DC0000", "#FFFFFF"

        def sbox(x, y, w, h, ec=F_MID, fc=F_WHITE, lw=0.55, ls="-", z=2):
            ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, ls=ls, joinstyle="miter", zorder=z))

        def fbox(x, y, w, h, fc=F_PALE, z=1):
            ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="none", zorder=z))

        def sline(xs, ys, c=F_INK, lw=0.55, ls="-", z=3):
            ax.plot(xs, ys, color=c, lw=lw, ls=ls, solid_capstyle="butt", dash_capstyle="butt", solid_joinstyle="miter", zorder=z)

        def sarrow(x0, y0, x1, y1, c=F_INK, lw=0.7, hl=1.5, hw=0.80, z=4):
            dx, dy = x1 - x0, y1 - y0
            ln = float(np.hypot(dx, dy))
            if ln <= 1e-6:
                return
            ux, uy = dx / ln, dy / ln
            nx, ny = -uy, ux
            ax.plot([x0, x1 - ux * hl * 0.85], [y0, y1 - uy * hl * 0.85], color=c, lw=lw,
                    solid_capstyle="butt", solid_joinstyle="miter", zorder=z)
            ax.add_patch(Polygon([(x1, y1),
                                  (x1 - ux * hl + nx * hw, y1 - uy * hl + ny * hw),
                                  (x1 - ux * hl - nx * hw, y1 - uy * hl - ny * hw)],
                                 closed=True, facecolor=c, edgecolor="none", zorder=z + 0.1))

        def tx(x, y, s, **kw):
            kw = {"fontsize": 6.35, "ha": "left", "va": "center", "color": F_INK, "zorder": 5, **kw}
            ax.text(x, y, s, **kw)

        # ---- Panel a: Monitoring network & causal data preparation ----
        tx(3.0, 122.0, "a", fontsize=8.0, fontweight="bold")
        tx(8.0, 122.0, "Monitoring network and causal data preparation", fontsize=7.0)

        # Top-left: quantitative 2-step station screening bar chart
        sbox(3.0, 94.0, 40.2, 25.0, ec=F_MID, fc=F_WHITE, lw=0.55)
        tx(5.3, 116.2, "Station screening (2021–2025)", fontsize=6.5, fontweight="bold")
        bx0, bw_max = 18.2, 17.6
        fbox(bx0, 110.4, bw_max, 2.9, fc=F_LIGHT, z=3)
        tx(5.3, 111.85, "All WEIS", fontsize=6.3, color=F_DARK)
        tx(bx0 + bw_max + 1.1, 111.85, f"{n_all}", fontsize=6.3, color=F_DARK)

        w37 = bw_max * (n_cov / n_all)
        fbox(bx0, 105.6, w37, 2.9, fc=F_MID, z=3)
        tx(5.3, 107.05, "Step 1 (rate)", fontsize=6.3, color=F_DARK)
        tx(bx0 + w37 + 1.1, 107.05, f"{n_cov}", fontsize=6.3, color=F_DARK)

        w20 = bw_max * (n_riv / n_all)
        w07 = bw_max * (n_res / n_all)
        fbox(bx0, 100.6, w20, 3.1, fc=F_NAVY, z=3)
        fbox(bx0 + w20, 100.6, w07, 3.1, fc=F_RED, z=3)
        tx(5.3, 102.15, "Step 2 (gap)", fontsize=6.3, fontweight="bold")
        tx(bx0 + w20 + w07 + 1.1, 102.15, f"{n_inc} ({n_riv}+{n_res})", fontsize=6.35, fontweight="bold")

        fbox(5.3, 96.3, 2.2, 2.2, fc=F_NAVY, z=3); tx(8.2, 97.4, f"{n_riv} river", fontsize=6.2, color=F_NAVY)
        fbox(22.0, 96.3, 2.2, 2.2, fc=F_RED, z=3); tx(24.9, 97.4, f"{n_res} reservoir", fontsize=6.2, color=F_RED)

        # Top-right: unified daily forecast inputs (WEIS + ERA5 = 17 core / 32 full ch)
        sbox(45.0, 94.0, 41.0, 25.0, ec=F_MID, fc=F_WHITE, lw=0.55)
        fbox(45.0, 94.0, 41.0, 5.4, fc=F_PALE, z=2.1)
        sline([45.0, 86.0], [99.4, 99.4], c=F_MID, lw=0.5, z=2.5)
        sline([45.0, 86.0], [108.8, 108.8], c=F_LIGHT, lw=0.5, z=2.5)
        tx(47.3, 115.8, "WEIS daily water quality", fontsize=6.5, fontweight="bold")
        tx(47.3, 112.1, "4 core / 9 full vars (13 / 28 ch)", fontsize=6.25, color=F_DARK)
        tx(47.3, 105.6, "ERA5 daily meteorology", fontsize=6.5, fontweight="bold")
        tx(47.3, 102.0, "Radiation, air T, rain, wind (4 ch)", fontsize=6.25, color=F_DARK)
        tx(65.5, 96.7, "30-d window: 17 core / 32 full ch", fontsize=6.3, fontweight="bold", ha="center")

        sarrow(23.1, 94.0, 23.1, 90.2, c=F_INK, lw=0.65)
        sarrow(65.5, 94.0, 65.5, 90.2, c=F_INK, lw=0.65)

        # Bottom tier of Panel a: Causal protocol & Fig. 1c proportional split bar
        sbox(3.0, 65.4, 83.0, 24.8, ec=F_GREY, fc=F_WHITE, lw=0.6)
        tx(5.4, 87.8, r"Causal protocol at issue day $t$ (data up to day $t$)", fontsize=6.5, fontweight="bold")
        tx(5.4, 84.4, r"Targets: measured Chl-a at $h = 1, 3, 7, 14$ d (never imputed)", fontsize=6.3)
        tx(5.4, 81.3, r"Inputs: $\leq$3-d forward fill, training median, measured flag & age", fontsize=6.3, color=F_DARK)
        tx(5.4, 78.2, "Training-split scaling & trailing multi-day means", fontsize=6.3, color=F_DARK)

        # Proportional 2021-2025 timeline strip (echoing Fig. 1c: 80% train, 20% hatched val, 2024-2025 test)
        sx0, sy0, sw, sh = 5.4, 67.6, 78.2, 3.9
        w_fit = sw * 0.60
        w_trn = w_fit * 0.80
        w_val = w_fit - w_trn
        w_tst = sw - w_fit
        fbox(sx0, sy0, w_trn, sh, fc=F_LIGHT, z=2.5)
        fbox(sx0 + w_trn, sy0, w_val, sh, fc=F_WHITE, z=2.5)
        for hk in range(6):
            hx0 = sx0 + w_trn + hk * 1.65
            hx1 = min(sx0 + w_fit, hx0 + 2.1)
            hy0 = sy0 + max(0.0, (hx0 + 2.1 - (sx0 + w_fit)) * (sh / 2.1)) if hx0 + 2.1 > sx0 + w_fit else sy0
            if hx0 < sx0 + w_fit:
                sline([hx0, hx1], [sy0 + sh, hy0], c=F_GREY, lw=0.45, z=2.8)
        sbox(sx0, sy0, w_fit, sh, ec=F_GREY, fc="none", lw=0.55, z=3)
        sline([sx0 + w_trn, sx0 + w_trn], [sy0, sy0 + sh], c=F_GREY, lw=0.5, z=3)
        sbox(sx0 + w_fit, sy0, w_tst, sh, ec=F_INK, fc=F_PALE, lw=0.65, z=3)
        xc_val = sx0 + w_trn + w_val / 2
        sline([xc_val, xc_val], [sy0 + sh, sy0 + sh + 0.9], c=F_GREY, lw=0.5, z=3)
        tx(sx0 + w_trn / 2, sy0 + sh / 2, "Fit 36-mo / 12-mo (80% train)", fontsize=6.15, ha="center")
        tx(xc_val, sy0 + sh + 2.2, "20% val", fontsize=6.0, color=F_DARK, ha="center")
        tx(sx0 + w_fit + w_tst / 2, sy0 + sh / 2, "Test 2024–2025 (69,748)", fontsize=6.15, fontweight="bold", ha="center")

        # Orthogonal connector a -> b (centered on the 4x4 LOBO matrix between F2 and F3)
        sline([86.0, 89.8, 89.8], [77.8, 77.8, 100.9], c=F_INK, lw=0.7, z=3.5)
        sarrow(89.8, 100.9, 94.0, 100.9, c=F_INK, lw=0.7)

        # ---- Panel b: Cross-basin transfer & constrained hybrid model ----
        tx(94.0, 122.0, "b", fontsize=8.0, fontweight="bold")
        tx(99.0, 122.0, "Cross-basin transfer and constrained hybrid model", fontsize=7.0)

        # Left sub-card: centered 4x4 Leave-one-basin-out (LOBO) matrix
        sbox(94.0, 85.8, 37.6, 33.2, ec=F_MID, fc=F_WHITE, lw=0.55)
        tx(96.3, 116.3, "4-fold basin CV (LOBO)", fontsize=6.5, fontweight="bold")
        gx0, gy0, cw, ch_, gap = 103.0, 105.0, 5.9, 3.1, 0.65
        b_heads = [("Han", b_cnt["Han"]), ("Nak.", b_cnt["Nakdong"]), ("Geum", b_cnt["Geum"]), ("YS", b_cnt["Yeongsan"])]
        for j, (bn, bc) in enumerate(b_heads):
            cx_ = gx0 + j * (cw + gap) + cw / 2
            tx(cx_, 112.4, bn, fontsize=6.1, fontweight="bold", ha="center")
            tx(cx_, 109.6, f"({bc})", fontsize=6.0, color=F_DARK, ha="center")
        for i in range(4):
            ry = gy0 - i * (ch_ + gap)
            tx(96.3, ry + ch_ / 2, f"F{i + 1}", fontsize=6.2, color=F_DARK)
            for j in range(4):
                cx_ = gx0 + j * (cw + gap)
                is_tgt = (i == j)
                sbox(cx_, ry, cw, ch_, ec=F_NAVY if is_tgt else F_TEAL, fc=F_PALE if is_tgt else F_WHITE, lw=0.80 if is_tgt else 0.55, z=3)
                tx(cx_ + cw / 2, ry + ch_ / 2, "T" if is_tgt else "S", fontsize=6.2, fontweight="bold", color=F_NAVY if is_tgt else F_TEAL, ha="center")
        sbox(96.3, 90.8, 2.1, 2.1, ec=F_TEAL, fc=F_WHITE, lw=0.55, z=3); tx(99.1, 91.85, "S: 3 source (19–22 stn)", fontsize=6.15, color=F_TEAL)
        sbox(96.3, 88.0, 2.1, 2.1, ec=F_NAVY, fc=F_PALE, lw=0.80, z=3); tx(99.1, 89.05, "T: 1 target (36-mo / 12-mo)", fontsize=6.15, color=F_NAVY)

        # Horizontal arrows LOBO -> Step 1 (teal) & LOBO -> Step 2 (navy)
        sarrow(131.6, 111.8, 136.6, 111.8, c=F_TEAL, lw=0.7)
        sarrow(131.6, 92.8, 136.6, 92.8, c=F_NAVY, lw=0.7)

        # Right sub-cards: Two-step hybrid fitting with collinear vertical spine at x = 158.3
        sbox(136.6, 104.4, 43.4, 14.6, ec=F_TEAL, fc=F_WHITE, lw=0.65)
        tx(138.8, 116.4, "Step 1: Source pre-training", fontsize=6.45, fontweight="bold", color=F_TEAL)
        tx(138.8, 113.1, "3 source basins (36-mo, 5 seeds)", fontsize=6.2, color=F_DARK)
        tx(138.8, 110.0, "Jointly fit 51,332 xLSTM weights,", fontsize=6.2)
        tx(138.8, 106.9, "gate & source constants", fontsize=6.2)

        sarrow(158.3, 104.4, 158.3, 99.8, c=F_TEAL, lw=0.7)
        tx(159.9, 102.1, "freeze 51,332 weights", fontsize=6.05, fontweight="bold", color=F_TEAL)

        sbox(136.6, 85.8, 43.4, 14.0, ec=F_NAVY, fc=F_WHITE, lw=0.65)
        tx(138.8, 97.4, "Step 2: Target-basin fitting", fontsize=6.45, fontweight="bold", color=F_NAVY)
        tx(138.8, 94.3, "5 constants/stn + 1 gate/basin", fontsize=6.2)
        tx(138.8, 91.3, "= 139 fitted values (27 stn, 4 basins)", fontsize=6.2, fontweight="bold", color=F_NAVY)
        tx(138.8, 88.3, "on 36-mo or 12-mo records", fontsize=6.15, color=F_DARK)

        # Collinear vertical arrow Step 2 -> Constrained hybrid transition (x = 158.3)
        sarrow(158.3, 85.8, 158.3, 81.8, c=F_NAVY, lw=0.75)

        # Bottom centerpiece of Panel b: Constrained forward hybrid transition (1.0 pt primary frame)
        sbox(94.0, 65.4, 86.0, 16.4, ec=F_NAVY, fc=F_PALE, lw=1.0)
        tx(96.4, 79.1, "Constrained hybrid transition (14 daily steps; Fig. 3)", fontsize=6.5, fontweight="bold")
        tx(177.6, 79.1, r"$x = \ln\,C$", fontsize=6.35, color=F_DARK, ha="right")
        tx(137.0, 73.7, r"$x_k = x_{k-1} - \kappa\,\theta^{\,T-20}(x_{k-1} - L_k) - \beta\,R\,e^{-k/3} + \varepsilon$",
           fontsize=7.2, ha="center")
        tx(96.4, 68.1, r"HC1: 5 sigmoid ranges ($\alpha,\omega,\kappa,\theta,\beta$)", fontsize=6.35, color=F_INK)
        tx(177.6, 68.1, r"HC2: $|\varepsilon| \leq 0.1$ d$^{-1}$ ($\tanh$; $10\,\varepsilon^2$ penalty)", fontsize=6.35, ha="right", color=F_TEAL)

        # Crossing-free Row 1 -> Row 2 connectors:
        # (1) a -> c: shared causal test protocol at x = 69.8
        sarrow(69.8, 65.4, 69.8, 54.4, c=F_INK, lw=0.7)
        tx(68.4, 61.8, "shared test", fontsize=6.0, color=F_GREY, ha="right")
        # (2) b -> c: fitted hybrid entering Benchmark suite (crossing-free elbow)
        sline([99.5, 99.5, 80.8], [65.4, 61.8, 61.8], c=F_NAVY, lw=0.7, z=3.5)
        sarrow(80.8, 61.8, 80.8, 54.4, c=F_NAVY, lw=0.7)
        tx(90.15, 63.4, "fitted hybrid", fontsize=6.0, color=F_NAVY, ha="center")
        # (3) b -> d: collinear vertical hybrid spine at x = 158.3 feeding unified H1-H3 block
        sarrow(158.3, 65.4, 158.3, 54.4, c=F_NAVY, lw=0.75)
        tx(159.9, 61.4, "hybrid", fontsize=6.0, color=F_NAVY)

        # ---- Panel c: Frameless 4-column visual benchmark matrix ----
        tx(3.0, 57.6, "c", fontsize=8.0, fontweight="bold")
        tx(8.0, 57.6, "Benchmark suite: Hybrid + 21 other single models", fontsize=7.0)

        fbox(3.0, 50.8, 82.4, 3.6, fc=F_PALE, z=1.5)
        fbox(3.0, 42.2, 82.4, 8.6, fc=F_PALE, z=1.2)
        fbox(3.0, 42.2, 1.15, 8.6, fc=F_NAVY, z=3)
        fbox(3.0, 3.8, 82.4, 4.0, fc=F_PALE, z=1.5)
        sline([3.0, 85.4], [54.4, 54.4], c=F_INK, lw=0.75, z=3)
        sline([3.0, 85.4], [50.8, 50.8], c=F_GREY, lw=0.5, z=3)
        sline([3.0, 85.4], [7.8, 7.8], c=F_MID, lw=0.45, z=3)
        sline([3.0, 85.4], [3.8, 3.8], c=F_INK, lw=0.75, z=3)

        tx(3.6, 52.6, "Model family", fontsize=6.3, fontweight="bold")
        tx(22.8, 52.6, "Formulation & transfer", fontsize=6.3, fontweight="bold")
        tx(64.4, 52.6, "Physics", fontsize=6.3, fontweight="bold", ha="center")
        tx(72.2, 52.6, "Models", fontsize=6.3, fontweight="bold")

        rows_c = [
            (42.2, 8.6, "Forward hybrid", "& process",
             "Hybrid (eq. + frozen xLSTM, 17 ch)", r"Standalone process eq. ($\varepsilon = 0$)",
             "Forecast", F_NAVY, "split_1_1", "1 + 1", True),
            (33.6, 8.6, "Reference &", "time-series",
             "Persistence, Fourier, damped;", "station ARIMA & harmonic ARIMA",
             "None", F_GREY, 5, "5", False),
            (25.0, 8.6, "Boosted trees", "(XGBoost)",
             "Station & cross-basin zero-shot", "(17 core & 32 full channels)",
             "None", F_GREY, 4, "4", False),
            (16.4, 8.6, "Standalone", "xLSTM",
             "Zero-shot & fine-tuned (17 & 32 ch);", "target joint (17, 32 & Chl-a ch)",
             "None", F_GREY, 7, "7", False),
            (7.8,  8.6, "Loss-term PINN", "& ensemble",
             "Monod loss, zero-shot & fine-tuned", "(17 & 32 ch); +6-model mean",
             "Loss", F_DARK, "plus_4_1", "4 (+1)", False),
        ]
        row_mids = []
        bx_m = 72.2
        for idx, (y0, rh, f1, f2, d1, d2, phy, pc, bar_spec, cnt_s, hi) in enumerate(rows_c):
            if idx > 0:
                sline([3.0, 85.4], [y0 + rh, y0 + rh], c=F_LIGHT, lw=0.45, z=2.5)
            ym = y0 + rh * 0.50
            row_mids.append(ym)
            lx = 5.0 if hi else 3.6
            tx(lx, y0 + rh * 0.67, f1, fontsize=6.35, fontweight="bold", color=F_NAVY if hi else F_INK)
            tx(lx, y0 + rh * 0.30, f2, fontsize=6.2, color=F_DARK)
            tx(22.8, y0 + rh * 0.67, d1, fontsize=6.15, fontweight="bold" if hi else "normal")
            tx(22.8, y0 + rh * 0.30, d2, fontsize=6.15, color=F_DARK)
            tx(64.4, ym, phy, fontsize=6.2, fontweight="bold" if hi or phy == "Loss" else "normal", color=pc, ha="center")
            if bar_spec == "split_1_1":
                fbox(bx_m, ym - 1.2, 1.15, 2.4, fc=F_NAVY, z=3)
                fbox(bx_m + 1.60, ym - 1.2, 1.15, 2.4, fc=F_GREY, z=3)
                bw_tot = 2.75
            elif bar_spec == "plus_4_1":
                fbox(bx_m, ym - 1.2, 3.8, 2.4, fc=F_MID, z=3)
                sbox(bx_m + 4.25, ym - 1.2, 1.0, 2.4, ec=F_GREY, fc=F_WHITE, lw=0.5, z=3)
                bw_tot = 5.25
            else:
                bw_tot = float(bar_spec) * 0.95
                fbox(bx_m, ym - 1.2, bw_tot, 2.4, fc=F_MID, z=3)
            tx(bx_m + bw_tot + 0.85, ym, cnt_s, fontsize=6.15, fontweight="bold" if hi else "normal",
               color=F_NAVY if hi else F_INK)

        tx(3.6, 5.8, "Total: 22 single models (ML on 17 & 32 ch) + 6-model ensemble mean", fontsize=6.15, color=F_DARK)

        # Right-side 5-row manifold bus spanning all 5 benchmark families -> H1 & Bounds
        sline([87.8, 87.8], [9.5, 48.3], c=F_INK, lw=0.65, z=3.5)
        for ym in row_mids:
            sline([85.4, 87.8], [ym, ym], c=F_INK, lw=0.55, z=3.5)
        sarrow(87.8, 48.3, 94.0, 48.3, c=F_INK, lw=0.7)
        sarrow(87.8, 9.5, 94.0, 9.5, c=F_INK, lw=0.7)

        # ---- Panel d: Unified H1-H3 block + separated Bounds card ----
        tx(94.0, 57.6, "d", fontsize=8.0, fontweight="bold")
        tx(99.0, 57.6, "Hypothesis tests (H1–H3) and boundary checks", fontsize=7.0)

        # Unified 3-row container for H1, H2, H3 (exact top y = 54.4)
        h_bot, h_top = 17.6, 54.4
        rh_d = (h_top - h_bot) / 3.0
        sbox(94.0, h_bot, 86.0, h_top - h_bot, ec=F_MID, fc=F_WHITE, lw=0.6)
        fbox(94.0, h_bot, 1.25, h_top - h_bot, fc=F_NAVY, z=3)
        h_rows = [
            ("H1", "Forecast parity & Monod parsimony", "Fig. 4; Tables 4, 6",
             r"Level: median $\Delta R^2 \geq {-}0.02$ at all 3, 7, 14 d (17 & 32 ch)",
             r"Improves: $\Delta R^2 > 0$, Wilcoxon $p < 0.05$ at $\geq 2$ horizons; +5-const. Monod"),
            ("H2", "Exact 6-term decomposition & turnover", "Figs. 5, 6; Table 5",
             r"Linear $\ln\,C$ split: smoothing, aging, mean & seasonal return, rain, $\varepsilon$",
             "20 °C return half-life & flushing: 7 reservoirs vs. 20 rivers"),
            ("H3", "Hard-constraint removals & perturbations", "Fig. 7; Table 6",
             "Single-constraint removals: free HC1 ranges vs. unbounded HC2",
             r"Range-exit check; +10 mm d$^{-1}$ rain & $\pm 3$ °C perturbations"),
        ]
        for idx, (b_tag, head, ref_tag, l1, l2) in enumerate(h_rows):
            ry0 = h_top - (idx + 1) * rh_d
            if idx > 0:
                sline([95.25, 180.0], [ry0 + rh_d, ry0 + rh_d], c=F_LIGHT, lw=0.45, z=2.5)
            tx(96.8, ry0 + 9.55, b_tag, fontsize=6.55, fontweight="bold", color=F_NAVY)
            tx(101.3, ry0 + 9.55, f"·  {head}", fontsize=6.5, fontweight="bold", color=F_INK)
            tx(177.8, ry0 + 9.55, ref_tag, fontsize=6.05, color=F_GREY, ha="right")
            tx(96.8, ry0 + 6.05, l1, fontsize=6.3, color=F_INK)
            tx(96.8, ry0 + 2.65, l2, fontsize=6.3, color=F_DARK)

        # Separated Bounds card (exact bottom y = 3.8 aligned with Panel c bottom rule)
        sbox(94.0, 3.8, 86.0, 11.4, ec=F_MID, fc=F_WHITE, lw=0.6)
        fbox(94.0, 3.8, 1.25, 11.4, fc=F_RED, z=3)
        tx(96.8, 12.65, "Bounds", fontsize=6.55, fontweight="bold", color=F_RED)
        tx(107.2, 12.65, "·  Short records, shifts, alerts & proxy", fontsize=6.5, fontweight="bold", color=F_INK)
        tx(177.8, 12.65, "Figs. 8, S1; Table 7", fontsize=6.05, color=F_GREY, ha="right")
        tx(96.8, 9.35, r"12-mo short records; post-record shifts; 15 & 25 mg m$^{-3}$ CSI", fontsize=6.3, color=F_INK)
        tx(96.8, 6.15, f"Geum cyanobacteria proxy: {n_alg:,} lab & {n_sen:,} sensor pairs (8 sites)", fontsize=6.3, color=F_DARK)

        save(fig, "fig_framework")
    finally:
        _mathtext.SHRINK_FACTOR = old_shrink


# ------------------------------------------------------------------------------------------------ Fig. 3: the model

def fig2():
    old_shrink = _mathtext.SHRINK_FACTOR
    _mathtext.SHRINK_FACTOR = 0.80
    try:
        H = 88.0
        fig = plt.figure(figsize=(WIDTH, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 183); ax.set_ylim(0, H); ax.axis("off")
        PROC, NET = "#3d3d3d", TEAL

        def box(x, y, w, h, ec=INK, fc="white", lw=0.6, ls="-"):
            ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, ls=ls, joinstyle="miter", zorder=2))

        def arrow(x0, y0, x1, y1, c=INK, lw=0.7, hl=1.55, hw=0.82):
            dx, dy = x1 - x0, y1 - y0
            ln = float(np.hypot(dx, dy))
            if ln <= 1e-6:
                return
            ux, uy = dx / ln, dy / ln
            nx, ny = -uy, ux
            ax.plot([x0, x1 - ux * hl * 0.85], [y0, y1 - uy * hl * 0.85], color=c, lw=lw,
                    solid_capstyle="butt", solid_joinstyle="miter", zorder=3)
            ax.add_patch(Polygon([(x1, y1),
                                  (x1 - ux * hl + nx * hw, y1 - uy * hl + ny * hw),
                                  (x1 - ux * hl - nx * hw, y1 - uy * hl - ny * hw)],
                                 closed=True, facecolor=c, edgecolor="none", zorder=3.1))

        def txt(x, y, s, **kw):
            kw = {"fontsize": 6.5, "ha": "left", "va": "center", "color": INK, "zorder": 4, **kw}
            ax.text(x, y, s, **kw)

        # inputs
        txt(3, 83.8, "a", fontsize=8, fontweight="bold"); txt(8, 83.8, "Inputs at the forecast day", fontsize=7)
        ins = [
            (63.2, 10.6, "Chl-a", "last measured value, its age,\n7-day mean"),
            (50.2, 10.8, "Station record", "mean $m$ and seasonal\ncurve $q$ of ln Chl-a"),
            (40.2, 8.0, "Water temperature", "7-day mean $T$"),
            (30.2, 8.0, "Rain", "7-day mean $R$"),
        ]
        for yb, hb_, head, sub in ins:
            box(3, yb, 40, hb_, ec=MID)
            if "\n" in sub:
                txt(5, yb + hb_ - 2.4, head, fontweight="bold")
                txt(5, yb + 3.5, sub, fontsize=6.25, color="#444444", linespacing=1.35)
            else:
                txt(5, yb + hb_ - 2.3, head, fontweight="bold")
                txt(5, yb + 2.3, sub, fontsize=6.25, color="#444444")
        box(3, 8.0, 40, 12.6, ec=NET)
        txt(5, 17.8, "Last 30 days, 17 channels", fontweight="bold", color=NET)
        txt(5, 12.3, "Chl-a, water temperature,\nTN, TP and four weather variables", fontsize=6.2, color="#444444", linespacing=1.32)

        # process equation
        txt(52, 83.8, "b", fontsize=8, fontweight="bold"); txt(57, 83.8, "Process equation: one step a day, 14 days ahead", fontsize=7)
        box(52, 27.6, 86, 51.4, ec=PROC, lw=0.8)
        txt(95, 73.2, r"$x_k \;=\; x_{k-1} \;-\; \kappa\,\theta^{\,T-20}\,(x_{k-1} - L_k) \;-\; \beta\,R\,e^{-k/3} \;+\; \varepsilon$", fontsize=7.8, ha="center")
        for x0_, x1_, c in ((79.1, 104.7, PROC), (105.8, 118.8, PROC), (119.9, 124.4, NET)):
            ax.plot([x0_, x0_, x1_, x1_], [70.9, 70.2, 70.2, 70.9], color=c, lw=0.5, solid_capstyle="butt", solid_joinstyle="miter", zorder=3)
        txt(87.5, 65.8, "return to the station level,\nfaster in warm water", fontsize=6.2, ha="center", color=PROC, linespacing=1.32)
        txt(111.0, 65.8, "flushing by rain,\n3-day e-folding", fontsize=6.15, ha="center", color=PROC, linespacing=1.32)
        txt(128.0, 65.8, "bounded\ncorrection", fontsize=6.2, ha="center", color=NET, linespacing=1.32)
        txt(54.8, 58.4, r"$x = \ln$ Chl-a;  $x_0 = \alpha \ln C_{\rm last} + (1-\alpha)\,\bar{x}_7$ (aged to $x_t$);  $L_k = m + \omega\,[\,q(t+k) - m\,]$", fontsize=6.25, ha="left", color="#444444")

        # hard constraint 1: ranges of the five station constants
        box(54.8, 29.8, 58.6, 24.4, ec=MID, fc=PALE, lw=0.45)
        txt(56.8, 51.4, "Hard constraint 1: ranges of the five station constants", fontsize=6.2, fontweight="bold")
        rows = [
            (r"$\alpha$", "weight of the last measured value", "0 to 1"),
            (r"$\omega$", "weight of the seasonal curve", "0 to 1"),
            (r"$\kappa$", "rate of return at 20 °C", r"0 to 0.5 d$^{-1}$"),
            (r"$\theta$", "temperature coefficient", "0.9 to 1.1"),
            (r"$\beta$", "flushing per mm of rain", r"0 to 0.05 mm$^{-1}$"),
        ]
        for i, (sym, what, rng_) in enumerate(rows):
            y = 47.6 - i * 3.75
            txt(57.2, y, sym, fontsize=6.8)
            txt(61.8, y, what, fontsize=6.2)
            txt(111.8, y, rng_, fontsize=6.35, ha="right")

        # hard constraint 2: bound on the correction
        box(115.6, 29.8, 20.2, 24.4, ec=NET, fc="white", lw=0.55)
        txt(125.7, 51.4, "Hard\nconstraint 2", fontsize=6.2, fontweight="bold", color=NET, ha="center", va="top", linespacing=1.28)
        txt(125.7, 41.2, r"$\varepsilon = 0.1\tanh(g\,z)$", fontsize=6.8, ha="center", color=NET)
        txt(125.7, 34.6, "at most 0.1\na day in ln $C$", fontsize=6.2, ha="center", color=NET, linespacing=1.35)
        arrow(127.5, 54.2, 128.0, 62.2, c=NET, lw=0.6)

        # network
        box(52, 8.0, 86, 12.6, ec=NET, lw=0.8)
        txt(54.5, 17.5, "Network: xLSTM with 51,332 weights, one number $z$ per station and day", fontsize=6.4, fontweight="bold", color=NET)
        txt(54.5, 12.0, "trained on the three other basins, then kept fixed at the target basin", fontsize=6.2, color="#444444")
        arrow(43, 14.3, 52, 14.3, c=NET); arrow(125.7, 20.6, 125.7, 29.8, c=NET); txt(127.3, 25.0, "$z$, gate $g$", fontsize=6.35, color=NET)
        for y in (68.5, 55.6, 44.2, 34.2):
            arrow(43, y, 52, y, c=MID)

        # output and reading
        txt(146, 83.8, "c", fontsize=8, fontweight="bold"); txt(151, 83.8, "Output", fontsize=7)
        arrow(138, 61.5, 146, 61.5, c=PROC, lw=0.8)
        box(146, 52.0, 34, 18.6, ec=INK, lw=0.8)
        txt(163, 65.8, "Forecast of Chl-a", fontweight="bold", ha="center")
        txt(163, 58.6, "1, 3, 7 and 14 days ahead,\nevery day, every station", fontsize=6.15, ha="center", color="#444444", linespacing=1.32)
        arrow(163, 52.0, 163, 45.8, c=INK, lw=0.6)
        box(146, 22.8, 34, 23.0, ec=MID, fc=PALE, lw=0.45)
        txt(163, 42.2, "Read term by term", fontweight="bold", ha="center")
        txt(163, 32.0, "the forecast change is an\nexact sum of named terms;\nthe constants give a half-life\nand a rain effect with units", fontsize=6.15, ha="center", color="#444444", linespacing=1.30)
        # training steps
        txt(146, 18.2, "Training", fontweight="bold")
        txt(146, 10.6, "1  other three basins: network,\n    gate and constants together\n2  target basin: five constants per\n    station and the gate only", fontsize=6.1, color="#444444", linespacing=1.28)
        save(fig, "fig2_model")
    finally:
        _mathtext.SHRINK_FACTOR = old_shrink


# ------------------------------------------------------------------------------------------------ Fig. 4: accuracy
DISPLAY = {"hybrid": "Hybrid", "core": "Process model", "persistence": "Persistence", "climatology": "Climatology", "damped anomaly": "Damped anomaly",
           "ARIMA": "ARIMA", "harmonic ARIMA": "Harmonic ARIMA", "XGBoost": "XGBoost", "XGBoost zero-shot": "XGBoost zero-shot",
           "xLSTM zero-shot": "xLSTM zero-shot", "xLSTM fine-tuned": "xLSTM fine-tuned", "PINN-sigmoid zero-shot": "Loss-term PINN zero-shot",
           "PINN-sigmoid fine-tuned": "Loss-term PINN fine-tuned", "bb-nut": "Joint network", "bb-chl": "Joint network, Chl-a only",
           "bb-all": "Joint network, 32 channels", "mean of six": "Mean of six"}
HX = {1: 0, 3: 1, 7: 2, 14: 3}


def fig3():
    S = pd.read_csv(RES / "stage7" / "test" / "more_summary.csv"); S = S[(S.record == 36) & S.model.isin(DISPLAY)].set_index("model")
    P = pd.read_csv(RES / "stage7" / "test" / "paired.csv"); P = P[(P.record == 36) & (P.model == "hybrid") & (P.horizon != "mean3_7_14")].assign(horizon=lambda d: d.horizon.astype(int))
    fig = plt.figure(figsize=(WIDTH, 84 * MM))
    # a: one strip per horizon, every single model as a tick
    others = [m for m in DISPLAY if m not in ("hybrid", "core", "persistence", "climatology", "mean of six")]
    lims = {1: (0.852, 0.882), 3: (0.578, 0.648), 7: (0.26, 0.43), 14: (-0.05, 0.27)}; x0, w, gap = 0.075, 0.064, 0.036
    for i, h in enumerate((1, 3, 7, 14)):
        ax = fig.add_axes([x0 + i * (w + gap), 0.20, w, 0.68]); c = f"median_r2_{h}"
        for m in others:
            ax.plot([-0.32, 0.32], [S.loc[m, c]] * 2, color=MID, lw=0.7, zorder=1)
        ax.plot([-0.46, 0.46], [S.loc["persistence", c]] * 2, color=INK, lw=1.0, ls=(0, (3, 1.5)), zorder=2)
        ax.plot([-0.46, 0.46], [S.loc["mean of six", c]] * 2, color=GREY, lw=1.0, ls=(0, (1, 1.2)), zorder=2)
        ax.plot([-0.46, 0.46], [S.loc["core", c]] * 2, color=TEAL, lw=1.4, zorder=3)
        ax.plot([-0.46, 0.46], [S.loc["hybrid", c]] * 2, color=NAVY, lw=1.8, zorder=4); ax.plot(0, S.loc["hybrid", c], "o", ms=3.6, mfc=NAVY, mec="white", mew=0.5, zorder=5)
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(*lims[h]); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
        ax.set_title(f"{h} day" + ("s" if h > 1 else ""), fontsize=7, pad=3); ax.tick_params(axis="y", labelsize=6, pad=1.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        if i == 0:
            ax.set_ylabel("$R^2$, median of the 27 stations", labelpad=2); letter(ax, "a", "Accuracy at each horizon", dx=-30, dy=14)
    fig.legend(handles=[Line2D([], [], color=NAVY, lw=1.8, marker="o", ms=3.6, mec="white", mew=0.5, label="Hybrid"),
                        Line2D([], [], color=TEAL, lw=1.4, label="Process model"),
                        Line2D([], [], color=MID, lw=0.9, label="Other single models (12)"),
                        Line2D([], [], color=GREY, lw=1.0, ls=(0, (1, 1.2)), label="Mean of six"),
                        Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 1.5)), label="Persistence")],
               loc="lower left", bbox_to_anchor=(0.045, 0.015), ncol=3, columnspacing=1.2, labelspacing=0.35, handletextpad=0.5)

    # b: hybrid minus each comparison model, median paired difference of the station R2
    bx = fig.add_axes([0.655, 0.135, 0.335, 0.745])
    order = [m for m in S.sort_values("median_r2_mean3_7_14", ascending=False).index if m not in ("hybrid", "climatology")]
    order = ["mean of six"] + [m for m in order if m != "mean of six"]
    mk = {3: "o", 7: "s", 14: "^"}; off = {3: -0.24, 7: 0.0, 14: 0.24}
    bx.axvspan(-0.065, -0.02, color=PALE, zorder=0); bx.axvline(0, color=INK, lw=0.5, zorder=1)
    for i, m in enumerate(order):
        for h in (3, 7, 14):
            r = P[(P.against == m) & (P.horizon == h)].iloc[0]; sig = r.wilcoxon_p < 0.05; c = NAVY if r.median_diff >= 0 else RED
            bx.plot(r.median_diff, i + off[h], mk[h], ms=3.4 if h != 14 else 3.8, mfc=c if sig else "white", mec=c, mew=0.7, zorder=3)
    bx.set_yticks(range(len(order))); bx.set_yticklabels([DISPLAY[m] for m in order], fontsize=6.3); bx.set_ylim(len(order) - 0.5, -0.6)
    bx.set_xlim(-0.065, 0.235); bx.set_xticks([-0.05, 0, 0.05, 0.10, 0.15, 0.20]); bx.set_xticklabels(["−0.05", "0", "0.05", "0.10", "0.15", "0.20"])
    bx.tick_params(axis="y", length=0, pad=3); bx.spines["left"].set_visible(False)
    bx.axhline(0.5, color=LIGHT, lw=0.5, zorder=0)
    bx.set_xlabel("Hybrid minus the model (median difference in station $R^2$)")
    bx.text(-0.0425, 8.0, "below −0.02: not level", fontsize=6.0, color=GREY, ha="center", va="center", rotation=90)
    bx.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.4, mfc="white", mec=INK, mew=0.7, label="3 days"),
                       Line2D([], [], marker="s", ls="", ms=3.4, mfc="white", mec=INK, mew=0.7, label="7 days"),
                       Line2D([], [], marker="^", ls="", ms=3.8, mfc="white", mec=INK, mew=0.7, label="14 days"),
                       Line2D([], [], marker="s", ls="", ms=3.4, mfc=INK, mec=INK, mew=0.7, label="filled: $p$ < 0.05")],
              loc="center right", bbox_to_anchor=(1.0, 0.40), labelspacing=0.4, handletextpad=0.4)
    letter(bx, "b", "Hybrid minus each comparison model", dx=-118, dy=14)
    save(fig, "fig3_accuracy")


# ------------------------------------------------------------------------------------------------ Fig. 5: terms and constants
TERM_ORDER = [("return_mean", "Return to the record mean", "#3d3d3d"), ("return_season", "Seasonal part of the station level", "#7f7f7f"), ("ageing", "Aging of the starting value", "#ababab"),
              ("flushing", "Rain flushing", "#c9c9c9"), ("smoothing", "Smoothing of the starting value", "#e6e6e6"), ("network", "Correction of the network", TEAL)]


def stack(ax, y, shares, h=0.62, text=True):
    left = 0.0
    for (k, _, c), v in zip(TERM_ORDER, shares):
        ax.barh(y, v, left=left, height=h, color=c, edgecolor="white", linewidth=0.4, zorder=2)
        if text and v >= 0.055:
            if c in ("#3d3d3d", "#7f7f7f", TEAL):
                ax.text(left + v / 2, y, f"{100 * v:.0f}", ha="center", va="center", fontsize=6, color=INK,
                        bbox=dict(boxstyle="square,pad=0.10", facecolor="#f1f1f1", edgecolor="none"), zorder=3)
            else:
                ax.text(left + v / 2, y, f"{100 * v:.0f}", ha="center", va="center", fontsize=6, color=INK, zorder=3)
        left += v


def fig4():
    T = pd.read_csv(RES / "explain" / "terms.csv"); T = T[T.record == 36]
    K = pd.read_csv(RES / "explain" / "constants_station.csv", dtype={"station": str}); K = K[K.record == 36].copy()
    typ = pd.read_csv(ROOT / "data" / "national" / "stations.csv", dtype=str).set_index("code")["type"]; K["type"] = K.station.map(typ)
    BT = pd.read_csv(RES / "strength" / "by_type_terms.csv")
    sh = lambda net, h: [float(T[(T.net == net) & (T.horizon == h) & (T.term == k)].share_abs_seed_mean.iloc[0]) for k, _, _ in TERM_ORDER]
    fig = plt.figure(figsize=(WIDTH, 96 * MM))
    ax = fig.add_axes([0.15, 0.215, 0.40, 0.70]); rows, y = [], 0.0
    for h in (3, 7, 14):
        stack(ax, y, sh("all", h)); rows.append((y, f"{h} days")); y += 1.0
    y += 0.7; y_b = y
    for k, full in [("han", "Han"), ("nakdong", "Nakdong"), ("geum", "Geum"), ("yeongsan", "Yeongsan–Seomjin")]:
        stack(ax, y, sh(k, 7)); rows.append((y, full)); y += 1.0
    y += 0.7; y_t = y
    for k, full in [("river", "River stations (20)"), ("lake", "Reservoir stations (7)")]:
        g = BT[(BT.record == 36) & (BT.type == k) & (BT.horizon == 7)]
        v = np.array([float(g[c].iloc[0]) for c, _, _ in TERM_ORDER]); stack(ax, y, list(v / v.sum())); rows.append((y, full)); y += 1.0
    ax.set_yticks([r[0] for r in rows]); ax.set_yticklabels([r[1] for r in rows]); ax.set_ylim(y - 0.4, -0.75); ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25", "50", "75", "100"]); ax.tick_params(axis="y", length=0, pad=3); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of the summed term sizes (%)")
    for yy, s in ((-0.62, "All stations, by horizon"), (y_b - 0.62, "By basin, 7 days"), (y_t - 0.62, "By station type, 7 days")):
        ax.text(0.0, yy, s, fontsize=6.3, color=GREY, ha="left", va="center", style="italic")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none", label=lab) for _, lab, c in TERM_ORDER], loc="lower left", bbox_to_anchor=(0.012, 0.005), ncol=2,
               columnspacing=1.2, labelspacing=0.35, handlelength=1.1, handletextpad=0.5)
    letter(ax, "a", "Additive components of forecast change", dx=-60, dy=6)

    # b: half-life of the return against water temperature, one line per station
    bx = fig.add_axes([0.655, 0.585, 0.33, 0.33]); Tw = np.linspace(2, 32, 121)
    hl = lambda k, th: np.log(0.5) / np.log1p(-np.minimum(k * th ** (Tw - 20), 0.9))
    for r in K.itertuples():
        is_res = r.type == "lake"
        bx.plot(Tw, hl(r.kappa, r.thk), color=RED if is_res else NAVY, ls=(0, (3, 1.5)) if is_res else "-",
                lw=0.5, alpha=0.45, zorder=2 if is_res else 1)
    for kind, c, ls_m in (("river", NAVY, "-"), ("lake", RED, (0, (4.2, 1.8)))):
        g = K[K.type == kind]; bx.plot(Tw, np.median(np.stack([hl(r.kappa, r.thk) for r in g.itertuples()]), 0), color=c, ls=ls_m, lw=1.8, zorder=3)
    bx.set_xlim(2, 32); bx.set_ylim(0, 40); bx.set_xlabel("Water temperature (°C)"); bx.set_ylabel("Half-life of the return (days)")
    bx.legend(handles=[Line2D([], [], color=RED, ls=(0, (4.2, 1.8)), lw=1.8, label="Reservoir stations, median"),
                       Line2D([], [], color=NAVY, ls="-", lw=1.8, label="River stations, median"),
                       Line2D([], [], color=GREY, lw=0.6, label="Single stations")], loc="upper right", labelspacing=0.35, borderaxespad=0.1)
    letter(bx, "b", "Return to the station level", dx=-30, dy=6)

    # c: effect of rain on the 7-day forecast, every station
    cx = fig.add_axes([0.655, 0.115, 0.33, 0.30]); g = K.sort_values("rain10mm_7d_pct").reset_index(drop=True)
    for idx_b, r_b in enumerate(g.itertuples()):
        if r_b.type == "lake":
            cx.add_patch(Rectangle((idx_b - 0.36, r_b.rain10mm_7d_pct), 0.72, -r_b.rain10mm_7d_pct, facecolor="white", edgecolor=RED, lw=0.85, joinstyle="miter", zorder=2.2))
        else:
            cx.add_patch(Rectangle((idx_b - 0.36, r_b.rain10mm_7d_pct), 0.72, -r_b.rain10mm_7d_pct, facecolor=NAVY, edgecolor="none", zorder=2))
    cx.axhline(0, color=INK, lw=0.5); cx.set_xlim(-0.8, len(g) - 0.2); cx.set_ylim(float(g.rain10mm_7d_pct.min()) * 1.08, 0.6); cx.set_xticks([]); cx.spines["bottom"].set_visible(False)
    cx.set_ylabel("Change of the 7-day\nforecast (%)"); cx.set_xlabel("The 27 stations, ordered by the effect")
    cx.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=NAVY, edgecolor="none", label="River"),
                       Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=RED, lw=0.85, label="Reservoir")],
              loc="lower right", handlelength=1.0, labelspacing=0.3)
    letter(cx, "c", "Effect of 10 mm of rain a day in the last week", dx=-30, dy=6)
    save(fig, "fig4_terms")


# ------------------------------------------------------------------------------------------------ Fig. 6: two stations
def fig5():
    import matplotlib.dates as mdates
    C = pd.read_csv(RES / "explain" / "case_series.csv", dtype={"station": str}, parse_dates=["anchor_date", "target_date"])
    fig = plt.figure(figsize=(WIDTH, 112 * MM)); cols = [("S03009", "Hyeondo, a river station of the Geum basin", 0.065), ("S04007", "Tamjin, a reservoir station of the Yeongsan–Seomjin basin", 0.555)]
    for j, (st, name, x0) in enumerate(cols):
        g = C[C.station == st].sort_values("target_date"); w = 0.415
        ax = fig.add_axes([x0, 0.50, w, 0.41]); obs = g[g.observed]
        ax.axhline(float(np.exp(g.record_mean_ln.iloc[0])), color=MID, lw=0.6, ls=(0, (4, 2)), zorder=1)
        ax.plot(g.target_date, g.base_raw, color=LIGHT, lw=0.9, zorder=2)
        ax.plot(obs.target_date, obs.y_true_raw, "o", ms=1.5, mfc=INK, mec="none", zorder=3)
        ax.plot(g.target_date, g.forecast, color=NAVY, lw=1.0, zorder=4)
        ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-07")); ax.set_ylim(0, None); ax.set_ylabel("Chl-a (mg m$^{-3}$)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.set_xticklabels([])
        letter(ax, "ab"[j], name, dx=-28, dy=6)
        if j == 1:
            ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=2.2, mfc=INK, mec="none", label="Measured"), Line2D([], [], color=NAVY, lw=1.0, label="Forecast made 7 days before"),
                               Line2D([], [], color=LIGHT, lw=1.2, label="Last measured value 7 days before"), Line2D([], [], color=MID, lw=0.6, ls=(0, (4, 2)), label="Record mean of the station")],
                      loc="upper right", labelspacing=0.3, borderaxespad=0.1, handletextpad=0.4)
        bx = fig.add_axes([x0, 0.10, w, 0.32]); bx.axhline(0, color=INK, lw=0.4, zorder=1)
        other = g.ageing + g.smoothing
        bx.plot(g.target_date, g.return_mean, color="#3d3d3d", lw=1.0, zorder=3); bx.plot(g.target_date, g.return_season, color="#8a8a8a", lw=0.9, zorder=2)
        bx.plot(g.target_date, g.flushing, color=RED, lw=0.7, zorder=2); bx.plot(g.target_date, g.network, color=TEAL, lw=0.9, zorder=4)
        bx.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-07")); bx.set_ylabel("Term of the 7-day change (ln $C$)")
        bx.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); bx.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y")); lim = float(np.nanmax(np.abs(g[["return_mean", "return_season", "flushing", "network"]].to_numpy()))) * 1.08
        bx.set_ylim(-lim, lim)
        if j == 1:
            bx.legend(handles=[Line2D([], [], color="#3d3d3d", lw=1.0, label="Return to the record mean"), Line2D([], [], color="#8a8a8a", lw=0.9, label="Seasonal part of the station level"),
                               Line2D([], [], color=RED, lw=0.7, label="Rain flushing"), Line2D([], [], color=TEAL, lw=0.9, label="Correction of the network")],
                      loc="upper left", bbox_to_anchor=(0.0, 1.06), labelspacing=0.3, borderaxespad=0.1, ncol=2, columnspacing=1.0, handletextpad=0.4)
    save(fig, "fig5_stations")


# ------------------------------------------------------------------------------------------------ Fig. 7: what the constraints give
def fig6():
    M = RES / "stage8" / "more"
    T = pd.read_csv(M / "terms.csv"); T = T[(T.record == 36) & (T.net == "all") & (T.horizon == 7)].set_index(["model", "term"])
    W = pd.read_csv(M / "constants_free.csv", dtype={"station": str}); W = W[W.record == 36]
    R = pd.read_csv(RES / "response" / "response.csv"); R = R[(R.scope == "all") & (R.perturbation == "rain +10 mm/d") & (R.part == "all")].set_index(["model", "horizon"])
    S = pd.read_csv(M / "spread.csv"); S = S[S.record == 36].set_index("model"); F = pd.read_csv(M / "spread_forecast.csv"); F = F[(F.record == 36) & (F.horizon == 7)].set_index("model")
    fig = plt.figure(figsize=(WIDTH, 128 * MM))

    # a: size of the process terms, of the network's correction and of the forecast change
    ax = fig.add_axes([0.075, 0.60, 0.385, 0.33]); groups = [("main", "Hybrid"), ("free-eps", "Without the bound\non the correction"), ("free-const", "Without the ranges\nof the constants")]
    for i, (m, lab) in enumerate(groups):
        pr, ne, to = (float(T.loc[(m, k), "mean_abs_seed_mean"]) for k in ("process_all", "network", "total_change")); cv = float(T.loc[(m, "network"), "cov_share_seed_mean"])
        ax.bar(i - 0.26, pr, 0.24, color="#4d4d4d", zorder=2); ax.bar(i, ne, 0.24, color=TEAL, zorder=2); ax.bar(i + 0.26, to, 0.24, facecolor="white", edgecolor=INK, lw=0.7, zorder=2)
        for dx, v in ((-0.26, pr), (0.0, ne), (0.26, to)):
            ax.text(i + dx, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.text(i, 0.455, f"{100 * cv:+.0f}%".replace("-", "−"), ha="center", va="center", fontsize=7, color=TEAL, fontweight="bold")
    ax.text(-0.52, 0.497, "Share of the network in the variance of the forecast change:", ha="left", va="center", fontsize=6.3, color=TEAL)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[1] for g in groups]); ax.set_ylim(0, 0.52); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4]); ax.set_xlim(-0.55, 2.55); ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("Mean absolute size at 7 days (ln $C$)")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, facecolor="#4d4d4d", edgecolor="none", label="Process terms together"), Rectangle((0, 0), 1, 1, facecolor=TEAL, edgecolor="none", label="Correction of the network"),
                        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, lw=0.7, label="Forecast change")], loc="lower left", bbox_to_anchor=(0.06, 0.492), ncol=3, columnspacing=1.0,
               handlelength=1.0, handletextpad=0.4, borderaxespad=0.0)
    letter(ax, "a", "The bound keeps the forecast in the process terms", dx=-30, dy=6)

    # b: two constants at every station, with and without the ranges
    for j, (c, lab, lo, hi, ylim, bound, sc) in enumerate([("beta", "Rain constant $\\beta$ (10$^{-3}$ mm$^{-1}$)", 0.0, 50.0, (-8.5, 13.5), 0.0, 1000.0),
                                                           ("alpha", "Weight of the last value $\\alpha$", 0.0, 1.0, (0.84, 1.21), 1.0, 1.0)]):
        bx = fig.add_axes([0.585 + j * 0.235, 0.60, 0.165, 0.33]); g = W.sort_values(f"{c}_main").reset_index(drop=True); xs = np.arange(len(g)); vm, vf = g[f"{c}_main"] * sc, g[f"{c}_free"] * sc
        bx.axhspan(max(lo, ylim[0]), min(hi, ylim[1]), color=PALE, zorder=0); bx.axhline(bound, color=INK, lw=0.5, zorder=1)
        out = (vf < lo) | (vf > hi)
        bx.vlines(xs, vm, vf, color=LIGHT, lw=0.6, zorder=1)
        bx.plot(xs[~out], vf[~out], "o", ms=2.8, mfc="white", mec=RED, mew=0.7, zorder=3); bx.plot(xs[out], vf[out], "o", ms=2.8, mfc=RED, mec=RED, mew=0.7, zorder=3)
        bx.plot(xs, vm, "o", ms=2.6, mfc=NAVY, mec="white", mew=0.3, zorder=4)
        bx.set_ylim(*ylim); bx.set_xlim(-1, len(g)); bx.set_xticks([]); bx.spines["bottom"].set_visible(False); bx.set_ylabel(lab, labelpad=2); bx.set_xlabel("27 stations")
        if j == 0:
            bx.text(0.97, 0.05, f"{int(out.sum())} stations\nbelow zero", transform=bx.transAxes, fontsize=6, color=RED, ha="right", va="bottom", linespacing=1.22)
            letter(bx, "b", "The ranges keep the constants meaningful", dx=-30, dy=6)
        else:
            bx.text(0.04, 0.96, f"{int(out.sum())} stations\nabove one", transform=bx.transAxes, fontsize=6, color=RED, ha="left", va="top", linespacing=1.22)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=2.8, mfc=NAVY, mec="white", mew=0.3, label="Hybrid"), Line2D([], [], marker="o", ls="", ms=2.8, mfc="white", mec=RED, mew=0.7, label="Without the ranges, inside"),
                        Line2D([], [], marker="o", ls="", ms=2.8, mfc=RED, mec=RED, mew=0.7, label="Outside the range"), Rectangle((0, 0), 1, 1, facecolor=PALE, edgecolor=LIGHT, lw=0.3, label="Range")],
               loc="lower left", bbox_to_anchor=(0.555, 0.492), ncol=4, columnspacing=0.8, handletextpad=0.2, handlelength=1.0, borderaxespad=0.0)

    # c: share of the forecasts that rise when rain is added
    cx = fig.add_axes([0.075, 0.095, 0.30, 0.30]); xs = [0, 1, 2]
    lines = [("free-const", "Without the ranges", RED, (0, (4, 1.5)), "s"), ("bb-nut", "Joint network", GREY, "-", "D"), ("free-eps", "Without the bound", RED, "-", "^"),
             ("main", "Hybrid", NAVY, "-", "o"), ("core", "Process model", TEAL, "-", "s")]
    for m, lab, c, ls, mk in lines:
        v = [100 * float(R.loc[(m, h), "share_increase"]) for h in (3, 7, 14)]
        cx.plot(xs, v, color=c, ls=ls, lw=1.6 if m == "main" else 1.1, marker=mk, ms=3.6 if m == "main" else 3.0, mec="white", mew=0.4, zorder=4 if m == "main" else 2)
        cx.annotate(lab, xy=(2, v[-1]), xytext=(5, {"free-const": 3, "bb-nut": -3, "free-eps": 0, "main": 2, "core": -1}[m]), textcoords="offset points", fontsize=6.3, color=c, va="center", ha="left")
    cx.set_xticks(xs); cx.set_xticklabels(["3", "7", "14"]); cx.set_xlim(-0.2, 2.2); cx.set_ylim(-2, 50); cx.set_xlabel("Forecast horizon (days)"); cx.set_ylabel("Forecasts raised by added rain (%)")
    letter(cx, "c", "Direction of the response to rain", dx=-30, dy=6)

    # d: score of single runs and of the averaged forecast against the spread of the forecast between runs
    dx_ = fig.add_axes([0.585, 0.095, 0.395, 0.30])
    pts = [("main", "Hybrid", NAVY, "o", (-5, 9), "right", "single", None), ("free-eps", "Without the bound", RED, "^", None, "left", "single", (0.034, 0.3915)),
           ("free-const", "Without the ranges", RED, "s", None, "left", "single", (0.034, 0.3835)), ("growth", "With the growth term", INK, "D", (7, 0), "left", "ens", None),
           ("no-pre", "Without training\non other basins", INK, "v", (7, 5), "left", "ens", None), ("core", "Process model", TEAL, "s", (6, 0), "left", "single", None),
           ("bb-nut", "Joint network", GREY, "D", (7, 0), "left", "single", None)]
    for m, lab, c, mk, off, ha, at, far in pts:
        x = float(F.loc[m, "sd_mg_median"]); y, e, ens = float(S.loc[m, "single_mean"]), float(S.loc[m, "single_sd"]), float(S.loc[m, "seed_mean_forecast"])
        dx_.plot([x, x], [y, ens], color=c, lw=0.5, ls=(0, (1, 1)), zorder=1)
        dx_.errorbar(x, y, yerr=e, fmt="none", color=c, ecolor=c, elinewidth=0.8, capsize=1.5, capthick=0.6, zorder=2); dx_.plot(x, y, mk, ms=3.8, mfc=c, mec="white", mew=0.4, zorder=3)
        dx_.plot(x, ens, mk, ms=3.8, mfc="white", mec=c, mew=0.8, zorder=3)
        if far is None:
            dx_.annotate(lab, xy=(x, y if at == "single" else ens), xytext=off, textcoords="offset points", fontsize=6.3, color=c, va="center", ha=ha, linespacing=1.22)
        else:                                                                       # label in free space with a thin leader line
            dx_.annotate(lab, xy=(x * 0.93, y - 0.0012), xytext=far, textcoords="data", fontsize=6.3, color=c, va="center", ha="left",
                         arrowprops=dict(arrowstyle="-", color=c, lw=0.4, shrinkA=1, shrinkB=2))
    dx_.set_xscale("log"); dx_.set_xlim(0.02, 4.5); dx_.set_xticks([0.03, 0.1, 0.3, 1.0]); dx_.set_xticklabels(["0.03", "0.1", "0.3", "1"]); dx_.xaxis.set_minor_locator(plt.NullLocator())
    dx_.set_ylim(0.335, 0.435); dx_.set_xlabel("Difference between runs in the 7-day forecast (mg m$^{-3}$)"); dx_.set_ylabel("$R^2$, mean of 3 to 14 days")
    dx_.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.8, mfc=INK, mec="white", mew=0.4, label="Single runs (mean ± SD)"), Line2D([], [], marker="o", ls="", ms=3.8, mfc="white", mec=INK, mew=0.8, label="Forecast averaged over five runs")],
               loc="lower left", bbox_to_anchor=(0.0, 0.0), labelspacing=0.3, handletextpad=0.3, borderaxespad=0.2)
    letter(dx_, "d", "Accuracy and stability of single runs", dx=-30, dy=6)
    save(fig, "fig6_constraints")


# ------------------------------------------------------------------------------------------------ Fig. 8: limits
def fig7():
    S = pd.read_csv(RES / "stage7" / "test" / "more_summary.csv"); S = S[S.model.isin(DISPLAY)].pivot(index="model", columns="record", values="median_r2_mean3_7_14")
    D = pd.read_csv(RES / "explain" / "diagnosis.csv", dtype={"station": str}); D = D[(D.horizon == 7) & (D.model == "hybrid")]
    A = pd.read_csv(RES / "explain" / "alert.csv"); A = A[(A.scope == "all") & (A.record == 36) & (A.threshold == 25.0) & A.model.isin(DISPLAY)]
    fig = plt.figure(figsize=(WIDTH, 74 * MM))

    # a: score with 36-month and with 12-month records
    ax = fig.add_axes([0.075, 0.14, 0.210, 0.74])
    named = {"hybrid": (NAVY, 1.8, "-"), "core": (TEAL, 1.3, "-"), "xLSTM fine-tuned": (RED, 1.3, "-"), "persistence": (INK, 1.0, (0, (3, 1.5))), "mean of six": (GREY, 1.0, (0, (1, 1.2)))}
    for m in S.index:
        if m not in named and m != "climatology":
            ax.plot([0, 1], [S.loc[m, 36], S.loc[m, 12]], color=LIGHT, lw=0.8, zorder=1)
    for m, (c, lw, ls) in named.items():
        ax.plot([0, 1], [S.loc[m, 36], S.loc[m, 12]], color=c, lw=lw, ls=ls, marker="o" if m in ("hybrid", "core", "xLSTM fine-tuned") else None, ms=3.2, mec="white", mew=0.4, zorder=3)
    lab = {"mean of six": ("Mean of six", 5), "xLSTM fine-tuned": ("xLSTM fine-tuned", 0), "hybrid": ("Hybrid", 0), "core": ("Process model", 0), "persistence": ("Persistence", 0)}
    for m, (s, dy) in lab.items():
        ax.annotate(s, xy=(1, S.loc[m, 12]), xytext=(5, dy), textcoords="offset points", fontsize=6.3, color=named[m][0], va="center", ha="left")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["36-month\nrecords", "12-month\nrecords"]); ax.set_xlim(-0.12, 1.12); ax.set_ylim(0.21, 0.45); ax.set_yticks([0.25, 0.30, 0.35, 0.40, 0.45])
    ax.set_ylabel("$R^2$, mean of 3 to 14 days"); ax.text(-0.08, 0.222, "Gray lines: other single models", fontsize=6, color=GREY, ha="left", va="center")
    letter(ax, "a", "One year of records", dx=-30, dy=6)

    # b: level shift of each station after the record against the mean error of the 7-day forecast of the hybrid
    bx = fig.add_axes([0.480, 0.14, 0.200, 0.74])
    for rec, fc in ((36, True), (12, False)):
        g = D[D.record == rec]
        for nak in (False, True):
            q = g[(g.net == "nakdong") == nak]; c = RED if nak else GREY
            bx.plot(q.level_shift_ln, 100 * q.mean_error / q.test_mean, "o" if fc else "o", ms=3.2, mfc=c if fc else "white", mec=c, mew=0.7, ls="", zorder=3 if nak else 2)
    bx.axhline(0, color=INK, lw=0.4, zorder=1); bx.axvline(0, color=INK, lw=0.4, zorder=1)
    bx.set_xlabel("Change of the station level after the record (ln $C$)"); bx.set_ylabel("Mean error of the 7-day forecast\n(% of the station mean)", labelpad=2, linespacing=1.22)
    bx.set_ylim(-70, 18); bx.set_xlim(-1.3, 1.3)
    bx.text(0.50, -13.5, "Nakdong\nstations", fontsize=6.3, color=RED, ha="left", va="center", linespacing=1.22); bx.text(-1.22, 15.5, "Other stations", fontsize=6.3, color=GREY, ha="left", va="center")
    bx.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.2, mfc=INK, mec=INK, mew=0.7, label="36 months"), Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec=INK, mew=0.7, label="12 months")],
              loc="lower left", labelspacing=0.3, handletextpad=0.3, borderaxespad=0.2, title="Record", title_fontsize=6.3, alignment="left")
    letter(bx, "b", "A level that shifts after the record", dx=-30, dy=6)

    # c: critical success index at 25 mg m-3
    cx = fig.add_axes([0.780, 0.14, 0.200, 0.74]); xs = [0, 1, 2, 3]
    v = lambda m: [float(A[(A.model == m) & (A.horizon == h)].csi.iloc[0]) for h in (1, 3, 7, 14)]
    for m in sorted(set(A.model)):
        if m not in ("hybrid", "persistence", "climatology", "mean of six"):
            cx.plot(xs, v(m), color=LIGHT, lw=0.8, zorder=1)
    cx.plot(xs, v("persistence"), color=INK, lw=1.0, ls=(0, (3, 1.5)), zorder=3); cx.plot(xs, v("hybrid"), color=NAVY, lw=1.8, marker="o", ms=3.4, mec="white", mew=0.4, zorder=4)
    cx.set_xticks(xs); cx.set_xticklabels(["1", "3", "7", "14"]); cx.set_xlim(-0.2, 3.2); cx.set_ylim(0.3, 0.85); cx.set_xlabel("Forecast horizon (days)"); cx.set_ylabel("Critical success index at 25 mg m$^{-3}$")
    cx.legend(handles=[Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 1.5)), label="Persistence"), Line2D([], [], color=NAVY, lw=1.8, marker="o", ms=3.4, mec="white", mew=0.4, label="Hybrid"),
                       Line2D([], [], color=LIGHT, lw=0.9, label="Other single models")], loc="upper right", labelspacing=0.3, borderaxespad=0.1)
    letter(cx, "c", "Days above a threshold", dx=-30, dy=6)
    save(fig, "fig7_limits")


# ------------------------------------------------------------------------------------------------ graphical abstract
def ga():
    """Graphical abstract, 13 cm x 5 cm at print size (drawn at 260 mm x 100 mm; written as PDF, PNG and TIFF at 300 dpi)."""
    T = pd.read_csv(RES / "explain" / "terms.csv"); T = T[(T.record == 36) & (T.net == "all") & (T.horizon == 7)].set_index("term").share_abs_seed_mean
    S = pd.read_csv(RES / "stage7" / "test" / "more_summary.csv"); S = S[(S.record == 36) & S.model.isin(DISPLAY)].set_index("model").median_r2_mean3_7_14
    O = pd.read_csv(RES / "stage8" / "more" / "constants_outside.csv").set_index(["record", "model", "constant"])
    W, H = 260.0, 100.0; fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    tx = lambda x, y, s, **kw: ax.text(x, y, s, **{"fontsize": 11, "ha": "left", "va": "center", "color": INK, **kw})
    tx(8, 91, "Hard constraints make an accurate chlorophyll-a forecast interpretable", fontsize=16, fontweight="bold")
    # 1 the model
    tx(8, 77, "1  The equation makes the forecast", fontsize=12.5, fontweight="bold")
    tx(8, 64, r"$x_k = x_{k-1} - \kappa\,\theta^{\,T-20}(x_{k-1} - L_k) - \beta R\,e^{-k/3} + \varepsilon$", fontsize=13)
    tx(8, 53.5, "return to the station level", fontsize=10, color="#3d3d3d"); tx(8, 47, "flushing by rain", fontsize=10, color="#3d3d3d"); tx(8, 40.5, "bounded correction of a network", fontsize=10, color=TEAL)
    ax.add_patch(Rectangle((8, 8), 82, 25, facecolor=PALE, edgecolor="none"))
    tx(11, 27, "Two hard constraints", fontsize=10.5, fontweight="bold"); tx(11, 20, "five station constants inside ranges", fontsize=10); tx(11, 13.5, "correction at most 0.1 a day in ln Chl-a", fontsize=10)
    ax.plot([97, 97], [8, 80], color=LIGHT, lw=0.8)
    # 2 the reading
    tx(104, 77, "2  The 7-day forecast, term by term", fontsize=12.5, fontweight="bold")
    parts = [("Return to the station's\nown level", T["return_mean"] + T["return_season"], "#3d3d3d"), ("Rain, start value", T["flushing"] + T["ageing"] + T["smoothing"], "#b5b5b5"), ("Network", T["network"], TEAL)]
    x0, w, left = 104.0, 68.0, 0.0
    for lab, v, c in parts:
        ax.add_patch(Rectangle((x0 + left * w, 52), v * w, 14, facecolor=c, edgecolor="white", lw=1.0))
        if v > 0.2:
            tx(x0 + (left + v / 2) * w, 59, f"{100 * v:.0f}%", fontsize=15, fontweight="bold", color="white", ha="center")
        left += v
    tx(104, 44, "85% is the return of chlorophyll-a to the", fontsize=10); tx(104, 38, "station's own level, with a half-life of", fontsize=10); tx(104, 32, "about 10 days at 20 °C", fontsize=10)
    tx(104, 22.5, f"rain and start value {100 * float(parts[1][1]):.0f}%", fontsize=10, color="#6a6a6a"); tx(104, 16.5, f"network {100 * float(parts[2][1]):.0f}%", fontsize=10, color=TEAL)
    tx(104, 9, "27 stations, four river basins, South Korea", fontsize=9, color=GREY)
    ax.plot([180, 180], [8, 80], color=LIGHT, lw=0.8)
    # 3 accuracy and constraints
    tx(187, 77, "3  Accurate and interpretable", fontsize=12.5, fontweight="bold")
    others = [m for m in S.index if m not in ("hybrid", "climatology", "mean of six")]; lo, hi = 0.26, 0.42; sx = lambda v: 187 + (v - lo) / (hi - lo) * 64
    ax.plot([187, 251], [58, 58], color=INK, lw=0.6)
    for m in others:
        ax.plot([sx(S[m])] * 2, [56, 60], color=MID, lw=1.0)
    ax.plot(sx(S["hybrid"]), 58, "o", ms=9, mfc=NAVY, mec="white", mew=1.0, zorder=5)
    tx(sx(S["hybrid"]), 66.5, f"Hybrid {S['hybrid']:.2f}", fontsize=10.5, color=NAVY, ha="right", fontweight="bold"); tx(187, 50.5, "$R^2$ of 21 other single models (ticks)", fontsize=9, color=GREY)
    n_out = int(O.loc[(36, "free-const", "any"), "stations_below"])
    tx(187, 39, "Without the hard constraints", fontsize=10.5, fontweight="bold"); tx(187, 32, "accuracy is the same, but", fontsize=10)
    tx(187, 25, f"constants leave their ranges", fontsize=10, color=RED); tx(187, 18.5, f"at {n_out} of 27 stations", fontsize=10, color=RED)
    tx(187, 9, "36-month records, test years 2024–2025", fontsize=9, color=GREY)
    out = ROOT / "output" / "graphical_abstract"; out.mkdir(exist_ok=True)
    fig.savefig(out / "graphical_abstract.pdf"); fig.savefig(out / "graphical_abstract.png", dpi=300); fig.savefig(out / "graphical_abstract.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)
    print("wrote", out / "graphical_abstract.pdf")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, f in {"fig1": fig1, "fig_framework": fig_framework, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7, "ga": ga}.items():
        if what in ("all", k):
            f()
