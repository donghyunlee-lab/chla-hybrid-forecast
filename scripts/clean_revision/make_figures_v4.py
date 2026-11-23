"""Figures of the six-station analysis (Geum River basin); fig6_algae_proxy is Figure S1 of the manuscript. All numbers come
from the frozen result tables
(results/clean_revision_v4/FREEZE_20261010/results); the peak examples are 7-day forecasts taken from the frozen runs
(results/clean_revision_v4/figure_data/peak_examples.csv). Map positions are listed with their sources at STATIONS below.
The basin outline in Fig. 1 needs HydroBASINS level 6 for Asia (hybas_as_lev06_v1c, www.hydrosheds.org; Lehner and Grill, 2013)
in src/gis/HydroBASINS/. Writes output/figures/fig{1..6}_*.{pdf,png}. The data figures are drawn at their
print width (140 mm) with fixed margins, so the manuscript includes them without rescaling.
Usage: python make_figures_v4.py [fig3 fig4 ...]   (no argument: all figures)"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.dates as mdates
import matplotlib.transforms as mtrans
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
warnings.filterwarnings("ignore")
from matplotlib import _mathtext


def _measured_xheight(self, fontname, fontsize, dpi):
    # Arial's PCLT table gives matplotlib an x-height about twice too large, which lifts every superscript;
    # the measured height of "x" places superscripts and subscripts correctly.
    return self.get_metrics(fontname, plt.rcParams["mathtext.default"], "x", fontsize, dpi).iceberg


_mathtext.TruetypeFonts.get_xheight = _measured_xheight

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "clean_revision_v4" / "FREEZE_20261010" / "results"
FD = ROOT / "results" / "clean_revision_v4" / "figure_data"
OUT = ROOT / "output" / "figures"
MM = 1 / 25.4
WIDTH = 140 * MM
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 1.1, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.handlelength": 1.4,
    "pdf.fonttype": 42, "axes.unicode_minus": True})
EN = {"현도": "Hyeondo", "남면": "Nammyeon", "공주": "Gongju", "부여": "Buyeo", "용담호": "Yongdam", "대청호": "Daecheong"}
# one colour per model in every figure; other categories use greys
INK, GREY, LIGHT, REF = "#222222", "#7a7a7a", "#e3e3e3", "#9a9a9a"
PINN, XLSTM, SOFT, NONE = "#D55E00", "#0072B2", "#CC79A7", "#009E73"
H = [1, 3, 7, 14]


def save(fig, name, tight=False):
    OUT.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(OUT / f"{name}.pdf", dpi=600, **kw); fig.savefig(OUT / f"{name}.png", dpi=300, **kw); plt.close(fig)


def title(ax, letter, text, dx=-14, x_fig=None):
    """Panel letter and a short title above the top-left corner; dx is in points from the left spine, or x_fig places
    the title at that fraction of the figure width (used for panels with long category labels)."""
    if x_fig is None:
        ax.annotate(f"({letter})  {text}", xy=(0, 1), xycoords="axes fraction", xytext=(dx, 7), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7.5)
    else:
        ax.annotate(f"({letter})  {text}", xy=(x_fig, 1), xycoords=("figure fraction", "axes fraction"), xytext=(0, 7),
                    textcoords="offset points", ha="left", va="bottom", fontsize=7.5)


def group_xlabel(fig, axes, text):
    """One x-axis label centred under several panels, at the height of an ordinary axis label."""
    first = axes[0]; first.set_xlabel(text); fig.canvas.draw()
    top = first.xaxis.label.get_window_extent().y1
    centre = fig.transFigure.transform(((axes[0].get_position().x0 + axes[-1].get_position().x1) / 2, 0))[0]
    first.xaxis.set_label_coords(*first.transAxes.inverted().transform((centre, top)))


def spread(values, gap):
    """Label positions at least `gap` apart and as close as possible to the values."""
    v = np.asarray(values, float); order = np.argsort(v); p = v[order].copy()
    for _ in range(2):
        for i in range(1, len(p)):
            p[i] = max(p[i], p[i - 1] + gap)
        p += (v[order] - p).mean()
    for i in range(1, len(p)):
        p[i] = max(p[i], p[i - 1] + gap)
    out = np.empty_like(p); out[order] = p
    return out


def interval(ax, pos, est, lo, hi, color, hollow=False, horizontal=False, ms=3.0):
    """Point estimate with a 95% interval as a plain line (no caps)."""
    pos, est, lo, hi = (np.asarray(v, float) for v in (pos, est, lo, hi))
    if horizontal:
        ax.hlines(pos, lo, hi, color=color, lw=0.9, zorder=2); xy = (est, pos)
    else:
        ax.vlines(pos, lo, hi, color=color, lw=0.9, zorder=2); xy = (pos, est)
    ax.plot(*xy, "o", ms=ms, color=color, mfc="white" if hollow else color, mec=color, mew=0.8, zorder=3, ls="none")


def plain_log(ax, axis):
    """Log axis with plain numbers (0.01, 0.1, 1, 10, 1,000) instead of powers of ten."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10)); a.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10))); a.set_minor_formatter(NullFormatter())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))


def models():
    return pd.read_csv(R / "stats" / "models.csv")


def comps():
    return pd.read_csv(R / "stats" / "comparisons.csv")


# ---- Fig. 1: study area ----
def tm_korea(lat, lon):
    """Korea 2000 unified TM (GRS80, lat0 38, lon0 127.5, k0 0.9996, FE 1e6, FN 2e6), Snyder (1987) series. The local
    pyproj installation cannot open its database, so the six station points are projected here; the shapefiles are
    already in this system."""
    a, f = 6378137.0, 1 / 298.257222101; e2 = 2 * f - f * f; ep2 = e2 / (1 - e2); k0, lat0, lon0 = 0.9996, np.radians(38.0), np.radians(127.5)
    def M(p):
        return a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * p - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * p)
                    + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * p) - (35 * e2 ** 3 / 3072) * np.sin(6 * p))
    p, l = np.radians(lat), np.radians(lon); N = a / np.sqrt(1 - e2 * np.sin(p) ** 2); T = np.tan(p) ** 2; C = ep2 * np.cos(p) ** 2; A = (l - lon0) * np.cos(p)
    x = 1e6 + k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120)
    y = 2e6 + k0 * (M(p) - M(lat0) + N * np.tan(p) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24 + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


GIS = ROOT / "src" / "gis"
# name, lat, lon, kind, label offset (km east, km north), label alignment. No official coordinates are published for the
# automatic stations; their positions were located from the official addresses (2025 water monitoring plan, annex 1) on
# OpenStreetMap roads (about 0.05-0.5 km). Dams and weirs are the OpenStreetMap structure lines (dams checked against
# Wikidata/GNS within 0.15 km). Map credit: (c) OpenStreetMap contributors.
STATIONS = [("Hyeondo", 36.4614, 127.4624, "river", (-1.0, -3.4), "right"), ("Nammyeon", 36.4800, 127.2694, "river", (1.0, 3.0), "left"),
            ("Gongju", 36.4630, 127.1407, "river", (0.0, 3.4), "center"), ("Buyeo", 36.3356, 126.9682, "river", (2.6, 0.6), "left"),
            ("Daecheong", 36.4310, 127.5553, "reservoir", (2.8, 0.0), "left"), ("Yongdam", 35.9370, 127.4860, "reservoir", (-2.8, 0.0), "right")]
# drawn as symbols only; the caption names them
STRUCTURES = [("Daecheong Dam", 36.4766, 127.4813, "dam"), ("Yongdam Dam", 35.9445, 127.5259, "dam"), ("Sejong weir", 36.4716, 127.2618, "weir"),
              ("Gongju weir", 36.4622, 127.0993, "weir"), ("Baekje weir", 36.3191, 126.9390, "weir")]
WATER, WATER_LIGHT, WATER_OUT, SEA, OUTSIDE = "#3f88c5", "#a9cde8", "#c9d8e4", "#e8eff5", "#efefef"


def read_geoms(path, clip=None):
    import fiona
    from shapely.geometry import shape
    shp = path if path.suffix == ".shp" else next(path.glob("*.shp"))
    with fiona.open(shp) as src:
        gs = [shape(f["geometry"]) for f in src if f["geometry"] is not None]
    return [g for g in gs if clip is None or g.intersects(clip)]


def fill(ax, geoms, rasterized=False, **kw):
    """(Multi)polygons, holes included, as one collection."""
    from matplotlib.collections import PathCollection
    from matplotlib.path import Path as MPath
    paths = []
    for g in geoms:
        for q in getattr(g, "geoms", [g]):
            if q.geom_type != "Polygon" or q.is_empty:
                continue
            rings = [np.asarray(r.coords)[:, :2] for r in [q.exterior, *q.interiors] if len(r.coords) >= 4]
            if not rings:
                continue
            codes = np.concatenate([[MPath.MOVETO] + [MPath.LINETO] * (len(r) - 2) + [MPath.CLOSEPOLY] for r in rings])
            paths.append(MPath(np.concatenate(rings), codes))
    ax.add_collection(PathCollection(paths, rasterized=rasterized, **kw))


def geum_basin():
    """Geum River basin from HydroBASINS level 6 (Lehner and Grill, 2013): every sub-basin that drains to the same river
    mouth as the sub-basin of the Buyeo station. Returns None when the file is missing."""
    shp = GIS / "HydroBASINS" / "hybas_as_lev06_v1c.shp"
    if not shp.exists():
        return None
    import fiona
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union, transform
    with fiona.open(shp) as src:
        recs = [(f["properties"], shape(f["geometry"])) for f in src.filter(bbox=(125.5, 34.8, 129.0, 37.8))]
    mouth = next(p["MAIN_BAS"] for p, g in recs if g.contains(Point(126.9372, 36.2731)))
    basin = unary_union([g for p, g in recs if p["MAIN_BAS"] == mouth])
    return transform(lambda lon, lat, z=None: tm_korea(np.asarray(lat), np.asarray(lon)), basin)


def graticule(ax, ext, lons, lats):
    x0, y0, x1, y1 = ext
    deg = lambda v, h: f"{int(v)}°{int(round((v % 1) * 60)):02d}′{h}" if v % 1 else f"{int(v)}°{h}"
    for lon in lons:
        lat = np.linspace(33, 39, 400); x, y = tm_korea(lat, np.full_like(lat, lon))
        ax.plot(x, y, color="#cfcfcf", lw=0.3, zorder=1.5)
        xb = np.interp(y0, y, x)
        if x0 < xb < x1:
            ax.annotate(deg(lon, "E"), xy=(xb, y0), xytext=(0, -3), textcoords="offset points", ha="center", va="top", fontsize=6.3, annotation_clip=False)
    for lat in lats:
        lon = np.linspace(124, 131, 400); x, y = tm_korea(np.full_like(lon, lat), lon)
        ax.plot(x, y, color="#cfcfcf", lw=0.3, zorder=1.5)
        yl = np.interp(x0, x, y)
        if y0 < yl < y1:
            ax.annotate(deg(lat, "N"), xy=(x0, yl), xytext=(-3, 0), textcoords="offset points", ha="right", va="center", fontsize=6.3,
                        rotation=90, rotation_mode="anchor", annotation_clip=False)


def scalebar(ax, x, y, km=(0, 10, 20), h=900):
    for i in range(len(km) - 1):
        ax.add_patch(Rectangle((x + km[i] * 1e3, y), (km[i + 1] - km[i]) * 1e3, h, facecolor=INK if i % 2 == 0 else "white", edgecolor=INK, lw=0.5, zorder=9))
    for k in km:
        ax.text(x + k * 1e3, y + h + 700, f"{k}", ha="center", va="bottom", fontsize=6.3, zorder=9)
    ax.text(x + km[-1] * 1e3 + 1500, y + h / 2, "km", ha="left", va="center", fontsize=6.3, zorder=9)


def north_arrow(ax, x, y, size=7000):
    ax.annotate("", xy=(x, y + size), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=8), zorder=9)
    ax.text(x, y + size + 700, "N", ha="center", va="bottom", fontsize=7.5, zorder=9)


def fig1_map():
    from shapely.geometry import box
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    basin = geum_basin()
    if basin is not None:
        bx0, by0, bx1, by1 = basin.bounds; ext = (bx0 - 5e3, by0 - 5e3, bx1 + 5e3, by1 + 5e3)
    else:
        xs, ys = tm_korea(np.array([35.55, 35.55, 36.98, 36.98]), np.array([126.62, 127.98, 126.62, 127.98])); ext = (xs.min(), ys.min(), xs.max(), ys.max())
    clip = box(*ext)
    sido_all = read_geoms(GIS / "시도"); rivers = read_geoms(GIS / "Rivers_National", clip); local = read_geoms(GIS / "Basin_Standard", clip)
    # boundaries simplified to what the print size can show (about 0.03 mm on the main map, 0.3 mm on the inset)
    sido = [g.intersection(clip.buffer(2e3)).simplify(40) for g in sido_all if g.intersects(clip)]; sido_k = [g.simplify(800) for g in sido_all]
    map_w = 92 * MM; map_h = map_w * (ext[3] - ext[1]) / (ext[2] - ext[0]); fig_h = map_h + 11 * MM
    fig = plt.figure(figsize=(WIDTH, fig_h))
    ax = fig.add_axes([7 * MM / WIDTH, 7 * MM / fig_h, map_w / WIDTH, map_h / fig_h]); ax.set_facecolor(SEA)
    if basin is not None:
        fill(ax, sido, facecolor=OUTSIDE, edgecolor="none", zorder=1); fill(ax, [basin], facecolor="white", edgecolor="none", zorder=1.2)
        local = [g.intersection(basin) for g in local if g.intersects(basin)]
        fill(ax, [g.difference(basin) for g in rivers], facecolor=WATER_OUT, edgecolor=WATER_OUT, linewidths=0.2, zorder=2, rasterized=True)
        rivers = [g.intersection(basin) for g in rivers if g.intersects(basin)]
    else:
        fill(ax, sido, facecolor="white", edgecolor="none", zorder=1)
    fill(ax, sido, facecolor="none", edgecolor="#bdbdbd", linewidths=0.35, zorder=1.6)
    graticule(ax, ext, [126.5, 127.0, 127.5, 128.0], [35.5, 36.0, 36.5, 37.0])
    fill(ax, local, facecolor=WATER_LIGHT, edgecolor=WATER_LIGHT, linewidths=0.2, zorder=2, rasterized=True)
    fill(ax, rivers, facecolor=WATER, edgecolor=WATER, linewidths=0.25, zorder=3, rasterized=True)
    if basin is not None:
        fill(ax, [basin], facecolor="none", edgecolor=INK, linewidths=0.8, zorder=4)
    for _, lat, lon, kind in STRUCTURES:
        x, y = tm_korea(lat, lon)
        ax.plot(x, y, "s" if kind == "dam" else "D", ms=3.6 if kind == "dam" else 2.8, color=INK, mec="white", mew=0.4, zorder=6)
    for name, lat, lon, kind, (dx, dy), ha in STATIONS:
        x, y = tm_korea(lat, lon)
        ax.plot(x, y, "o" if kind == "river" else "^", ms=5.2 if kind == "river" else 6.0, mfc="white", mec=INK, mew=1.0, zorder=8)
        ax.text(x + dx * 1e3, y + dy * 1e3, name, ha=ha, va="center", fontsize=7, zorder=8, path_effects=halo)
    ax.set_xlim(ext[0], ext[2]); ax.set_ylim(ext[1], ext[3]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6)
    scalebar(ax, ext[2] - 33e3, ext[1] + 5e3); north_arrow(ax, ext[0] + 9e3, ext[3] - 17e3)
    # inset and legend in the right-hand column
    left = (7 + 92 + 6) * MM / WIDTH; colw = 1 - left - 0.01
    kx0, ky0 = tm_korea(33.05, 125.0); kx1, ky1 = tm_korea(38.65, 129.65)
    ins_h = colw * WIDTH * (ky1 - ky0) / (kx1 - kx0) / fig_h
    axk = fig.add_axes([left, 1 - 4 * MM / fig_h - ins_h, colw, ins_h])
    fill(axk, sido_k, facecolor="#e6e6e6", edgecolor="#a6a6a6", linewidths=0.25)
    if basin is not None:
        fill(axk, [basin], facecolor="#6b6b6b", edgecolor="none", zorder=3)
    else:
        axk.add_patch(Rectangle((ext[0], ext[1]), ext[2] - ext[0], ext[3] - ext[1], fill=False, edgecolor=INK, linewidth=0.8, zorder=3))
    axk.set_xlim(kx0, kx1); axk.set_ylim(ky0, ky1); axk.set_aspect("equal"); axk.set_xticks([]); axk.set_yticks([])
    for s in axk.spines.values():
        s.set_visible(True); s.set_linewidth(0.5)
    axk.text(0.05, 0.96, "South Korea", transform=axk.transAxes, ha="left", va="top", fontsize=6.5)
    hs = [Line2D([], [], ls="none", marker="o", ms=5.2, mfc="white", mec=INK, mew=1.0), Line2D([], [], ls="none", marker="^", ms=6.0, mfc="white", mec=INK, mew=1.0),
          Line2D([], [], ls="none", marker="s", ms=3.6, color=INK), Line2D([], [], ls="none", marker="D", ms=2.8, color=INK),
          Rectangle((0, 0), 1, 1, facecolor=WATER, edgecolor="none")]
    labs = ["River station\n(source)", "Reservoir station\n(target)", "Dam", "Weir", "River and reservoir"]
    if basin is not None:
        hs.append(Line2D([], [], color=INK, lw=0.8)); labs.append("Geum River basin")
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(left - 0.005, 1 - 8 * MM / fig_h - ins_h), fontsize=6.5, handlelength=1.3, handletextpad=0.6,
               labelspacing=0.75, borderaxespad=0)
    save(fig, "fig1_study_area")


# ---- Fig. 2: accuracy ----
def fig2_accuracy():
    m = models(); s = m[m.group == "source"].pivot_table(index="model", columns="horizon", values="r2_pooled")
    c = comps(); c = c[c.group == "source"]
    fig = plt.figure(figsize=(WIDTH, 58 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 0.72], wspace=0.45, left=0.075, right=0.985, bottom=0.2, top=0.86)
    a, b, d = (fig.add_subplot(gs[0, i]) for i in range(3)); x = np.arange(4)
    others = s.loc[["LSTM", "iTransformer", "ModernTCN", "Mamba"], H]
    a.fill_between(x, others.min().values, others.max().values, color=LIGHT, lw=0, zorder=1)
    series = [("Harmonic ARIMA", "Harmonic ARIMA", INK, "-", 1.3), ("XGBoost", "XGBoost residual", GREY, "-", 1.0),
              ("ARIMA", "ARIMA", GREY, (0, (3, 2)), 0.9), ("xLSTM", "xLSTM", XLSTM, "-", 1.1), ("PINN", "PINN sigmoid w0.3", PINN, "-", 1.1)]
    for (name, key, col, ls, lw), yl in zip(series, spread([s.loc[k, 14] for _, k, *_ in series], 0.036)):
        a.plot(x, s.loc[key, H].values, color=col, ls=ls, lw=lw, marker="o", ms=2.2, zorder=3)
        a.text(3.15, yl, name, color=col, va="center", fontsize=6.8)
    a.set_xlim(-0.15, 4.75); a.set_ylim(0.3, 1.0); a.set_xticks(x, H); a.spines["bottom"].set_bounds(0, 3); a.set_ylabel("Pooled $R^2$")
    for ax in (b, d):
        ax.axhline(0, color=REF, lw=0.5, zorder=0); ax.set_xticks(x, H); ax.set_xlim(-0.5, 3.5)
    for key, off, col in (("xLSTM - Harmonic ARIMA", -0.13, XLSTM), ("XGBoost residual - Harmonic ARIMA", 0.13, GREY)):
        t = c[c.comparison == key].set_index("horizon").loc[H]
        interval(b, x + off, t.diff_pooled, t.ci30_pooled_lo, t.ci30_pooled_hi, col)
    b.text(0.06, 0.2, "xLSTM", color=XLSTM, transform=b.transAxes, fontsize=6.8)
    b.text(0.06, 0.1, "XGBoost", color=GREY, transform=b.transAxes, fontsize=6.8)
    b.set_ylim(-0.31, 0.07); b.set_ylabel(r"$\Delta R^2$")
    t = c[c.comparison == "PINN sigmoid w0.3 - xLSTM"].set_index("horizon").loc[H]
    interval(d, x, t.diff_pooled, t.ci30_pooled_lo, t.ci30_pooled_hi, PINN); d.set_ylim(-0.065, 0.015)
    title(a, "a", "Accuracy"); title(b, "b", "Model − harmonic ARIMA"); title(d, "c", "PINN − xLSTM")
    group_xlabel(fig, [a, b, d], "Forecast horizon (days)")
    save(fig, "fig2_accuracy")


# ---- Fig. 3: bloom events ----
def fig3_events():
    sk = pd.read_csv(R / "events" / "skill.csv"); sk = sk[(sk.group == "source") & (sk.threshold == 25.0)]
    pk = pd.read_csv(FD / "peak_examples.csv", parse_dates=["target_date"])
    fig = plt.figure(figsize=(WIDTH, 60 * MM))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.3, 2.0], wspace=0.24, left=0.07, right=0.985, bottom=0.2, top=0.86)
    a = fig.add_subplot(outer[0]); inner = outer[1].subgridspec(1, 2, wspace=0.2); x = np.arange(4)
    series = [("Harmonic ARIMA", "Harmonic ARIMA", INK, 1.3), ("XGBoost", "XGBoost residual", GREY, 1.0), ("xLSTM", "xLSTM", XLSTM, 1.1),
              ("PINN", "PINN sigmoid w0.3", PINN, 1.1)]
    ends = spread([sk[(sk.model == k) & (sk.horizon == 14)].csi.iloc[0] for _, k, _, _ in series], 0.04)
    for (name, key, col, lw), yl in zip(series, ends):
        d = sk[sk.model == key].set_index("horizon").loc[H]
        a.plot(x, d.csi.values, color=col, lw=lw, marker="o", ms=2.2, zorder=3)
        a.text(3.15, yl, name, color=col, va="center", fontsize=6.8)
    a.set_xlim(-0.15, 5.0); a.set_ylim(0.2, 0.9); a.set_xticks(x, H); a.spines["bottom"].set_bounds(0, 3)
    a.set_xlabel("Forecast horizon (days)"); a.set_ylabel("Critical success index")
    title(a, "a", "Chl-a ≥ 25 mg $\\mathrm{m}^{-3}$")
    cases = [("부여 2025", "Buyeo 2025, 7-day", [("Harmonic ARIMA", INK), ("xLSTM", XLSTM), ("PINN sigmoid w0.3", PINN)]),
             ("대청호 2024", "Daecheong 2024, 7-day", [("Harmonic ARIMA", INK), ("fine-tune pure", XLSTM), ("fine-tune PINN", PINN)])]
    for j, (case, ttl, ms) in enumerate(cases):
        ax = fig.add_subplot(inner[j]); d = pk[pk.case == case]
        obs = d.drop_duplicates("target_date").sort_values("target_date"); obs = obs[obs.observed.notna()]
        ax.axhline(25, color=REF, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
        ax.plot(obs.target_date, obs.observed, "o", ms=1.9, color="#a3a3a3", mec="none", zorder=2)
        for key, col in ms:
            e = d[d.model == key].sort_values("target_date"); ax.plot(e.target_date, e.forecast, color=col, lw=0.9, zorder=3)
        ax.set_ylim(0, 155); ax.set_yticks([0, 50, 100, 150])
        ax.set_xlim(obs.target_date.min() - pd.Timedelta(days=2), obs.target_date.max() + pd.Timedelta(days=2))
        ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if j == 0:
            ax.set_ylabel("Chl-a (mg $\\mathrm{m}^{-3}$)")
            hs = [Line2D([], [], ls="none", marker="o", ms=2.5, color="#a3a3a3", mec="none")] + [Line2D([], [], color=c_, lw=1.0) for c_ in (INK, XLSTM, PINN)]
            ax.legend(hs, ["Observed", "Harmonic ARIMA", "xLSTM", "PINN"], loc="upper left", borderaxespad=0.1, handlelength=1.2, labelspacing=0.3)
            title(ax, "b", ttl)
        else:
            ax.tick_params(labelleft=False); title(ax, "c", ttl, dx=-4)
    save(fig, "fig3_events")


# ---- Fig. 4: kinetic parameters ----
def fig4_parameters():
    rp = pd.read_csv(R / "identifiability" / "run_parameters.csv")
    rp = rp[(rp.estimate == "converged") & rp.target.isna() & (rp.w_max == 0.3)]
    fig = plt.figure(figsize=(WIDTH, 102 * MM))
    rows = fig.add_gridspec(2, 1, hspace=0.62, left=0.095, right=0.985, bottom=0.1, top=0.94)
    top = rows[0].subgridspec(1, 2, width_ratios=[2.05, 1], wspace=0.36); bot = rows[1].subgridspec(1, 3, wspace=0.1)
    a = fig.add_subplot(top[0]); kb = fig.add_subplot(top[1])
    groups = [(0.0, "0.3", (rp.physics == "sigmoid") & (rp.kn_max == 0.3), PINN, 0.3), (1.0, "0.5", (rp.physics == "sigmoid") & (rp.kn_max == 0.5), PINN, 0.5),
              (2.0, "1.0", (rp.physics == "sigmoid") & (rp.kn_max == 1.0), PINN, 1.0), (3.4, "Soft\npenalty", rp.physics == "soft", SOFT, None),
              (4.4, "No\nconstraint", rp.physics == "none", NONE, None)]
    a.axhspan(0.01, 0.5, color=LIGHT, lw=0, zorder=0); a.text(-0.48, 0.0125, "Literature range", color=GREY, fontsize=6.5, va="bottom")
    for xpos, _, mask, col, ub in groups:
        v = np.sort(rp[mask].K_N.values)
        a.plot(xpos + np.linspace(-0.13, 0.13, len(v)), v, "o", ms=3.4, color=col, mec=col, zorder=3, ls="none")
        if ub:
            a.plot([xpos - 0.32, xpos + 0.32], [ub, ub], color=INK, lw=0.7, ls=(0, (2.5, 1.5)), zorder=2)
    a.set_yscale("log"); a.set_ylim(0.005, 3000); plain_log(a, "y"); a.set_xlim(-0.55, 4.85)
    a.set_xticks([g[0] for g in groups], [g[1] for g in groups]); a.tick_params(axis="x", length=0)
    a.annotate("Sigmoid, upper bound (mg $\\mathrm{L}^{-1}$)", xy=(1.0, 0), xycoords=mtrans.blended_transform_factory(a.transData, a.transAxes),
               xytext=(0, -12.5), textcoords="offset points", ha="center", va="top", fontsize=7)
    a.set_ylabel("$K_N$ (mg $\\mathrm{L}^{-1}$)"); title(a, "a", "Fitted $K_N$")
    km = pd.read_csv(R / "identifiability" / "kinetic_multiday.csv"); km = km[km["mode"] == "sigmoid"]; ks = [1, 7, 14]; xs = np.arange(3)
    kinds = [("rivers (source)", "Rivers", INK, INK), ("Daecheong Reservoir", "Daecheong", GREY, GREY), ("Yongdam Reservoir", "Yongdam", GREY, "white")]
    vals = [[km[(km.group == g) & (km.k_days == k)].r2_test.iloc[0] for k in ks] for g, *_ in kinds]
    kb.axhline(0, color=REF, lw=0.5, zorder=0)
    for (_, lab, col, mfc), v, yl in zip(kinds, vals, spread([v[-1] for v in vals], 0.022)):
        kb.plot(xs, v, color=col, lw=1.0, marker="o", ms=3, mfc=mfc, mec=col, mew=0.8, zorder=3)
        kb.text(2.18, yl, lab, color=col, va="center", fontsize=6.8)
    kb.set_xticks(xs, ks); kb.set_xlim(-0.25, 3.6); kb.spines["bottom"].set_bounds(0, 2); kb.set_ylim(-0.13, 0.06)
    kb.set_xlabel("Change interval (days)"); kb.set_ylabel("Kinetic $R^2$"); title(kb, "b", "Explained change")
    pf = pd.read_csv(R / "identifiability" / "profiles.csv"); pf = pf[pf.objective == "kinetic_regression"]
    lit = [("K_N", 0.01, 0.5, "Nitrogen", "$K_N$ (mg $\\mathrm{L}^{-1}$)"), ("K_P", 0.001, 0.05, "Phosphorus", "$K_P$ (mg $\\mathrm{L}^{-1}$)"),
           ("K_I", 50, 1000, "Light", "$K_I$ (μmol $\\mathrm{m}^{-2}\\,\\mathrm{s}^{-1}$)")]
    thr = {p: 100 * 3.841 / pf[pf.param == p].n_distinct_pairs.iloc[0] for p, *_ in lit}; ymax = max(0.5, 1.6 * max(thr.values()))
    first = None
    for j, (p, lo, hi, ttl, xl) in enumerate(lit):
        ax = fig.add_subplot(bot[j], sharey=first); first = first or ax
        ax.axvspan(lo, hi, color=LIGHT, lw=0, zorder=0)
        for _, d in pf[pf.param == p].groupby("seed"):
            d = d.sort_values("value"); ax.plot(d.value, 100 * (d.loss / d.loss_min_free - 1), color=PINN, lw=0.9, alpha=0.85, zorder=3)
        ax.axhline(thr[p], color=INK, lw=0.6, ls=(0, (1, 1.5)), zorder=2)
        ax.set_xscale("log"); plain_log(ax, "x"); ax.set_xlabel(xl); ax.set_ylim(0, ymax)
        if j == 0:
            ax.set_ylabel("Loss increase (%)")
            ax.text(0.98, thr[p] + 0.012, "95% limit", transform=mtrans.blended_transform_factory(ax.transAxes, ax.transData), ha="right", va="bottom", fontsize=6.5)
            title(ax, "c", ttl)
        else:
            ax.tick_params(labelleft=False); title(ax, "cde"[j], ttl, dx=-4)
    save(fig, "fig4_parameters")


# ---- Fig. 5: what the constraint and the physics loss change ----
def fig5_constraint_transfer():
    sc = pd.read_csv(R / "stats" / "station_comparisons.csv"); sc = sc[(sc.group == "source") & (sc.horizon == 7)]
    c = comps(); t = c[(c.group == "transfer") & (c.record == "24m")]
    fig = plt.figure(figsize=(WIDTH, 60 * MM))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.1, 2.0], wspace=0.3, left=0.105, right=0.985, bottom=0.2, top=0.86)
    a = fig.add_subplot(outer[0]); inner = outer[1].subgridspec(1, 2, wspace=0.1)
    order = ["현도", "남면", "공주", "부여"]; y = np.arange(4)[::-1].astype(float)
    a.axvline(0, color=REF, lw=0.5, zorder=0)
    for cmp, col, off in (("PINN none w0.3 - PINN sigmoid w0.3", NONE, 0.15), ("PINN soft w0.3 - PINN sigmoid w0.3", SOFT, -0.15)):
        d = sc[sc.comparison == cmp].set_index("station").loc[order]
        interval(a, y + off, d["diff"], d.ci_lo, d.ci_hi, col, horizontal=True)
    a.set_yticks(y, [EN[s] for s in order]); a.tick_params(axis="y", length=0); a.spines["left"].set_visible(False)
    a.set_xlim(-1.8, 0.25); a.set_ylim(-0.6, 3.6); a.set_xlabel(r"$\Delta R^2$ from sigmoid, 7 days")
    hs = [Line2D([], [], color=c_, lw=0.9, marker="o", ms=3) for c_ in (NONE, SOFT)]
    a.legend(hs, ["No constraint", "Soft penalty"], loc="lower left", borderaxespad=0.1, handlelength=1.6, labelspacing=0.3)
    title(a, "a", "Parameter handling", x_fig=0.012)
    xx = np.arange(4); panels = []
    for j, tg in enumerate(["용담호", "대청호"]):
        ax = fig.add_subplot(inner[j], sharey=panels[0] if panels else None); panels.append(ax)
        ax.axhline(0, color=REF, lw=0.5, zorder=0)
        for cmp, col, hollow, off in (("fine-tune PINN - Harmonic ARIMA", PINN, False, -0.2), ("fine-tune placebo - Harmonic ARIMA", PINN, True, 0.0),
                                      ("fine-tune pure - Harmonic ARIMA", XLSTM, False, 0.2)):
            d = t[(t.target == tg) & (t.comparison == cmp)].set_index("horizon").loc[H]
            interval(ax, xx + off, d.diff_pooled, d.ci30_pooled_lo, d.ci30_pooled_hi, col, hollow=hollow)
        ax.set_xticks(xx, H); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.13, 0.2)
        if j == 0:
            ax.set_ylabel(r"$\Delta R^2$ from harmonic ARIMA")
            hs = [Line2D([], [], ls="none", marker="o", ms=3, color=PINN, mec=PINN, mew=0.8), Line2D([], [], ls="none", marker="o", ms=3, color=PINN, mfc="white", mew=0.8),
                  Line2D([], [], ls="none", marker="o", ms=3, color=XLSTM, mec=XLSTM, mew=0.8)]
            ax.legend(hs, ["PINN", "Placebo", "xLSTM"], loc="upper left", borderaxespad=0.1, handletextpad=0.2, labelspacing=0.3)
            title(ax, "b", f"{EN[tg]} Reservoir")
        else:
            ax.tick_params(labelleft=False); title(ax, "c", f"{EN[tg]} Reservoir", dx=-4)
    group_xlabel(fig, panels, "Forecast horizon (days)")
    save(fig, "fig5_constraint_transfer")


# ---- Fig. 6: chlorophyll-a as a proxy for cyanobacteria ----
def fig6_algae_proxy():
    e = pd.read_csv(R / "e9_algae_proxy" / "summary.csv"); e = e[e.season == "all months"]
    units = [("대청호", "Daecheong", 4.6), ("용담호", "Yongdam", 3.6), ("남면", "Nammyeon\n(Sejong weir)", 2.0), ("공주", "Gongju\n(Gongju weir)", 1.0), ("부여", "Buyeo\n(Baekje weir)", 0.0)]
    fig = plt.figure(figsize=(WIDTH, 56 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.1, left=0.15, right=0.960, bottom=0.2, top=0.86)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1], sharey=a); y = np.array([u[2] for u in units])
    for cmp, off, hollow in (("sensor Chl-a vs cells", 0.15, False), ("lab Chl-a vs cells (same sample)", -0.15, True)):
        r = [e[e.unit.str.startswith(k) & (e.comparison == cmp)].iloc[0] for k, _, _ in units]
        interval(a, y + off, [q.spearman for q in r], [q.spearman_lo for q in r], [q.spearman_hi for q in r], INK, hollow=hollow, horizontal=True)
        interval(b, y + off, [q.auc_alert for q in r], [q.auc_lo for q in r], [q.auc_hi for q in r], INK, hollow=hollow, horizontal=True)
    a.axvline(0, color=REF, lw=0.5, zorder=0); b.axvline(0.5, color=REF, lw=0.5, zorder=0)
    a.set_xlim(-0.15, 0.78); b.set_xlim(0.45, 1.0); a.set_ylim(-0.55, 5.55)
    a.set_yticks(y, [u[1] for u in units]); a.tick_params(axis="y", length=0); a.spines["left"].set_visible(False)
    b.tick_params(axis="y", length=0, labelleft=False); b.spines["left"].set_visible(False)
    for yy, txt in ((5.2, "Reservoirs"), (2.6, "Weir reaches")):
        a.annotate(txt, xy=(0.012, yy), xycoords=("figure fraction", "data"), color=GREY, fontsize=6.5, va="center")
    a.set_xlabel(r"Spearman $\rho$"); b.set_xlabel("AUC")
    hs = [Line2D([], [], ls="none", marker="o", ms=3, color=INK, mec=INK, mew=0.8), Line2D([], [], ls="none", marker="o", ms=3, color=INK, mfc="white", mew=0.8)]
    b.legend(hs, ["Sensor Chl-a (forecast target)", "Laboratory Chl-a"], loc="upper left", bbox_to_anchor=(0.12, 1.0), borderaxespad=0.1, handletextpad=0.2, labelspacing=0.3)
    a.annotate("a", xy=(0.012, 1), xycoords=("figure fraction", "axes fraction"), xytext=(0, 7), textcoords="offset points", ha="left", va="bottom", fontsize=8.0, fontweight="bold")
    a.annotate("Correlation with cell counts", xy=(0.012, 1), xycoords=("figure fraction", "axes fraction"), xytext=(10, 7.3), textcoords="offset points", ha="left", va="bottom", fontsize=7.0)
    b.annotate("b", xy=(0, 1), xycoords="axes fraction", xytext=(-4, 7), textcoords="offset points", ha="left", va="bottom", fontsize=8.0, fontweight="bold")
    b.annotate("Samples ≥ 1,000 cells $\\mathrm{mL}^{-1}$", xy=(0, 1), xycoords="axes fraction", xytext=(6, 7.3), textcoords="offset points", ha="left", va="bottom", fontsize=7.0)
    save(fig, "fig6_algae_proxy")


# ---- Graphical abstract (separate submission file, 2.5:1, read from left to right) ----
def graphical_abstract():
    rp = pd.read_csv(R / "identifiability" / "run_parameters.csv")
    rp = rp[(rp.estimate == "converged") & rp.target.isna() & (rp.w_max == 0.3)]
    m = models(); s = m[m.group == "source"].pivot_table(index="model", columns="horizon", values="r2_pooled")
    W, Hgt = 130, 52                                            # mm
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "mathtext.bfit": "Arial:bold:italic",
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6}):
        fig = plt.figure(figsize=(W * MM, Hgt * MM))
        col = W / 3
        heads = ["The constraint keeps $\\mathbfit{K}_\\mathbfit{N}$\nin its literature range", "but $\\mathbfit{K}_\\mathbfit{N}$ follows the bound,\nnot the data",
                 "and a seasonal baseline is as\naccurate as any model at rivers"]
        axes = []
        for k, h in enumerate(heads):
            fig.text((k + 0.5) / 3, 0.975, h, ha="center", va="top", fontsize=8, fontweight="bold", linespacing=1.15)
            axes.append(fig.add_axes([(k * col + 13) / W, 10.5 / Hgt, 26 / W, 30.5 / Hgt]))
        a, b, c = axes
        # 1: K_N under the three ways of handling it
        a.axhspan(0.01, 0.5, color=LIGHT, lw=0, zorder=0)
        a.text(2.5, 0.012, "literature\nrange", ha="right", va="bottom", fontsize=7, color=GREY, linespacing=1.0)
        for x, (mask, colr) in enumerate([(rp.physics == "none", NONE), (rp.physics == "soft", SOFT), ((rp.physics == "sigmoid") & (rp.kn_max == 0.5), PINN)]):
            v = np.sort(rp[mask].K_N.values)
            a.plot(x + np.linspace(-0.16, 0.16, len(v)), v, "o", ms=3.8, color=colr, mec=colr, ls="none", zorder=3)
        a.set_yscale("log"); a.set_ylim(0.006, 3000); a.set_yticks([0.01, 1, 100]); a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_formatter(NullFormatter()); a.set_xlim(-0.55, 2.6); a.tick_params(axis="x", length=0)
        a.set_xticks([0, 1, 2], ["None", "Soft", "Sigmoid"]); a.set_xlabel("Constraint"); a.set_ylabel("$K_N$ (mg $\\mathrm{L}^{-1}$)", labelpad=1)
        # 2: the bound-shift test
        b.plot([0.2, 1.1], [0.2, 1.1], color=REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
        b.text(1.08, 0.27, "fitted =\nbound", ha="right", va="bottom", fontsize=7, color=GREY, style="italic", linespacing=1.0)
        for ub in (0.3, 0.5, 1.0):
            v = rp[(rp.physics == "sigmoid") & (rp.kn_max == ub)].K_N.values
            b.plot(np.full(len(v), ub), v, "o", ms=4.2, color=PINN, mec=PINN, ls="none", zorder=3)
        b.set_xlim(0.2, 1.1); b.set_ylim(0.2, 1.1); b.set_xticks([0.3, 0.5, 1.0]); b.set_yticks([0.3, 0.5, 1.0])
        b.set_xlabel("Upper bound of $K_N$"); b.set_ylabel("Fitted $K_N$", labelpad=1)
        # 3: forecast accuracy at the river stations
        x = np.arange(4); hv, pv, xv = s.loc["Harmonic ARIMA", H].values, s.loc["PINN sigmoid w0.3", H].values, s.loc["xLSTM", H].values
        c.plot(x, hv, color=INK, lw=1.5, marker="o", ms=2.6); c.plot(x, xv, color=XLSTM, lw=1.1, marker="o", ms=2.2); c.plot(x, pv, color=PINN, lw=1.3, marker="o", ms=2.6)
        lad = pd.read_csv(ROOT / "results" / "clean_revision_v4" / "v5_additions" / "4_ladder_models.csv")   # from v5_additions.py
        dv = lad[(lad.group == "source") & (lad.model == "Damped anomaly")].set_index("horizon").loc[H].r2_pooled.values
        c.plot(x, dv, color=GREY, lw=1.3, ls=(0, (3, 1.5)), marker="o", ms=2.2, zorder=4)
        c.text(2.1, 0.835, "Seasonal baseline,\nharmonic ARIMA", ha="center", va="bottom", fontsize=7.0, color=INK, linespacing=1.0)
        c.text(1.55, pv[2] - 0.09, "PINN", ha="center", va="top", fontsize=7.5, color=PINN)
        c.text(1.55, pv[2] - 0.17, "xLSTM", ha="center", va="top", fontsize=7.5, color=XLSTM)
        c.set_xticks(x, H); c.set_ylim(0.3, 1.0); c.set_yticks([0.4, 0.6, 0.8, 1.0]); c.set_xlabel("Forecast horizon (days)")
        c.set_ylabel("Pooled $R^2$", labelpad=1)
        out = OUT.parent / "graphical_abstract"; out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "graphical_abstract.pdf"); fig.savefig(out / "graphical_abstract.png", dpi=600)
        fig.savefig(out / "graphical_abstract.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)


FIGS = {"fig1": fig1_map, "fig2": fig2_accuracy, "fig3": fig3_events, "fig4": fig4_parameters, "fig5": fig5_constraint_transfer,
        "fig6": fig6_algae_proxy, "ga": graphical_abstract}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(FIGS)):
        FIGS[k](); print("done", k, flush=True)
