#!/usr/bin/env python3
"""Generate SVG charts for the dense-ft-rgcn cross-dataset report.

Style matches report/assets/2wiki_dataset/*.svg: grouped bar charts, light
gridlines, per-bar value labels on the top group, bottom legend.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "report" / "assets" / "dense_ft_rgcn_cross_dataset"
OUT.mkdir(parents=True, exist_ok=True)

# Method order, labels, colors (palette consistent with existing reports)
METHODS = ["Dense", "Dense-FT", "Dense-RGCN", "Dense-FT-RGCN"]
COLORS = {
    "Dense": "#2563EB",
    "Dense-FT": "#F97316",
    "Dense-RGCN": "#14B8A6",
    "Dense-FT-RGCN": "#DC2626",
}

FONT = 'font-family="Arial, sans-serif"'


def esc(v, nd=3):
    return f"{v:.{nd}f}"


def grouped_bar_chart(title, groups, series, ymax, value_fmt=esc, label_all=False):
    """groups: list[str] metric names. series: dict[method] -> list[float] per group."""
    W, H = 1080, 560
    left, right = 90, 1050
    top, base = 70.0, 410.0  # y for value ymax and value 0
    plot_w = right - left

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" {FONT} font-size="22" font-weight="700" fill="#111827">{title}</text>',
    ]

    # gridlines: 0, 0.25, 0.5, 0.75, 1.0 fraction of ymax
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = base - (base - top) * frac
        val = ymax * frac
        parts.append(f'<line x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#E5E7EB"/>')
        parts.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" {FONT} font-size="12" fill="#4B5563">{val:.2f}</text>'
        )

    n_groups = len(groups)
    n_series = len(METHODS)
    group_w = plot_w / n_groups
    bar_w = 17.6
    gap = 4.0
    cluster_w = n_series * bar_w + (n_series - 1) * gap

    for gi, g in enumerate(groups):
        gx_center = left + group_w * (gi + 0.5)
        cluster_x0 = gx_center - cluster_w / 2
        parts.append(
            f'<text x="{gx_center:.1f}" y="460" text-anchor="middle" {FONT} font-size="13" fill="#111827">{g}</text>'
        )
        for mi, m in enumerate(METHODS):
            v = series[m][gi]
            bx = cluster_x0 + mi * (bar_w + gap)
            bh = (base - top) * (v / ymax)
            by = base - bh
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{COLORS[m]}" rx="2"/>'
            )
            if label_all or m == METHODS[-1]:
                parts.append(
                    f'<text x="{bx+bar_w/2:.1f}" y="{by-5:.1f}" text-anchor="middle" {FONT} font-size="10" fill="#374151">{value_fmt(v)}</text>'
                )

    # legend
    leg_y = [490, 518]
    leg_x = [90, 340, 590, 840]
    for i, m in enumerate(METHODS):
        x = leg_x[i % 2 * 0 + i] if False else leg_x[i]
        y = leg_y[0]
        parts.append(f'<rect x="{x}" y="{y}" width="14" height="14" fill="{COLORS[m]}" rx="2"/>')
        parts.append(f'<text x="{x+22}" y="{y+12}" {FONT} font-size="13" fill="#111827">{m}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------- Data (from authoritative CSVs) ----------------

# Main metrics groups
MAIN_GROUPS = ["Recall@2", "Recall@5", "Recall@10", "Full Support@5", "Full Support@10", "MRR"]

hotpot_main = {
    "Dense":         [0.5448, 0.7698, 0.8886, 0.5320, 0.7538, 0.8530],
    "Dense-FT":      [0.6143, 0.8548, 0.9436, 0.6896, 0.8675, 0.8932],
    "Dense-RGCN":    [0.6665, 0.8871, 0.9600, 0.7599, 0.9090, 0.9156],
    "Dense-FT-RGCN": [0.6961, 0.9088, 0.9687, 0.7998, 0.9279, 0.9305],
}
wiki_main = {
    "Dense":         [0.5383, 0.6776, 0.7953, 0.3628, 0.5586, 0.8633],
    "Dense-FT":      [0.7019, 0.9190, 0.9814, 0.8143, 0.9615, 0.9444],
    "Dense-RGCN":    [0.7277, 0.9351, 0.9742, 0.8764, 0.9506, 0.9525],
    "Dense-FT-RGCN": [0.7867, 0.9706, 0.9906, 0.9444, 0.9824, 0.9746],
}

# Structure metrics
STRUCT_GROUPS = ["Conn.Ev.Recall@5", "Conn.Ev.Recall@10", "Q-E Connectivity@10"]
hotpot_struct = {
    "Dense":         [0.3773, 0.5944, 0.7471],
    "Dense-FT":      [0.4969, 0.6914, 0.8546],
    "Dense-RGCN":    [0.5711, 0.7343, 0.8958],
    "Dense-FT-RGCN": [0.5982, 0.7524, 0.9135],
}
wiki_struct = {
    "Dense":         [0.2010, 0.3968, 0.5535],
    "Dense-FT":      [0.5554, 0.7246, 0.9432],
    "Dense-RGCN":    [0.5754, 0.6944, 0.9325],
    "Dense-FT-RGCN": [0.6341, 0.7304, 0.9627],
}

# 2Wiki path/edge (only RGCN variants have values; baselines N/A -> 0 bar)
WIKI_PATH_GROUPS = ["Path Recall@10", "Edge Recall@10"]
wiki_path = {
    "Dense":         [0.0, 0.0],
    "Dense-FT":      [0.0, 0.0],
    "Dense-RGCN":    [0.8943, 0.8561],
    "Dense-FT-RGCN": [0.9399, 0.8892],
}

# Latency (ms/query)
LAT_GROUPS = ["Retrieval Latency / Query (ms)"]
hotpot_lat = {
    "Dense":         [27.41],
    "Dense-FT":      [28.93],
    "Dense-RGCN":    [53.02],
    "Dense-FT-RGCN": [55.30],
}
wiki_lat = {
    "Dense":         [22.18],
    "Dense-FT":      [21.84],
    "Dense-RGCN":    [34.73],
    "Dense-FT-RGCN": [37.70],
}


def ms_fmt(v):
    return f"{v:.1f}"


charts = {
    "hotpot_main_metrics.svg": grouped_bar_chart(
        "HotpotQA — Main Retrieval Metrics", MAIN_GROUPS, hotpot_main, 1.0),
    "wiki_main_metrics.svg": grouped_bar_chart(
        "2WikiMultiHopQA — Main Retrieval Metrics", MAIN_GROUPS, wiki_main, 1.0),
    "hotpot_structure_metrics.svg": grouped_bar_chart(
        "HotpotQA — Structural Connectivity", STRUCT_GROUPS, hotpot_struct, 1.0, label_all=True),
    "wiki_structure_metrics.svg": grouped_bar_chart(
        "2WikiMultiHopQA — Structural Connectivity", STRUCT_GROUPS, wiki_struct, 1.0, label_all=True),
    "wiki_path_edge_metrics.svg": grouped_bar_chart(
        "2WikiMultiHopQA — Path / Edge Recovery (RGCN only)", WIKI_PATH_GROUPS, wiki_path, 1.0, label_all=True),
    "hotpot_latency.svg": grouped_bar_chart(
        "HotpotQA — Retrieval Latency", LAT_GROUPS, hotpot_lat, 60.0, value_fmt=ms_fmt, label_all=True),
    "wiki_latency.svg": grouped_bar_chart(
        "2WikiMultiHopQA — Retrieval Latency", LAT_GROUPS, wiki_lat, 60.0, value_fmt=ms_fmt, label_all=True),
}

for name, svg in charts.items():
    (OUT / name).write_text(svg, encoding="utf-8")
    print(f"wrote {OUT / name}")
