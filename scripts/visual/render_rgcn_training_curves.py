"""
Render Phase 2 R-GCN training curves from `train_metrics.jsonl`.

Usage:
    uv run python scripts/visual/render_rgcn_training_curves.py \
      --metrics runs/quick_valid_100/learned/rgcn_gpu_100x100_e5_base_5ep/train_metrics.jsonl \
      --output report/phase2_rgcn_gpu_training_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Font = ImageFont.ImageFont | ImageFont.FreeTypeFont


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metric rows found: {path}")
    return rows


def metric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row[key]
        if not isinstance(value, (int, float)):
            raise ValueError(f"Metric must be numeric: {key}")
        values.append(float(value))
    return values


def epoch_values(rows: list[dict[str, object]]) -> list[int]:
    values: list[int] = []
    for row in rows:
        value = row["epoch"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("Epoch must be an integer.")
        values.append(value)
    return values


def draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    title: str,
    epochs: list[int],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    font: Font,
    small_font: Font,
) -> None:
    left, top, right, bottom = box
    plot_left = left + 58
    plot_top = top + 42
    plot_right = right - 24
    plot_bottom = bottom - 54
    draw.rectangle((left, top, right, bottom), outline=(210, 214, 220), width=1)
    draw.text((left + 16, top + 12), title, fill=(20, 25, 35), font=font)

    all_values = [value for _, values, _ in series for value in values]
    y_min = min(all_values)
    y_max = max(all_values)
    padding = max((y_max - y_min) * 0.12, 0.02)
    y_min -= padding
    y_max += padding

    for i in range(5):
        y = plot_top + i * (plot_bottom - plot_top) / 4
        value = y_max - i * (y_max - y_min) / 4
        draw.line((plot_left, y, plot_right, y), fill=(232, 235, 240), width=1)
        draw.text((left + 12, y - 7), f"{value:.2f}", fill=(90, 96, 108), font=small_font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(120, 128, 140), width=1)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(120, 128, 140), width=1)

    def xy(epoch_index: int, value: float) -> tuple[float, float]:
        x_span = max(1, len(epochs) - 1)
        x = plot_left + epoch_index * (plot_right - plot_left) / x_span
        y = plot_bottom - (value - y_min) * (plot_bottom - plot_top) / (y_max - y_min)
        return x, y

    for label, values, color in series:
        points = [xy(index, value) for index, value in enumerate(values)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        else:
            draw.line(points, fill=color, width=3)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    for index, epoch in enumerate(epochs):
        x, _ = xy(index, y_min)
        draw.text((x - 6, plot_bottom + 10), str(epoch), fill=(90, 96, 108), font=small_font)

    legend_x = plot_left
    legend_y = bottom - 32
    for label, _, color in series:
        draw.rectangle((legend_x, legend_y + 4, legend_x + 16, legend_y + 14), fill=color)
        draw.text((legend_x + 22, legend_y), label, fill=(50, 56, 68), font=small_font)
        legend_x += 150


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Phase 2 R-GCN training curves.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.metrics))
    epochs = epoch_values(rows)
    width, height = 1180, 760
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    small_font = ImageFont.load_default(size=15)
    title_font = ImageFont.load_default(size=26)

    draw.text((32, 24), "Phase 2 R-GCN GPU Training Curves", fill=(15, 23, 42), font=title_font)
    draw.text(
        (32, 58),
        "100 train tasks / 100 dev tasks, intfloat/e5-base-v2, 5 epochs",
        fill=(75, 85, 99),
        font=small_font,
    )

    draw_panel(
        draw,
        box=(32, 100, 1148, 390),
        title="Loss",
        epochs=epochs,
        series=[
            ("train_loss", metric_values(rows, "train_loss"), (37, 99, 235)),
            ("dev_loss", metric_values(rows, "dev_loss"), (220, 86, 35)),
        ],
        font=font,
        small_font=small_font,
    )
    draw_panel(
        draw,
        box=(32, 420, 1148, 720),
        title="Validation Metrics",
        epochs=epochs,
        series=[
            ("Recall@5", metric_values(rows, "dev_recall_at_5"), (22, 163, 74)),
            ("FullSupport@5", metric_values(rows, "dev_full_support_at_5"), (147, 51, 234)),
            ("MRR", metric_values(rows, "dev_mrr"), (217, 119, 6)),
            ("BestMetric", metric_values(rows, "best_dev_metric"), (14, 116, 144)),
        ],
        font=font,
        small_font=small_font,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
