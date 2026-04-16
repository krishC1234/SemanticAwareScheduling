#!/usr/bin/env python3
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

CSV_IN = Path(__file__).parent / "data" / "benchmark.csv"
CSV_OUT = Path(__file__).parent / "data" / "scaling_dataset.csv"
FAMILIES_JSON = Path(__file__).parent / "data" / "model_families.json"


def load_families():
    """Read data/model_families.json -> (ordered family list, model->family map).

    JSON layout: {family_name: [model_folder, ...], ...}. Keys starting with
    "_" (e.g. "_comment") are ignored. Insertion order defines one-hot column
    order. Edit the JSON to add eval models.
    """
    raw = json.loads(FAMILIES_JSON.read_text())
    families = [k for k in raw]
    model_to_family = {}
    for fam in families:
        for model in raw[fam]:
            model_to_family[model] = fam
    return families, model_to_family


def fit_k(gpu_counts, total_times):
    """Fit log(total_time) = log(a) - k*log(N). Returns (k, r_squared).

    k > 0 means time decreases with more GPUs (good scaling).
    k = 1 is perfect linear scaling.
    """
    xs = [math.log(n) for n in gpu_counts]
    ys = [math.log(t) for t in total_times]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan"), float("nan")
    slope = sxy / sxx
    k = -slope
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else float("nan")
    return k, r2


def main():
    if not CSV_IN.exists():
        sys.exit(f"missing {CSV_IN}")
    if not FAMILIES_JSON.exists():
        sys.exit(f"missing {FAMILIES_JSON}")
    families, model_to_family = load_families()

    # group: (model, config) -> list of (gpu_count, total_time_sec, batch_size, param_count)
    groups = defaultdict(list)
    with CSV_IN.open() as f:
        for row in csv.DictReader(f):
            try:
                gc = int(row["gpu_count"])
                tt = float(row["total_time_sec"])
                bs = int(row["batch_size"])
                pc = int(row["param_count"])
            except (ValueError, KeyError):
                continue
            if tt <= 0:
                continue
            groups[(row["model"], row["config"])].append((gc, tt, bs, pc))

    unknown = sorted({m for (m, _) in groups if m not in model_to_family})
    if unknown:
        print(f"WARN: no family mapping for {len(unknown)} model(s) "
              f"(add to {FAMILIES_JSON.name}): {unknown}", file=sys.stderr)

    # model_name is a non-feature key column kept so downstream training
    # can do GroupKFold (avoid leaking 4 configs of the same model across folds).
    fieldnames = (
        ["model_name", "batch_size", "param_count"]
        + [f"family_{fam}" for fam in families]
        + ["k"]
    )

    rows = []
    for (model, config), pts in sorted(groups.items()):
        if len(pts) < 2:
            continue
        gpu_counts, total_times, batch_sizes, param_counts = zip(*pts)
        if len(set(batch_sizes)) > 1 or len(set(param_counts)) > 1:
            print(f"WARN: inconsistent batch/param within {model}/{config}", file=sys.stderr)
        k, r2 = fit_k(gpu_counts, total_times)
        if (r2 < 0.9): continue # Skip datapoints where the scaling curve is weak
        family = model_to_family.get(model, "Other")
        row = {
            "model_name": model,
            "batch_size": batch_sizes[0],
            "param_count": param_counts[0],
            "k": round(k, 6)
        }
        for fam in families:
            row[f"family_{fam}"] = 1 if fam == family else 0
        rows.append(row)

    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {CSV_OUT}")


if __name__ == "__main__":
    main()
