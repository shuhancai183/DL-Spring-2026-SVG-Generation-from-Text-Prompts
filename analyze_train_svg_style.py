import os
import re
import json
import math
import argparse
from collections import Counter, defaultdict
from statistics import mean, median

import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
NS_PREFIX = "{http://www.w3.org/2000/svg}"

PATH_CMD_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
HEX_COLOR_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
RGB_COLOR_RE = re.compile(r"^rgba?\(")

ANALYZE_TAGS = [
    "svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon",
    "defs", "use", "symbol", "clipPath", "mask", "linearGradient", "radialGradient",
    "stop", "pattern", "image", "text"
]

PRIMITIVE_TAGS = {"rect", "circle", "ellipse", "line", "polyline", "polygon"}
SHAPE_TAGS = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}


def strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def safe_percentile(arr, q):
    if not arr:
        return None
    s = sorted(arr)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] * (hi - pos) + s[hi] * (pos - lo))


def summarize_numeric(arr):
    if not arr:
        return {}
    return {
        "count": len(arr),
        "min": min(arr),
        "max": max(arr),
        "mean": float(mean(arr)),
        "median": float(median(arr)),
        "p90": safe_percentile(arr, 90),
        "p95": safe_percentile(arr, 95),
        "p99": safe_percentile(arr, 99),
    }


def classify_color(x: str) -> str:
    if x is None:
        return "none"
    x = str(x).strip()
    if x == "":
        return "empty"
    if x.lower() == "none":
        return "none"
    if x.lower() == "currentcolor":
        return "currentColor"
    if HEX_COLOR_RE.match(x):
        return "hex"
    if RGB_COLOR_RE.match(x.lower()):
        return "rgb"
    return "named_or_other"


def get_tree_depth(elem, depth=0):
    if len(list(elem)) == 0:
        return depth
    return max(get_tree_depth(child, depth + 1) for child in list(elem))


def analyze_svg(svg_text: str):
    result = {
        "valid": False,
        "error": None,

        "svg_length": len(svg_text),
        "tag_count_total": 0,
        "max_depth": 0,

        "tag_counter": Counter(),
        "attr_counter": Counter(),

        "path_count": 0,
        "primitive_count": 0,
        "shape_count": 0,

        "has_g": False,
        "has_defs": False,
        "has_style_attr": False,
        "has_transform_attr": False,
        "has_use": False,
        "has_symbol": False,

        "fill_values": [],
        "stroke_values": [],
        "color_type_counter": Counter(),

        "path_cmd_counter": Counter(),
        "path_cmd_count_per_svg": 0,

        "root_attrs": {},
    }

    try:
        root = ET.fromstring(svg_text)
    except Exception as e:
        result["error"] = repr(e)
        return result

    result["valid"] = True
    result["max_depth"] = get_tree_depth(root)

    nodes = [root] + list(root.iter())
    seen = set()
    uniq_nodes = []
    for n in nodes:
        if id(n) not in seen:
            seen.add(id(n))
            uniq_nodes.append(n)

    result["tag_count_total"] = len(uniq_nodes)

    for k, v in root.attrib.items():
        kk = strip_ns(k)
        result["root_attrs"][kk] = v

    for node in uniq_nodes:
        tag = strip_ns(node.tag)
        result["tag_counter"][tag] += 1

        if tag == "g":
            result["has_g"] = True
        if tag == "defs":
            result["has_defs"] = True
        if tag == "use":
            result["has_use"] = True
        if tag == "symbol":
            result["has_symbol"] = True

        if tag == "path":
            result["path_count"] += 1
        if tag in PRIMITIVE_TAGS:
            result["primitive_count"] += 1
        if tag in SHAPE_TAGS:
            result["shape_count"] += 1

        for k, v in node.attrib.items():
            kk = strip_ns(k)
            result["attr_counter"][kk] += 1

            if kk == "style":
                result["has_style_attr"] = True
            if kk == "transform":
                result["has_transform_attr"] = True

            if kk == "fill":
                result["fill_values"].append(v)
                result["color_type_counter"][f"fill::{classify_color(v)}"] += 1
            if kk == "stroke":
                result["stroke_values"].append(v)
                result["color_type_counter"][f"stroke::{classify_color(v)}"] += 1

        if tag == "path":
            d = node.attrib.get("d", "")
            cmds = PATH_CMD_RE.findall(d)
            result["path_cmd_count_per_svg"] += len(cmds)
            for c in cmds:
                result["path_cmd_counter"][c.upper()] += 1

    return result


def topk_counter(counter: Counter, k=30):
    return [{"key": k_, "count": int(v)} for k_, v in counter.most_common(k)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="train.csv with columns id,prompt,svg")
    parser.add_argument("--output_dir", type=str, default="./train_svg_style_report")
    parser.add_argument("--sample_limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    assert "svg" in df.columns, "train.csv 缺少 svg 列"
    if "id" not in df.columns:
        df["id"] = range(len(df))
    if "prompt" not in df.columns:
        df["prompt"] = ""

    if args.sample_limit is not None and args.sample_limit > 0:
        df = df.head(args.sample_limit).copy()

    svg_lengths = []
    path_counts = []
    primitive_counts = []
    shape_counts = []
    tag_counts_total = []
    max_depths = []
    path_cmd_counts_per_svg = []

    valid_count = 0
    invalid_count = 0

    global_tag_counter = Counter()
    global_attr_counter = Counter()
    global_path_cmd_counter = Counter()
    global_color_type_counter = Counter()
    global_fill_counter = Counter()
    global_stroke_counter = Counter()

    binary_flags = Counter()

    invalid_rows = []
    per_row_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing SVGs"):
        row_id = row["id"]
        prompt = str(row["prompt"])
        svg = str(row["svg"])

        r = analyze_svg(svg)

        if not r["valid"]:
            invalid_count += 1
            invalid_rows.append({
                "id": row_id,
                "prompt": prompt,
                "error": r["error"],
                "svg_length": len(svg),
            })
            continue

        valid_count += 1

        svg_lengths.append(r["svg_length"])
        path_counts.append(r["path_count"])
        primitive_counts.append(r["primitive_count"])
        shape_counts.append(r["shape_count"])
        tag_counts_total.append(r["tag_count_total"])
        max_depths.append(r["max_depth"])
        path_cmd_counts_per_svg.append(r["path_cmd_count_per_svg"])

        global_tag_counter.update(r["tag_counter"])
        global_attr_counter.update(r["attr_counter"])
        global_path_cmd_counter.update(r["path_cmd_counter"])
        global_color_type_counter.update(r["color_type_counter"])
        global_fill_counter.update([str(x).strip() for x in r["fill_values"] if str(x).strip() != ""])
        global_stroke_counter.update([str(x).strip() for x in r["stroke_values"] if str(x).strip() != ""])

        if r["has_g"]:
            binary_flags["has_g"] += 1
        if r["has_defs"]:
            binary_flags["has_defs"] += 1
        if r["has_style_attr"]:
            binary_flags["has_style_attr"] += 1
        if r["has_transform_attr"]:
            binary_flags["has_transform_attr"] += 1
        if r["has_use"]:
            binary_flags["has_use"] += 1
        if r["has_symbol"]:
            binary_flags["has_symbol"] += 1
        if r["path_count"] > 0 and r["primitive_count"] == 0:
            binary_flags["path_only_svg"] += 1
        if r["primitive_count"] > 0:
            binary_flags["has_primitive"] += 1

        per_row_rows.append({
            "id": row_id,
            "prompt": prompt,
            "svg_length": r["svg_length"],
            "path_count": r["path_count"],
            "primitive_count": r["primitive_count"],
            "shape_count": r["shape_count"],
            "tag_count_total": r["tag_count_total"],
            "max_depth": r["max_depth"],
            "path_cmd_count_per_svg": r["path_cmd_count_per_svg"],
            "has_g": int(r["has_g"]),
            "has_defs": int(r["has_defs"]),
            "has_style_attr": int(r["has_style_attr"]),
            "has_transform_attr": int(r["has_transform_attr"]),
            "has_use": int(r["has_use"]),
            "has_symbol": int(r["has_symbol"]),
        })

    total = len(df)
    report = {
        "meta": {
            "train_csv": args.train_csv,
            "output_dir": args.output_dir,
            "total_rows": total,
            "valid_rows": valid_count,
            "invalid_rows": invalid_count,
        },
        "length_stats": summarize_numeric(svg_lengths),
        "path_count_stats": summarize_numeric(path_counts),
        "primitive_count_stats": summarize_numeric(primitive_counts),
        "shape_count_stats": summarize_numeric(shape_counts),
        "tag_count_total_stats": summarize_numeric(tag_counts_total),
        "max_depth_stats": summarize_numeric(max_depths),
        "path_cmd_count_per_svg_stats": summarize_numeric(path_cmd_counts_per_svg),
        "binary_flags_ratio": {
            k: (v / valid_count if valid_count > 0 else 0.0)
            for k, v in binary_flags.items()
        },
        "top_tags": topk_counter(global_tag_counter, 50),
        "top_attrs": topk_counter(global_attr_counter, 50),
        "top_path_commands": topk_counter(global_path_cmd_counter, 30),
        "top_color_types": topk_counter(global_color_type_counter, 30),
        "top_fill_values": topk_counter(global_fill_counter, 30),
        "top_stroke_values": topk_counter(global_stroke_counter, 30),
    }

    # 保存 JSON 总报告
    report_path = os.path.join(args.output_dir, "svg_style_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 保存逐条明细
    per_row_df = pd.DataFrame(per_row_rows)
    per_row_df.to_csv(os.path.join(args.output_dir, "per_svg_metrics.csv"), index=False)

    invalid_df = pd.DataFrame(invalid_rows)
    invalid_df.to_csv(os.path.join(args.output_dir, "invalid_svg_rows.csv"), index=False)

    pd.DataFrame(topk_counter(global_tag_counter, 200)).to_csv(
        os.path.join(args.output_dir, "tag_distribution.csv"), index=False
    )
    pd.DataFrame(topk_counter(global_attr_counter, 200)).to_csv(
        os.path.join(args.output_dir, "attr_distribution.csv"), index=False
    )
    pd.DataFrame(topk_counter(global_path_cmd_counter, 50)).to_csv(
        os.path.join(args.output_dir, "path_command_distribution.csv"), index=False
    )
    pd.DataFrame(topk_counter(global_fill_counter, 200)).to_csv(
        os.path.join(args.output_dir, "fill_values.csv"), index=False
    )
    pd.DataFrame(topk_counter(global_stroke_counter, 200)).to_csv(
        os.path.join(args.output_dir, "stroke_values.csv"), index=False
    )

    print("\n========== TRAIN SVG STYLE SUMMARY ==========")
    print(f"total rows      : {total}")
    print(f"valid rows      : {valid_count}")
    print(f"invalid rows    : {invalid_count}")
    print("\n--- svg length ---")
    print(report["length_stats"])
    print("\n--- path count ---")
    print(report["path_count_stats"])
    print("\n--- primitive count ---")
    print(report["primitive_count_stats"])
    print("\n--- total tag count ---")
    print(report["tag_count_total_stats"])
    print("\n--- max depth ---")
    print(report["max_depth_stats"])
    print("\n--- path cmd per svg ---")
    print(report["path_cmd_count_per_svg_stats"])

    print("\n--- binary flags ratio ---")
    for k, v in sorted(report["binary_flags_ratio"].items()):
        print(f"{k}: {v:.4f}")

    print("\n--- top tags ---")
    for x in report["top_tags"][:20]:
        print(x)

    print("\n--- top attrs ---")
    for x in report["top_attrs"][:20]:
        print(x)

    print("\n--- top path commands ---")
    for x in report["top_path_commands"][:20]:
        print(x)

    print(f"\nreport saved to: {report_path}")
    print("============================================\n")


if __name__ == "__main__":
    main()