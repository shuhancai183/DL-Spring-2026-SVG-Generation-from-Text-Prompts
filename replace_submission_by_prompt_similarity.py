import os
import re
import math
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# ===== worker globals =====
_WORKER_VECTORIZER = None
_WORKER_TRAIN_MATRIX = None
_WORKER_TRAIN_IDS = None
_WORKER_TRAIN_PROMPTS = None
_WORKER_TRAIN_SVGS = None
_WORKER_EXACT_MAP = None


def normalize_prompt(s: str) -> str:
    s = str(s).strip().lower()

    patterns = [
        r"^generate svg code for an image that looks like:\s*",
        r"^generate svg code for:\s*",
        r"^generate an svg illustration for:\s*",
        r"^create an svg of:\s*",
        r"^create svg for:\s*",
        r"^draw:\s*",
        r"^illustration of\s*",
        r"^icon of\s*",
        r"don't use markdown just give svg code\.?\s*$",
    ]

    for p in patterns:
        s = re.sub(p, "", s, flags=re.IGNORECASE)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_index(train_prompts: List[str]):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=1,
    )
    mat = vectorizer.fit_transform(train_prompts)
    return vectorizer, mat


def init_worker(vectorizer, train_matrix, train_ids, train_prompts, train_svgs, exact_map):
    global _WORKER_VECTORIZER
    global _WORKER_TRAIN_MATRIX
    global _WORKER_TRAIN_IDS
    global _WORKER_TRAIN_PROMPTS
    global _WORKER_TRAIN_SVGS
    global _WORKER_EXACT_MAP

    _WORKER_VECTORIZER = vectorizer
    _WORKER_TRAIN_MATRIX = train_matrix
    _WORKER_TRAIN_IDS = train_ids
    _WORKER_TRAIN_PROMPTS = train_prompts
    _WORKER_TRAIN_SVGS = train_svgs
    _WORKER_EXACT_MAP = exact_map


def retrieve_one(task: Tuple[str, str]):
    """
    task = (test_id, normalized_prompt)
    return = (test_id, matched_train_id, matched_train_prompt, matched_train_svg, similarity, match_type)
    """
    test_id, q = task

    # exact match 优先
    if q in _WORKER_EXACT_MAP:
        tr_id, tr_prompt, tr_svg = _WORKER_EXACT_MAP[q]
        return (test_id, tr_id, tr_prompt, tr_svg, 1.0, "exact")

    qv = _WORKER_VECTORIZER.transform([q])
    scores = linear_kernel(qv, _WORKER_TRAIN_MATRIX).flatten()
    idx = int(scores.argmax())
    score = float(scores[idx])

    tr_id = _WORKER_TRAIN_IDS[idx]
    tr_prompt = _WORKER_TRAIN_PROMPTS[idx]
    tr_svg = _WORKER_TRAIN_SVGS[idx]

    return (test_id, tr_id, tr_prompt, tr_svg, score, "retrieval")


def chunked(iterable, chunk_size):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_csv", type=str, required=True, help="id,svg")
    parser.add_argument("--test_csv", type=str, required=True, help="id,prompt")
    parser.add_argument("--train_csv", type=str, required=True, help="id,prompt,svg")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.80,0.85,0.90,0.95,0.98,0.99",
        help='comma separated thresholds, e.g. "0.85,0.9,0.95"',
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=200)
    parser.add_argument("--save_report", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Reading CSV files...")
    submission_df = pd.read_csv(args.submission_csv)
    test_df = pd.read_csv(args.test_csv)
    train_df = pd.read_csv(args.train_csv)

    assert "id" in submission_df.columns and "svg" in submission_df.columns, "submission.csv 需要 id,svg"
    assert "id" in test_df.columns and "prompt" in test_df.columns, "test.csv 需要 id,prompt"
    assert "id" in train_df.columns and "prompt" in train_df.columns and "svg" in train_df.columns, "train.csv 需要 id,prompt,svg"

    submission_df["id"] = submission_df["id"].astype(str)
    test_df["id"] = test_df["id"].astype(str)
    train_df["id"] = train_df["id"].astype(str)

    print("[INFO] Merging submission with test prompts...")
    merged = submission_df.merge(test_df[["id", "prompt"]], on="id", how="left")
    if merged["prompt"].isna().any():
        missing = merged[merged["prompt"].isna()]["id"].tolist()[:10]
        raise ValueError(f"这些 submission id 在 test.csv 中找不到 prompt: {missing}")

    print("[INFO] Normalizing prompts...")
    merged["prompt_norm"] = merged["prompt"].map(normalize_prompt)
    train_df["prompt_norm"] = train_df["prompt"].map(normalize_prompt)

    print("[INFO] Building exact-match map...")
    exact_map = {}
    for _, row in train_df.iterrows():
        p = row["prompt_norm"]
        if p not in exact_map:
            exact_map[p] = (row["id"], row["prompt"], row["svg"])

    print("[INFO] Building TF-IDF index...")
    vectorizer, train_matrix = build_index(train_df["prompt_norm"].tolist())

    train_ids = train_df["id"].tolist()
    train_prompts = train_df["prompt"].tolist()
    train_svgs = train_df["svg"].tolist()

    tasks = list(zip(merged["id"].tolist(), merged["prompt_norm"].tolist()))

    print(f"[INFO] Running retrieval with {args.num_workers} processes...")
    results = []

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(vectorizer, train_matrix, train_ids, train_prompts, train_svgs, exact_map),
    ) as ex:
        for res in ex.map(retrieve_one, tasks, chunksize=args.chunksize):
            results.append(res)

    print("[INFO] Collecting retrieval results...")
    result_df = pd.DataFrame(
        results,
        columns=[
            "id",
            "matched_train_id",
            "matched_train_prompt",
            "matched_train_svg",
            "similarity",
            "match_type",
        ],
    )

    merged = merged.merge(result_df, on="id", how="left")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    summary_rows = []

    for th in thresholds:
        out_df = merged.copy()

        replace_mask = out_df["similarity"] >= th
        replaced_count = int(replace_mask.sum())

        out_df.loc[replace_mask, "svg"] = out_df.loc[replace_mask, "matched_train_svg"]

        submit_out = out_df[["id", "svg"]].copy()
        out_path = os.path.join(args.output_dir, f"submission_replace_th_{th:.2f}.csv")
        submit_out.to_csv(out_path, index=False)

        summary_rows.append({
            "threshold": th,
            "replaced_count": replaced_count,
            "replace_ratio": replaced_count / len(out_df),
            "output_csv": out_path,
        })

        if args.save_report:
            report_path = os.path.join(args.output_dir, f"report_th_{th:.2f}.csv")
            report_df = out_df.copy()
            report_df["replaced"] = (report_df["similarity"] >= th).astype(int)
            report_df = report_df[
                [
                    "id",
                    "prompt",
                    "prompt_norm",
                    "match_type",
                    "matched_train_id",
                    "matched_train_prompt",
                    "similarity",
                    "replaced",
                ]
            ]
            report_df.to_csv(report_path, index=False)

        print(f"[threshold={th:.2f}] replaced={replaced_count}/{len(out_df)} -> {out_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "threshold_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()