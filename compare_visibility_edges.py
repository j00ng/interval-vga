#!/usr/bin/env python3
"""Compare two visibility-edge CSV exports without changing either pipeline.

Expected columns are RefFrom/RefTo (depthmapX) or source/target. The default is
an undirected comparison suitable for the standard 2D VGA baseline. Use
``--directed`` only for observer/target-conditioned experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _find_columns(fieldnames: list[str]) -> tuple[str, str]:
    lowered = {name.lower().replace("_", ""): name for name in fieldnames}
    choices = [
        ("reffrom", "refto"),
        ("source", "target"),
        ("from", "to"),
    ]
    for left, right in choices:
        if left in lowered and right in lowered:
            return lowered[left], lowered[right]
    if len(fieldnames) >= 2:
        return fieldnames[0], fieldnames[1]
    raise ValueError("edge CSV requires at least two columns")


def load_edges(path: str, *, directed: bool = False) -> set[tuple[int, int]]:
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        source_col, target_col = _find_columns(reader.fieldnames)
        result = set()
        for line, row in enumerate(reader, 2):
            try:
                source = int(row[source_col])
                target = int(row[target_col])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid edge at {path}:{line}") from error
            if source == target:
                continue
            if not directed and source > target:
                source, target = target, source
            result.add((source, target))
    return result


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, implemented without SciPy."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _degree(edges: set[tuple[int, int]], directed: bool) -> Counter:
    degree = Counter()
    for source, target in edges:
        degree[source] += 1
        if not directed:
            degree[target] += 1
    return degree


def compare_edges(
    reference: set[tuple[int, int]], candidate: set[tuple[int, int]], *, directed: bool = False
) -> dict:
    true_positive = len(reference & candidate)
    false_positive = len(candidate - reference)
    false_negative = len(reference - candidate)
    precision = true_positive / len(candidate) if candidate else float(reference == candidate)
    recall = true_positive / len(reference) if reference else float(reference == candidate)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(reference | candidate)

    ref_degree = _degree(reference, directed)
    cand_degree = _degree(candidate, directed)
    nodes = sorted(set(ref_degree) | set(cand_degree))
    if len(nodes) >= 2:
        ref_values = np.array([ref_degree[node] for node in nodes], dtype=float)
        cand_values = np.array([cand_degree[node] for node in nodes], dtype=float)
        ref_rank, cand_rank = _rank(ref_values), _rank(cand_values)
        if ref_rank.std() and cand_rank.std():
            degree_rank_correlation = float(np.corrcoef(ref_rank, cand_rank)[0, 1])
        else:
            degree_rank_correlation = float(ref_degree == cand_degree)
    else:
        degree_rank_correlation = float(ref_degree == cand_degree)

    return {
        "directed": directed,
        "reference_edges": len(reference),
        "candidate_edges": len(candidate),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": true_positive / union if union else 1.0,
        "degree_rank_correlation": degree_rank_correlation,
        "reference_only_sample": sorted(reference - candidate)[:20],
        "candidate_only_sample": sorted(candidate - reference)[:20],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="depthmapX pointmap-connections CSV")
    parser.add_argument("candidate", help="Python visibility edge CSV")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--out", default=None, help="optional JSON output")
    args = parser.parse_args(argv)
    result = compare_edges(
        load_edges(args.reference, directed=args.directed),
        load_edges(args.candidate, directed=args.directed),
        directed=args.directed,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
