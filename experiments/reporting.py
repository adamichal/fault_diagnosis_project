from __future__ import annotations

from typing import Dict, Any, List


def print_markdown_diagnosis_table(results: Dict[str, Dict[str, Dict[str, Any]]],
                                   metrics: List[str] | None = None) -> None:
    if metrics is None:
        metrics = ["WE", "NWE", "EXAM", "MRR"]

    methods = ["baseline", "rec_min", "rec_max", "rec_weighted"]

    header = ["Benchmark"] + [f"{m}-{metr}" for m in methods for metr in metrics]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")

    for bench_name, res in results.items():
        row = [bench_name]
        for m in methods:
            diag = res[m]["diagnosis"]
            for metr in metrics:
                row.append(f"{float(diag.get(metr, 0.0)):.3f}")
        print("| " + " | ".join(row) + " |")


def print_markdown_reconstruction_table(results: Dict[str, Dict[str, Dict[str, Any]]],
                                        which: str = "union") -> None:
    methods = ["rec_min", "rec_max", "rec_weighted"]
    if which == "macro":
        cols = ["macro_precision", "macro_recall", "macro_f1"]
        label = "macro"
    else:
        cols = ["union_precision", "union_recall", "union_f1"]
        label = "union"

    header = ["Benchmark"] + [f"{m}-{c.replace('_', '-')}" for m in methods for c in cols]
    print(f"\n### Reconstruction ({label})\n")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")

    for bench_name, res in results.items():
        row = [bench_name]
        for m in methods:
            recon = res[m].get("reconstruction", {})
            for c in cols:
                row.append(f"{float(recon.get(c, 0.0)):.3f}")
        print("| " + " | ".join(row) + " |")
