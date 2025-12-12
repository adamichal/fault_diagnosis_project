from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict, Any, Optional

import numpy as np

from sfl_diagnoser.data_models import SpectrumData
from sfl_diagnoser.diagnosis import run_sfl
from sfl_diagnoser.dynamic_analysis import ExecutionTracer
from sfl_diagnoser.evaluation import EvaluationSuite, reconstruction_quality, jaccard_similarity
from sfl_diagnoser.reconstruction import rec_min, rec_max, rec_weighted
from sfl_diagnoser.static_analysis import build_call_graph_from_files


@dataclass
class BenchmarkConfig:
    name: str
    file_paths: List[str]
    tests: List[Tuple[str, Callable[[], None]]]
    faulty_component: str

    visible_ratio: float = 0.3
    random_seed: int = 0

    hidden_mode: str = "random"  # "random" | "by_module"
    visible_modules: Optional[List[str]] = None

    weighted_threshold: float = 0.5
    aggregate_over: str = "failing"  # "failing" | "all"


def _build_call_graph_and_root(file_paths: List[str]):
    abs_paths = [os.path.abspath(p) for p in file_paths]
    common_root = os.path.commonpath(abs_paths)
    G = build_call_graph_from_files(abs_paths)
    return G, common_root


def _trace_tests(tests: List[Tuple[str, Callable[[], None]]], project_root: str):
    tracer = ExecutionTracer(project_root=project_root)
    T, e, full_rows = [], [], []

    for t_name, t_func in tests:
        T.append(t_name)
        tracer.executed_functions.clear()
        try:
            tracer.run_function(t_func)
            e.append(0)
        except Exception:
            e.append(1)
        full_rows.append(set(tracer.executed_functions))

    return T, e, full_rows


def _build_full_spectrum(T: List[str], full_rows: List[set], graph_nodes: List[str]):
    dynamic_components = set().union(*full_rows) if full_rows else set()
    C = sorted(dynamic_components.union(graph_nodes))

    comp_to_idx = {c: j for j, c in enumerate(C)}
    A_full = np.zeros((len(T), len(C)), dtype=int)

    for i, executed in enumerate(full_rows):
        for comp in executed:
            j = comp_to_idx.get(comp)
            if j is not None:
                A_full[i, j] = 1

    return C, A_full


def _choose_hidden_components(C: List[str], visible_ratio: float, random_seed: int,
                              mode: str = "random", visible_modules: Optional[List[str]] = None) -> set:
    if mode == "by_module":
        visible_modules = visible_modules or []
        visible = [c for c in C if any(c.startswith(m + ".") or c == m for m in visible_modules)]
        if visible:
            return set(c for c in C if c not in set(visible))

    rng = random.Random(random_seed)
    num_visible = max(1, int(len(C) * visible_ratio))
    visible = set(rng.sample(C, num_visible))
    return set(c for c in C if c not in visible)


def _predicted_traces_from_recon(A_recon: np.ndarray, C: List[str], threshold: float) -> Dict[int, set]:
    traces: Dict[int, set] = {}
    for i in range(A_recon.shape[0]):
        traces[i] = {C[j] for j in range(A_recon.shape[1]) if float(A_recon[i, j]) >= threshold}
    return traces


def _actual_traces_from_full(A_full: np.ndarray, C: List[str]) -> Dict[int, set]:
    traces: Dict[int, set] = {}
    for i in range(A_full.shape[0]):
        traces[i] = {C[j] for j in range(A_full.shape[1]) if int(A_full[i, j]) == 1}
    return traces


def _filter_test_indices(e: List[int], mode: str) -> List[int]:
    if mode == "all":
        return list(range(len(e)))
    return [i for i, out in enumerate(e) if out == 1]  # default: failing


def _union_over_indices(traces: Dict[int, set], indices: List[int]) -> set:
    agg = set()
    for i in indices:
        agg |= traces.get(i, set())
    return agg


def _run_method(
        method_name: str,
        base: SpectrumData,
        G,
        faulty_component: str,
        actual_traces: Dict[int, set],
        actual_union_focus: set,
        focus_indices: List[int],
        recon_threshold: float,
        recon_fn,
):
    # fresh SpectrumData copy (avoid A_recon leakage)
    s = SpectrumData(
        T=list(base.T),
        e=base.e.copy(),
        C=list(base.C),
        H=set(base.H),
        A_partial=base.A_partial.copy(),
    )

    s = recon_fn(s, G)
    ranking = run_sfl(s, metric="ochiai")

    pred_traces = _predicted_traces_from_recon(s.A_recon, s.C, threshold=recon_threshold)
    pred_union_focus = _union_over_indices(pred_traces, focus_indices)

    suite = EvaluationSuite(
        ranking=ranking,
        faulty_component=faulty_component,
        spectrum=s,
        predicted_traces=pred_traces,
        actual_traces=actual_traces,
    )

    diag = suite.diagnosis_metrics(top_k=[1, 3, 5])
    recon = suite.reconstruction_metrics()

    p, r, f = reconstruction_quality(pred_union_focus, actual_union_focus, set(s.H))
    recon["union_precision"] = p
    recon["union_recall"] = r
    recon["union_f1"] = f
    recon["jaccard_union_focus"] = jaccard_similarity(pred_union_focus, actual_union_focus)

    if method_name == "rec_weighted":
        recon["weighted_threshold"] = recon_threshold

    return {"diagnosis": diag, "reconstruction": recon}


def run_single_benchmark(cfg: BenchmarkConfig) -> Dict[str, Dict[str, Any]]:
    G, common_root = _build_call_graph_and_root(cfg.file_paths)
    T, e_list, full_rows = _trace_tests(cfg.tests, project_root=common_root)

    C, A_full = _build_full_spectrum(T, full_rows, graph_nodes=G.nodes)
    H = _choose_hidden_components(C, cfg.visible_ratio, cfg.random_seed, cfg.hidden_mode, cfg.visible_modules)

    spectrum = SpectrumData.from_full_spectra(T=T, e=e_list, C=C, H=H, A_full=A_full)

    # baseline
    baseline_ranking = run_sfl(spectrum, metric="ochiai")
    suite_baseline = EvaluationSuite(baseline_ranking, cfg.faulty_component, spectrum=spectrum)
    baseline = {"diagnosis": suite_baseline.diagnosis_metrics(top_k=[1, 3, 5])}

    # reconstruction evaluation targets
    actual_traces = _actual_traces_from_full(A_full, C)
    focus_indices = _filter_test_indices(e_list, cfg.aggregate_over)
    actual_union_focus = _union_over_indices(actual_traces, focus_indices)

    rec_min_res = _run_method(
        "rec_min", spectrum, G, cfg.faulty_component,
        actual_traces, actual_union_focus, focus_indices,
        recon_threshold=0.5, recon_fn=rec_min
    )

    rec_max_res = _run_method(
        "rec_max", spectrum, G, cfg.faulty_component,
        actual_traces, actual_union_focus, focus_indices,
        recon_threshold=0.5, recon_fn=rec_max
    )

    rec_w_res = _run_method(
        "rec_weighted", spectrum, G, cfg.faulty_component,
        actual_traces, actual_union_focus, focus_indices,
        recon_threshold=cfg.weighted_threshold, recon_fn=rec_weighted
    )

    return {
        "baseline": baseline,
        "rec_min": rec_min_res,
        "rec_max": rec_max_res,
        "rec_weighted": rec_w_res,
        "_meta": {
            "name": cfg.name,
            "num_tests": len(T),
            "num_components": len(C),
            "visible_ratio": cfg.visible_ratio,
            "hidden_mode": cfg.hidden_mode,
            "aggregate_over": cfg.aggregate_over,
        }
    }


def run_multiple_benchmarks(configs: List[BenchmarkConfig]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {cfg.name: run_single_benchmark(cfg) for cfg in configs}
