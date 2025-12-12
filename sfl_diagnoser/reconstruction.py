"""
Trace reconstruction algorithms for partial spectra.

Implements the three reconstruction techniques from:
"Spectrum-based fault diagnosis with partial traces"
B. Sotto-Mayor et al., Journal of Systems & Software, 2026.

Algorithms:
    1. Rec-Min       – conservative, adds only unavoidable hidden components.
    2. Rec-Max       – aggressive, adds hidden components that may appear.
    3. Rec-Weighted  – probabilistic reconstruction used by extended SFL.

Each function takes a SpectrumData and an ExecutionGraph and writes the
reconstructed spectrum A_recon back into the SpectrumData instance.
"""

from __future__ import annotations

from collections import deque, defaultdict
from typing import Dict, Set, List, Tuple

import networkx as nx
import numpy as np

from sfl_diagnoser.data_models import SpectrumData, ExecutionGraph, Component


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _initialize_full_matrix_from_partial(data: SpectrumData) -> np.ndarray:
    """
    Build a full-size reconstruction matrix A_recon from A_partial.

    Shapes:
        A_partial : (N, |C \\ H|)
        A_recon   : (N, |C|)

    Visible components copy their column from A_partial. Hidden components
    are initialised to 0 and will be filled by the reconstruction step.
    """
    num_tests = data.num_tests
    num_components = data.num_components_full

    A_recon = np.zeros((num_tests, num_components), dtype=float)

    visible = data.visible_components
    for j_vis, comp in enumerate(visible):
        j_full = data.C.index(comp)
        A_recon[:, j_full] = data.A_partial[:, j_vis]

    return A_recon


def _get_observed_components(data: SpectrumData, t_idx: int) -> List[Component]:
    """
    Return the visible components observed in test t_idx.

    A value of 1 in A_partial indicates that the component was seen
    in the partial trace of that test.
    """
    row = data.A_partial[t_idx, :]
    visible = data.visible_components
    return [visible[j] for j, val in enumerate(row) if val == 1]


def _topo_sort_subset(G: ExecutionGraph, nodes: Set[Component]) -> List[Component]:
    """
    Order a subset of nodes according to a topological sort of the graph.

    If the graph is not a DAG, fall back to the raw node order.
    This is used to choose a consistent start/end among observed nodes.
    """
    try:
        order = list(nx.topological_sort(G.raw_graph))
    except nx.NetworkXUnfeasible:
        order = list(G.raw_graph.nodes)

    return [n for n in order if n in nodes]


def _is_connected_without(
        G: ExecutionGraph,
        n1: Component,
        n2: Component,
        blocked: Component,
) -> bool:
    """
    Check if n1 and n2 remain connected when `blocked` is removed.

    Directly encodes the condition in Rec-Min (Algorithm 1): if removing
    c breaks all paths between some pair of observed nodes, then c must
    be in every consistent full trace.
    """
    return (
            G.has_path(n1, n2, blocked=blocked) or
            G.has_path(n2, n1, blocked=blocked)
    )


# ---------------------------------------------------------------------------
# Rec-Min  (Algorithm 1 – conservative reconstruction)
# ---------------------------------------------------------------------------

def rec_min(data: SpectrumData, G: ExecutionGraph) -> SpectrumData:
    """
    Rec-Min: conservative reconstruction (Algorithm 1).

    For each test t and hidden component c ∈ H:
        - Temporarily "remove" c from the execution graph.
        - If there exists a pair of observed components (n1, n2) such that
          n1 and n2 are not connected without c, then c must be present in
          every full trace consistent with the partial trace of t.

    This minimizes false positives but may miss some executed hidden
    components (false negatives).
    """
    A_recon = _initialize_full_matrix_from_partial(data)
    num_tests = data.num_tests

    for t_idx in range(num_tests):
        # Only consider observed components that appear in the graph
        observed = [
            comp for comp in _get_observed_components(data, t_idx)
            if comp in G.nodes
        ]

        # Need at least two observed nodes to define a pair
        if len(observed) < 2:
            continue

        for c in data.H:
            if c not in G.nodes:
                continue

            must_add = False
            # Check all pairs of observed nodes
            for i in range(len(observed)):
                for j in range(i + 1, len(observed)):
                    n1, n2 = observed[i], observed[j]
                    # If removing c breaks connectivity, c is unavoidable
                    if not _is_connected_without(G, n1, n2, blocked=c):
                        must_add = True
                        break
                if must_add:
                    break

            if must_add:
                j_full = data.C.index(c)
                A_recon[t_idx, j_full] = 1.0

    data.set_reconstructed_matrix(A_recon)
    return data


# ---------------------------------------------------------------------------
# Rec-Max  (graph-based aggressive reconstruction)
# ---------------------------------------------------------------------------

def rec_max(data: SpectrumData, G: ExecutionGraph) -> SpectrumData:
    """
    Rec-Max: aggressive reconstruction.

    Idea:
        For a test t with observed components O(t), we add a hidden
        component c ∈ H if c can appear on at least one execution path
        that is compatible with the partial trace, i.e. if there exists
        an observed node n ∈ O(t) such that:

            n → ... → c   or   c → ... → n

        in the execution graph.

    This increases recall (fewer false negatives) at the cost of more
    false positives.
    """
    A_recon = _initialize_full_matrix_from_partial(data)
    num_tests = data.num_tests

    for t_idx in range(num_tests):
        observed = [
            comp for comp in _get_observed_components(data, t_idx)
            if comp in G.nodes
        ]
        if not observed:
            continue

        for c in data.H:
            if c not in G.nodes:
                continue

            add_c = False

            # Check if c lies on some path related to the observed nodes
            for n in observed:
                if G.has_path(n, c) or G.has_path(c, n):
                    add_c = True
                    break

            if add_c:
                j_full = data.C.index(c)
                A_recon[t_idx, j_full] = 1.0

    data.set_reconstructed_matrix(A_recon)
    return data


# ---------------------------------------------------------------------------
# Rec-Weighted  (Algorithms 2 & 3 – probabilistic reconstruction)
# ---------------------------------------------------------------------------

def _calculate_edge_weights(
        G: ExecutionGraph,
        end_node: Component,
        partial_components: Set[Component],
) -> Dict[Tuple[Component, Component], int]:
    """
    Algorithm 2 – compute edge weights (bottom-up).

    We propagate a counter of "partial-trace hits" backwards from end_node.
    For each edge (u, v), the weight w(u, v) is the maximum number of
    observed components that can still be collected on a best path from
    v to the end node.
    """
    edge_weights: Dict[Tuple[Component, Component], int] = defaultdict(int)
    best_count: Dict[Component, int] = defaultdict(int)

    queue: deque[Component] = deque()
    queue.append(end_node)
    best_count[end_node] = 0  # start from 0, as in the paper

    while queue:
        v = queue.popleft()
        current = best_count[v]

        if v in partial_components:
            current += 1

        for u in G.predecessors(v):
            # Edge (u, v) inherits the best count through v
            edge_weights[(u, v)] = max(edge_weights[(u, v)], current)

            # Propagate improved counts to predecessors
            if current > best_count[u]:
                best_count[u] = current
                queue.append(u)

    return edge_weights


def _calculate_node_hit_probabilities(
        G: ExecutionGraph,
        edge_weights: Dict[Tuple[Component, Component], int],
        start_node: Component,
) -> Dict[Component, float]:
    """
    Algorithm 3 – compute node hit probabilities (top-down).

    We start with P(start_node) = 1 and push probability forward. At
    each node u, we only follow outgoing edges with maximal weight, and
    we split P(u) equally among those successors.

    The result is a distribution P over nodes, approximating the
    probability that a component belongs to the full trace given the
    partial trace.
    """
    P: Dict[Component, float] = defaultdict(float)
    P[start_node] = 1.0

    # Work only with nodes reachable from start_node
    reachable = nx.descendants(G.raw_graph, start_node) | {start_node}

    # Indegree within the reachable subgraph, based on weighted edges
    indegree: Dict[Component, int] = {v: 0 for v in reachable}
    for u, v in edge_weights.keys():
        if u in reachable and v in reachable:
            indegree[v] += 1

    queue: deque[Component] = deque()
    queue.append(start_node)
    visited_preds: Dict[Component, int] = defaultdict(int)

    while queue:
        u = queue.popleft()

        succs = [v for v in G.successors(u) if v in reachable]
        if not succs:
            continue

        # Keep only successors along edges with maximal weight
        max_w = None
        best_succs: List[Component] = []
        for v in succs:
            w = edge_weights.get((u, v), 0)
            if max_w is None or w > max_w:
                max_w = w
                best_succs = [v]
            elif w == max_w:
                best_succs.append(v)

        if not best_succs:
            continue

        share = P[u] / len(best_succs)
        for v in best_succs:
            P[v] += share
            visited_preds[v] += 1
            if visited_preds[v] >= indegree[v]:
                queue.append(v)

    return P


def rec_weighted(data: SpectrumData, G: ExecutionGraph) -> SpectrumData:
    """
    Rec-Weighted: probabilistic reconstruction (Algorithms 2–3).

    For each test t:
        1. Take the set of observed components in its partial trace.
        2. Use the graph order to pick:
               start = earliest observed component,
               end   = latest observed component.
        3. Run Algorithm 2 from `end` (edge weights).
        4. Run Algorithm 3 from `start` (node probabilities).
        5. Store P(c) for all components c in the reconstructed matrix.

    A_recon is thus a probability spectrum that is later consumed by
    the extended SFL diagnosis.
    """
    num_tests = data.num_tests
    num_components = data.num_components_full

    A_recon = np.zeros((num_tests, num_components), dtype=float)

    for t_idx in range(num_tests):
        observed = _get_observed_components(data, t_idx)
        if not observed:
            continue

        observed_set = set(observed)

        topo_observed = _topo_sort_subset(G, observed_set)
        if not topo_observed:
            # Graph does not contain these nodes in a consistent order
            continue

        start_node = topo_observed[0]
        end_node = topo_observed[-1]

        edge_weights = _calculate_edge_weights(
            G,
            end_node=end_node,
            partial_components=observed_set,
        )

        P = _calculate_node_hit_probabilities(
            G,
            edge_weights=edge_weights,
            start_node=start_node,
        )

        for j, comp in enumerate(data.C):
            A_recon[t_idx, j] = float(P.get(comp, 0.0))

    data.set_reconstructed_matrix(A_recon)
    return data
