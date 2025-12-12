from __future__ import annotations

"""
Core data models for the partial-trace SFL framework.

We define two main abstractions:

  1. SpectrumData  – tests, outcomes, components and spectra.
  2. ExecutionGraph – directed graph G(V, E) used by reconstruction.
"""

from dataclasses import dataclass, field
from typing import List, Set, Sequence, Iterable, Optional

import networkx as nx
import numpy as np

# Type aliases
Component = str
TestCase = str


# ---------------------------------------------------------------------------
# Spectrum data
# ---------------------------------------------------------------------------

@dataclass
class SpectrumData:
    """
    Container for all SFL data of a program / benchmark.

    Attributes:
        T         : ordered list of test identifiers.
        e         : pass/fail vector (1 = fail, 0 = pass), shape (|T|,).
        C         : ordered list of all components in the system.
        H         : set of hidden components (H ⊆ C).
        A_partial : spectrum over visible components C \\ H, shape (|T|, |C \\ H|).
        A_recon   : optional reconstructed / probabilistic spectrum over C,
                    shape (|T|, |C|). Can be binary (Rec-Min / Rec-Max)
                    or probabilistic (Rec-Weighted).
    """

    T: List[TestCase]
    e: np.ndarray
    C: List[Component]
    H: Set[Component]
    A_partial: np.ndarray
    A_recon: Optional[np.ndarray] = None

    # ------------------------- basic properties -------------------------

    @property
    def num_tests(self) -> int:
        """Return |T|, the number of tests."""
        return len(self.T)

    @property
    def num_components_full(self) -> int:
        """Return |C|, the number of components."""
        return len(self.C)

    @property
    def visible_components(self) -> List[Component]:
        """
        Return the visible components C \\ H.

        The order matches the columns of A_partial and the order in C.
        """
        return [c for c in self.C if c not in self.H]

    # ---------------------- construction helpers -----------------------

    @classmethod
    def from_full_spectra(
            cls,
            T: Sequence[TestCase],
            e: Sequence[int],
            C: Sequence[Component],
            H: Iterable[Component],
            A_full: np.ndarray,
    ) -> "SpectrumData":
        """
        Build SpectrumData from a full spectrum matrix A_full.

        Args:
            T      : test identifiers.
            e      : pass/fail outcomes for each test.
            C      : all components in the system.
            H      : hidden components (H ⊆ C).
            A_full : full binary spectrum over all components,
                     shape (|T|, |C|).

        Returns:
            SpectrumData with A_partial obtained by projecting A_full
            to visible components C \\ H. A_recon is not set.
        """
        T_list = list(T)
        e_arr = np.asarray(e, dtype=int).reshape(-1)
        C_list = list(C)
        H_set = set(H)

        if A_full.shape != (len(T_list), len(C_list)):
            raise ValueError(
                f"A_full has shape {A_full.shape}, expected "
                f"({len(T_list)}, {len(C_list)})"
            )

        # Visible components and their indices in C
        visible = [c for c in C_list if c not in H_set]
        if not visible:
            raise ValueError("Visible component set is empty (H == C).")

        visible_indices = [C_list.index(c) for c in visible]
        A_partial = A_full[:, visible_indices].astype(int)

        return cls(
            T=T_list,
            e=e_arr,
            C=C_list,
            H=H_set,
            A_partial=A_partial,
        )

    # --------------------- reconstruction handling ---------------------

    def set_reconstructed_matrix(self, A_recon: np.ndarray) -> None:
        """
        Attach a reconstructed / probabilistic spectrum A_recon.

        A_recon must have shape (|T|, |C|). For Rec-Min / Rec-Max it is
        binary; for Rec-Weighted it stores probabilities in [0, 1].
        """
        if A_recon.shape != (self.num_tests, self.num_components_full):
            raise ValueError(
                f"A_recon has shape {A_recon.shape}, expected "
                f"({self.num_tests}, {self.num_components_full})"
            )
        self.A_recon = A_recon

    @property
    def active_spectrum(self) -> np.ndarray:
        """
        Return the spectrum currently used for SFL.

        If A_recon is set, use it; otherwise use A_partial.
        """
        if self.A_recon is not None:
            return self.A_recon
        return self.A_partial


# ---------------------------------------------------------------------------
# Execution graph
# ---------------------------------------------------------------------------

@dataclass
class ExecutionGraph:
    """
    Directed execution graph G(V, E) used by reconstruction.

    Nodes:
        Components (e.g. fully-qualified function names).

    Edges:
        Possible control-flow (e.g. static call graph edges).

    Internally this wraps a NetworkX DiGraph, but only a small API is
    exposed to keep the rest of the code simple.
    """

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    # ---------------------- basic manipulation -------------------------

    def add_node(self, component: Component) -> None:
        """Add a component node if it does not exist."""
        self.graph.add_node(component)

    def add_edge(self, src: Component, dst: Component) -> None:
        """Add a directed edge src → dst."""
        self.graph.add_edge(src, dst)

    # --------------------------- queries -------------------------------

    @property
    def nodes(self) -> List[Component]:
        """Return all nodes in the graph."""
        return list(self.graph.nodes)

    def successors(self, node: Component) -> List[Component]:
        """Return successors of node."""
        return list(self.graph.successors(node))

    def predecessors(self, node: Component) -> List[Component]:
        """Return predecessors of node."""
        return list(self.graph.predecessors(node))

    # --------------------------- paths ---------------------------------

    def has_path(
            self,
            src: Component,
            dst: Component,
            blocked: Optional[Component] = None,
    ) -> bool:
        """
        Check if there is a path src → dst.

        If blocked is given, treat that node as removed. The original
        graph is not modified; we work on a temporary copy.
        """
        if src not in self.graph or dst not in self.graph:
            return False

        # No blocked node
        if blocked is None or blocked not in self.graph:
            try:
                return nx.has_path(self.graph, src, dst)
            except nx.NetworkXError:
                return False

        # Work on a temporary copy without the blocked node
        temp = self.graph.copy()
        temp.remove_node(blocked)

        try:
            return nx.has_path(temp, src, dst)
        except nx.NetworkXError:
            return False

    # --------------------------- raw access ----------------------------

    @property
    def raw_graph(self) -> nx.DiGraph:
        """Expose the underlying NetworkX DiGraph (for debugging/visualisation)."""
        return self.graph
