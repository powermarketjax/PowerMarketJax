"""Voltage sensitivity matrices of the linearised DistFlow model.

Setup-time numpy, float64, run once per case.  Nothing here is jittable and
nothing here needs to be: the matrices are constants of the feeder, and the
clearing operator consumes them as such.

    R[n, m] = sum of  r_l  over the lines shared by the path from the
              substation to bus n and the path from the substation to bus m
    X[n, m] = the same with reactance
    A[l, m] = 1 when bus m lies downstream of line l, else 0

`A` is the coefficient of the clearing's flow expression and `R`, `X` are the
coefficients of its squared-voltage expression, so the three together are
everything the clearing problem needs from the network.

The line ratings are converted here as well: `line_cap` is a thermal rating
in MVA, while `R`, `X` and every flow derived from them are per unit.
Assembling the thermal constraint against the raw rating leaves the limit
larger than the flows by the base power, so the constraint never binds and
the market silently becomes voltage driven -- the linear program stays
feasible, converges, and returns finite quantities without reporting the
error.  Converting once here means no caller has to remember.

Both matrices are computed as one product against the path indicator matrix,

    P[n, l] = 1 when line l lies on the path from the substation to bus n
    R = P diag(r) P'

which is the same arithmetic as summing over the shared path, because the
product picks out exactly the lines that both rows carry.

The arithmetic runs in float64, and that is the rule that matters.  Every
array of a `CaseData` is float32, the line parameters included, so the
float64 buys arithmetic precision rather than data precision: upcasting the
vendored `BFSTopoData` and multiplying in float64 reproduces this module bit
for bit, while carrying the same product out in float32 does not.

The tree is nonetheless built here rather than taken from `prepare_bfs`, for
reasons unrelated to precision: `line_index` is needed to read `line_cap` and
the other per-line registers, the spanning-tree check below is needed and
`build_radial_topology` does not make it, and the clearing operator wants a
setup-time numpy object rather than a device array.

Lines registered out of service are removed before the tree is built,
matching `prepare_bfs`.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoltageSensitivity:
    """Network constants of one feeder, all float64.

    Attributes:
        R: ``(n_bus, n_bus)`` shared-path resistance, per unit.
        X: ``(n_bus, n_bus)`` shared-path reactance, per unit.
        A: ``(n_line, n_bus)`` downstream indicator, ``A[l, m] = 1`` when bus
            ``m`` is downstream of line ``l``.
        r: ``(n_line,)`` resistance of each in-service line, per unit.
        x: ``(n_line,)`` reactance of each in-service line, per unit.
        p_max: ``(n_line,)`` thermal rating of each in-service line, **per
            unit**, converted from the case's `line_cap` in MVA.
        line_index: ``(n_line,)`` position of each in-service line in the
            case's original line arrays, for reading any other per-line
            register.
        slack: index of the substation bus.
        base_mva: system base power, the factor between per unit and MVA.
    """

    R: np.ndarray
    X: np.ndarray
    A: np.ndarray
    r: np.ndarray
    x: np.ndarray
    p_max: np.ndarray
    line_index: np.ndarray
    slack: int
    base_mva: float

    @property
    def n_bus(self) -> int:
        """Number of buses of the feeder, the substation included."""
        return self.R.shape[0]

    @property
    def n_line(self) -> int:
        """Number of in-service lines, which `build_voltage_sensitivity` pins
        at ``n_bus - 1``; lines registered out of service are not counted, and
        ``line_index`` maps the remaining ones back to the case's numbering."""
        return self.r.shape[0]


def build_voltage_sensitivity(case) -> VoltageSensitivity:
    """Build the voltage sensitivity matrices for one distribution case.

    Args:
        case: a `CaseData` carrying `line_r`, `line_x`, `line_from_idx`,
            `line_to_idx`, `line_status` and `slack_bus_idx`.

    Returns:
        `VoltageSensitivity`, all arrays float64.

    Raises:
        ValueError: if the in-service lines do not form a spanning tree of the
            buses.  A feeder with a cycle would have lines outside the tree,
            and dropping them silently would model a different network:
            radial operation requires ``n_line == n_bus - 1``.
    """
    n_bus = int(case.n_nodes)
    slack = int(case.slack_bus_idx)

    # float64 from the entry onwards.  Every array of the case is float32, so
    # the upcast recovers no data precision; what it buys is arithmetic
    # precision in the two matrix products below.
    r = np.asarray(case.line_r, np.float64)
    x = np.asarray(case.line_x, np.float64)
    frm = np.asarray(case.line_from_idx, np.int64)
    to = np.asarray(case.line_to_idx, np.int64)
    line_index = np.arange(len(frm), dtype=np.int64)

    if case.line_status is not None:
        live = np.asarray(case.line_status) > 0
        r, x, frm, to, line_index = r[live], x[live], frm[live], to[live], line_index[live]

    n_line = len(line_index)
    if n_line != n_bus - 1:
        raise ValueError(
            f"in-service lines do not form a spanning tree: {n_line} lines for "
            f"{n_bus} buses, expected {n_bus - 1}")

    parent_line = _spanning_tree(n_bus, frm, to, slack)

    # P[n, l] = 1 for every line on the path from the substation to bus n.
    # Walking up from each bus is O(n_bus * depth); on 533 buses that is
    # nothing, and it keeps the construction readable.
    P = np.zeros((n_bus, n_line), np.float64)
    for bus in range(n_bus):
        node = bus
        while node != slack:
            line = parent_line[node]
            P[bus, line] = 1.0
            node = frm[line] if to[line] == node else to[line]

    R = (P * r) @ P.T
    X = (P * x) @ P.T
    base_mva = float(case.base_mva)
    p_max = np.asarray(case.line_cap, np.float64)[line_index] / base_mva
    return VoltageSensitivity(R=R, X=X, A=P.T, r=r, x=x, p_max=p_max,
                              line_index=line_index, slack=slack, base_mva=base_mva)


def _spanning_tree(n_bus: int, frm: np.ndarray, to: np.ndarray, slack: int) -> np.ndarray:
    """Depth-first tree rooted at the substation; returns the parent line of each bus.

    The traversal order does not matter: the in-service lines have already been
    checked to number one fewer than the buses, so a connected result is a tree
    and its parent structure is unique.  ``parent_line[slack]`` is -1 and is
    never read, since the walk above stops at the substation.
    """
    adjacency: list = [[] for _ in range(n_bus)]
    for line, (a, b) in enumerate(zip(frm, to)):
        adjacency[a].append((b, line))
        adjacency[b].append((a, line))

    parent_line = np.full(n_bus, -1, np.int64)
    seen = np.zeros(n_bus, bool)
    seen[slack] = True
    queue = [slack]
    while queue:
        node = queue.pop()
        for neighbour, line in adjacency[node]:
            if not seen[neighbour]:
                seen[neighbour] = True
                parent_line[neighbour] = line
                queue.append(neighbour)

    if not seen.all():
        raise ValueError(
            f"buses unreachable from the substation: {np.flatnonzero(~seen).tolist()}")
    return parent_line
