# Written for this repository -- no upstream counterpart.
# The four solver modules in this package are verbatim copies of
# powerzoojax/envs/grid/{power_flow,bfs_power_flow,bfs_3phase_power_flow,
# ac_power_flow}.py at upstream commit a7641de (2026-05-07), copied 2026-08-05.
# Each solver's own header is the authoritative vendoring record and all four
# now list deviations: bfs_power_flow and ac_power_flow from earlier changes,
# and all four from the documentation-only additions of 2026-08-20.
# Upstream's envs/grid/__init__.py also exported grid *environments*
# (DistGridEnv, TransGridEnv, dc_opf, unit_commitment, ...), which this
# repository does not vendor -- hence this file is written fresh rather
# than copied.
"""Power-flow solvers.

Each solver splits into a NumPy ``prepare_*`` that runs once outside JIT and a
pure-JAX runtime function that is ``jit`` / ``vmap`` compatible.

- ``power_flow``            — DC power flow, thermal and safety checks
- ``ac_power_flow``         — Newton AC power flow (Ybus based)
- ``bfs_power_flow``        — backward/forward sweep for radial feeders
- ``bfs_3phase_power_flow`` — three-phase unbalanced BFS

These are physics, not environments: array in, array out, no episode state.
"""

from powermarketjax.physics.power_flow import (
    dc_power_flow,
    dc_power_flow_with_check,
    safety_check,
    ac_thermal_check,
    compute_generation_cost,
    proportional_dispatch,
)
from powermarketjax.physics.ac_power_flow import (
    ACPFSetup,
    ACPFResult,
    branch_admittances,
    build_ybus,
    prepare_acpf,
    ac_power_flow,
    ac_power_flow_with_check,
    calc_branch_flows,
)
from powermarketjax.physics.bfs_power_flow import (
    BFSTopoData,
    BFSResult,
    MIN_V_PU_FLOOR,
    build_radial_topology,
    prepare_bfs,
    backward_sweep,
    forward_sweep,
    bfs_power_flow,
)
from powermarketjax.physics.bfs_3phase_power_flow import (
    ThreePhaseTopoData,
    BFS3PhResult,
    build_3phase_topology,
    bfs_3phase_power_flow,
)

__all__ = [
    # DC
    "dc_power_flow",
    "dc_power_flow_with_check",
    "safety_check",
    "ac_thermal_check",
    "compute_generation_cost",
    "proportional_dispatch",
    # AC
    "ACPFSetup",
    "ACPFResult",
    "branch_admittances",
    "build_ybus",
    "prepare_acpf",
    "ac_power_flow",
    "ac_power_flow_with_check",
    "calc_branch_flows",
    # BFS (radial, single phase)
    "BFSTopoData",
    "BFSResult",
    "MIN_V_PU_FLOOR",
    "build_radial_topology",
    "prepare_bfs",
    "backward_sweep",
    "forward_sweep",
    "bfs_power_flow",
    # BFS (radial, three phase)
    "ThreePhaseTopoData",
    "BFS3PhResult",
    "build_3phase_topology",
    "bfs_3phase_power_flow",
]
