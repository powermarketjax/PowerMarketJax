"""L2 numerical equivalence for the low-rank ``monitored_lines``/``kkt`` path:
both `make_clearing` and `make_relax` gained two
keywords.  ``monitored_lines`` is a *modelling* choice -- ``None`` carries
every line's limit row, a 1-D int array carries only those rows, dropping the
rest from the LP rather than relaxing them.  ``kkt`` is a *linear-algebra*
choice among `clearing.KKT_ROUTES` -- ``"dense"`` is today's block-tridiagonal
sweep, ``"lowrank"`` is the new Schur-complement route, and ``"auto"`` picks
dense when ``monitored_lines`` is ``None`` and lowrank otherwise.  The two are
orthogonal: `kkt` never changes which LP is solved, only how.

The **solver** comparison in this file is therefore always same-LP,
same-``monitored_lines``, route forced on both sides:

    dense = make_clearing(case, ..., monitored_lines=monitored, kkt="dense")
    new   = make_clearing(case, ..., monitored_lines=monitored)   # kkt="auto" -> lowrank

For 29gb/73rts ``monitored`` is every line, so this is also every line's own
existing dense path.  For 813nem ``monitored`` is the 7 lines that can ever
bind (the other 1271 sit at 1e6 MW and, with max|PTDF| = 1 against 39 164 MW of
installed capacity, never do) -- **not** ``None``, because comparing a 7-row
solve against an unrelated 1278-row solve is not a solver check at all, it is
the modelling difference `test_monitored_rows_move_only_the_degenerate_face`
exists to document separately.  This was this file's bug on its first run: two
813nem tests failed not because the low-rank route was wrong, but because their
"dense" comparator carried every line while "new" carried 7 -- a different LP,
not the same LP through two routes.  Measured then (coordinator, T=2): the two
813nem LPs park on different points of the same near-degenerate face, objective
agreeing to 4.5e-7 relative and shed to 5e-17 while individual awards move up
to 6.2 MW between two units tied on offer within 0.01 $/MWh, and 168/1626
bus-periods move the LMP by more than `ATOL_LMP`.  That is a modelling fact
about 813nem's tie structure, asserted on directly below, not something the
solver comparison should have to tolerate.

The tolerance for the solver comparisons is the L2 one, already measured in
`test_clearing_l2.py` and reused here rather than re-derived:

    ATOL_LMP = 1e-4   (measured 6.2e-5)
    RTOL_LMP = 1e-6   (measured 1.8e-7)
    ATOL_MW  = 1e-6   (measured 1.6e-10 on award)

Three same-LP comparisons use it, all on the monitored-lines LP just defined:

1. **new vs dense**, all three cases -- an implementation error in the new
   route, since both sides solve the identical LP.
2. **new vs the numpy reference** (`reference.clear`, independently assembled,
   on the *same* monitored-lines LP -- for 813nem this means a
   `case.replace(PTDF=..., line_cap=...)` restricted to the 7 rows before
   calling it, not the unmodified 1278-row reference), all three cases -- the
   same check `test_clearing_l2.py` runs against the dense path, now run
   against the new one so a bug that happens to agree with the dense path's
   own error cannot hide.
3. The relaxed-commitment path (`make_relax`), new vs dense only, same-LP the
   same way -- no HiGHS reference is re-run here, since `test_relax_l2.py`
   already establishes that the dense relax matches HiGHS and this file's job
   is only to certify that `monitored_lines`/`kkt` do not change what the
   relax solves.

As in `test_clearing_l2.py`, **per-bus shed is not compared**: when several
buses shed simultaneously they all price at VOLL and the split between them is
not unique, so only the per-period total is asserted.

Both `make_clearing` and `make_relax` are checked independently for whether
they have grown the keyword yet, since the two land separately: a module-level
skip covers `make_clearing` (nothing in this file can run without it, since
even the relax tests build their offer inputs the same way `test_clearing_l2`
does), and each relax test additionally skips on its own if `make_relax` alone
has not caught up.
"""
import inspect
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import VOLL, make_clearing, segment_costs
from powermarketjax.envs.day_ahead.relax import ROUND_EPS, make_relax, round_commitment

from . import reference
from .test_clearing_l2 import ATOL_LMP, ATOL_MW, RTOL_LMP

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from commitment import precommit as PC          # noqa: E402

#: Both horizon and segmentation are the smallest the L2 tests elsewhere in
#: this directory use, on purpose: the numpy reference is O(n^3) dense.  On the
#: full 1278-row LP that made 813nem's reference measured at ~230s at T=2
#: (coordinator's estimate was 1-3 minutes; this file no longer runs that
#: solve -- see `_reference_clear` -- but the constant is kept small since the
#: monitored-only reference is still dense-O(n^3) in the same n).
T, K = 2, 1

#: `cap_scale`, `ramp_scale` per case.  29gb is the L2 `base` scenario
#: (0.6 / 1.0); 73rts is the RTS scenario measured separately
#: (0.42 / 1.0); 813nem has no registered scenario yet, so both are 1.0 -- its
#: network only ever congests on the 7 finite-capacity lines below regardless
#: of `cap_scale`, since the other 1271 lines are 1e6 MW.
CASE_SCALE = {
    "29gb": (0.6, 1.0),
    "73rts": (0.42, 1.0),
    "813nem": (1.0, 1.0),
}
CASE_NAMES = ["29gb", "73rts", "813nem"]

#: `np.where(np.asarray(case.line_cap) < 1e5)[0]` on 813nem -- measured by the
#: coordinator, re-derive with that command if the case data ever changes.
CASE_813NEM_LINES = np.array([10, 134, 188, 361, 363, 738, 1040], dtype=np.int64)


def _demand_for(case_name, case):
    """Total system demand for each case's scenario, MW.

    29gb reuses `test_clearing_l2.py`'s own `base` demand so the two files are
    talking about the same scenario by construction.  73rts and 813nem have no
    such existing scenario, so their demand is a fraction of installed
    capacity, per the coordinator's measurements: 0.6x for 73rts (RTS
    scenario), 0.5x for 813nem.
    """
    pmax = np.asarray(case.unit_p_max, np.float64)
    if case_name == "29gb":
        return 33466.0
    if case_name == "73rts":
        return 0.6 * float(pmax.sum())          # measured 4845.6 MW
    if case_name == "813nem":
        return 0.5 * float(pmax.sum())          # measured 19582.0 MW
    raise ValueError(case_name)


def _monitored_lines(case_name, n_lines):
    """The line subset each case's low-rank path monitors.

    29gb and 73rts have every line finite, so "monitored" is the full set: the
    LP is then identical to the dense one, which is what
    `test_dense_path_is_unchanged_by_the_keyword` and the reference comparison
    below both rely on.
    """
    if case_name == "813nem":
        return CASE_813NEM_LINES
    return np.arange(n_lines, dtype=np.int64)


def _feasible_u_ones(case, demand):
    """All-on commitment, de-committed just enough to leave room under `demand`.

    Not needed for any of the three configurations below -- measured:
    813nem's p_min sums to 10 725 MW against a 19 582 MW demand (u = ones is
    already feasible), and 29gb/73rts are further still from their p_min
    ceiling at their scenario demands -- but kept in case a future re-scale of
    any of the three makes u = ones infeasible.  De-commits the largest-p_min
    units first, since those free the most room per unit de-committed.
    """
    p_min = np.asarray(case.unit_p_min, np.float64)
    u = np.ones_like(p_min)
    if float((p_min * u).sum()) >= demand:
        for i in np.argsort(-p_min):
            u[i] = 0.0
            if float((p_min * u).sum()) < demand:
                break
    return u


def _as_bid_obj(offer, award, shed, p_min, u):
    """As-bid objective from `award`/`shed` alone, K=1 only.

    `clearing.clear` never exposes its solver's `c . x`, so this reconstructs
    the LP's own objective from the outputs it does return.  It is valid only
    at K=1: `offer[:, 0, :]` prices a single segment per unit and period, and
    `award - p_min * u` is exactly that segment's accepted quantity because
    `award = p_min * u + sum_k g_k` (see `clearing.py`'s docstring) collapses
    to `p_min * u + g_0` when there is only one segment. At K>1 the same
    difference would still be the *total* accepted quantity but priced at a
    single rate, which is wrong whenever more than one segment clears.
    """
    above_must_run = np.asarray(award) - p_min[:, None] * u
    return float((offer[:, 0, :] * above_must_run).sum() + VOLL * np.asarray(shed).sum())


def _assert_clear_matches(new, other, label):
    np.testing.assert_allclose(np.asarray(new["lmp"]), np.asarray(other["lmp"]),
                               rtol=RTOL_LMP, atol=ATOL_LMP, err_msg=f"lmp ({label})")
    np.testing.assert_allclose(np.asarray(new["award"]), np.asarray(other["award"]),
                               rtol=1e-6, atol=ATOL_MW, err_msg=f"award ({label})")
    # total only; the split between simultaneously-shedding buses is degenerate
    np.testing.assert_allclose(np.asarray(new["shed"]).sum(1),
                               np.asarray(other["shed"]).sum(1),
                               rtol=1e-6, atol=1e-4, err_msg=f"shed total ({label})")


def _reference_clear(case, monitored, n_lines_all, offer, u, dem, p_init, cap_scale,
                     ramp_scale):
    """`reference.clear`, restricted to `monitored`'s line rows.

    `reference.py` takes no `monitored_lines` argument -- it is the fixed
    independently-assembled reference, not the thing under test -- so the
    restriction is applied to its *input* instead: `CaseData` is a frozen
    `flax.struct.dataclass` (`case_data.py`), and `.replace(PTDF=..., line_cap=...)`
    with both sliced to `monitored` produces a case whose every other field
    (units, buses, ramp rates, ...) is untouched and whose `build_lp` sees
    exactly `len(monitored)` line rows -- the same reduction `make_clearing`
    applies internally, reused here instead of re-deriving `reference.clear`'s
    dual-unpacking with a hand-rolled row mask.  When `monitored` already is
    every line (29gb, 73rts) this reduces to the plain, unmodified case, so the
    two existing L2 configurations there are untouched.

    Measured (813nem, T=2, this file's config): 8.1-8.3s on the 7-row LP
    against ~230s on the full 1278-row one -- both the row count driving the
    dense KKT factorisation and the O(n_lines) Python assembly loop in
    `reference.build_lp` shrink, which is more than the row-count ratio alone
    would predict.
    """
    if monitored.size == n_lines_all:
        return reference.clear(case, offer, u, dem, p_init, cap_scale, ramp_scale)
    case_mon = case.replace(
        PTDF=jnp.asarray(np.asarray(case.PTDF)[monitored]),
        line_cap=jnp.asarray(np.asarray(case.line_cap)[monitored]))
    return reference.clear(case_mon, offer, u, dem, p_init, cap_scale, ramp_scale)


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


_CLEARING_HAS_LOWRANK = "monitored_lines" in inspect.signature(make_clearing).parameters
if not _CLEARING_HAS_LOWRANK:
    pytest.skip("low-rank clearing path (monitored_lines) not implemented yet",
               allow_module_level=True)

#: Checked independently of the module-level skip above: `make_relax` may gain
#: the keyword on its own schedule, separately from `make_clearing`.
_RELAX_HAS_LOWRANK = "monitored_lines" in inspect.signature(make_relax).parameters


def _build_clear_inputs(case_name):
    case = load_case(case_name)
    cap_scale, ramp_scale = CASE_SCALE[case_name]
    demand = _demand_for(case_name, case)
    n_lines = np.asarray(case.line_cap).shape[0]
    monitored = _monitored_lines(case_name, n_lines)

    _, cost = segment_costs(case, K)
    offer = cost[:, :, None] * np.ones(T)
    p_min = np.asarray(case.unit_p_min, np.float64)
    u = np.broadcast_to(_feasible_u_ones(case, demand)[:, None], (len(p_min), T)).copy()
    p_init = p_min * u[:, 0]
    dem = np.full(T, demand)
    return case, cap_scale, ramp_scale, monitored, offer, u, p_init, dem, p_min


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_clear_lowrank_matches_dense_and_reference(case_name):
    case, cap_scale, ramp_scale, monitored, offer, u, p_init, dem, p_min = \
        _build_clear_inputs(case_name)
    n_lines_all = np.asarray(case.line_cap).shape[0]

    # same-LP: both sides carry exactly `monitored`'s line rows, only the
    # route differs.  For 29gb/73rts `monitored` is every line; for 813nem it
    # is the 7 that can ever bind (see the module docstring for why this must
    # not be `monitored_lines=None` on one side).
    dense_clear, _ = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                                   ramp_scale=ramp_scale, monitored_lines=monitored,
                                   kkt="dense")
    new_clear, _ = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                                 ramp_scale=ramp_scale, monitored_lines=monitored)
    args = (jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem), jnp.asarray(p_init))
    dense = jax.jit(dense_clear)(*args)
    new = jax.jit(new_clear)(*args)

    t0 = time.perf_counter()
    want = _reference_clear(case, monitored, n_lines_all, offer, u, dem, p_init,
                            cap_scale, ramp_scale)
    ref_wall = time.perf_counter() - t0
    if case_name == "813nem":
        print(f"\n813nem numpy reference wall time (7-row LP): {ref_wall:.1f}s "
             f"mu={want['mu']:.2e}")

    assert float(dense["mu"]) < 1e-8, "dense (monitored) path did not converge"
    assert float(new["mu"]) < 1e-8, "low-rank path did not converge"
    assert want["mu"] < 1e-8, "numpy reference did not converge"

    # new vs dense: the identical LP through the two routes, all three cases.
    _assert_clear_matches(new, dense, "new vs dense, same monitored LP")
    # new vs the independently-assembled numpy reference, same monitored LP.
    _assert_clear_matches(new, want, "new vs reference, same monitored LP")

    obj_new = _as_bid_obj(offer, new["award"], new["shed"], p_min, u)
    obj_dense = _as_bid_obj(offer, dense["award"], dense["shed"], p_min, u)
    obj_ref = _as_bid_obj(offer, want["award"], want["shed"], p_min, u)
    assert abs(obj_new - obj_dense) / abs(obj_dense) < 1e-8, (
        f"as-bid objective new vs dense: {obj_new} vs {obj_dense}")
    assert abs(obj_new - obj_ref) / abs(obj_ref) < 1e-8, (
        f"as-bid objective new vs reference: {obj_new} vs {obj_ref}")


def test_dense_path_is_unchanged_by_the_keyword():
    """Positive control: `monitored_lines=None` must be indistinguishable from
    not passing the keyword at all -- bit-identical, not merely L2-close."""
    case_name = "29gb"
    case, cap_scale, ramp_scale, _, offer, u, p_init, dem, _ = \
        _build_clear_inputs(case_name)

    no_kw, _ = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                             ramp_scale=ramp_scale)
    explicit_none, _ = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                                     ramp_scale=ramp_scale, monitored_lines=None)
    args = (jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem), jnp.asarray(p_init))
    a = jax.jit(no_kw)(*args)
    b = jax.jit(explicit_none)(*args)

    assert set(a.keys()) == set(b.keys())
    for key in a:
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), (
            f"'{key}' differs between the no-keyword call and monitored_lines=None")


def test_monitored_subset_that_drops_a_binding_line_is_detected():
    """Negative control: dropping the one line that binds at this config must
    move the solution by more than the L2 tolerance, proving the comparisons
    above can fail rather than passing vacuously.

    Measured on this branch's dense path at 29gb's base config: line index 42
    binds in the down direction (`line_dual_dn.max()` ~= 368.6 $/MWh), no other
    line binds either direction. If a future case-data or scenario change makes
    that no longer true, the first assertion below fails loudly rather than the
    test silently checking nothing.
    """
    case_name = "29gb"
    case, cap_scale, ramp_scale, _, offer, u, p_init, dem, _ = \
        _build_clear_inputs(case_name)

    dense_clear, spec = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                                      ramp_scale=ramp_scale)
    args = (jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem), jnp.asarray(p_init))
    dense = jax.jit(dense_clear)(*args)

    up = np.asarray(dense["line_dual_up"])
    dn = np.asarray(dense["line_dual_dn"])
    binding = np.where((up.max(0) > 1e-6) | (dn.max(0) > 1e-6))[0]
    assert binding.size > 0, (
        "no line binds at this config any more; re-pick cap_scale so at least "
        "one does before this negative control means anything (see "
        "test_clearing_l2.py's CONFIGS for scenarios that do)")

    n_lines = spec["n_l"]
    monitored = np.setdiff1d(np.arange(n_lines, dtype=np.int64), binding)
    new_clear, _ = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                                 ramp_scale=ramp_scale, monitored_lines=monitored)
    new = jax.jit(new_clear)(*args)

    # Measured 2026-09-16 on this config: dropping line 42 moves the awards by
    # 158 MW and puts its flow at 1.033 x cap, while the LMP moves by 2.7e-12
    # $/MWh -- that line's PTDF row is uniform enough over the buses that its
    # 368.6 $/MWh dual is absorbed by lambda.  So the quantity that must move
    # is the award (and the flow), not the price.
    award_diff = float(np.max(np.abs(np.asarray(new["award"]) - np.asarray(dense["award"]))))
    assert award_diff > 1.0, (
        f"dropping the only binding line (idx {binding.tolist()}) should move "
        f"the awards by MW, but max |delta award| = {award_diff:.3e} MW; the "
        "comparisons in this file would not have caught a low-rank path that "
        "silently ignored `monitored_lines`")
    # and the dropped line is now overloaded: the constraint really is gone
    ptdf = np.asarray(case.PTDF)
    unit_bus = np.asarray(case.unit_node_idx)
    inj = np.zeros((T, spec["n_buses"]))
    for t in range(T):
        np.add.at(inj[t], unit_bus, np.asarray(new["award"])[:, t])
    net = inj - (spec["demand_share"][None, :] * dem[:, None] - np.asarray(new["shed"]))
    flow = net @ ptdf.T
    cap = np.asarray(case.line_cap) * cap_scale
    assert np.abs(flow[:, binding]).max() > 1.01 * cap[binding].min(), (
        "the dropped line is not overloaded, so the row was not really dropped")


def test_monitored_rows_move_only_the_degenerate_face():
    """813nem only: (7 rows, dense) vs (all 1278 rows, dense) documents the
    *modelling* difference dropping never-binding rows leaves behind, instead
    of asserting the two are numerically identical -- they are not, and the
    solver comparisons above (same LP, both sides) are not the place to find
    that out.

    Measured 2026-09-16 (coordinator's `dbg_813.py`, reproduced independently
    here): the two LPs converge (mu ~3.7e-11 monitored, ~5.3e-9 all-rows) to
    different points of the same near-degenerate face.  Units 144 and 145 tie
    on their period-0 offer (9.3368 vs 9.3468 $/MWh, a 0.01 gap) and split
    6.23 MW differently between the two runs; objective still agrees to
    4.5e-7 relative and shed totals to 5e-17.  This is the ADR-quotable
    reading of the modelling choice, not a bug: every unit whose award moves
    beyond the noise floor has another unit tied with it on offer, and that
    tie -- not a solver defect -- is why the two runs disagree at all.
    """
    case_name = "813nem"
    case, cap_scale, ramp_scale, monitored, offer, u, p_init, dem, p_min = \
        _build_clear_inputs(case_name)

    mon_dense, _ = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                                 ramp_scale=ramp_scale, monitored_lines=monitored,
                                 kkt="dense")
    all_dense, _ = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                                 ramp_scale=ramp_scale, kkt="dense")
    args = (jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem), jnp.asarray(p_init))
    mon = jax.jit(mon_dense)(*args)
    allr = jax.jit(all_dense)(*args)

    assert float(mon["mu"]) < 1e-8, "7-row dense did not converge"
    assert float(allr["mu"]) < 1e-8, "all-row dense did not converge"

    obj_mon = _as_bid_obj(offer, mon["award"], mon["shed"], p_min, u)
    obj_all = _as_bid_obj(offer, allr["award"], allr["shed"], p_min, u)
    assert abs(obj_mon - obj_all) / abs(obj_all) < 1e-6, (
        f"objective, 7-row vs all-row dense: {obj_mon} vs {obj_all} "
        f"(measured 2026-09-16: 4.5e-7 relative)")

    shed_mon = np.asarray(mon["shed"]).sum(1)
    shed_all = np.asarray(allr["shed"]).sum(1)
    np.testing.assert_allclose(shed_mon, shed_all, atol=1e-4, rtol=1e-6,
                               err_msg="shed total, 7-row vs all-row dense")

    # every unit whose award moved beyond noise must have a tied competitor;
    # that tie is the mechanism, so absence of one would mean this is no
    # longer the degenerate-face story the docstring tells.
    award_mon, award_all = np.asarray(mon["award"]), np.asarray(allr["award"])
    da = np.abs(award_mon - award_all)
    offer0 = offer[:, 0, 0]                       # period-0, segment-0 offer
    moved = np.where(da.max(1) > 1e-3)[0]
    for i in moved:
        others = np.arange(len(offer0)) != i
        tie = np.any(np.abs(offer0[others] - offer0[i]) < 0.02)
        assert tie, (
            f"unit {i} moved award by {da[i].max():.3f} MW between the two "
            "row sets with no other unit tied within 0.02 $/MWh on its "
            "period-0 offer")

    # printed, not asserted: this is the reading the ticket said the ADR will
    # quote, not a pass/fail criterion on its own.
    lmp_diff = np.abs(np.asarray(mon["lmp"]) - np.asarray(allr["lmp"]))
    n_over = int((lmp_diff > ATOL_LMP).sum())
    print(f"\n813nem 7-row vs all-row dense: max|delta lmp| = {lmp_diff.max():.3e} "
         f"$/MWh, {n_over}/{lmp_diff.size} bus-periods exceed ATOL_LMP "
         f"({ATOL_LMP:.0e}) (measured 2026-09-16: 0.108 $/MWh, 168/1626)")


def _build_relax_inputs(case_name):
    case = load_case(case_name)
    cap_scale, ramp_scale = CASE_SCALE[case_name]
    demand = _demand_for(case_name, case)
    n_lines = np.asarray(case.line_cap).shape[0]
    monitored = _monitored_lines(case_name, n_lines)

    boundary = PC.first_boundary(case, demand, cap_scale=cap_scale,
                                 ramp_scale=ramp_scale, K=K, delta_h=1.0)
    _, cost = segment_costs(case, K)
    offer = np.repeat(cost[:, :, None], T, axis=2)
    demand_arr = np.full(T, demand)
    return case, cap_scale, ramp_scale, monitored, offer, demand_arr, boundary


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_relax_lowrank_matches_dense(case_name):
    if not _RELAX_HAS_LOWRANK:
        pytest.skip("low-rank relax path (monitored_lines) not implemented yet")

    case, cap_scale, ramp_scale, monitored, offer, demand_arr, boundary = \
        _build_relax_inputs(case_name)

    # same-LP, as in the clearing test above: the dense comparator must carry
    # exactly `monitored`'s rows too, not every line -- see the module
    # docstring for why an unrestricted dense comparator is not a solver check
    # on 813nem.
    dense_relax, _ = make_relax(case, T, n_segments=K, cap_scale=cap_scale,
                                ramp_scale=ramp_scale, monitored_lines=monitored,
                                kkt="dense")
    new_relax, _ = make_relax(case, T, n_segments=K, cap_scale=cap_scale,
                              ramp_scale=ramp_scale, monitored_lines=monitored)
    args = (jnp.asarray(offer), jnp.asarray(demand_arr), jnp.asarray(boundary["p_init"]),
           jnp.asarray(boundary["u_prev"]), jnp.asarray(boundary["up_time"].astype(float)),
           jnp.asarray(boundary["down_time"].astype(float)))
    dense = jax.jit(dense_relax)(*args)
    new = jax.jit(new_relax)(*args)

    assert float(dense["mu"]) < 1e-6, "dense relax did not converge"
    assert float(new["mu"]) < 1e-6, "low-rank relax did not converge"

    np.testing.assert_allclose(float(new["obj"]), float(dense["obj"]), rtol=1e-9,
                               err_msg="relax objective, new vs dense")
    np.testing.assert_allclose(np.asarray(new["u"]), np.asarray(dense["u"]), atol=1e-7,
                               err_msg="relax u, new vs dense")
    np.testing.assert_allclose(np.asarray(new["p"]), np.asarray(dense["p"]), atol=1e-6,
                               err_msg="relax p, new vs dense")

    rounded_new = np.asarray(round_commitment(new["u"]))
    rounded_dense = np.asarray(round_commitment(dense["u"]))
    assert np.array_equal(rounded_new, rounded_dense), (
        f"rounded commitment differs in "
        f"{int(np.abs(rounded_new - rounded_dense).sum())} of {rounded_new.size} "
        "unit-periods between the low-rank and dense relax")
