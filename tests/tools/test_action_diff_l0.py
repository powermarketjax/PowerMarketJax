# Written for this repository on 2026-09-06 -- no upstream counterpart.
"""L0: the reported action-range numbers are still in the products behind them.

The test names carry short labels for four groups of reported numbers:
`c01` the collapse of the shared-parameter policy's per-unit action range on
market 01, `c03` the per-agent-parameter archives and their profits, `c27` the
market-02 recomputation, `c56` the float64 cross-unit dedup reading.  Until 2026-09-06 the apparatus
for these claims (594 lines) and every one of its products lived in a
directory that `.gitignore` excludes -- the failure mode that has already cost
this repository one set of products.  The device was then COPIED to
`tools/day_ahead/action_diff.py` and five products to `tests/fixtures/`; the
originals were left in place, because the 02/03 half -- ingested later, as
`tools/real_time/action_diff_rt.py` -- imported from the original copy by name
at the time.  This file is what stops the claims and the repository's
copies from drifting apart afterwards.

WHAT IS RECOMPUTED HERE, AND FROM WHAT
---------------------------------------
Not read off the summary blocks the device wrote.  Every median below is
recomputed from `range_per_unit` -- the per-unit array -- and the 30-unit
denominator comes from a DIFFERENT product: the `ever_on` mask in
`da_alpha_cross_section_wd052_iter110_seed0.npz`, which
was written by the truthful arm, not by any checkpoint.  So the 30-unit numbers
here are the agreement of two products written by two devices, not one product
agreeing with itself.  `test_the_recomputation_reproduces_the_devices_own_summary`
then asserts the recomputation matches the summary the device stored, which is
the only place the two paths are allowed to be compared.

TOLERANCES ARE DERIVED FROM THE REPORTED VALUES' PRINTED WIDTH, NOT PINNED TO A MEASUREMENT
-------------------------------------------------------------------------------------------
`REL_4SF` and `REL_7SF` are the largest relative error a correct number can have
after being rounded to that many significant figures (0.5 in the last place of a
leading `1`), so a failure means the quantity moved, not that a digit was
re-rounded (a quantity made of ULPs is bounded by its ULP magnitude, not by
one day's reading).  Measured 2026-09-06 on CPU, market 01 at `29gb`,
`cap_scale = 0.60`, `ramp_scale = 1.00`, `markup_max = 2.0`, 12 evaluation days:
the largest deviation actually seen against a 4-figure claimed value is 2.0092e-4
(the market-01 pilot200 pair) and against a 7-figure one 3.7033e-7 (the shared
seed-2 profit).  Both sit under their bound with room, and neither bound is one
of those readings.

WHY THIS FILE DOES NOT IMPORT THE TOOL
---------------------------------------
`tools/day_ahead/action_diff.py` runs `jax.config.update("jax_enable_x64", True)`
at import, and `tests/tools/test_runtime_stamp_l0.py::test_this_session_runs_without_x64`
asserts the opposite for this session; importing it here would break that
assertion depending on collection order.  The cross-section tool hit the same
wall and was split in two.  Nothing here needs the tool: the quantities are medians of arrays
the products already carry, and `numpy` computes them.

**THE MEASURED "IT BITES" DATUM, as required before a check counts.**  Measured
2026-09-06 against a corrupted copy of `da_action_diff_01_wd052_controls.npz`:
one entry of the pilot200 row's `range_per_unit`, unit 8, which sits inside the
30-unit `ever_on` set and at the upper of the two tied values at the middle of
the sorted order, changed from `1.3322676295501878e-15` to
`1.3322676295501878e-13` -- one digit of the exponent, a value still far below
anything a market can act on, every other byte identical.  The 30-unit median
moves to `1.4432899320127035e-15`, an 8.3% shift.  Run against it, this file
gives **3 failed, 12 passed**:

  `test_c01_claim_sentence_numbers_on_both_denominators` -- the 66-unit
      assertion is evaluated first and passes, then the 30-unit one fails
      against the reported 1.332e-15;
  `test_the_recomputation_reproduces_the_devices_own_summary` -- the summary the
      device stored still reads 1.3323e-15, so the two paths part company;
  `test_the_products_are_the_ones_the_stamps_say_they_are` -- the sha256 over
      the carried arrays no longer matches the one taken at ingest.

**The twelve that still pass are the point of the paragraph.**  Every `c03`
check passes, because those numbers live in a different product; the 66-unit
`c01` assertion inside the failing test passes, because the corrupted unit sits
above the 66-unit median and moving it further up cannot move that median;
`c27` and `c56` pass for the same reason.  A version of this file written
around the 66-unit denominator alone -- the one the market-01 numbers lead
with -- would go green against a product whose 30-unit half is wrong.  That is
why every median here is asserted on both denominators.

**The first version of this file was itself green against that corruption, and
the bite run is the only reason anyone knows.**  Every `approx` had been written
without `abs=`, and `pytest.approx` defaults to `abs=1e-12` while comparing
against `max(rel * expected, abs)` -- so on the quantities this file exists to
guard, all of which live at 1e-8 or below, the relative tolerance never applied.
The run reported 2 failed instead of 3.  See `ABS_0` below.
"""
import hashlib
import json
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"

#: Largest relative error a value rounded to 4 / 7 significant figures can carry.
#: Derived from the printed width, not pinned to a reading -- see the docstring.
REL_4SF = 5e-4
REL_7SF = 5e-7

#: **Every `approx` below passes `abs=0`, and that is load-bearing.**
#: `pytest.approx` defaults to `abs=1e-12` and compares against
#: `max(rel * expected, abs)`, so on quantities that live at 1e-8 and below --
#: which is most of what this file checks -- the default absolute tolerance
#: swallows the relative one and the assertion becomes vacuous.  Measured
#: 2026-09-06: written without `abs=0`, this file went GREEN against the
#: corrupted product described in the docstring, and the "it bites" run reported
#: 2 failed instead of 3.  A gate that cannot fail is not a gate.
ABS_0 = 0.0

#: The products, by the number groups they carry.
CONTROLS = "da_action_diff_01_wd052_controls.npz"          # c01
SWEEP200 = "da_action_diff_01_wd052_sweep200.npz"          # c56
PER_AGENT = "da_action_diff_01_per_agent_final.npz"        # c03 range medians
LAMBDA = "da_lambda_01_branches.npz"                       # c03 profits
RT02 = "rt_action_diff_02_seed0_iter200.npz"               # c27
CROSS = "da_alpha_cross_section_wd052_iter110_seed0.npz"   # the ever_on mask

ALL_STAMPED = (CONTROLS, SWEEP200, PER_AGENT, LAMBDA, RT02)

#: sha256 over the carried arrays (key name then bytes, keys sorted), taken
#: 2026-09-06 at ingest and re-taken 2026-09-22 after the working-directory
#: prefix was removed from the path strings in `rows` / `payload`; every
#: numeric array was compared bit for bit across that edit.  Not a substitute for the checks below -- it says the
#: bytes did not move, they say the bytes still mean what the claim quotes.
ARRAY_SHA256 = {
    CONTROLS: "139b1df123c6f1629b9aab62d7dd0486eb40f99da0ce44657a2d8c965ee79c3d",
    SWEEP200: "d72cd580da1cbfcc97b675fd874edd509a2e4b104fed26f6898293f723fd292d",
    PER_AGENT: "ee61e432f166fe2d7c44f0ca6d17a7e043696cb4d5f6b28a2b035fb67d64942d",
    LAMBDA: "274987109b000b45d29c1549fe7af24a4cd5dc9a810f9f6290b36091457cdaa6",
    RT02: "80e5c517f03c1c602a3118e13ad22566f9655bdf0d5d0edd28660a36917a91ac",
}

N_UNITS, N_ON, N_OBS_01 = 66, 30, 12

#: The market-01 headline numbers, both denominators.  "untrained" is the control and
#: "pilot200" the 200-iteration archive; the fifteen orders of magnitude reported
#: are the gap between the two rows.
C01 = {"untrained": {66: 0.8518, 30: 0.8534},
       "pilot200": {66: 2.220e-16, 30: 1.332e-15}}

#: The per-agent archives.  **The order is seed 1 / seed 2 / `wd052`, not by
#: seed number** -- it is reported in bold because the highest-profit branch
#: is `wd052`, which is seed 0.  Asserting the paths is what makes the order a
#: check rather than a convention this file happens to share.
C03_ROWS = [
    ("pa_s1_0400", "r1_01_pa_s1/checkpoints/seed1_iter0400.npz",
     {66: 1.719792e-7, 30: 8.341439e-9}),
    ("pa_s2_0400", "r1_01_pa_s2/checkpoints/seed2_iter0400.npz",
     {66: 4.667881e-7, 30: 1.273011e-8}),
    ("pa_wd052_0200", "r1_01_pa_wd052/checkpoints/seed0_iter0200.npz",
     {66: 2.903380e-8, 30: 3.478883e-8}),
]

#: The per-agent versus shared twelve-day profits, in US dollars, per-agent side then shared side,
#: each in the same seed 1 / seed 2 / `wd052` order.
C03_PROFIT = {"r1_01_pa_s1": 1.933438e8, "r1_01_pa_s2": 2.197372e8,
              "r1_01_pa_wd052": 2.273100e8, "r1_01_s1": 1.087779e8,
              "r1_01_s2": 1.164325e8, "rl_01_wd052": 1.244120e8}

#: The market-02 recomputation.
C27 = {"untrained": dict(r_med=7.480e-1, l_med=576, dedup=(65, 65)),
       "pilot200": dict(r_med=1.443e-7, l_med=17, dedup=(23, 53))}

#: The float64 reading of the cross-unit dedup on the `wd = 0.52` side.
#: The reported "2 to 44" is the float32 reading; recomputed, this side reads 19 to 64 when the forward pass is recomputed in float64,
#: and this product is that float64 recomputation, at iteration 200.
C56_DEDUP_AT_200 = (19, 64)


def load(name):
    return np.load(FIXTURES / name, allow_pickle=False)


@pytest.fixture(scope="module")
def ever_on():
    """The 30-unit denominator, from the cross-section product rather than from this one.

    Set by the truthful arm over the 12 evaluation days, so it belongs to the
    run point and not to any checkpoint -- which is why the same mask is the
    right one for the shared and the per-agent products alike.
    """
    m = np.asarray(load(CROSS)["ever_on"], bool)
    assert m.size == N_UNITS and int(m.sum()) == N_ON
    return m


@pytest.fixture(scope="module")
def rows():
    return {n: {r["label"]: r for r in json.loads(str(load(n)["rows"]))}
            for n in (CONTROLS, SWEEP200, PER_AGENT, RT02)}


@pytest.fixture(scope="module")
def branches():
    return json.loads(str(load(LAMBDA)["payload"]))["branches"]


def medians(row, mask):
    """(66-unit, 30-unit) median of one row's per-unit action range.

    Recomputed from the per-unit array; the summary blocks the device wrote are
    not consulted here, which is what lets the two be compared afterwards.
    """
    rp = np.asarray(row["range_per_unit"], np.float64)
    assert rp.size == N_UNITS
    return {66: float(np.median(rp)), 30: float(np.median(rp[mask]))}


# --------------------------------------------------------------------------
# the products, and what they say about themselves
# --------------------------------------------------------------------------
def test_the_products_are_the_ones_the_stamps_say_they_are():
    """A copy that cannot be told from a re-run is not a copy, it is a claim.

    A binary product's stamp can only be written by its writer, so the
    stamp attached at ingest says `REPACKAGED` in its own key and never touches the
    `meta` the device wrote.  The sha256 here is over the carried arrays, so it
    catches an edit to the arrays even though the file's own hash changed when
    the stamp was added.
    """
    for name in ALL_STAMPED:
        d = load(name)
        st = json.loads(str(d["fixture_stamp"]))
        assert st["stamp_kind"] == "REPACKAGED, not a writer stamp", name
        assert st["recomputed"] is False, name
        assert st["ingested_on"].startswith("2026-09-06"), name
        assert len(st["ingested_from_sha256"]) == 64, name
        assert st["ingested_from"].endswith(".npz"), name
        assert not st["ingested_from"].startswith(("/", "scratch/")), name
        h = hashlib.sha256()
        for k in sorted(k for k in d.files if k != "fixture_stamp"):
            h.update(k.encode())
            h.update(np.ascontiguousarray(d[k]).tobytes())
        assert h.hexdigest() == ARRAY_SHA256[name], (
            f"{name}: the carried arrays are not the ones ingested")


def test_the_observation_grid_is_non_trivial():
    """The market-01 numbers' second refutation condition, as a check rather than a footnote.

    "or the grid's non-triviality precondition fails, i.e. some unit sees fewer
    than two distinct observations on the grid".  A device handed the same
    observation twelve times reports a zero range for every policy, which reads
    exactly like the collapse being measured -- so this has to be asserted
    before any range median is believed.
    """
    for name in (CONTROLS, SWEEP200, PER_AGENT):
        n = np.asarray(load(name)["obs_grid_distinct_per_unit"], np.int64)
        assert n.size == N_UNITS, name
        assert n.min() >= 2, f"{name}: some unit sees one observation"
        assert set(n.tolist()) == {N_OBS_01}, (
            f"{name}: distinct observations per unit are not all {N_OBS_01}")
    n02 = np.asarray(load(RT02)["obs_grid_distinct_per_unit"], np.int64)
    assert n02.size == N_UNITS and n02.min() >= 2


# --------------------------------------------------------------------------
# c01: market 01, shared parameters
# --------------------------------------------------------------------------
def test_c01_claim_sentence_numbers_on_both_denominators(rows, ever_on):
    """The four market-01 headline numbers, recomputed on 66 and on 30 units.

    The report tells the reader not to quote one pair without its denominator;
    this is that instruction as a check.  0.8518 -> 2.220e-16 on 66 units and
    0.8534 -> 1.332e-15 on 30 are the fifteen orders of magnitude.
    """
    for label, want in C01.items():
        got = medians(rows[CONTROLS][label], ever_on)
        for n in (66, 30):
            assert got[n] == pytest.approx(want[n], rel=REL_4SF, abs=ABS_0), (
                f"market 01 {label} on {n} units: {got[n]:.6e}, "
                f"reported {want[n]:.4e}")


def test_c01_the_two_rows_are_fifteen_orders_apart_on_both_denominators(rows, ever_on):
    """The claim is the RATIO, and a ratio survives both numbers moving together.

    Asserted separately from the four values because the failure it catches is a
    different one: a re-measurement that shifted both rows would keep this green
    and fail the check above, and one that shifted only the trained row would do
    the reverse.
    """
    un = medians(rows[CONTROLS]["untrained"], ever_on)
    tr = medians(rows[CONTROLS]["pilot200"], ever_on)
    for n in (66, 30):
        assert np.log10(un[n] / tr[n]) >= 14.0, (
            f"on {n} units the gap is 10^{np.log10(un[n] / tr[n]):.1f}, "
            "the reported gap is fifteen orders of magnitude")


def test_c01_the_30_unit_denominator_is_the_larger_side_at_both_ends(rows, ever_on):
    """The two denominators disagree in SIZE, not only in value.

    The trained end is six times larger on 30 units than on 66, and the
    untrained end is larger too by a hair.  Only the direction is asserted here:
    the "six times" is already pinned by the four values above, to four figures
    each, and a second check on their ratio would need a tolerance nobody
    measured for a quantity nobody reports.  What this catches is a mask that
    had silently become a different 30 units -- an arbitrary subset would sit on
    either side of the 66-unit median, not reliably above it at both ends.
    """
    for label in ("untrained", "pilot200"):
        m = medians(rows[CONTROLS][label], ever_on)
        assert m[30] > m[66], (
            f"market 01 {label}: 30-unit median {m[30]:.6e} is not above the "
            f"66-unit {m[66]:.6e}")


def test_the_recomputation_reproduces_the_devices_own_summary(rows, ever_on):
    """Two paths to the same number, and the only place they may be compared.

    The medians above come from `range_per_unit` and a mask from another
    product; these come from the summary blocks the device computed when it ran.
    They are equal bit for bit, which is what says the recomputation is of the
    same quantity and not of a neighbouring one.
    """
    for name in (CONTROLS, PER_AGENT):
        for label, row in rows[name].items():
            got = medians(row, ever_on)
            assert got[66] == row["all_units"]["range_median"], (name, label)
            assert row["ever_on"]["n_units"] == N_ON, (name, label)
            assert got[30] == row["ever_on"]["range_median"], (name, label)


def test_the_device_reported_two_paths_agreeing_to_float64_roundoff(rows):
    """The reported recomputation: "max absolute difference 4.441e-16".

    The device computes every action twice -- once through the package under
    `jit` and once in plain numpy -- and stores the largest disagreement. It is
    a property of the run, so it is asserted rather than recomputed; a product
    whose two paths had diverged would be reporting a bug, not a policy.
    """
    for name in (CONTROLS, SWEEP200, PER_AGENT, RT02):
        for label, row in rows[name].items():
            assert row["two_path_max_abs_diff"] <= 4.441e-16, (
                f"{name} {label}: the jit and numpy paths differ by "
                f"{row['two_path_max_abs_diff']:.3e}")


# --------------------------------------------------------------------------
# c03: market 01, per-agent parameters
# --------------------------------------------------------------------------
def test_c03_the_product_holds_the_three_archives_the_ledger_names(rows):
    """The three per-agent final archives, in the order they are reported.

    **Not seed order**: seed 1, seed 2, then `wd052` which is seed 0.  The
    report sets this out in bold because the highest-profit branch is the last
    one, and a reader who assumed seed order would attribute it to seed 2.
    """
    got = [(lab, r["path"]) for lab, r in rows[PER_AGENT].items()]
    assert got == [(lab, path) for lab, path, _ in C03_ROWS]
    for lab, _, _ in C03_ROWS:
        assert rows[PER_AGENT][lab]["per_agent_params"] is True, lab


def test_c03_the_three_range_medians_on_both_denominators(rows, ever_on):
    """The three per-agent range medians on 30 units, and their 66-unit twins.

    The 30-unit trio is the headline and the 66-unit one the qualification, with "the two denominators may not be quoted interchangeably"
    spelled out; both are recomputed here for that reason.  Until the ingest these
    six numbers were in no product that was named -- only the three
    archives and the device were, and the product that actually carried the
    read-out went unnamed.
    """
    for lab, _, want in C03_ROWS:
        got = medians(rows[PER_AGENT][lab], ever_on)
        for n in (66, 30):
            assert got[n] == pytest.approx(want[n], rel=REL_7SF, abs=ABS_0), (
                f"per-agent {lab} on {n} units: {got[n]:.7e}, "
                f"reported {want[n]:.7e}")


def test_c03_the_response_did_not_come_back_while_the_dedup_pinned_at_66(rows):
    """The half of the per-agent result that is a contrast, not a number.

    "cross-unit dedup pinned at 66 at the same time" -- 66 distinct action rows,
    one per unit, on every observation, while the per-unit range across
    observations stays at 1e-8.  That pair is the claim: the action's VARIETY is
    maximal and its RESPONSE to the state is not there.  Asserted per
    observation and not only on the min/max, because a row that dipped on one
    day would leave both summaries at 66.
    """
    for lab, _, _ in C03_ROWS:
        row = rows[PER_AGENT][lab]
        assert (row["cross_unit_dedup_min"], row["cross_unit_dedup_max"]) == (66, 66), lab
        assert row["cross_unit_dedup"] == [66] * N_OBS_01, lab


def test_c03_the_twelve_day_profits_recompute_from_the_lambda_branches(branches):
    """The six per-agent and shared profit figures, summed from the per-day rows.

    The recomputation says these were "summed per day from
    `archive`"; this is that sum, on the product now in the repository.  The
    per-agent side is roughly twice the shared side, which is the result.
    """
    got = {}
    for name, want in C03_PROFIT.items():
        pd = np.asarray(branches[name]["archive"]["profit_per_day"], np.float64)
        assert pd.size == N_OBS_01, name
        got[name] = float(pd.sum())
        assert got[name] == pytest.approx(want, rel=REL_7SF, abs=ABS_0), (
            f"profit {name}: {got[name]:.7e}, reported {want:.7e}")
    for pa, sh in (("r1_01_pa_s1", "r1_01_s1"), ("r1_01_pa_s2", "r1_01_s2"),
                   ("r1_01_pa_wd052", "rl_01_wd052")):
        assert got[pa] / got[sh] == pytest.approx(2.0, abs=0.25, rel=0.0), (
            f"{pa} over {sh} is {got[pa] / got[sh]:.3f}; the reported ratio is "
            "roughly double")


def test_c03_the_six_branches_are_paired_by_layout_and_nothing_else(branches):
    """The pairing the per-agent result rests on: same seed, same iteration count, same decay.

    The result is "per-agent parameters roughly double the profit", which is only
    about the layout if the two sides of each pair differ in the layout alone.
    The known exceptions are (a commit difference shared by
    all three pairs, and one optimiser restart on the `wd052` pair); what this
    asserts is the part that must hold for the pairing to exist at all.
    """
    for pa, sh in (("r1_01_pa_s1", "r1_01_s1"), ("r1_01_pa_s2", "r1_01_s2"),
                   ("r1_01_pa_wd052", "rl_01_wd052")):
        a, b = branches[pa]["meta"], branches[sh]["meta"]
        assert a["per_agent_params"] is True and b["per_agent_params"] is False
        assert a["seed"] == b["seed"], (pa, sh)
        assert a["weight_decay"] == b["weight_decay"], (pa, sh)


# --------------------------------------------------------------------------
# c27 and c56: the other two rows re-pointed at the ingest
# --------------------------------------------------------------------------
def test_c27_the_market_02_recomputation_column(rows):
    """Market 02: range medians 7.480e-1 and 1.443e-7, level medians 576 and 17.

    576 levels is one per observation -- the untrained policy answers to every
    one of the 12 x 48 grid points -- and 17 is what is left after 200
    iterations, while the cross-unit dedup still reports 23 to 53.  That pair is
    the point: the two criteria are furthest apart here.
    """
    for label, want in C27.items():
        row = rows[RT02][label]
        rp = np.asarray(row["range_per_unit"], np.float64)
        lv = np.asarray(row["levels_per_unit"], np.float64)
        assert row["n_obs"] == row["n_days"] * row["n_periods"] == 576, label
        assert float(np.median(rp)) == pytest.approx(want["r_med"], rel=REL_4SF,
                                                     abs=ABS_0), (
            f"market 02 {label} range median {np.median(rp):.6e}, "
            f"reported {want['r_med']:.4e}")
        assert float(np.median(lv)) == want["l_med"], label
        assert (row["cross_unit_dedup_min"],
                row["cross_unit_dedup_max"]) == want["dedup"], label


def test_c56_the_float64_dedup_reading_on_the_wd052_side(rows):
    """This side reads 19 to 64 in float64.

    The reported "2 to 44" is a float32 reading from a different
    product; recomputing the forward pass in float64 turns it into 19 to 64,
    and the two must not be mixed.
    This product is the float64 one, so 19 to 64 is what it has to say.
    """
    row = rows[SWEEP200]["0200"]
    assert (row["cross_unit_dedup_min"],
            row["cross_unit_dedup_max"]) == C56_DEDUP_AT_200
    assert len(rows[SWEEP200]) == 20
    assert row["per_agent_params"] is False


def test_the_fixtures_load_without_allow_pickle():
    """A reference nobody can open without trusting it is not a reference.

    `rows`, `meta`, `payload` and `fixture_stamp` are JSON strings for exactly
    this reason: `np.load(..., allow_pickle=True)` executes what it reads.
    """
    for name in ALL_STAMPED:
        d = np.load(FIXTURES / name, allow_pickle=False)
        assert d.files
        for key in ("rows", "payload", "meta", "fixture_stamp"):
            if key in d.files:
                assert json.loads(str(d[key])) is not None
