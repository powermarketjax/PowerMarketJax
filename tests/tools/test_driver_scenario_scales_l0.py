"""L0: the four benchmark drivers take their scenario scales from the run.

`run_eval_02.py`, `run_rl_02.py`, `run_eval_03.py` and `run_rl_03.py` are the
drivers markets 02 and 03 are run through.  Until 2026-09-12 each of them held
`cap_scale` and `ramp_scale` as module constants at `case29gb`'s adopted values,
and none of them applied the third scale, `p_min_scale`, at all.  That is two
separate defects with opposite failure modes, and they are fixed together
because the second one is unreachable while the first one stands.

**The first is loud.**  `case73rts` adopts `cap_scale = 0.424`
and each driver refuses a
position fixture whose scenario disagrees with its own.  Measured 2026-09-12,
before the flags existed::

    run_eval_02.py --position <73rts position>
        -> SystemExit: fixture meta cap_scale=0.424 disagrees with 0.6
    run_eval_03.py --case 73rts --position <73rts position>
        -> SystemExit: fixture meta cap_scale=0.424 disagrees with 0.6

So markets 02 and 03 could only ever be pointed at `case29gb`.

**The second is silent, which is why it is the worse of the two.**  The position
fixture records the `p_min_scale` its commitment was built at; the drivers
loaded the registered case and ignored it.  Nothing raises: the interior-point
method diverges rather than failing, every array comes back well-formed, the
product writes, and its `meta` stamps the fixture's scale beside a dispatch that
ran at another.  Measured 2026-09-12 on `case73rts`, 2020-01-01, from
a day-ahead position fixture (25 of the 73 units committed, arithmetic only, no
solver in the evidence):

===================  =====================  =======================  ==========
`p_min_scale`        committed floor (MW)   floor > demand, periods  worst (MW)
===================  =====================  =======================  ==========
1.0 (what ran)       2 680.0                **38 / 48**              +180.0
0.80 (the fixture)   2 144.0                **0 / 48**               -356.0
===================  =====================  =======================  ==========

against that day's demand of 2 500.0 .. 3 339.5 MW.  The clearing's `worst mu`
went to 1.733e+293 on the unscaled case and 1.167e-11 on the scaled one.

**Why this file checks by parse and not only by running.**  A flag that is
declared and then not handed to the operator is silent in exactly the same way:
the run prints the value it was asked for, every gate passes, and the product
records a scenario it did not clear.  `compile()` is blind to it, and so is a
smoke run at the default, which is the one value where wired and unwired agree.
The runtime half of the evidence is below; the end-to-end half was a
`case73rts` clearing run separately.

The mutation tests at the bottom are these checks' own "it bites" data: each
undoes one piece of wiring in a copy of the source held in a string -- the files
are never written -- and asserts the matching check goes red.
"""
import ast
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "tools" / "benchmark"

#: The four drivers, and the name each one's env-construction call goes by.
DRIVERS = {
    "run_eval_02.py": "make_env",
    "run_rl_02.py": "make_env",
    "run_eval_03.py": "make_ancillary_env",
    "run_rl_03.py": "make_ancillary_env",
}
SOURCE = {name: (BENCH / name).read_text() for name in DRIVERS}

FLAGS = {"--cap-scale": "CAP_SCALE", "--ramp-scale": "RAMP_SCALE"}


# ── predicates, each taking a parsed tree so a mutated copy can be asked ─────

def _tree(name, src=None):
    return ast.parse(SOURCE[name] if src is None else src)


def _add_argument_calls(tree):
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            out[node.args[0].value] = node
    return out


def _kwarg(call, key):
    for kw in call.keywords:
        if kw.arg == key:
            return kw.value
    return None


def _is_args_attr(node, attr):
    return (isinstance(node, ast.Attribute) and node.attr == attr
            and isinstance(node.value, ast.Name) and node.value.id == "args")


def _is_run_ramp_scale(tree, node):
    """`args.ramp_scale`, or the one named exception that stands for it.

    The exception is a name assigned exactly once in the module, and that
    assignment is ``args.ramp_scale if args.clearing_ramp_scale is None else
    ...``: a counterfactual flag that is off by default and, when off, leaves
    the run's own ramp scale in force.  Any other name, or any other value for
    that name, is refused -- this is not "any name will do".
    """
    if _is_args_attr(node, "ramp_scale"):
        return True
    if not isinstance(node, ast.Name):
        return False
    assigns = [a for a in ast.walk(tree) if isinstance(a, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == node.id for t in a.targets)]
    if len(assigns) != 1:
        return False
    v = assigns[0].value
    return (isinstance(v, ast.IfExp) and _is_args_attr(v.body, "ramp_scale")
            and isinstance(v.test, ast.Compare) and len(v.test.ops) == 1
            and isinstance(v.test.ops[0], ast.Is)
            and _is_args_attr(v.test.left, "clearing_ramp_scale")
            and isinstance(v.test.comparators[0], ast.Constant)
            and v.test.comparators[0].value is None)


def _calls(tree, name):
    return [c for c in ast.walk(tree)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == name]


def _flag_default(tree, flag):
    """The `default=` of `flag`, or None when the flag is not declared."""
    call = _add_argument_calls(tree).get(flag)
    return None if call is None else _kwarg(call, "default")


def _scenario_dicts(tree):
    """Every `dict(...)` in the module that stamps `cap_scale`.

    Globbed rather than listed on purpose, and the opposite choice from
    `test_da_position_scenario_flags_l0.py`: there the call sites are four and
    naming them makes a new one a deliberate addition, here they are eleven
    across four files with three different names (`run_point`, `scenario_meta`,
    `curve_meta`, and the checkpoint's own `scenario` blob), and the failure
    being guarded against is precisely one of them being forgotten.
    """
    return [c for c in ast.walk(tree) if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name) and c.func.id == "dict"
            and any(kw.arg == "cap_scale" for kw in c.keywords)]


def p_flag_declared(tree, flag):
    return flag in _add_argument_calls(tree)


def p_flag_defaults_to_constant(tree, flag):
    d = _flag_default(tree, flag)
    return isinstance(d, ast.Name) and d.id == FLAGS[flag]


def p_no_bare_constant(tree):
    """`CAP_SCALE` / `RAMP_SCALE` are read nowhere but as a flag's default.

    This is the check that catches the half-fix -- flag declared, operator still
    on the constant -- without having to enumerate the operators.  Their own
    assignment (a Store) and `run_rl_03`'s from-import (an alias, not a Name)
    are not reads and do not count.
    """
    defaults = {id(_flag_default(tree, f)) for f in FLAGS}
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Name) and n.id in set(FLAGS.values())
             and isinstance(n.ctx, ast.Load) and id(n) not in defaults]
    return not reads


def p_env_call_takes_the_flags(tree, env_call):
    call, = _calls(tree, env_call)
    return (_is_args_attr(_kwarg(call, "cap_scale"), "cap_scale")
            and _is_run_ramp_scale(tree, _kwarg(call, "ramp_scale")))


def p_fixture_check_reads_the_flags(tree):
    """The scenario check compares the fixture against this run, not the constant."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple):
            continue
        pairs = {e.elts[0].value: e.elts[1] for e in node.elts
                 if isinstance(e, ast.Tuple) and len(e.elts) == 2
                 and isinstance(e.elts[0], ast.Constant)}
        if "cap_scale" in pairs and "ramp_scale" in pairs:
            return (_is_args_attr(pairs["cap_scale"], "cap_scale")
                    and _is_args_attr(pairs["ramp_scale"], "ramp_scale"))
    return False


def p_every_stamp_takes_the_flags(tree):
    ds = _scenario_dicts(tree)
    return bool(ds) and all(
        _is_args_attr(_kwarg(d, "cap_scale"), "cap_scale")
        and _is_run_ramp_scale(tree, _kwarg(d, "ramp_scale")) for d in ds)


def p_every_stamp_records_p_min_scale(tree):
    ds = _scenario_dicts(tree)
    return bool(ds) and all(
        isinstance(_kwarg(d, "p_min_scale"), ast.Name)
        and _kwarg(d, "p_min_scale").id == "p_min_scale" for d in ds)


def p_no_stamp_names_the_case_constant(tree):
    """A stamp must not record `CASE` while the run took `args.case`.

    The 03 drivers took a `--case` flag on 2026-09-12 and one stamp was left
    behind on the constant -- `run_rl_03.py`'s checkpoint `scenario` blob.  Same
    shape as the scales: what was decided and what was in force, in one product,
    disagreeing.
    """
    for d in _scenario_dicts(tree):
        v = _kwarg(d, "case")
        if isinstance(v, ast.Name) and v.id == "CASE":
            return False
    return True


def p_case_is_scaled(tree):
    """Every `load_case(...)` is an argument of a `scale_min_output(...)`.

    The whole of the second defect, in one predicate: a bare `load_case` is the
    registered minimum-output level whatever the fixture says.
    """
    loads = _calls(tree, "load_case")
    if not loads:
        return False
    wrapped = {id(c) for sms in _calls(tree, "scale_min_output")
               for c in _calls(sms, "load_case")}
    return all(id(c) in wrapped for c in loads)


def p_the_scale_comes_from_the_fixture(tree):
    """...and the scale it is given is read from the fixture's own `meta`.

    Not a flag, and not a constant: the commitment in the fixture was built at
    that level.  `P_MIN_SCALE_IMPLIED_PRIOR` is the reading of an absent field,
    which `precommit.IMPLIED_PRIOR` already fixes at 1.0.
    """
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "p_min_scale"):
            calls = [c for c in ast.walk(node.value) if isinstance(c, ast.Call)]
            gets = [c for c in calls if isinstance(c.func, ast.Attribute)
                    and c.func.attr == "get"
                    and isinstance(c.func.value, ast.Name)
                    and c.func.value.id == "meta"]
            if not gets:
                return False
            g, = gets
            return (len(g.args) == 2 and isinstance(g.args[0], ast.Constant)
                    and g.args[0].value == "p_min_scale"
                    and isinstance(g.args[1], ast.Name)
                    and g.args[1].id == "P_MIN_SCALE_IMPLIED_PRIOR")
    return False


# ── the checks ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("driver", sorted(DRIVERS))
@pytest.mark.parametrize("flag", sorted(FLAGS))
def test_the_flag_exists(driver, flag):
    assert p_flag_declared(_tree(driver), flag), (
        f"{driver} does not take {flag}, so it can only be run at case29gb's "
        f"scenario -- a case73rts position fixture is refused before it clears")


@pytest.mark.parametrize("driver", sorted(DRIVERS))
@pytest.mark.parametrize("flag", sorted(FLAGS))
def test_the_flag_defaults_to_the_module_constant(driver, flag):
    """A default typed out beside the constant is a second answer to one question.

    It is also what keeps every existing GB command line meaning what it meant:
    changing a module-level default changes every line that runs without the
    flag, and these four are the most-run drivers in the tree.
    """
    assert p_flag_defaults_to_constant(_tree(driver), flag), (
        f"{driver}'s {flag} does not default to {FLAGS[flag]}")


@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_the_constants_are_read_nowhere_but_as_the_defaults(driver):
    assert p_no_bare_constant(_tree(driver)), (
        f"{driver} still reads CAP_SCALE/RAMP_SCALE somewhere other than a "
        f"flag default, so --cap-scale would not reach that consumer")


@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_the_environment_is_built_at_the_run_s_scales(driver):
    assert p_env_call_takes_the_flags(_tree(driver), DRIVERS[driver])


@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_the_fixture_check_compares_against_the_run(driver):
    assert p_fixture_check_reads_the_flags(_tree(driver)), (
        f"{driver} checks the fixture's scenario against its module constants, "
        f"so --cap-scale 0.424 would be rejected by the driver that was asked "
        f"for it")


@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_every_product_stamp_records_what_ran(driver):
    """Every `dict(... cap_scale=...)` in the file, not a named list of them."""
    assert p_every_stamp_takes_the_flags(_tree(driver))
    assert p_every_stamp_records_p_min_scale(_tree(driver)), (
        f"{driver} writes a scenario stamp without p_min_scale; the product "
        f"could not say which minimum-output level it cleared at")
    assert p_no_stamp_names_the_case_constant(_tree(driver))


@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_the_case_is_scaled_by_the_fixture_s_p_min_scale(driver):
    assert p_case_is_scaled(_tree(driver)), (
        f"{driver} calls load_case without scale_min_output, so a commitment "
        f"built at p_min_scale=0.80 would be cleared against the registered "
        f"minimum output -- silently")
    assert p_the_scale_comes_from_the_fixture(_tree(driver))


# ── the mutation half: every check above must be able to go red ─────────────

#: `(driver, predicate key, text to replace, replacement)`.  Each undoes exactly
#: one piece of wiring.  The source file is never written: the mutation lives in
#: a string, instead of editing a shared file to take a measurement.
REVERSIONS = [
    ("run_eval_03.py", "env-call",
     "    ramp_eff = (args.ramp_scale if args.clearing_ramp_scale is None\n"
     "                else float(args.clearing_ramp_scale))",
     "    ramp_eff = 0.25"),
    ("run_eval_03.py", "stamp-flags",
     "    ramp_eff = (args.ramp_scale if args.clearing_ramp_scale is None\n"
     "                else float(args.clearing_ramp_scale))",
     "    ramp_eff = 0.25"),
    ("run_eval_02.py", "flag:--cap-scale",
     '    ap.add_argument("--cap-scale", type=float, default=CAP_SCALE)\n', ""),
    ("run_eval_03.py", "default:--ramp-scale",
     'ap.add_argument("--ramp-scale", type=float, default=RAMP_SCALE)',
     'ap.add_argument("--ramp-scale", type=float, default=1.00)'),
    ("run_rl_02.py", "bare-constant",
     "                               cap_scale=args.cap_scale,",
     "                               cap_scale=CAP_SCALE,"),
    ("run_eval_02.py", "env-call",
     "                         cap_scale=args.cap_scale,",
     "                         cap_scale=0.60,"),
    ("run_rl_03.py", "env-call",
     "                             ramp_scale=args.ramp_scale,",
     "                             ramp_scale=1.00,"),
    ("run_eval_03.py", "fixture-check",
     '    for key, given in (("cap_scale", args.cap_scale),',
     '    for key, given in (("cap_scale", 0.60),'),
    ("run_rl_02.py", "stamp-flags",
     "    run_point = dict(cap_scale=args.cap_scale,",
     "    run_point = dict(cap_scale=0.60,"),
    ("run_rl_03.py", "stamp-p-min",
     "        p_min_scale=p_min_scale, voll=VOLL_IN_EFFECT,",
     "        voll=VOLL_IN_EFFECT,"),
    ("run_rl_03.py", "stamp-case",
     "            case=args.case, theta=list(THETA),",
     "            case=CASE, theta=list(THETA),"),
    ("run_eval_02.py", "scaled-case",
     '    case = scale_min_output(load_case(meta["case"]), p_min_scale)',
     '    case = load_case(meta["case"])'),
    ("run_eval_03.py", "scale-from-fixture",
     '    p_min_scale = float(meta.get("p_min_scale", P_MIN_SCALE_IMPLIED_PRIOR))',
     '    p_min_scale = 1.0'),
]

#: Which predicate each reversion is supposed to turn red.
PREDICATE = {
    "flag:--cap-scale": lambda t, d: p_flag_declared(t, "--cap-scale"),
    "default:--ramp-scale": lambda t, d: p_flag_defaults_to_constant(t, "--ramp-scale"),
    "bare-constant": lambda t, d: p_no_bare_constant(t),
    "env-call": lambda t, d: p_env_call_takes_the_flags(t, DRIVERS[d]),
    "fixture-check": lambda t, d: p_fixture_check_reads_the_flags(t),
    "stamp-flags": lambda t, d: p_every_stamp_takes_the_flags(t),
    "stamp-p-min": lambda t, d: p_every_stamp_records_p_min_scale(t),
    "stamp-case": lambda t, d: p_no_stamp_names_the_case_constant(t),
    "scaled-case": lambda t, d: p_case_is_scaled(t),
    "scale-from-fixture": lambda t, d: p_the_scale_comes_from_the_fixture(t),
}


@pytest.mark.parametrize("driver,key,old,new", REVERSIONS,
                         ids=[f"{d.split('.')[0]}-{k}" for d, k, _o, _n in REVERSIONS])
def test_the_check_bites(driver, key, old, new):
    src = SOURCE[driver]
    assert src.count(old) == 1, (
        f"the anchor for {key} in {driver} matched {src.count(old)} times, not "
        f"once; this reversion no longer describes the file")
    mutated = ast.parse(src.replace(old, new, 1))
    assert PREDICATE[key](_tree(driver), driver), "the check is red before the mutation"
    assert not PREDICATE[key](mutated, driver), (
        f"reverting {key} in {driver} left the check green, so it is not "
        f"checking it")


# ── the runtime half ────────────────────────────────────────────────────────

def test_the_unit_scale_returns_the_same_object():
    """Identity, not equality -- it is what makes the GB path bitwise unchanged.

    `tests/envs/day_ahead/test_p_min_scale_l0.py::test_unit_scale_is_the_identity`
    already asserts the arrays are equal at 1.0.  The drivers need the stronger
    statement: `case29gb` records `p_min_scale = 1.0` (or nothing, which reads
    as 1.0), so the line these four just gained must hand the clearing the very
    object it had before, not a copy that happens to compare equal.
    """
    from powermarketjax.case import load_case, scale_min_output
    gb = load_case("29gb")
    assert scale_min_output(gb, 1.0) is gb
    assert scale_min_output(gb, 0.80) is not gb


def test_an_absent_field_reads_as_the_registered_case():
    """And all five devices that read it agree on that reading.

    The GB position fixtures in `tests/fixtures/` predate the field, so the
    drivers meet the absent case on every existing command line.
    """
    import sys
    for mod_dir in (BENCH, REPO / "tools" / "commitment"):
        sys.path.insert(0, str(mod_dir))
    import precommit
    prior = precommit.IMPLIED_PRIOR["p_min_scale"]
    assert prior == 1.0
    for name in ("run_eval_02.py", "run_eval_03.py", "run_rl_02.py"):
        tree = _tree(name)
        value, = [n.value.value for n in ast.walk(tree)
                  if isinstance(n, ast.Assign) and len(n.targets) == 1
                  and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == "P_MIN_SCALE_IMPLIED_PRIOR"]
        assert value == prior, (
            f"{name} reads an absent p_min_scale as {value} and precommit reads "
            f"it as {prior}; two devices disagreeing about an absent field is "
            f"worse than both reading it wrong")
    # run_rl_03 imports the constant rather than restating it
    imported = [a.name for n in ast.walk(_tree("run_rl_03.py"))
                if isinstance(n, ast.ImportFrom) and n.module == "run_eval_03"
                for a in n.names]
    assert "P_MIN_SCALE_IMPLIED_PRIOR" in imported


def test_the_scale_decides_whether_the_committed_minimum_can_follow_the_trough():
    """The measured consequence, on the registered case and the real series.

    `case73rts`, 2020-01-01, at the adopted demand run point.  The commitment
    here is **every unit**, stated rather than solved for: this test owns no
    solver and a commitment read off a fixture would be reporting that fixture's
    property.  It is therefore an upper bound on any commitment, and the
    narrower measurement -- the adopted commitment of 25 of the 73 units, 38 of 48
    periods against 0 of 48 -- is in the module docstring with its provenance.

    Measured here 2026-09-12: 3 745.0 MW against 2 996.0 MW, infeasible in 48
    of 48 periods against 42 of 48.  A six-period gap rather than the 38 the
    adopted commitment shows, because committing every unit is a far worse
    starting point than the one `precommit` produces -- the direction is the
    same, and it is the direction this file can state without a solver.

    What it shows is the thing the drivers were blind to: at the registered
    minimum output this case cannot be dispatched down to its own trough, and at
    the adopted 0.80 it can be dispatched further down than at 1.0 in every
    period.  A scale that changed nothing here would make the missing line
    harmless, and it is not.
    """
    from powermarketjax.case import load_case, scale_min_output
    from powermarketjax.envs.real_time.demand import load_rts_demand_half_hourly

    case = load_case("73rts")
    hh, _dates = load_rts_demand_half_hourly(
        floor_mw=2500.0, netted=["hydro", "pv", "rtpv", "wind"])
    demand = np.asarray(hh[0], np.float64)
    assert demand.shape == (48,)
    assert abs(float(demand.min()) - 2500.0) < 1e-9
    assert abs(float(demand.max()) - 3339.541015625) < 1e-6

    floors = {}
    for scale in (1.0, 0.80):
        p_min = np.asarray(scale_min_output(case, scale).unit_p_min, np.float64)
        floors[scale] = float(p_min.sum())
    assert abs(floors[1.0] - 3745.0) < 1e-9, floors
    # the case carries p_min in float32, so the product of the scale is exact
    # only to that: measured 4.58e-05 MW out of 2 996 MW, 1.5e-08 relative
    assert abs(floors[0.80] - 0.80 * floors[1.0]) < 1e-3, floors

    over = {s: int((f > demand).sum()) for s, f in floors.items()}
    assert over == {1.0: 48, 0.80: 42}, (
        f"the all-committed floor clears {over} periods of 48; if this moved, "
        f"either the case data or the demand run point did, and the driver "
        f"numbers in this file's docstring were taken against the old one")
