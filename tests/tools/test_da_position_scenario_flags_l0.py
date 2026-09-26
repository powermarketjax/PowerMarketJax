"""L0: `da_position.py` takes its three scenario scales from the command line.

Until 2026-09-12 `cap_scale` and `ramp_scale` were module constants in that file
and `p_min_scale` did not appear in it at all, so the day-ahead position could
only ever be built at `case29gb`'s adopted scenario.  `case73rts` adopts
`cap_scale = 0.424`, `ramp_scale = 1.00` and `p_min_scale = 0.80`
(2026-09-04 and 2026-09-05), and
markets 02 and 03 consume the position fixture, so without the flags those two
markets cannot be run on either new case.

**The measured "it bites" data, as required before a check counts.**  All three
measured 2026-09-12 on `case29gb`, CPU, `jax` 0.10.2, two days from a
pre-commitment fixture built at the matching scenario, against the same two days
at `cap 0.60 / ramp 1.00 / p_min 1.0`:

    --cap-scale   0.60 -> 0.50   q_da moves 4 431 MW (56.6% of its own max),
                                 lmp_da 46.76 $/MWh (41.8%), 50 of 3 168
                                 commitment cells flip
    --ramp-scale  1.00 -> 0.50   q_da 1 550.9 MW (19.8%), lmp_da 7.75 $/MWh
                                 (6.9%), 2 of 3 168 commitment cells flip
    --p-min-scale 1.0  -> 0.80   q_da 637.6 MW (8.1%), lmp_da 5.51 $/MWh
                                 (4.9%), 1 of 3 168 commitment cells flips

So none of the three is a relabelling: each changes the dispatch and the price
by orders of magnitude more than any solver tolerance in this repository.

**And the default is unchanged, verified by effect rather than by inspection**:
the same command with no new flag, run before and
after the change, writes ten arrays that are bitwise identical -- `u`, `q_da`,
`s_da`, `lmp_da`, `d_da`, `mu`, `shed_mwh`, `line_dual_up`, `line_dual_dn`,
`shed_dual` -- with `meta` differing by exactly one added key, `p_min_scale: 1.0`.
A control of two *pre-change* runs against each other was taken first, so
"bitwise identical" is known to be a statement this comparison can make.

**Why the wiring is checked by parse and not only by running.**  A flag that is
declared and then not passed to the operator is silent: the run prints the value
it was asked for, every gate passes, and the product records a scenario it did
not run.  `compile()` is blind to that and so is a smoke run at the default,
which is the one value where wired and unwired agree.
"""
import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "tools" / "commitment" / "da_position.py"
SOURCE = SRC.read_text()
TREE = ast.parse(SOURCE)

#: The three flags and the module constant each defaults to.
FLAGS = {"--cap-scale": "CAP_SCALE", "--ramp-scale": "RAMP_SCALE",
         "--p-min-scale": "P_MIN_SCALE"}


def _func(name, tree=None):
    for node in ast.walk(TREE if tree is None else tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{SRC.name} has no function {name}")


def _calls(node, name):
    """Every `name(...)` call inside `node`, by parse."""
    return [c for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == name]


def _kwarg(call, key):
    for kw in call.keywords:
        if kw.arg == key:
            return kw.value
    return None


def _is_args_attr(node, attr):
    return (isinstance(node, ast.Attribute) and node.attr == attr
            and isinstance(node.value, ast.Name) and node.value.id == "args")


def _add_argument_calls(tree=None):
    out = {}
    for node in ast.walk(TREE if tree is None else tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            out[node.args[0].value] = node
    return out


@pytest.mark.parametrize("flag", sorted(FLAGS))
def test_the_flag_exists(flag):
    assert flag in _add_argument_calls(), (
        f"{SRC.name} does not take {flag}, so the position can only be built at "
        f"case29gb's scenario")


@pytest.mark.parametrize("flag", sorted(FLAGS))
def test_the_flag_defaults_to_the_module_constant(flag):
    """A default typed out beside the constant is a second answer to one question."""
    default = _kwarg(_add_argument_calls()[flag], "default")
    assert isinstance(default, ast.Name) and default.id == FLAGS[flag], (
        f"{flag}'s default is not the module constant {FLAGS[flag]}; two places "
        f"would then state the scenario and they can disagree")


def _harvest_reads_no_constant(tree=None):
    h = _func("harvest", tree)
    used = {n.id for b in h.body for n in ast.walk(b) if isinstance(n, ast.Name)}
    return not used & {"CAP_SCALE", "RAMP_SCALE"}


def _harvest_hands_monitored(tree=None):
    """`{callee: bool}` -- does `harvest` hand **both** operators the line set?

    `da_position.py` builds two operators: `make_env` for the chain and a
    standalone `make_clearing` for the per-period re-solve, and the
    "re-solve reproduces the env" gate compares exactly those two.  A flag that
    reaches one and not the other leaves that gate comparing a seven-line
    re-solve against a 1 278-line environment **and reporting the difference as
    a numerical residual** -- the gate stays green-looking while measuring two
    different objects.  So the wiring is asserted per callee, not once.
    """
    h = _func("harvest", tree)
    out = {}
    for callee in ("make_env", "make_clearing"):
        call, = _calls(h, callee)
        kw = _kwarg(call, "monitored_lines")
        out[callee] = isinstance(kw, ast.Name) and kw.id == "monitored_lines"
    return out


def _main_passes(tree=None):
    """`{consumer: bool}` -- does main() hand each consumer the flag's value?"""
    m = _func("main", tree)
    hv, = _calls(m, "harvest")
    lfo, = _calls(m, "line_flow_overshoot")
    sms, = _calls(m, "scale_min_output")
    meta, = [c for c in _calls(m, "dict")
             if any(kw.arg == "stage" for kw in c.keywords)]
    return {
        "harvest.cap_scale": _is_args_attr(_kwarg(hv, "cap_scale"), "cap_scale"),
        "harvest.ramp_scale": _is_args_attr(_kwarg(hv, "ramp_scale"), "ramp_scale"),
        "overshoot.cap_scale": _is_args_attr(_kwarg(lfo, "cap_scale"), "cap_scale"),
        "case.p_min_scale": _is_args_attr(sms.args[1], "p_min_scale"),
        "meta.cap_scale": _is_args_attr(_kwarg(meta, "cap_scale"), "cap_scale"),
        "meta.ramp_scale": _is_args_attr(_kwarg(meta, "ramp_scale"), "ramp_scale"),
        "meta.p_min_scale": _is_args_attr(_kwarg(meta, "p_min_scale"), "p_min_scale"),
    }


def test_harvest_takes_the_two_operator_scales_and_reads_no_constant():
    """The failure this catches: flag declared, operator still on the constant."""
    h = _func("harvest")
    names = [a.arg for a in h.args.args + h.args.kwonlyargs]
    assert "cap_scale" in names and "ramp_scale" in names
    # the body only: the signature's *defaults* are the module constants on
    # purpose, and that is what makes the no-flag call unchanged
    assert _harvest_reads_no_constant(), (
        "harvest still reads the module constants, so its parameters are "
        "decorative and a --cap-scale on the command line would not reach the "
        "clearing operator")


def test_main_passes_the_flags_to_every_consumer():
    """Each of the four places the scales are consumed takes `args.*`.

    Listed rather than globbed: these are the call sites, and a new one is meant
    to be added here deliberately rather than swept in.
    """
    passed = _main_passes()
    assert all(passed.values()), f"unwired consumers: {[k for k, v in passed.items() if not v]}"
    # p_min_scale is applied to the case, not to the operators: that is
    # `scale_min_output`'s own contract, and `precommit.py` makes the same call
    sms, = _calls(_func("main"), "scale_min_output")
    assert _calls(sms, "load_case"), "scale_min_output is not wrapping load_case" 


@pytest.mark.parametrize("key", ["cap_scale", "ramp_scale", "p_min_scale"])
def test_the_product_records_what_the_run_asked_for(key):
    """`meta` must carry `args.*`, not the constant: that is the other half."""
    assert _main_passes()[f"meta.{key}"], (
        f"meta records {key} from somewhere other than args.{key}; a product "
        f"built at one scenario would be stamped with another")


#: One reversion per wiring the checks above are supposed to hold, written as the
#: edit that undoes it.  `(what it breaks, old text, text to put back)`.
REVERSIONS = [
    #: 2026-09-16: `--monitored-lines` added one more argument to `harvest(...)`,
    #: so the closing parenthesis of these two anchors' original
    #: `...args.ramp_scale)` no longer followed directly; they matched 0 times and
    #: the assertion went red.  **The anchors follow the file; the tests themselves
    #: are not deleted** -- they must keep biting on the new file.
    ("harvest.cap_scale",
     "cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,\n"
     "                        monitored_lines=monitored)",
     "cap_scale=CAP_SCALE, ramp_scale=args.ramp_scale,\n"
     "                        monitored_lines=monitored)"),
    ("harvest.ramp_scale",
     "cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,\n"
     "                        monitored_lines=monitored)",
     "cap_scale=args.cap_scale, ramp_scale=RAMP_SCALE,\n"
     "                        monitored_lines=monitored)"),
    ("overshoot.cap_scale", 'pos["d_da"][d], cap_scale=args.cap_scale)',
     'pos["d_da"][d], cap_scale=CAP_SCALE)'),
    ("case.p_min_scale", "scale_min_output(load_case(args.case), args.p_min_scale)",
     "scale_min_output(load_case(args.case), P_MIN_SCALE)"),
    ("meta.cap_scale", "cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,\n        p_min_scale=args.p_min_scale",
     "cap_scale=CAP_SCALE, ramp_scale=args.ramp_scale,\n        p_min_scale=args.p_min_scale"),
    ("meta.ramp_scale", "cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,\n        p_min_scale=args.p_min_scale",
     "cap_scale=args.cap_scale, ramp_scale=RAMP_SCALE,\n        p_min_scale=args.p_min_scale"),
    ("meta.p_min_scale", "cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,\n        p_min_scale=args.p_min_scale",
     "cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,\n        p_min_scale=P_MIN_SCALE"),
]


@pytest.mark.parametrize("consumer,old,new", REVERSIONS, ids=[r[0] for r in REVERSIONS])
def test_the_wiring_check_bites(consumer, old, new):
    """Undo one wiring in a copy of the source; the check must go red.

    This is the "it bites" datum for the parse-side checks, executable rather
    than measured once: a check that cannot fail is the shape of defect this
    repository has recorded repeatedly.  The source is never written -- the
    mutation lives in a string, instead of editing a shared file to take a
    measurement.
    """
    assert SOURCE.count(old) == 1, (
        f"the anchor for {consumer} matched {SOURCE.count(old)} times, not once; "
        f"this reversion no longer describes the file")
    mutated = ast.parse(SOURCE.replace(old, new, 1))
    assert not _main_passes(mutated)[consumer], (
        f"reverting {consumer} left the check green, so it is not checking it")


def test_harvest_hands_monitored_lines_to_both_operators():
    """The failure this catches: the flag reaches the chain but not the re-solve."""
    assert _harvest_hands_monitored() == {"make_env": True, "make_clearing": True}


def test_the_monitored_wiring_check_bites():
    """Drop the argument from the re-solve operator; the check must go red."""
    anchor = ("period_hours=1.0, max_iter=MAX_ITER,\n"
              "                                 monitored_lines=monitored_lines)")
    assert SOURCE.count(anchor) == 1, SOURCE.count(anchor)
    mutated = ast.parse(SOURCE.replace(anchor, "period_hours=1.0, max_iter=MAX_ITER)", 1))
    assert not _harvest_hands_monitored(mutated)["make_clearing"], (
        "dropping monitored_lines from the re-solve operator left the check green")


def test_the_harvest_constant_check_bites():
    """The same, for the one check that reads a function body."""
    anchor = ("cap_scale=cap_scale, ramp_scale=ramp_scale,\n"
              "                         monitored_lines=monitored_lines)")
    assert SOURCE.count(anchor) == 1, SOURCE.count(anchor)
    mutated = ast.parse(SOURCE.replace(
        anchor,
        "cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,\n"
        "                         monitored_lines=monitored_lines)", 1))
    assert not _harvest_reads_no_constant(mutated), (
        "putting the module constants back inside harvest left the check green")


# ── the runtime half: the one scale `make_env` is blind to ──────────────────

@pytest.fixture(scope="module")
def da_position():
    """Import the driver, and put the global JAX config back afterwards.

    **Importing it is not free.**  `da_position.py` runs
    ``jax.config.update("jax_enable_x64", True)`` and
    ``jax.config.update("jax_default_matmul_precision", "highest")`` at module
    level, which is right for a driver that owns its process and wrong for a
    test session: the flags are global, they outlive this file, and every test
    that runs afterwards then builds float64 arrays where it meant float32.
    Measured 2026-09-12: without this restore, `pytest -q tests/tools` fails
    `test_p2p_external_sac_l0.py::test_the_parameter_tree_is_the_packages_own_tree`
    on a leaf that should be ``log(0.2) = -1.609438``, while the same file alone
    passes 25/25 -- so the failure reads as a defect in a file nobody touched.
    """
    import sys

    import jax
    before = (jax.config.jax_enable_x64,
              jax.config.jax_default_matmul_precision)
    sys.path.insert(0, str(REPO / "tools" / "commitment"))
    import da_position as mod
    yield mod
    jax.config.update("jax_enable_x64", before[0])
    jax.config.update("jax_default_matmul_precision", before[1])


def test_make_env_already_refuses_a_cap_or_ramp_mismatch():
    """Why `check_boundary_scenario` covers only `p_min_scale`.

    Measured 2026-09-12: asking `harvest` for `cap_scale=0.50` against the
    shipped `cap_scale=0.60` fixture raises in `envs/day_ahead/env.py` --
    ``fixture was built at cap_scale=0.6, ramp_scale=1.0, K=1, but this
    environment asks for 0.5, 1.0, 1``.  A second copy of that check in the
    driver would be a check that never fires.
    """
    src = (REPO / "powermarketjax" / "envs" / "day_ahead" / "env.py").read_text()
    assert "fixture was built at cap_scale=" in src, (
        "the raise this test defers to has moved; da_position's own check "
        "covers only p_min_scale on the strength of it")


def test_a_p_min_scale_mismatch_is_refused(da_position):
    """The gate, and that it bites: `make_env` never sees `p_min_scale`."""
    d = da_position
    meta = {"cap_scale": 0.6, "ramp_scale": 1.0, "p_min_scale": 1.0}
    d.check_boundary_scenario(meta, 1.0)                       # agrees, no raise
    with pytest.raises(SystemExit, match="p_min_scale"):
        d.check_boundary_scenario(meta, 0.80)
    # and the asymmetry that makes the check worth having: the env-level check
    # names three fields and this is not one of them
    env_src = (REPO / "powermarketjax" / "envs" / "day_ahead" / "env.py").read_text()
    assert "p_min_scale" not in env_src


def test_a_fixture_without_the_field_reads_as_the_registered_case(da_position):
    """Absent means 1.0, the reading `precommit.IMPLIED_PRIOR` already records."""
    d = da_position
    assert d.P_MIN_SCALE_IMPLIED_PRIOR == 1.0
    d.check_boundary_scenario({"cap_scale": 0.6, "ramp_scale": 1.0}, 1.0)
    with pytest.raises(SystemExit, match="p_min_scale=1.0"):
        d.check_boundary_scenario({"cap_scale": 0.6, "ramp_scale": 1.0}, 0.80)

    import sys
    sys.path.insert(0, str(REPO / "tools" / "commitment"))
    import precommit
    assert precommit.IMPLIED_PRIOR["p_min_scale"] == d.P_MIN_SCALE_IMPLIED_PRIOR, (
        "two devices read an absent p_min_scale differently, which is worse "
        "than both reading it wrong")


def test_the_cap_scale_reaches_the_line_ratings(da_position):
    """`line_flow_overshoot` against two ratings, on inputs with no solver in them."""
    import numpy as np
    d = da_position
    from powermarketjax.case import load_case
    case = load_case("29gb")
    T = 3
    q = np.zeros((int(case.n_units) if hasattr(case, "n_units")
                  else len(case.unit_p_min), T), np.float64)
    s = np.zeros((T, int(case.n_nodes)), np.float64)
    dd = np.zeros((T, int(case.n_nodes)), np.float64)
    a = d.line_flow_overshoot(case, q, s, dd, cap_scale=0.60)
    b = d.line_flow_overshoot(case, q, s, dd, cap_scale=0.50)
    cap = np.asarray(case.line_cap, np.float64)
    # zero flow, so the overshoot is exactly -F and the difference is exactly
    # the change in the ratings: 0.10 * line_cap, per line
    assert np.allclose(a - b, -0.10 * cap[None, :], rtol=0, atol=1e-9)
    assert float(np.abs(a - b).max()) > 1.0, "the two ratings are indistinguishable"
