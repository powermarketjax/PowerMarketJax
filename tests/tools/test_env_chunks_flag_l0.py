"""L0: `--env-chunks` reaches the learner and the products in the four drivers.

`env_chunks` is a keyword of `make_ippo` (`tests/learning/test_ippo_env_chunks_l0.py`
is its L0), and the drivers are where it can be lost: declared, defaulted, and
then not handed on, or handed to the rollout but not to
`observation_statistics`, which steps the same `n_envs` environments before
the first iteration and is where `case813nem` at `n_envs=64` fails first.
Either half-wiring is silent at the default -- 1 is the one value where wired
and unwired agree -- so, as `test_driver_scenario_scales_l0.py` does for the
scenario scales, this file checks the source by parse rather than by running
it, and the mutation tests at the bottom undo one piece of wiring in a copy
held in a string to show each check goes red.

`run_rl_04.py` is the odd one: its learner comes from
`wrappers.p2p.make_pooled_ippo`, which does not take the keyword, so that
driver declares the flag for a uniform stamp and refuses any value but 1
(the flag's own comment says what would wire it).  What is checked there is
the declaration, the default, the stamp and the refusal.
"""
import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "tools" / "benchmark"

#: The three drivers whose learner is `make_ippo` directly.
WIRED = ("run_rl_01.py", "run_rl_02.py", "run_rl_03.py")
ALL = WIRED + ("run_rl_04.py",)
SOURCE = {name: (BENCH / name).read_text() for name in ALL}
FLAG = "--env-chunks"


def _tree(name, src=None):
    return ast.parse(SOURCE[name] if src is None else src)


def _add_argument(tree, flag):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == flag):
            return node
    return None


def _kwarg(call, key):
    for kw in call.keywords:
        if kw.arg == key:
            return kw.value
    return None


def _is_args_attr(node, attr):
    return (isinstance(node, ast.Attribute) and node.attr == attr
            and isinstance(node.value, ast.Name) and node.value.id == "args")


def _calls(tree, name):
    return [c for c in ast.walk(tree)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == name]


def _assigned_dicts(tree, target):
    """Every `dict(...)` assigned to `target` (e.g. `curve_meta = dict(...)`)."""
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == target
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "dict"):
            out.append(node.value)
    return out


# ── predicates ───────────────────────────────────────────────────────────────

def p_declared_with_default_one(tree):
    call = _add_argument(tree, FLAG)
    if call is None:
        return False
    d = _kwarg(call, "default")
    t = _kwarg(call, "type")
    return (isinstance(d, ast.Constant) and d.value == 1 and type(d.value) is int
            and isinstance(t, ast.Name) and t.id == "int")


def p_make_ippo_takes_the_flag(tree):
    calls = _calls(tree, "make_ippo")
    return bool(calls) and all(
        _is_args_attr(_kwarg(c, "env_chunks"), "env_chunks") for c in calls)


def p_observation_statistics_takes_the_flag(tree):
    calls = _calls(tree, "observation_statistics")
    return bool(calls) and all(
        _is_args_attr(_kwarg(c, "env_chunks"), "env_chunks") for c in calls)


def _stamps(tree, target):
    ds = _assigned_dicts(tree, target)
    if not ds:
        return False
    for d in ds:
        v = _kwarg(d, "env_chunks")
        # `env_chunks=int(args.env_chunks)`: the int() is the stamp's own
        # guard that a string never reaches the product
        if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                and v.func.id == "int" and len(v.args) == 1
                and _is_args_attr(v.args[0], "env_chunks")):
            return False
    return True


def p_curve_meta_stamps_it(tree):
    """`curve_meta` carries it directly, or through a `**scenario_meta` that does."""
    if _stamps(tree, "curve_meta"):
        return True
    cm = _assigned_dicts(tree, "curve_meta")
    spreads = {kw.value.id for d in cm for kw in d.keywords
               if kw.arg is None and isinstance(kw.value, ast.Name)}
    return "scenario_meta" in spreads and _stamps(tree, "scenario_meta")


def p_run_point_stamps_it(tree):
    return _stamps(tree, "run_point")


def p_refuses_any_value_but_one(tree):
    """A `SystemExit` raised under `args.env_chunks != 1` -- the 04 arrangement
    and the SAC branches of the other three.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if (isinstance(t, ast.Compare) and _is_args_attr(t.left, "env_chunks")
                and len(t.ops) == 1 and isinstance(t.ops[0], ast.NotEq)
                and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value == 1):
            if any(isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
                   and isinstance(n.exc.func, ast.Name)
                   and n.exc.func.id == "SystemExit" for n in ast.walk(node)):
                return True
    return False


# ── the checks ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("driver", ALL)
def test_the_flag_is_declared_as_an_int_defaulting_to_one(driver):
    assert p_declared_with_default_one(_tree(driver)), driver


@pytest.mark.parametrize("driver", WIRED)
def test_the_learner_takes_the_flag(driver):
    assert p_make_ippo_takes_the_flag(_tree(driver)), driver


@pytest.mark.parametrize("driver", WIRED)
def test_the_observation_statistics_take_the_flag(driver):
    assert p_observation_statistics_takes_the_flag(_tree(driver)), driver


@pytest.mark.parametrize("driver", ALL)
def test_the_curve_meta_line_stamps_it(driver):
    assert p_curve_meta_stamps_it(_tree(driver)), driver


@pytest.mark.parametrize("driver", WIRED)
def test_the_run_point_stamps_it(driver):
    assert p_run_point_stamps_it(_tree(driver)), driver


@pytest.mark.parametrize("driver", ALL)
def test_the_path_that_cannot_take_it_refuses_it(driver):
    """04 for every value but 1; 01-03 on their SAC branch.  A flag that a
    branch accepts and drops leaves no trace in any product."""
    assert p_refuses_any_value_but_one(_tree(driver)), driver


# ── the mutations: each undoes one piece of wiring in a string copy ──────────

@pytest.mark.parametrize("driver, old, new, predicate", [
    ("run_rl_01.py", "env_chunks=args.env_chunks)\n    else:",
     ")\n    else:", p_make_ippo_takes_the_flag),
    # `cfg.horizon`, not `SHARED.horizon`: this driver's statistics batch moved
    # to `cfg` on 2026-09-16, when `--n-envs` arrived and made the two differ.
    # The anchor names the line above the one it is really testing, so it has to
    # follow that rename; the check itself is unchanged.  Four more spaces since
    # 9b198c2 (2026-09-18): `--init-params` put the call under an `else:` so an
    # archive's own statistics can be used without refitting, and the anchor
    # carries the indentation with it -- a stale anchor makes the mutation a
    # no-op and this test reads "stayed green" as the wiring's fault.
    ("run_rl_02.py", "cfg.horizon,\n                                                   env_chunks=args.env_chunks)",
     "cfg.horizon)", p_observation_statistics_takes_the_flag),
    ("run_rl_03.py", "        env_chunks=int(args.env_chunks),\n", "",
     p_run_point_stamps_it),
    ("run_rl_04.py", "        env_chunks=int(args.env_chunks),\n", "",
     p_curve_meta_stamps_it),
    ("run_rl_04.py", "if args.env_chunks != 1:", "if args.env_chunks != 2:",
     p_refuses_any_value_but_one),
    ("run_rl_01.py", 'ap.add_argument("--env-chunks", type=int, default=1,',
     'ap.add_argument("--env-chunks", type=int, default=2,',
     p_declared_with_default_one),
])
def test_each_check_bites(driver, old, new, predicate):
    src = SOURCE[driver]
    assert src.count(old) >= 1, f"{driver}: mutation anchor not found: {old!r}"
    mutated = src.replace(old, new)
    assert predicate(_tree(driver)), f"{driver}: {predicate.__name__} is red on the real file"
    assert not predicate(_tree(driver, mutated)), (
        f"{driver}: {predicate.__name__} stayed green after {old!r} -> {new!r}")
