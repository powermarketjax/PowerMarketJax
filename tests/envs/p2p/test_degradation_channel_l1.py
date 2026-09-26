"""L0 + L1 for the battery-degradation channel this market publishes.

Why this exists.  Market 04's success criterion in the five-market baseline
matrix is a displacement in **system
cost**, and battery degradation is the only real resource cost among the
modelled participants: there is no generation and no load shedding here, and
the auction leg is a transfer that nets to zero across them.  (Whether that
makes it the *whole* of this market's system cost is a judgement about where
the boundary sits, argued at `make_settlement`; nothing in this file rests on
it.)  `settle` formed `kappa * throughput` for `cost` and
then dropped it, and `step` dropped `sub["throughput"]` too, so the matrix cell
read **not available** and the entry point was recorded: the `info` dict, not a
recomputation.

Both quantities are published, not one.  `kappa` is per agent and is swept, so
a throughput alone cannot be priced by a reader, and a price alone cannot say
how much was cycled.

L0 here is the JAX contract on the two new keys: shape, dtype,
jit agreement and vmap.  L1 is what the numbers must satisfy: the published
cost is exactly the product of the published throughput and the `kappa` that
was in force, it is a component of `cost` and never of `reward`, and it moves
with `kappa` while the physics does not.

**What this does not catch**, so a green result is not read as more than it is:
whether `throughput` is the right physical quantity for the battery's degradation
(that is the action map's acceptance, `test_action_l1.py`), and whether `kappa`
is calibrated (that is a scenario question).
"""
import chex
import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.envs.p2p import make_p2p_env, make_p2p_params
from powermarketjax.resources.battery import make_battery_bundle

N = 8
N_PERIODS = 24
EPISODE_LEN = 6
PI_EXP, PI_RET = 4.1, 26.11
DELTA = 0.5


def build(kappa_value=3.0, seed=0):
    rng = np.random.default_rng(seed)
    battery = make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(2.0, 10.0, N).tolist(),
        power_mw=rng.uniform(0.5, 3.0, N).tolist())
    params = make_p2p_params(
        p_pv=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
        load=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
        battery=battery,
        kappa=np.full(N, kappa_value),
        learner_mask=np.ones(N, bool),
        episode_len=EPISODE_LEN)
    return make_p2p_env(N, PI_EXP, PI_RET, DELTA), params


def _rollout_one(kappa_value=3.0, seed=0, key_seed=5):
    (reset, step, _, _), params = build(kappa_value, seed)
    key = jax.random.PRNGKey(key_seed)
    _, state = reset(key, params)
    action = jax.random.uniform(jax.random.PRNGKey(key_seed + 1), (N, 2),
                                jnp.float32, -1.0, 1.0)
    return jax.jit(step)(key, state, action, params), params, state, action


# ------------------------------------------------------------------ L0

def test_both_keys_are_published_with_the_right_shape_and_dtype():
    (_o, _s, _r, _c, _d, info), _p, _st, _a = _rollout_one()
    for key in ("throughput", "degradation_cost"):
        assert key in info, (
            f"`info` lost {key!r}; market 04's system-cost cell goes back to "
            f"not available and the only other route is a recomputation")
        assert info[key].shape == (N,), key
        assert info[key].dtype == jnp.float32, key


def test_jit_matches_eager_on_the_two_keys():
    (reset, step, _, _), params = build()
    key = jax.random.PRNGKey(2)
    _, state = reset(key, params)
    action = jax.random.uniform(jax.random.PRNGKey(3), (N, 2), jnp.float32,
                                -1.0, 1.0)
    eager = step(key, state, action, params)[5]
    jitted = jax.jit(step)(key, state, action, params)[5]
    for k in ("throughput", "degradation_cost"):
        chex.assert_trees_all_equal(eager[k], jitted[k])


def test_vmaps_over_a_batch_of_states():
    """The rollout runs `vmap` over environments, so a key that does not batch
    is a key no experiment can record."""
    (reset, step, _, _), params = build()
    keys = jax.random.split(jax.random.PRNGKey(4), 5)
    _, states = jax.vmap(reset, in_axes=(0, None))(keys, params)
    actions = jax.random.uniform(jax.random.PRNGKey(5), (5, N, 2), jnp.float32,
                                 -1.0, 1.0)
    info = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None)))(
        keys, states, actions, params)[5]
    for k in ("throughput", "degradation_cost"):
        assert info[k].shape == (5, N), k


# ------------------------------------------------------------------ L1

def test_published_cost_is_the_published_throughput_times_the_kappa_in_force():
    """The identity is asserted per agent, on the dimension it is consumed on.

    Per agent rather than on the total: a total holds under any reallocation
    between agents, and `kappa` is a per-agent vector precisely so that it can
    differ between them.
    """
    kappa_value = 3.0
    (_o, _s, _r, _c, _d, info), params, _st, _a = _rollout_one(kappa_value)
    thr = np.asarray(info["throughput"], np.float32)
    deg = np.asarray(info["degradation_cost"], np.float32)
    # The reference multiply is done in float32, the precision this whole
    # market is written in.  Forming it in float64 instead leaves a one-ULP
    # gap (measured 2026-08-29: max absolute 1.19e-07, max relative 4.49e-08 on
    # this fixture) which says nothing about the wiring and would only invite a
    # tolerance pinned to one measurement.  At the market's own precision the
    # identity is bitwise, so that is what is asserted.
    np.testing.assert_array_equal(
        deg, np.asarray(params.kappa, np.float32) * thr)
    # non-triviality: an identity over an all-zero throughput holds vacuously
    assert (thr > 0.0).any(), (
        "no agent cycled its battery in this fixture, so the identity above is "
        "vacuous; the check is on a run point that does not exist")


def test_it_tracks_kappa_while_the_physics_does_not():
    """Doubling `kappa` doubles the published cost and moves neither the
    throughput nor the state of charge.  This is the injection that separates
    'the cost was published' from 'the cost was published for the kappa that
    was actually in force' -- a stale capture would pass the identity above and
    fail here."""
    (_o1, s1, _r1, _c1, _d1, i1), _p1, _st, _a = _rollout_one(3.0)
    (_o2, s2, _r2, _c2, _d2, i2), _p2, _st2, _a2 = _rollout_one(6.0)
    np.testing.assert_array_equal(np.asarray(i1["throughput"]),
                                  np.asarray(i2["throughput"]))
    np.testing.assert_array_equal(np.asarray(s1.soc), np.asarray(s2.soc))
    np.testing.assert_allclose(np.asarray(i2["degradation_cost"], np.float64),
                               2.0 * np.asarray(i1["degradation_cost"],
                                                np.float64),
                               rtol=1e-6, atol=0)


def test_it_is_a_component_of_cost_and_never_of_reward():
    """`cost` is what a participant really pays and reaches
    `reward` through `profit`; `costs` is the CMDP constraint vector and never
    reaches `reward`.  Degradation is of the first kind, so raising `kappa`
    must lower `reward` by exactly the extra degradation and leave `costs`
    untouched."""
    (_o1, _s1, r1, c1, _d1, i1), _p1, _st, _a = _rollout_one(3.0)
    (_o2, _s2, r2, c2, _d2, i2), _p2, _st2, _a2 = _rollout_one(6.0)
    np.testing.assert_array_equal(np.asarray(c1), np.asarray(c2))
    drop = np.asarray(r1, np.float64) - np.asarray(r2, np.float64)
    extra = (np.asarray(i2["degradation_cost"], np.float64)
             - np.asarray(i1["degradation_cost"], np.float64))
    np.testing.assert_allclose(drop, extra, rtol=1e-5, atol=1e-6)
    assert (extra > 0.0).any(), "vacuous: no agent cycled, so nothing degraded"


# ------------------------------------------------------- the driver's unpack

#: Every tool that unpacks `run_rl_04.evaluate`, with its number of call sites
#: (counted 2026-09-23).
KNOWN_EVALUATE_CALLERS = {"bestfixed_04_grid.py": 1, "compare_04_paths.py": 3,
                          "curve_04_convergence.py": 1, "pairwise_04_seeds.py": 1,
                          "run_rl_04.py": 1}

def test_every_evaluate_call_site_unpacks_the_third_array():
    """`run_rl_04.evaluate` returns three arrays since the degradation channel
    landed, and the tools in `KNOWN_EVALUATE_CALLERS` call it.

    A return arity is not a signature, so
    `tests/tools/test_shared_helper_arity_l0.py` -- which binds call sites
    against live signatures -- cannot see this one.  The failure is loud
    (`ValueError` on unpack) but it fires *after* the evaluation has run, and
    for this market that is the expensive half: the same shape of "green suite,
    driver dead at the line that writes its first product" that file was
    written for.
    """
    import ast
    import pathlib

    def sites(tree):
        """(line, number of names unpacked) for every `... = evaluate(...)`."""
        return [(node.lineno, len(getattr(node.targets[0], "elts", [None]))
                 if isinstance(node.targets[0], ast.Tuple) else 1)
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "evaluate"]

    root = pathlib.Path(__file__).resolve().parents[3] / "tools" / "benchmark"
    for path in sorted(root.glob("*.py")):
        for line, n in sites(ast.parse(path.read_text())):
            assert n == 3, (
                f"{path.name}:{line} unpacks {n} names from evaluate(), which "
                f"returns 3 (return, constraint, degradation)")
    # Not vacuous: the known callers, each with its number of call sites.  Not
    # every checkout carries all of them, so an absent file is not counted --
    # but a present one must still hold all its sites, and one must be present.
    present = {n: k for n, k in KNOWN_EVALUATE_CALLERS.items() if (root / n).exists()}
    assert present, "none of the known evaluate() callers is in this checkout"
    for name, k in present.items():
        got = len(sites(ast.parse((root / name).read_text())))
        assert got == k, f"{name}: {got} evaluate() call sites, expected {k}"
    # and the check bites: one known call site unpacking two names is refused
    name = sorted(present)[0]
    tree = ast.parse((root / name).read_text())
    hit = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and isinstance(node.value, ast.Call)
               and getattr(node.value.func, "id", None) == "evaluate")
    hit.targets[0].elts = hit.targets[0].elts[:2]
    assert [n for _, n in sites(tree) if n != 3] == [2], name
