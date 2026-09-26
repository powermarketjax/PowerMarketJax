"""L0 for `--per-agent-params` on the OUT-OF-PACKAGE market 04 drivers.

**These two files had no test of any kind.**  `grep -rl constrained_baseline
tests/` returned nothing on 2026-09-09, and the published 04 learning numbers
come from exactly that file.  So the layout switch this module pins is a switch on the device the
paper's numbers came from, and it is pinned here rather than only on the
in-package driver: two arms differing in the parameter layout AND in which
program produced them are not a contrast in the layout.

What the four groups below are for, and what each of them would catch:

* the shared layout is still the shared layout -- shapes and scalar count, so
  that adding the switch cannot quietly move the arm every published number was
  produced on;
* the per-agent layout adds ONE leading axis and that axis tracks `n_agents`,
  which is the same shape relation `powermarketjax.learning.ippo` produces.  The
  scalar count is asserted as an exact multiple, because the four
  zero-initialised biases do not depend on the mapped key and a `vmap` that
  dropped their output axis would leave a tree that still looks per-agent at a
  glance;
* the two layouts do not compute the same thing, and each participant's network
  is wired to that participant's own row.  Shapes alone cannot see either: a
  forward pass that ignored the leading axis, or one that used participant 0's
  network for everybody, produces arrays of exactly the right shape;
* the DRIVER hands back what it was asked for.  The layout is read off the
  parameter tree that came back, never off the boolean that went in -- "I passed
  the flag" and "the learner received it" are two claims and only the second is
  worth an assertion.

**Measured judgement power** (2026-09-09, CPU, `jax` 0.10.2).  Against a copy of
the two modules in which `run` accepts `per_agent` and then calls
`R.init_policy` unconditionally -- the flag wired in and dropped -- three of the
tests below go red and the shared-arm tests stay green; the reds report `w1` at
`(15, 64)` where `(4, 15, 64)` was required, `1` distinct leading axis value
where `4` was required, and an action difference of exactly `0.0`.
"""
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The out-of-package drivers reach `distrax`/`rlax`/`optax` through the `rl`
# extra, which CI's `dev` install does not carry; a bare import would turn an
# absent optional dependency into a failure rather than a skip.
pytest.importorskip("optax", reason="the out-of-package driver needs the rl extra")
pytest.importorskip("distrax", reason="preliminary_reference builds its policy on distrax")
pytest.importorskip("rlax", reason="the losses come from rlax")

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "p2p_experiment"))
import constrained_baseline as CB                                # noqa: E402
import preliminary_reference as R                                # noqa: E402

#: The shared tree's shapes, written out rather than recomputed from the module
#: constants.  A test that derived them from `HIDDEN` and `OBS_DIM` would follow
#: those constants wherever they went, and what is being pinned here is that the
#: arm the published numbers came from did not move.
SHARED_SHAPES = {
    "w1": (15, 64), "b1": (64,),
    "w2": (64, 64), "b2": (64,),
    "wm": (64, 2), "bm": (2,),
    "wv": (64, 1), "bv": (1,),
    "log_std": (2,),
}
#: 960 + 64 + 4096 + 64 + 128 + 2 + 64 + 1 + 2.
SHARED_SCALARS = 5381


def shapes_of(policy):
    return {k: tuple(int(d) for d in jnp.shape(v)) for k, v in policy.items()}


def scalars_of(policy):
    return int(sum(int(np.prod(s)) if s else 1
                   for s in shapes_of(policy).values()))


# --------------------------------------------------------- the shared layout

def test_shared_layout_is_unchanged():
    """Nine leaves at the shapes every published 04 learning number came from."""
    policy = R.init_policy(jax.random.PRNGKey(0))
    assert shapes_of(policy) == SHARED_SHAPES
    assert scalars_of(policy) == SHARED_SCALARS


def test_shared_layout_ignores_the_participant_count():
    """`init_policy` takes no `n_agents`: the agent axis is the batch axis.

    This is the property the whole shared arm rests on, and it is what makes
    the per-agent count below a real difference rather than a renaming.
    """
    a = R.init_policy(jax.random.PRNGKey(0))
    b = R.init_policy(jax.random.PRNGKey(0), R.OBS_DIM)
    assert shapes_of(a) == shapes_of(b) == SHARED_SHAPES


# ------------------------------------------------------ the per-agent layout

@pytest.mark.parametrize("n_agents", [2, 3, 5])
def test_per_agent_adds_one_leading_axis_that_tracks_n(n_agents):
    """Every leaf is `(n_agents,) + its shared shape`, and nothing else moves."""
    policy = R.init_policy_per_agent(jax.random.PRNGKey(0), n_agents)
    got = shapes_of(policy)
    assert sorted(got) == sorted(SHARED_SHAPES), "the leaf SET must not change"
    for name, shared in SHARED_SHAPES.items():
        assert got[name] == (n_agents,) + shared, name


@pytest.mark.parametrize("n_agents", [2, 3, 5])
def test_per_agent_scalar_count_is_an_exact_multiple(n_agents):
    """`n_agents` times the shared count, exactly.

    Asserted on the count and not only on the shapes because the four biases
    are `jnp.zeros(...)` and do not depend on the mapped key.  A `vmap` that
    broadcast them away would leave `w1 w2 wm wv log_std` correctly per-agent
    and the ratio at 3.927 instead of 4 at N = 4 -- a tree that reads as
    per-agent everywhere a reader is likely to look.
    """
    policy = R.init_policy_per_agent(jax.random.PRNGKey(0), n_agents)
    assert scalars_of(policy) == n_agents * SHARED_SCALARS


def test_participants_do_not_start_identical():
    """Split keys, so no two participants are handed the same network."""
    policy = R.init_policy_per_agent(jax.random.PRNGKey(0), 4)
    w1 = np.asarray(policy["w1"])
    for j in range(1, 4):
        assert not np.array_equal(w1[0], w1[j]), j


# ------------------------------------------- the switch actually does something

def synthetic_obs(n_agents, key=7):
    """An `(n_agents, OBS_DIM)` observation of the size the scaled channels are.

    Not drawn from the environment: what is under test is the forward pass, and
    a rollout would put the market between the parameters and the assertion.
    """
    return jax.random.normal(jax.random.PRNGKey(key),
                             (n_agents, R.OBS_DIM), jnp.float32)


def test_the_two_layouts_give_different_actions_on_the_same_key():
    """Same init key, same observation, and not one component agrees.

    `tanh(mean)` is the greedy action the `learned_mean` arm submits, so this
    is the quantity the market would see, not an intermediate.  A switch that
    was accepted and then ignored lands here with a difference of exactly zero.
    """
    n_agents = 4
    key = jax.random.PRNGKey(0)
    obs = synthetic_obs(n_agents)
    shared = np.asarray(jnp.tanh(R.forward(R.init_policy(key), obs)[0]))
    per_agent = np.asarray(jnp.tanh(
        R.forward_per_agent(R.init_policy_per_agent(key, n_agents), obs)[0]))
    assert shared.shape == per_agent.shape == (n_agents, R.ACTION_DIM)
    assert not np.any(shared == per_agent), (
        "some action component is bit-identical across the two layouts, which "
        "is what a switch that never reached the forward pass looks like")
    assert np.abs(shared - per_agent).max() > 1e-4, (
        "the two layouts differ by less than 1e-4 in every component; both "
        "output heads are scaled by 0.01, so a difference this small would "
        "mean the assertion above is reading float noise")


def test_each_participant_gets_its_own_network_and_not_its_neighbours():
    """Row `j` of the output comes from leaf slice `j`, checked one at a time.

    The reference is built by the OTHER path: the shared `forward` applied to
    one sliced-out network and one observation row, with no `vmap` anywhere.  A
    per-agent forward that used participant 0's network for everybody would
    produce arrays of exactly the right shape and would fail only here.
    """
    n_agents = 4
    policy = R.init_policy_per_agent(jax.random.PRNGKey(1), n_agents)
    obs = synthetic_obs(n_agents)
    mean, _log_std, value = R.forward_per_agent(policy, obs)
    for j in range(n_agents):
        one = jax.tree_util.tree_map(lambda leaf: leaf[j], policy)
        want_mean, _, want_value = R.forward(one, obs[j])
        np.testing.assert_allclose(np.asarray(mean[j]), np.asarray(want_mean),
                                   rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(value[j]), np.asarray(want_value),
                                   rtol=0, atol=1e-6)
        for k in range(n_agents):
            if k == j:
                continue
            other = jax.tree_util.tree_map(lambda leaf: leaf[k], policy)
            wrong, _, _ = R.forward(other, obs[j])
            assert not np.allclose(np.asarray(mean[j]), np.asarray(wrong),
                                   rtol=0, atol=1e-6), (j, k)


def test_crossing_the_two_layouts_is_refused():
    """A shared tree through the per-agent forward raises rather than computes.

    `ippo.make_greedy_action` documents that this guard is `vmap`'s and not a
    check anyone wrote, and that it is blind where `n_agents == obs_dim`.  Four
    participants against fifteen channels is not that case, which is why the
    tests above use four.
    """
    with pytest.raises(Exception):
        R.forward_per_agent(R.init_policy(jax.random.PRNGKey(0)),
                            synthetic_obs(4))


# ------------------------------------------------------------- the driver

def test_run_hands_back_the_layout_it_was_asked_for():
    """`constrained_baseline.run` under both settings, read off what came back.

    One training iteration on two episodes: this is not a score, it is the
    smallest thing that puts the flag through `make_update`, the rollout, the
    loss and the exploration-width schedule.  The schedule is the one place a
    literal shape would silently collapse the per-agent `log_std` back to one
    shared row, so its shape is asserted after the loop and not at `init`.
    """
    n_agents = 4
    shared = CB.run(n_agents, 0.5, False, 0, 1, 2, 1, False)[1]
    per_agent = CB.run(n_agents, 0.5, False, 0, 1, 2, 1, True)[1]

    assert shapes_of(shared) == SHARED_SHAPES
    assert shapes_of(per_agent) == {
        k: (n_agents,) + v for k, v in SHARED_SHAPES.items()}
    assert per_agent["log_std"].shape == (n_agents, R.ACTION_DIM), (
        "the exploration-width schedule overwrote the per-agent log_std with "
        "one shared row")

    assert CB.parameter_layout(shared, n_agents)["per_agent_params"] is False
    got = CB.parameter_layout(per_agent, n_agents)
    assert got["per_agent_params"] is True
    assert got["scalars"] == n_agents * got["scalars_shared_layout"]
    assert got["leaves"] == 9


def test_parameter_layout_refuses_a_tree_in_neither_layout():
    """The readback has no third value, and does not file the odd one as shared.

    Reporting "not per-agent" for a tree it cannot place would put it in with
    the shared runs, which is the direction that loses information silently.
    """
    policy = dict(R.init_policy(jax.random.PRNGKey(0)))
    policy["w1"] = jnp.zeros((3, 15, 64), jnp.float32)
    with pytest.raises(SystemExit):
        CB.parameter_layout(policy, 4)


def test_the_cli_flag_runs_end_to_end_and_stamps_the_product(tmp_path, monkeypatch):
    """`--per-agent-params` through `main`, and the stamp lands in the JSON.

    `--iterations 0` is the untrained control: no update runs, so this exercises
    the flag's whole path -- parse, `run`, the readback guard in `main`, and the
    record written to disk -- at the cost of the evaluation alone.  The stamp is
    what lets a reader tell the two layouts apart from the file, which is the
    same arrangement markets 01 to 03 use for this flag.
    """
    out = tmp_path / "curves.json"
    monkeypatch.setattr(sys, "argv", [
        "constrained_baseline", "--agents", "3", "--seeds", "1",
        "--iterations", "0", "--eval-every", "1", "--initial-soc", "0.5",
        "--per-agent-params", "--out", str(out)])
    CB.main()

    records = json.loads(out.read_text())
    assert records, "main wrote no record"
    for record in records:
        layout = record["layout"]
        assert layout["per_agent_params"] is True
        assert layout["leaves"] == 9
        assert layout["scalars"] == 3 * SHARED_SCALARS
        assert layout["scalars_shared_layout"] == SHARED_SCALARS
        assert layout["leaf_shapes"]["w1"] == [3, 15, 64]
        assert layout["read_from"].startswith("the policy")
