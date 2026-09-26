"""L0 for `learner_dtype`: what the flag demotes, and what it must not reach.

The keyword exists because with `jax_enable_x64` on -- which every driver turns
on, because `envs/ancillary/clearing.py` refuses to be built without it -- flax's
float32 kernels against a float64 observation promote every matmul back to
float64.  So the property under test is not "the dtype I passed in came back":
it is the **element type of the matmul in the compiled module**.  A
float32 `obs_mean` alone changes nothing at all and reads as though it had, which
is a trap an earlier measurement fell into.

`tools/benchmark/hlo_audit.py` does the classification, and is imported rather
than restated so the criterion has one definition.

**The discriminating injection is the default arm.**  Each assertion
about the flag being on is paired with the same reading taken with the flag off,
which must come out the other way: the default arm still has float64 in the
gradient step, and still carries the two float64 parameter leaves.  Without that
pair, an audit that had silently stopped matching anything would satisfy every
"no float64" assertion here over an empty set.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("optax", reason="powermarketjax.learning needs the rl extra")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                      / "tools" / "benchmark"))
from hlo_audit import (CLEARING, GRADIENT_STEP, ROLLOUT_POLICY,  # noqa: E402
                       gradient_step_types, matmul_audit)

from powermarketjax.learning.ippo import (IPPOConfig, learner_cast,  # noqa: E402
                                          make_ippo)
from powermarketjax.learning.sac import SACConfig, make_sac  # noqa: E402
from tests.learning.test_ippo_per_agent_params_l0 import MARKETS  # noqa: E402

#: Market 03, the one every timing of the float32 learner is taken on.  One market for the
#: HLO readings rather than three: the keyword is dtype plumbing in shared
#: module code and does not branch on the market, while each compile of this
#: environment costs tens of seconds.
HLO_MARKET = "ancillary"


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


def _sac_cfg():
    return SACConfig(n_envs=2, horizon=2, buffer_size=8, batch_size=4,
                     utd_ratio=1.0, gamma=0.99, tau=0.005, policy_lr=3e-4,
                     q_lr=1e-3, alpha_lr=1e-3, init_alpha=0.2, hidden=(8, 8),
                     log_std_min=-5.0, log_std_max=2.0, reward_scale=1.0)


def _ippo_cfg():
    return IPPOConfig(n_envs=2, horizon=4, epochs=2, minibatches=2, lr=3e-4,
                      clip_eps=0.2, gamma=0.99, gae_lambda=0.95, vf_coef=0.5,
                      ent_coef=0.01, max_grad_norm=0.5, hidden=(8, 8),
                      init_scale=1.0, weight_decay=0.0)


def _build(algo, market, per_agent, dtype):
    """``(init, iterate, prm)`` for one cell."""
    four, bounds, prm, _res = MARKETS[market]()
    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    od = int(obs.shape[-1])
    kw = dict(obs_mean=jnp.zeros(od, obs.dtype), obs_std=jnp.ones(od, obs.dtype),
              per_agent_params=per_agent, learner_dtype=dtype)
    if algo == "sac":
        init, iterate = make_sac(four, bounds, _sac_cfg(), **kw)
    else:
        init, iterate = make_ippo(four, bounds, _ippo_cfg(), **kw)
    return init, iterate, prm


def _dtypes(tree):
    return sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(tree)})


@pytest.mark.parametrize("algo", ["sac", "ippo"])
@pytest.mark.parametrize("per_agent", [False, True])
@pytest.mark.parametrize("market", sorted(MARKETS))
def test_the_default_arm_keeps_the_float64_leaf_the_flag_exists_to_close(
        algo, per_agent, market):
    """Under `None`, `params` still holds float64 -- so the flag has work to do.

    This is the non-triviality of every "all float32" assertion below.  SAC's
    float64 leaf is `log_alpha` (`jnp.log` of a Python float under x64) and
    IPPO's is `log_std` (`nn.initializers.zeros` with no dtype); flax's `Dense`
    was already storing float32, so a test that only looked at the kernels would
    pass identically with and without the keyword.
    """
    init, _it, prm = _build(algo, market, per_agent, None)
    params, *_ = init(jax.random.PRNGKey(0), prm)
    assert "float64" in _dtypes(params), (
        f"{algo}/{market}: the default arm has no float64 parameter leaf, so "
        f"the flag's paired assertion below is comparing float32 with float32")


@pytest.mark.parametrize("algo", ["sac", "ippo"])
@pytest.mark.parametrize("per_agent", [False, True])
@pytest.mark.parametrize("market", sorted(MARKETS))
def test_every_parameter_and_optimiser_leaf_is_float32_under_the_flag(
        algo, per_agent, market):
    """`float32` reaches the parameters, their moments and nothing integral."""
    init, _it, prm = _build(algo, market, per_agent, "float32")
    params, _tx, learner, *_ = init(jax.random.PRNGKey(0), prm)
    assert _dtypes(params) == ["float32"], f"{algo}/{market}: {_dtypes(params)}"
    # SAC's `learner` also carries the replay buffer, which stays in the
    # environment's precision because the rollout writes it; IPPO's is the
    # optimiser state alone.
    opt = ({k: v for k, v in learner.items() if k != "buffer"}
           if algo == "sac" else learner)
    assert "float64" not in _dtypes(opt), f"{algo}/{market}: {_dtypes(opt)}"
    if algo == "sac":
        # the buffer is written by the rollout and keeps the rollout's
        # precision whatever the flag says: same leaf dtypes as the default arm
        # (which on the ancillary market means float64 leaves, on the float32
        # markets none -- since 2026-09-18 `pre`/`reward` follow the rollout)
        init0, _it0, _p0 = _build(algo, market, per_agent, None)
        _p, _t, learner0, *_ = init0(jax.random.PRNGKey(0), prm)
        assert _dtypes(learner["buffer"]) == _dtypes(learner0["buffer"]), (
            f"{market}: the replay buffer was demoted by the flag: "
            f"{_dtypes(learner['buffer'])} vs default {_dtypes(learner0['buffer'])}")
        if market == "ancillary":
            assert "float64" in _dtypes(learner["buffer"]), (
                "the ancillary rollout is float64, so its buffer must be too")


@pytest.mark.parametrize("algo", ["sac", "ippo"])
@pytest.mark.parametrize("per_agent", [False, True])
def test_the_gradient_step_has_no_float64_matmul_under_the_flag(algo, per_agent):
    """The criterion, read off the compiled module, with its own control.

    Flag off: the gradient step's matmuls are float64 (that is the whole reason
    for the keyword).  Flag on: none of them is.  Both readings come from the
    same audit, so a classification that had stopped matching would fail the
    first assertion rather than pass the second over nothing.
    """
    seen = {}
    for dtype in (None, "float32"):
        init, iterate, prm = _build(algo, HLO_MARKET, per_agent, dtype)
        state = init(jax.random.PRNGKey(0), prm)
        params, tx, learner, env_state, env_obs = state
        txt = jax.jit(iterate, static_argnums=(1,)).lower(
            params, tx, learner, env_state, env_obs,
            jax.random.PRNGKey(1), prm).compile().as_text()
        seen[dtype] = matmul_audit(txt)

    off, on = gradient_step_types(seen[None]), gradient_step_types(seen["float32"])
    assert off, ("the audit attributed no matmul to the gradient step, so the "
                 "assertion below would pass over an empty set")
    assert off.get("f64", 0) > 0, (
        f"{algo}: the default arm's gradient step is already free of f64 "
        f"matmuls ({off}), so this test cannot tell the flag from its absence")
    assert on and on.get("f64", 0) == 0, (
        f"{algo}: the gradient step still has float64 matmuls under the flag: "
        f"{on}")
    assert sum(off.values()) == sum(on.values()), (
        f"{algo}: the flag changed how many matmuls the gradient step runs "
        f"({sum(off.values())} -> {sum(on.values())}), which is more than a "
        f"change of precision")


@pytest.mark.parametrize("algo", ["sac", "ippo"])
def test_the_flag_does_not_reach_the_rollout_or_the_clearing(algo):
    """What must stay float64: the policy in the rollout and the market itself.

    The market's clearing needs float64 and says so, and the replay buffer and
    the recorded per-step series are written by the rollout, so demoting it
    would change quantities this keyword is documented not to touch.
    """
    init, iterate, prm = _build(algo, HLO_MARKET, False, "float32")
    params, tx, learner, env_state, env_obs = init(jax.random.PRNGKey(0), prm)
    rows = matmul_audit(jax.jit(iterate, static_argnums=(1,)).lower(
        params, tx, learner, env_state, env_obs, jax.random.PRNGKey(1),
        prm).compile().as_text())
    for scope in (ROLLOUT_POLICY, CLEARING):
        types = {et: c for (s, et), c in rows.items() if s == scope}
        assert types, f"{algo}: nothing was attributed to {scope}"
        assert types.get("f64", 0) > 0, (
            f"{algo}: {scope} has no float64 matmul left ({types}); the flag "
            f"reached past the gradient step")


@pytest.mark.parametrize("algo", ["sac", "ippo"])
@pytest.mark.parametrize("per_agent", [False, True])
def test_iterate_jits_and_the_pytree_is_stable_under_the_flag(algo, per_agent):
    """L0 proper: one `jit`, two chained calls, structure and dtypes unchanged.

    Two calls rather than one, for the reason
    `test_ippo_per_agent_params_l0` gives: a structure that changed inside the
    update would recompile or raise on the second call and not on the first.  A
    dtype that changed would do the same, and is the failure this keyword could
    plausibly introduce -- a cast applied on one pass and not the next.
    """
    init, iterate, prm = _build(algo, HLO_MARKET, per_agent, "float32")
    params, tx, learner, env_state, env_obs = init(jax.random.PRNGKey(1), prm)
    before = (jax.tree_util.tree_structure(params), _dtypes(params))
    step = jax.jit(iterate, static_argnums=(1,))
    key = jax.random.PRNGKey(2)
    for _ in range(2):
        key, k = jax.random.split(key)
        params, learner, env_state, env_obs, _k, _m = step(
            params, tx, learner, env_state, env_obs, k, prm)
    assert (jax.tree_util.tree_structure(params), _dtypes(params)) == before


@pytest.mark.parametrize("bad", ["int32", "bool"])
def test_a_non_floating_learner_dtype_is_refused(bad):
    """`learner_dtype` chooses an arithmetic precision and nothing else.

    An integer here would cast the parameters to integers and train on them,
    which runs and produces numbers; refusing at construction is the only place
    it is cheap to notice.
    """
    with pytest.raises(ValueError, match="not a floating type"):
        learner_cast(bad)


def test_learner_cast_is_none_for_the_default_so_the_default_arm_is_unwrapped():
    """`None` in, `None` out -- the callers branch in Python, not on a no-op cast."""
    assert learner_cast(None) is None
    f = learner_cast("float32")
    # the integer leaf is compared against its own dtype rather than against a
    # named one: under x64 `jnp.arange` gives int64 and without it int32, and
    # the claim is that the cast left it alone, not which width it was
    n_in = jnp.arange(3)
    out = f({"a": jnp.ones(3, jnp.float64), "n": n_in})
    assert out["a"].dtype == jnp.float32
    assert out["n"].dtype == n_in.dtype, "an integer leaf was cast"
