"""L0 for the replay buffer's `pre` / `reward` dtype: the rollout's, not the box's.

**What went wrong before** (2026-09-18).  `make_sac` sized the two leaves
on `jnp.result_type(low)`, and `bounds_for` builds the box in float64.  On the
markets whose observation is float32 (day-ahead, real-time) the actor samples a
float32 `pre` and the market pays a float32 reward; the FIFO write widened both
exactly, and at replay `_q_action(batch["pre"])` came back float64, so the two
critics' forward and backward pass on the replayed sample ran in float64 while
the target critics and the actor ran in float32.  On an Ada card that one
promotion was about 80% of the gradient step's time.

**What is asserted, and the pair each assertion comes with**:

* on a float32 market the buffer's `pre` and `reward` have the rollout's dtype,
  and the compiled gradient step has no float64 matmul; the discriminating
  injection is a buffer built the old way (`result_type(low)`), which must put
  float64 back into the gradient step -- so an audit that had stopped matching
  would fail the injection, not pass the assertion over an empty set;
* on the float64 market (ancillary) `init` is bit-identical to the old
  expression, leaf for leaf, dtype for dtype: the change is a no-op there.

`tools/benchmark/hlo_audit.py` classifies the matmuls, imported rather than
restated so the criterion has one definition.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("optax", reason="powermarketjax.learning needs the rl extra")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                      / "tools" / "benchmark"))
from hlo_audit import gradient_step_types, matmul_audit  # noqa: E402

from powermarketjax.learning.sac import SACConfig, make_sac  # noqa: E402
from tests.learning.test_ippo_per_agent_params_l0 import MARKETS  # noqa: E402

FLOAT32_MARKETS = ("day_ahead", "real_time")
FLOAT64_MARKET = "ancillary"


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


def _cfg():
    return SACConfig(n_envs=2, horizon=2, buffer_size=8, batch_size=4,
                     utd_ratio=1.0, gamma=0.99, tau=0.005, policy_lr=3e-4,
                     q_lr=1e-3, alpha_lr=1e-3, init_alpha=0.2, hidden=(8, 8),
                     log_std_min=-5.0, log_std_max=2.0, reward_scale=1.0)


def _build(market, per_agent=False):
    four, bounds, prm, _res = MARKETS[market]()
    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    od = int(obs.shape[-1])
    init, iterate = make_sac(four, bounds, _cfg(), obs_mean=jnp.zeros(od, obs.dtype),
                             obs_std=jnp.ones(od, obs.dtype), per_agent_params=per_agent)
    return four, bounds, prm, init, iterate


def _rollout_dtypes(four, prm):
    """What one step of the market actually produces: the reward's dtype, and
    the observation's (the actor's `pre` follows the observation through the
    float32 network)."""
    reset, _step, step_auto, spec = four
    obs, state = reset(jax.random.PRNGKey(1), prm)
    _o, _s, reward, *_ = step_auto(jax.random.PRNGKey(2), state,
                                   spec["baseline_action"], prm)
    return obs.dtype, reward.dtype


@pytest.mark.parametrize("per_agent", [False, True])
@pytest.mark.parametrize("market", FLOAT32_MARKETS)
def test_on_a_float32_market_pre_and_reward_take_the_rollouts_dtype(market, per_agent):
    four, bounds, prm, init, _it = _build(market, per_agent)
    obs_dt, reward_dt = _rollout_dtypes(four, prm)
    assert obs_dt == jnp.float32 and reward_dt == jnp.float32, (
        f"{market} is expected to roll out in float32; got obs {obs_dt}, "
        f"reward {reward_dt} -- the market changed, re-read the test's premise")
    _p, _t, learner, *_ = init(jax.random.PRNGKey(0), prm)
    buf = learner["buffer"]
    assert buf["pre"].dtype == jnp.float32, buf["pre"].dtype
    assert buf["reward"].dtype == reward_dt, buf["reward"].dtype
    assert buf["obs"].dtype == obs_dt and buf["next_obs"].dtype == obs_dt
    # the box is still float64: the buffer no longer follows it
    assert jnp.result_type(bounds[0]) == jnp.float64


def _gradient_step_types(four, bounds, prm, init, iterate, widen):
    """Element types of the compiled gradient step's matmuls; `widen=True`
    injects the old buffer (leaves cast to `result_type(low)`) into the same
    `iterate`, which is what the pre-2026-09-18 module did."""
    params, tx, learner, env_state, env_obs = init(jax.random.PRNGKey(0), prm)
    if widen:
        wide = jnp.result_type(bounds[0])
        buf = dict(learner["buffer"])
        buf["pre"] = buf["pre"].astype(wide)
        buf["reward"] = buf["reward"].astype(wide)
        learner = dict(learner, buffer=buf)
    step = jax.jit(iterate, static_argnums=(1,))
    text = step.lower(params, tx, learner, env_state, env_obs,
                      jax.random.PRNGKey(3), prm).compile().as_text()
    rows = matmul_audit(text)
    assert sum(rows.values()) > 0, "the audit matched no matmul at all"
    return gradient_step_types(rows)


def test_the_gradient_step_has_no_float64_matmul_on_a_float32_market_and_the_old_buffer_puts_it_back():
    market = "real_time"
    four, bounds, prm, init, iterate = _build(market)
    fixed = _gradient_step_types(four, bounds, prm, init, iterate, widen=False)
    assert fixed and "f64" not in fixed, fixed
    # the injection: the same learner with the buffer widened the old way must
    # bring float64 back into the gradient step (this is the reading that
    # proves the audit still bites)
    old = _gradient_step_types(four, bounds, prm, init, iterate, widen=True)
    assert "f64" in old and old["f64"] > 0, old


@pytest.mark.parametrize("per_agent", [False, True])
def test_on_the_float64_market_init_is_bit_identical_to_the_box_sized_buffer(per_agent):
    """Negative control: where the rollout is float64 the change is a no-op.
    Every leaf of `init`'s output is compared against the same tree with the
    two leaves rebuilt the old way; zero differences over a counted number of
    leaves, and the two leaves' dtypes are float64 on both sides."""
    four, bounds, prm, init, _it = _build(FLOAT64_MARKET, per_agent)
    obs_dt, reward_dt = _rollout_dtypes(four, prm)
    assert obs_dt == jnp.float64 and reward_dt == jnp.float64, (obs_dt, reward_dt)
    params, _tx, learner, env_state, env_obs = init(jax.random.PRNGKey(0), prm)
    buf = learner["buffer"]
    wide = jnp.result_type(bounds[0])
    old = dict(buf, pre=jnp.zeros(buf["pre"].shape, wide),
               reward=jnp.zeros(buf["reward"].shape, wide))
    n = 0
    for (pa, a), (pb, b) in zip(jax.tree_util.tree_leaves_with_path(buf),
                                jax.tree_util.tree_leaves_with_path(old)):
        assert pa == pb
        assert a.dtype == b.dtype, (jax.tree_util.keystr(pa), a.dtype, b.dtype)
        assert np.array_equal(np.asarray(a), np.asarray(b)), jax.tree_util.keystr(pa)
        n += 1
    assert n == len(buf) == 7, n
    assert buf["pre"].dtype == jnp.float64 and buf["reward"].dtype == jnp.float64
