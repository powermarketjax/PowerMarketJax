"""L0 for `env_chunks`: the same rollout in pieces, and the pieces put back.

The keyword exists for one number: 2.842 GiB of device memory per environment
on `case813nem`, so the shared `n_envs=64` needs 182 GiB and a 24 GB card
holds seven.  `env_chunks=c`
steps the environment axis in `c` consecutive pieces under `lax.map` instead
of one `vmap`, so one piece's clearing is live at a time.  What can go wrong
is an axis: the pieces are a reshape of the environment axis and a reshape
back, and a wrong one would hand environment `i` environment `j`'s key, state
or action while every shape stayed right.  The test that bites is therefore
"the same sample, bit for bit", and the mutation at the bottom shows that it
does bite: a chunker that reverses the piece order goes red on the same
comparison.

**Bit identity is asked of the two markets where it holds at this file's
shape, and measured on the one where it does not.**  On CPU float64 the
day-ahead and real-time iterations agree bit for bit between `c=1` and `c=2`
(measured 2026-09-16, `T=4`, `n_envs=4, horizon=2`: 52 and 49 leaves, none
differing; day-ahead also at `n_envs=8` for `c=2` and `c=4`).  That is a
statement about this shape, not about the operator: at the production shape
`T=24` the same comparison leaves every reward, action, value and the updated
parameters identical but moves the cancellation residual `shed_mwh`
(~1e-19 MWh) in its last bits, and one end-to-end driver iteration then
differs in its updated parameters by 5.5e-08 relative.
The ancillary
clearing forms its KKT matrix densely (`solvers/ipm._dense_kkt`), and that
matmul's rounding depends on the batch shape it is evaluated at: awards move
by 7.1e-16 relative and the prices, which the ill-conditioned final Newton
steps amplify, by 4.3e-08 (`lmp_prev`) and 5.9e-07 (`reserve_price_prev`),
`reward_mean` by 3.2e-09 -- the same order as the cross-process floor
measured for that market on GPU.  So for ancillary this file pins the
integer and boolean leaves bit for bit (the axis question has no rounding in
it) and the floating ones to bounds written beside their measured values; a
permuted axis would miss those bounds by many orders of magnitude.

**The default arm is not a wrapper.**  `env_chunks=1` must be the `jax.vmap`
object the rollout always was; that is asserted on the jaxpr of `iterate`
against the bare `vmap`, not on a belief about what XLA folds.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("optax", reason="powermarketjax.learning needs the rl extra")

from powermarketjax.learning import ippo  # noqa: E402
from powermarketjax.learning.ippo import (IPPOConfig, chunked_env_step,  # noqa: E402
                                          make_ippo, observation_statistics)
from powermarketjax.learning.policy import bounds_for  # noqa: E402
from tests.learning.test_ippo_per_agent_params_l0 import MARKETS  # noqa: E402

N_ENVS = 4
HORIZON = 2

#: The two markets whose environment step is bit-identical across `c` at this
#: file's shape on CPU float64 (module docstring).  Ancillary is measured
#: separately below.
BITWISE = ("day_ahead", "real_time")


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


def _cfg(n_envs=N_ENVS):
    return IPPOConfig(n_envs=n_envs, horizon=HORIZON, epochs=1, minibatches=1,
                      lr=3e-4, clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                      vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                      hidden=(8, 8), init_scale=1.0, weight_decay=0.0)


def _built(name, env_chunks, n_envs=N_ENVS):
    four, bounds, prm, _reserve = MARKETS[name]()
    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    od = int(obs.shape[-1])
    init, iterate = make_ippo(four, bounds, _cfg(n_envs),
                              obs_mean=jnp.zeros(od, obs.dtype),
                              obs_std=jnp.ones(od, obs.dtype),
                              env_chunks=env_chunks)
    return four, prm, init, iterate


def _one_iteration(init, iterate, prm):
    p, tx, opt, st, eobs = init(jax.random.PRNGKey(1), prm)
    return jax.jit(iterate, static_argnums=(1,))(p, tx, opt, st, eobs,
                                                 jax.random.PRNGKey(2), prm)


def _leaves(tree):
    return [(jax.tree_util.keystr(k), np.asarray(v))
            for k, v in jax.tree_util.tree_leaves_with_path(tree)]


def _differing(a, b):
    """Names of the leaves whose bytes differ; shapes and dtypes must agree."""
    la, lb = _leaves(a), _leaves(b)
    assert [k for k, _ in la] == [k for k, _ in lb], "the pytrees differ"
    out = []
    for (k, x), (_, y) in zip(la, lb):
        assert x.shape == y.shape and x.dtype == y.dtype, (k, x.shape, y.shape)
        if x.tobytes() != y.tobytes():
            out.append(k)
    return out


def test_the_default_arm_is_the_bare_vmap():
    """`env_chunks=1` returns `jax.vmap(step)` itself, not a function around it.

    Checked two ways: the object the helper returns traces to the same jaxpr
    as the bare `vmap` on the same inputs, and `make_ippo`'s default equals
    `env_chunks=1` explicitly.  The first is the claim the module docstring
    makes ("bound in Python at construction"); the second is that the default
    value is 1 and not something a caller has to know to ask for.
    """
    four, prm, _init, _it = _built("day_ahead", 1)
    reset, _step, step_auto_reset, spec = four
    keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)
    obs, state = jax.vmap(reset, in_axes=(0, None))(keys, prm)
    action = jnp.broadcast_to(spec["baseline_action"],
                              (N_ENVS,) + tuple(spec["action_shape"]))
    bare = jax.vmap(step_auto_reset, in_axes=(0, 0, 0, None))
    bound = chunked_env_step(step_auto_reset, N_ENVS, 1)
    assert str(jax.make_jaxpr(bound)(keys, state, action, prm)) == \
        str(jax.make_jaxpr(bare)(keys, state, action, prm))

    obs_dim = int(obs.shape[-1])
    kw = dict(obs_mean=jnp.zeros(obs_dim, obs.dtype),
              obs_std=jnp.ones(obs_dim, obs.dtype))
    b = bounds_for(spec)
    _i, it_default = make_ippo(four, b, _cfg(), **kw)
    _i, it_one = make_ippo(four, b, _cfg(), env_chunks=1, **kw)
    p, tx, opt, st, eobs = _init(jax.random.PRNGKey(1), prm)
    k = jax.random.PRNGKey(2)
    assert str(jax.make_jaxpr(it_default, static_argnums=(1,))(
        p, tx, opt, st, eobs, k, prm)) == \
        str(jax.make_jaxpr(it_one, static_argnums=(1,))(
            p, tx, opt, st, eobs, k, prm))


@pytest.mark.parametrize("bad", [0, -1, 3, 8, 1.0, True])
def test_a_non_divisor_or_a_non_integer_is_refused_at_construction(bad):
    """`n_envs % c != 0`, `c < 1` and non-integers fail in `make_ippo`, not later.

    3 and 8 do not divide 4; `1.0` and `True` are not the integer the keyword
    is, and `True == 1` is exactly the kind of value that would pass a lax
    check and then reshape by a boolean.
    """
    four, bounds, prm, _reserve = MARKETS["day_ahead"]()
    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    od = int(obs.shape[-1])
    with pytest.raises(ValueError, match="env_chunks"):
        make_ippo(four, bounds, _cfg(), obs_mean=jnp.zeros(od, obs.dtype),
                  obs_std=jnp.ones(od, obs.dtype), env_chunks=bad)
    with pytest.raises(ValueError, match="env_chunks"):
        observation_statistics(four, prm, jax.random.PRNGKey(0), N_ENVS,
                               HORIZON, env_chunks=bad)


@pytest.mark.parametrize("name", BITWISE)
def test_one_iteration_is_bit_identical_in_two_pieces(name):
    """Everything `iterate` returns -- new params, optimiser state, carried
    environment state and observation, key, metrics with every per-step series
    -- from the same start and the same key, `c=1` against `c=2`, byte for
    byte.  Measured 2026-09-16 on CPU float64: 52 leaves on day-ahead, 49 on
    real-time, none differing.
    """
    _f, prm, init1, it1 = _built(name, 1)
    _f, prm2, init2, it2 = _built(name, 2)
    out1 = _one_iteration(init1, it1, prm)
    out2 = _one_iteration(init2, it2, prm2)
    diff = _differing(out1, out2)
    assert not diff, f"{name}: {len(diff)} leaves differ between c=1 and c=2: {diff}"


@pytest.mark.parametrize("name", BITWISE)
@pytest.mark.parametrize("env_chunks", [2, 4])
def test_observation_statistics_are_bit_identical_in_pieces(name, env_chunks):
    """`obs_mean` / `obs_std` standardise every observation the policy ever
    sees, so a chunked fit that moved them would be a different policy from
    the same weights (`write_params`' docstring).  `c=4` is one environment
    per piece, the extreme the reshape has to survive too.
    """
    four, _b, prm, _r = MARKETS[name]()
    ref = observation_statistics(four, prm, jax.random.PRNGKey(3), N_ENVS, HORIZON)
    got = observation_statistics(four, prm, jax.random.PRNGKey(3), N_ENVS, HORIZON,
                                 env_chunks=env_chunks)
    assert not _differing(ref, got), f"{name}: statistics moved at c={env_chunks}"


def test_ancillary_keeps_the_axis_and_moves_only_by_rounding():
    """The market where bit identity does not hold, and how far it does not.

    The non-floating leaves (the state's `cursor` and `step_in_episode`,
    Adam's `count`, the `step_cursor` and `step_converged` series) carry the
    axis question and no rounding, so they are byte for byte.  The floating ones take bounds written beside the values measured
    2026-09-16 on CPU float64 at `n_envs=4, horizon=2`: awards (`p_prev`,
    `award_prev`) 7.1e-16 relative, `lmp_prev` 4.3e-08, `reserve_price_prev`
    5.9e-07, `profit_prev` and `step_reward` 8.2e-08, `reward_mean` 3.2e-09.
    Each bound is two orders of magnitude above its measurement and many below
    what a permuted environment axis produces (the mutation test shows O(1)
    on day-ahead).  The metric terms that cancel to near zero (`costs_mean`
    here is ~1e-16) are not compared relatively for the reason
    `test_ippo_per_agent_params_l0` gives.
    """
    _f, prm, init1, it1 = _built("ancillary", 1)
    _f, prm2, init2, it2 = _built("ancillary", 2)
    _p1, o1, st1, obs1, _k1, m1 = _one_iteration(init1, it1, prm)
    _p2, o2, st2, obs2, _k2, m2 = _one_iteration(init2, it2, prm2)

    ints = [(k, x) for k, x in _leaves((st1, o1, m1["step_cursor"],
                                         m1["step_converged"]))
            if x.dtype.kind in "iub"]
    ints2 = [(k, x) for k, x in _leaves((st2, o2, m2["step_cursor"],
                                          m2["step_converged"]))
             if x.dtype.kind in "iub"]
    assert len(ints) >= 3, "no integer/boolean leaves found; the axis check is empty"
    for (k, x), (_, y) in zip(ints, ints2):
        assert x.tobytes() == y.tobytes(), f"ancillary: {k} differs"

    def rel(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        return float(np.abs(x - y).max() / max(np.abs(x).max(), 1e-300))

    bounds = {"p_prev": 1e-13, "award_prev": 1e-13, "lmp_prev": 1e-5,
              "reserve_price_prev": 1e-5, "profit_prev": 1e-5}
    for field, bound in bounds.items():
        r = rel(getattr(st1, field), getattr(st2, field))
        assert r <= bound, f"ancillary: {field} moved {r:.3e} relative (bound {bound:.0e})"
    assert rel(m1["step_reward"], m2["step_reward"]) <= 1e-5
    assert rel(m1["reward_mean"], m2["reward_mean"]) <= 1e-6
    assert rel(obs1, obs2) <= 1e-5


def test_a_chunker_that_permutes_the_pieces_is_caught(monkeypatch):
    """The bite: the same comparison goes red when the pieces come back in the
    wrong order.  `chunked_env_step` is rebuilt with a `lax.map` whose
    stacked output has its piece axis reversed -- environment `i` is then
    handed to slot `n-1-i` -- and the day-ahead iteration, bit-identical
    above, must now differ in the carried state.  Without this, a bitwise
    assertion that had quietly started comparing an object with itself would
    pass over nothing.
    """
    real_map = jax.lax.map

    def reversed_map(f, xs, *a, **kw):
        return jax.tree.map(lambda x: x[::-1], real_map(f, xs, *a, **kw))

    # `lax.map` is called when `iterate` is TRACED, not when `make_ippo` binds
    # the chunker, so the patch has to be live through the jit below.  Nothing
    # else under `powermarketjax/` calls `lax.map` (grepped 2026-09-16), and
    # the `c=1` arm never reaches it, so the patch touches only the pieces.
    monkeypatch.setattr(ippo.jax.lax, "map", reversed_map)
    _f, prm, init2, it2 = _built("day_ahead", 2)
    out2 = _one_iteration(init2, it2, prm)
    monkeypatch.undo()
    _f, prm1, init1, it1 = _built("day_ahead", 1)
    out1 = _one_iteration(init1, it1, prm1)
    diff = _differing(out1, out2)
    assert any("cursor" in k for k in diff), (
        f"reversing the pieces changed nothing the comparison sees "
        f"({len(diff)} leaves differ: {diff}); the bitwise test above has no "
        f"content")
