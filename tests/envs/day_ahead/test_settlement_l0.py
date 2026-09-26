"""L0 JAX contract for the day-ahead settlement.

jit, vmap, pytree stability, no NaN, and callability from inside lax.scan over
days.  Domain correctness is L1 (test_settlement_l1.py).

Settlement is arithmetic, not a solve, so these tests feed it arrays directly
instead of running the clearing -- L0 is about the contract, and a solve here
would only make it slow.

x64 is set by an autouse fixture rather than at import time because **other
modules in this suite turn it off globally** (see the L0 clearing module).
Settlement does not need float64 the way the solver does, but it is called on
the solver's output, so this is the regime it actually runs in.
"""
import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import make_settlement

T = 4


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def case(x64):
    return load_case("29gb")


@pytest.fixture(scope="module")
def args(case):
    """Plausible-shaped clearing output: award within capacity, prices positive."""
    n_u, n_b = len(np.asarray(case.unit_p_min)), int(case.n_nodes)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    p_max = jnp.asarray(np.asarray(case.unit_p_max, np.float64))
    award = jax.random.uniform(k1, (n_u, T), jnp.float64) * p_max[:, None]
    lmp = 20.0 + 60.0 * jax.random.uniform(k2, (T, n_b), jnp.float64)
    commitment = jnp.ones((n_u, T), jnp.float64)
    commitment_status = jnp.zeros((n_u,), jnp.float64)
    return award, lmp, commitment, commitment_status


def test_shapes_and_dtype(case, args):
    n_u = len(np.asarray(case.unit_p_min))
    out = make_settlement(case)(*args)
    # Exact set equality, not a subset check: a settlement that quietly
    # grew a key is a changed contract, and this assertion is what makes
    # the change deliberate.  It fired on 2026-08-19 when a change added
    # the three components of `cost`, which is the behaviour intended --
    # the keys are listed again here rather than relaxed to `>=`.
    assert set(out) == {"revenue", "cost", "profit", "reward",
                        "energy_cost", "no_load_cost", "startup_cost"}
    # The split must reconstruct the total it was split from, or the three
    # components are three numbers that merely travel beside `cost`.
    # Measured 2026-08-19 on the 29gb case: dropping `no_load_cost` from
    # this sum fails it by 6.68e+05 on the worst single unit and 1.09e+07
    # summed, while the full expression leaves a residual of exactly 0.
    # So the check bites, and it bites on the component carrying about a
    # third of system cost rather than on rounding.
    parts = out["energy_cost"] + out["no_load_cost"] + out["startup_cost"]
    np.testing.assert_allclose(np.asarray(parts), np.asarray(out["cost"]),
                               rtol=1e-12, atol=0.0)
    for k, v in out.items():
        assert v.shape == (n_u,), k
        # settlement runs inside the clearing's float64 scope; casting down to
        # the environment's float32 (§15) belongs to `step`, not here
        assert v.dtype == jnp.float64, k
        assert np.isfinite(np.asarray(v)).all(), k


def test_jit_matches_eager(case, args):
    """Not bit for bit: XLA fuses the reductions differently under jit, which on
    these 66-unit sums is a 2.7e-15 relative difference."""
    settle = make_settlement(case)
    a, b = settle(*args), jax.jit(settle)(*args)
    assert tu.tree_structure(a) == tu.tree_structure(b)
    for k in a:
        np.testing.assert_allclose(np.asarray(a[k]), np.asarray(b[k]), rtol=1e-13)


def test_vmap_over_batch(case, args):
    """Batched settlement equals per-lane settlement, lane by lane."""
    settle = make_settlement(case)
    award, lmp, commitment, commitment_status = args
    batch = 3
    scale = jnp.arange(1, batch + 1, dtype=jnp.float64)[:, None, None]
    awards = award[None] * scale
    lmps = jnp.broadcast_to(lmp, (batch,) + lmp.shape)
    us = jnp.broadcast_to(commitment, (batch,) + commitment.shape)
    inits = jnp.broadcast_to(commitment_status, (batch,) + commitment_status.shape)

    out = jax.vmap(settle)(awards, lmps, us, inits)
    for i in range(batch):
        one = settle(awards[i], lmp, commitment, commitment_status)
        for k in one:
            np.testing.assert_allclose(np.asarray(out[k][i]), np.asarray(one[k]),
                                       rtol=1e-12)


def test_inside_lax_scan(case, args):
    """A day is one scan step, and the commitment carries across the boundary."""
    settle = make_settlement(case)
    award, lmp, commitment, commitment_status = args
    n_days = 5

    def day(carry, _):
        out = settle(award, lmp, commitment, carry)
        return commitment[:, -1], out["reward"]

    _, rewards = jax.lax.scan(day, commitment_status, None, length=n_days)
    assert rewards.shape == (n_days, len(np.asarray(case.unit_p_min)))
    assert np.isfinite(np.asarray(rewards)).all()
    # start-up is charged on day 0 only: after it the unit is already on
    assert (np.asarray(rewards)[0] <= np.asarray(rewards)[1] + 1e-9).all()


def test_agent_axis_is_the_partition(case, args):
    """`unit_to_agent` sets the length of every returned array."""
    n_u = len(np.asarray(case.unit_p_min))
    mapping = np.arange(n_u) % 4
    out = make_settlement(case, unit_to_agent=mapping)(*args)
    for k, v in out.items():
        assert v.shape == (4,), k
