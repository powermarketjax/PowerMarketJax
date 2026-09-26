"""The gradient clip is JOINT across agents, and most agents do not need it.

**Why this is a test and not a note.**  Under `per_agent_params=True` the
question a reader will ask of any result is "did per-agent parameters restore
state response".  There is a device-side answer that has to be excluded first:
`optax.clip_by_global_norm` bounds the norm of the WHOLE parameter pytree, so
with one network per agent the factor is set by the aggregate and applied to
everyone.  If most agents' updates are being scaled down by a factor they did
not earn, "per-agent parameters changed nothing" is a statement about the
optimiser chain and not about parameter sharing.  That is the reading this file
exists to keep available, so the numbers live in the tree rather than in a
scratch report that a later reader has no reason to open.

**What is asserted, and what each one would catch.**  The clip binds at all; a
strict majority of agents are inside the clip radius on their own; every agent's
clipped gradient is the same single factor times its own, which is what "joint"
means and is exactly what a per-agent clip would break; and that factor is small
enough that the majority is scaled by more than a thousandfold.  Swap
`clip_by_global_norm` for a per-agent clip and the third fails; make the
gradients homogeneous and the second fails; remove the clip and the first fails.

**Coverage.**  Market 01 only, at the initial parameters, on one batch.  The
effect is a property of the optimiser chain and so is not market-specific, but
this file measures it in one place and does not claim the others.  It is also
not a claim about training: what the factor does over 200 iterations, once Adam
has moments and the value head has stopped dominating the gradient, is not
measured here.

`_loss` is read out of `iterate`'s closure, the way
`tests/learning/test_ippo_gae_terminal_l1.py` reads `_rollout` and `_gae`: the
gradient is not part of the public return, and exporting it from `ippo.py` to
make it testable would be editing the object under test.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("optax", reason="powermarketjax.learning needs the rl extra")
import optax

from tests.learning.test_ippo_per_agent_params_l0 import _built, _cfg


@pytest.fixture(scope="module", autouse=True)
def x64():
    """float64, module scope.

    The helpers are imported from the sibling file but its autouse fixture is
    not, and the day-ahead clearing refuses to build without float64 -- so
    without this the file fails on the environment rather than on the property.
    """
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


def _cells(fn):
    return dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))


def _loss_of(iterate):
    """`(_rollout, _gae, _loss)` out of the closure chain, or a loud failure."""
    top = _cells(iterate)
    for n in ("_rollout", "_gae", "_update"):
        assert n in top, (
            f"make_ippo's `iterate` no longer closes over {n}; its free "
            f"variables are {sorted(top)}. This file reads the gradient from "
            f"that closure because `iterate` returns metrics and not grads; a "
            f"rename means this has to be re-pointed, not deleted")
    upd = _cells(top["_update"].cell_contents)
    assert "_loss" in upd, (
        f"`_update` no longer closes over `_loss`: {sorted(upd)}")
    return (top["_rollout"].cell_contents, top["_gae"].cell_contents,
            upd["_loss"].cell_contents)


def _per_agent_norms(grads):
    """Each agent's own gradient norm, from its slice of every leaf."""
    leaves = jax.tree_util.tree_leaves(grads)
    n_agents = leaves[0].shape[0]
    return np.array([
        float(jnp.sqrt(sum(jnp.sum(x[i] ** 2) for x in leaves)))
        for i in range(n_agents)])


def _whole_norm(grads):
    """The pytree's norm, by the same formula, so the two are comparable.

    `optax.global_norm` would do this as well, but it is deprecated and, more to
    the point, computing it here with the same reduction as `_per_agent_norms`
    keeps the comparison below between the factor optax APPLIED and the factor
    this file EXPECTS, rather than between two different norm implementations.
    """
    return float(jnp.sqrt(sum(jnp.sum(x ** 2)
                              for x in jax.tree_util.tree_leaves(grads))))


def test_the_clip_is_joint_and_most_agents_did_not_need_it():
    """Measured 2026-08-27, CPU, float64, market 01 at the initial parameters.

    The four assertions are ordered so the first failure is the most
    informative: if the clip does not bind, nothing below it means anything.
    """
    cfg = _cfg(epochs=1, minibatches=1)
    four, bounds, prm, _c, _obs_dim, init, iterate = _built("day_ahead", True,
                                                            cfg)
    n_agents = int(four[3]["n_agents"])
    params, tx, _opt, st, obs = init(jax.random.PRNGKey(1), prm)

    rollout, gae, loss = _loss_of(iterate)
    st, obs, _k, traj, last = rollout(params, st, obs, jax.random.PRNGKey(7),
                                      prm)
    adv, ret = gae(traj, last)
    n = cfg.horizon * cfg.n_envs
    batch = jax.tree.map(
        lambda x: x.reshape((n,) + x.shape[2:]),
        dict(obs=traj["obs"], pre=traj["pre"], logp=traj["logp"], adv=adv,
             ret=ret))
    grads = jax.grad(loss, has_aux=True)(params, batch)[0]

    joint = _whole_norm(grads)
    own = _per_agent_norms(grads)
    c = cfg.max_grad_norm

    # 1. the clip binds at all.  Everything below is about what it does, and
    #    none of it has content if it never fires.
    assert joint > c, (
        f"the joint gradient norm is {joint:.4e} against max_grad_norm={c}, so "
        f"the clip does not fire here and this file measures nothing; the "
        f"operating point has moved and the numbers below have to be retaken")

    # 2. a strict majority of agents are INSIDE the radius on their own.  This
    #    is the whole point: they are scaled anyway.
    inside = int((own <= c).sum())
    assert inside > n_agents // 2, (
        f"only {inside} of {n_agents} agents are inside the clip radius on "
        f"their own, so 'most agents are scaled by a factor they did not earn' "
        f"is not what is happening here; per-agent norms span "
        f"{own.min():.3e} to {own.max():.3e}")

    # 3. THE structural assertion: one factor, applied to everyone.  A per-agent
    #    clip would leave every agent in (2) untouched and pin the rest at `c`,
    #    which compresses the ratios; a joint clip preserves them exactly.
    clipped, _ = optax.clip_by_global_norm(c).update(grads, None)
    own_clipped = _per_agent_norms(clipped)
    scale = c / joint
    expected = own * scale
    dev = float(np.max(np.abs(own_clipped - expected)
                       / np.maximum(expected, 1e-300)))
    # The bound comes from the measurement and from what it has to separate.
    # Measured 2026-08-27: 9.4e-08, which is uniformity to seven digits and not
    # float64 round-off -- `optax.clip_by_global_norm` reduces the tree in its
    # own order, and these leaves span four orders of magnitude.  The bound is
    # then checked against the alternative it exists to exclude, in this same
    # test rather than from memory: under a PER-AGENT clip the agents already
    # inside the radius would be left alone, so their departure from `expected`
    # would be 1/scale, of order 1e+04 here.
    counterfactual = np.minimum(own, c)
    cf_dev = float(np.max(np.abs(counterfactual - expected)
                          / np.maximum(expected, 1e-300)))
    assert dev < 1e-6, (
        f"the clip did not scale every agent by the same factor {scale:.4e}; "
        f"the largest relative departure is {dev:.3e}. A clip that treats "
        f"agents separately would land here, and that is a different device "
        f"from the one every measurement so far was taken on")
    assert cf_dev > 1e3 * 1e-6, (
        f"a per-agent clip would depart from the joint one by only "
        f"{cf_dev:.3e}, which the bound above ({1e-6:.0e}) would not separate "
        f"from the measured {dev:.3e}; on this operating point the assertion "
        f"cannot tell the two devices apart and has no content")

    # 4. and the factor is not a rounding-level adjustment.
    assert scale < 1e-3, (
        f"the joint clip scales every agent by {scale:.4e}, which is close "
        f"enough to 1 that the majority in (2) are not materially held back; "
        f"the concern this file records does not apply at this operating point")

    # The numbers themselves, so a reader gets them without running anything.
    # Measured 2026-08-27 on market 01 at the initial parameters: 42 of 66
    # agents INSIDE the radius and 24 above it, per-agent norms spanning
    # 7.528e-04 to 7.692e+03 (a factor of 1.02e+07), joint norm 1.2270e+04
    # against a radius of 0.5, so the joint clip multiplies every one of the 66
    # by 4.0750e-05.  The uniformity departure is 9.399e-08 against the 1e-6
    # bound, and the per-agent-clip counterfactual departs by 2.454e+04, i.e.
    # 2.6e+11 times further -- which is the margin that makes the bound a
    # discriminating one rather than a loose one.  These are NOT stable across
    # edits to `ippo.py`: an earlier reading on the same market gave 44/22 and
    # a joint norm of 1.4323e+04, and only the rollout upstream had changed.
    # They are recorded as an order of magnitude, not as a fixture.
    assert own.max() / max(own.min(), 1e-300) > 10.0, (
        f"the per-agent gradient norms span only "
        f"{own.max() / max(own.min(), 1e-300):.2f}x, so 'the largest agent sets "
        f"the factor for everyone' is not a meaningful description here")
