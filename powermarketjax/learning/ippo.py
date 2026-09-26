"""IPPO: rollout, GAE and the clipped update, over one policy or `n_agents` of them.

**Parameter sharing is a keyword, not a config field.** `per_agent_params=False`
is the default and is the path every archive in the repository was produced on;
`True` gives each agent its own copy of the same architecture, which is the
control the collapse work needs.
The switch is a keyword of `make_ippo` and `make_greedy_action` rather than a
field of `IPPOConfig` because the three drivers write `vars(cfg)` into every
checkpoint's `hyperparams`, so a new field would alter every product already on
disk.  Each of the four places the two paths differ binds its function at
construction, in Python, so the shared arm is the expression it always was and
not that expression with an axis moved by zero.

**One thing genuinely changes under `per_agent_params=True`, and it is not the
network.** `optax.clip_by_global_norm` bounds the norm of the WHOLE parameter
pytree, so with one network per agent it clips the `n_agents` gradients jointly
rather than each on its own: an agent with a large gradient shrinks its
neighbours' updates.  That is a property of the optimiser chain, not of
per-agent parameters as such, and it is left in place rather than quietly
replaced by a per-agent clip, because swapping it would make the two paths
differ in two ways at once.

Independent learners.  Each agent sees only its own observation and optimises
only its own profit; there is no centralised critic, since one would feed the
opponents' information into the value function through the back door, and this
benchmark exists to ask whether a participant can learn to bid from what it can
actually see.

The whole iteration runs inside one `jit`: the rollout is a `lax.scan` over the
horizon with `vmap` over the environment axis, and the update is a `lax.scan`
over epochs and minibatches.  No Python loop may drive the rollout and no
`pure_callback` may sit on that path, so the only Python loop here is the one
over training iterations, outside `jit`.

**`env_chunks` trades rollout parallelism for peak memory and changes nothing
else.**  The dominant buffers of a day-ahead rollout are the clearing's
per-period Newton systems, and they scale with the `vmap` width alone: 2.842
GiB per environment on `case813nem`, so the shared `n_envs=64` needs 182 GiB
and one 24 GB card holds seven.
`env_chunks=c` splits the environment axis of the environment step into `c`
pieces of `n_envs // c`, runs the same `vmap` on each piece under `lax.map`
(a `scan`, so one piece's working set is live at a time) and joins the pieces
back, inside the same horizon `scan`.  The keys are still split once over the
full `n_envs` and only regrouped, so environment `i` sees the same key, state
and action in either arm; the policy forward (`_act`) is not chunked, because
the memory is not there and chunking it would change the row count of its
matmul.  **`1`, the default, is the bare `vmap` bound in Python at
construction** -- the same arrangement as `per_agent_params` and
`learner_dtype`, and for the same reason.  What `c > 1` may change is
rounding inside the clearing, since XLA selects kernels by batch shape; how
much, against the cross-process floor of the same card, was measured
rather than assumed to be zero.

**The observation is standardised against a frozen reference**, taken once from
a rollout of the truthful action and then held fixed.  A running normaliser is
deliberately not used: it would make the observation a function of training
history, so two runs differing only in seed would see different observations.

**`learner_dtype` demotes the gradient step and nothing else.**  Every driver
runs under `jax_enable_x64` because the ancillary clearing refuses to be built
without it (`envs/ancillary/clearing.py:148`), and with x64 on, an f32 parameter
against an f64 observation promotes the matmul back to f64 -- so the whole
update runs in float64 although flax already stores the kernels in float32.
`learner_dtype="float32"` casts, at construction, the four float64 quantities
that reach the loss (`obs_mean`, `obs_std`, `low`, `high`), the float64
parameter leaf flax's default initialiser produces (`policy.SharedActorCritic`'s
`log_std`, from `nn.initializers.zeros` under x64), and the minibatch itself;
the rollout, the environment, GAE and the recorded per-step series keep the
precision they had.  **`None` is not the same code path with a no-op cast**: it
is bound in Python at construction, so the default arm is the expression it
always was.  What it is worth is measured, not asserted -- the gradient step
and the rollout are timed separately, and on the markets whose clearing
dominates an iteration this keyword buys almost nothing.
"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax

from .policy import (SharedActorCritic, entropy, log_prob, sample_pre,
                     to_action)


@dataclass(frozen=True)
class IPPOConfig:
    """Every field is required; none of these has a defensible default here.

    The values a run adopts belong in that run's record, not in this file: a
    default carried across markets would be a calibration nobody performed.
    """

    n_envs: int
    horizon: int
    epochs: int
    minibatches: int
    lr: float
    clip_eps: float
    gamma: float
    gae_lambda: float
    vf_coef: float
    ent_coef: float
    max_grad_norm: float
    hidden: Sequence[int]
    init_scale: float
    #: Decoupled weight decay (`optax.adamw`).  Zero selects `optax.adam`
    #: exactly, not an approximation of it.  Required like every other field
    #: here: a default would be a calibration nobody performed, and this one
    #: in particular changes whether the weight norm has a stationary
    #: distribution at all.
    weight_decay: float


#: What each `spec["termination"]` value selects at the episode boundary, and
#: why.  Written as a table rather than as `== "terminal"` so that a market
#: declaring something new fails at construction instead of quietly taking the
#: bootstrap branch -- which is the branch that is wrong precisely for the
#: markets whose declaration nobody has read yet.
_BOOTSTRAP_AT_DONE = {
    # 01, 02, 03, 05: nothing is settled at the boundary, so the continuation
    # value is real and wiping it would throw away the rest of the process.
    "truncation": True,
    # 04: `envs/p2p/env.py` pays for the stock left in the battery with a
    # terminal leg inside `reward`, so a continuation value on top of it is a
    # second payment for the end of the episode.
    "terminal": False,
    # A CartPole control task outside the package: `done` there conflates a
    # real termination with the time limit, so NEITHER branch below is right for
    # it -- masking on `done` would wipe the continuation value of an episode
    # that had not ended.  The repair that task needs reads `info["terminated"]`,
    # which a variant of this module carries through `extra_info_keys`
    # and this shared module deliberately does not: a market-specific `info`
    # name appearing here is the warning sign `_require` describes.  So this
    # value keeps the arithmetic it has always had, which is also what makes
    # `run_control_ippo.py --gae asis` the control it is recorded as.
    "termination+truncation": True,
}


def _require(info: dict, key: str, asked_by: str = "extra_info_keys"):
    """`info[key]`, but the failure says which market's dictionary this is.

    `extra_info_keys` and `convergence_key` are both supplied by the caller so
    that a market-specific name never sits in this shared module, at the cost
    that a typo or a name borrowed from another market reaches here undetected
    until this raises; the message lists which keys are actually available and
    which parameter asked for the missing one.
    """
    if key not in info:
        raise KeyError(
            f"{asked_by} asked for info[{key!r}], which this market's step "
            f"does not return; available keys: {sorted(info)}")
    return info[key]


def learner_cast(learner_dtype):
    """`None`, or a function casting every floating leaf of a pytree to one dtype.

    Shared by `make_ippo` and `make_sac`, which face the same promotion: with
    `jax_enable_x64` on, one float64 operand anywhere in a loss drags the whole
    expression back to float64, and a cast applied to only some of the inputs
    reads as though it had demoted the learner while changing nothing.  The
    element type of the `dot` in the compiled HLO is what says whether it took,
    so it is read back rather than the dtype that was passed in being trusted.

    Returned as `None` rather than as the identity so that every caller
    branches in **Python** at construction: under `learner_dtype=None` the
    learner is the expression it always was, and not that expression with a
    `convert_element_type` that happens to be a no-op.  That is the arrangement
    `per_agent_params` uses, for the reason its own comment gives -- bit
    identity across the arrival of an option should hold by construction, not
    by a belief about what XLA folds away.

    Integer and boolean leaves pass through.  `done` is a flag, `cursor` and
    Adam's `count` are counters, and demoting a counter is a different change
    from demoting an arithmetic precision.
    """
    if learner_dtype is None:
        return None
    dt = jnp.dtype(learner_dtype)
    if dt.kind != "f":
        raise ValueError(
            f"learner_dtype={learner_dtype!r} resolves to {dt}, which is not a "
            f"floating type; this keyword chooses the precision the gradient "
            f"step runs in and has no meaning for any other kind")
    return lambda tree: jax.tree.map(
        lambda x: x.astype(dt) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        tree)


def chunked_env_step(step_auto_reset, n_envs: int, env_chunks: int,
                     action_batched: bool = True):
    """`vmap(step_auto_reset)` over `n_envs`, in `env_chunks` sequential pieces.

    Returns ``f(keys, state, action, env_params)`` with every output's leading
    axis ``n_envs`` in either arm.  `action_batched` says whether `action`
    carries an environment axis (the rollout) or is one array every
    environment takes (`observation_statistics`, the truthful action).

    **`env_chunks=1` returns the `jax.vmap` object itself**, not a wrapper
    around it, so the default arm is the expression the rollout always was
    (module docstring).  For `c > 1` the two reshapes are row-major and
    copy-free and the pieces are consecutive blocks of the environment axis,
    so piece `j` holds environments ``j*n .. (j+1)*n-1`` and the join puts
    them back where they were.  `lax.map` and not a Python loop: the trip
    count is static and the whole thing stays inside the horizon `scan`.

    `n_envs % env_chunks != 0` is refused rather than padded: a padded
    environment would be a rollout sample that no key, no reset and no
    hyperparameter accounts for.
    """
    if not isinstance(env_chunks, int) or isinstance(env_chunks, bool) \
            or env_chunks < 1:
        raise ValueError(
            f"env_chunks={env_chunks!r} must be a positive integer; it is the "
            f"number of sequential pieces the environment axis is stepped in")
    if n_envs % env_chunks:
        raise ValueError(
            f"env_chunks={env_chunks} does not divide n_envs={n_envs}; the "
            f"environment axis is split into equal consecutive pieces and is "
            f"not padded")
    vstep = jax.vmap(step_auto_reset,
                     in_axes=(0, 0, 0 if action_batched else None, None))
    if env_chunks == 1:
        return vstep
    per = n_envs // env_chunks

    def split(x):
        return x.reshape((env_chunks, per) + x.shape[1:])

    def join(x):
        return x.reshape((n_envs,) + x.shape[2:])

    def chunked(keys, state, action, env_params):
        if action_batched:
            xs = jax.tree.map(split, (keys, state, action))
            body = lambda a: vstep(a[0], a[1], a[2], env_params)
        else:
            xs = jax.tree.map(split, (keys, state))
            body = lambda a: vstep(a[0], a[1], action, env_params)
        return jax.tree.map(join, jax.lax.map(body, xs))

    return chunked


def action_layout(spec: dict, bounds: Tuple[jnp.ndarray, jnp.ndarray]):
    """Split a market's action into ``(n_agents, act_dim, act_shape, low, high)``.

    `act_dim` is the size of ONE agent's action, derived from the market's own
    `action_shape` together with `n_agents` rather than read off the last axis
    of `bounds`, which is ambiguous for a market whose action is one number per
    unit.  Axis 0 is the agent axis in every market, and everything after it is
    that agent's action, so `act_dim` is the product of the trailing dimensions.
    The check is loud because a market whose axis 0 is not the agent axis would
    otherwise train a shared policy over the wrong axis and still run to
    completion.

    `low` and `high` come back reshaped to ``(n_agents, act_dim)``, which is the
    space the policy works in; `act_shape` is what the environment expects.
    """
    n_agents = int(spec["n_agents"])
    act_shape = tuple(int(d) for d in spec["action_shape"])
    if not act_shape or act_shape[0] != n_agents:
        raise ValueError(
            f"spec['action_shape']={act_shape} does not start with the agent "
            f"axis (n_agents={n_agents}); IPPO shares one policy across agents "
            f"and has no way to know which axis to share over")
    act_dim = 1
    for d in act_shape[1:]:
        act_dim *= d
    low, high = bounds
    return (n_agents, act_dim, act_shape,
            low.reshape(n_agents, act_dim), high.reshape(n_agents, act_dim))


def _apply_per_agent(net, params, z):
    """`net.apply` with the agent axis of `z` mapped against the parameter axis.

    `z` is ``(..., n_agents, obs_dim)`` at every call site: un-batched in
    `make_greedy_action`, one environment's batch inside `_act`, and a
    minibatch of environment-steps inside `_loss`.  The agent axis is therefore
    always the one before `obs_dim`, which is what lets a single helper serve
    all three; it is moved to the front, `vmap`ped against the leading axis of
    `params`, and moved back, so every caller keeps the shapes it had under
    shared parameters.

    `log_std` comes back as ``(n_agents, act_dim)`` and is deliberately NOT
    moved.  It is a raw module parameter carrying no batch axes, and that shape
    broadcasts against `mean` in exactly the places the shared ``(act_dim,)``
    did; moving it would put the agent axis where the callers' `pre` and `low`
    do not have one.
    """
    zz = jnp.moveaxis(z, -2, 0)
    mean, log_std, value = jax.vmap(net.apply)(params, zz)
    return jnp.moveaxis(mean, 0, -2), log_std, jnp.moveaxis(value, 0, -1)


def make_greedy_action(spec: dict,
                       bounds: Tuple[jnp.ndarray, jnp.ndarray],
                       cfg: IPPOConfig,
                       obs_mean: jnp.ndarray,
                       obs_std: jnp.ndarray,
                       per_agent_params: bool = False) -> Callable:
    """`(params, obs) -> action`: the policy mean, in the market's action shape.

    Evaluation must report the policy, not the policy plus its exploration
    noise, so it needs this deterministic entry point.

    `per_agent_params` must match the `params` being passed.  Crossing the two
    is caught, but by `vmap` rather than by anything here: the leading axis of a
    shared kernel is `obs_dim`, so mapping it against `n_agents` observations
    raises on the inconsistent sizes.  That guard fails only where those two
    numbers coincide, which is also why the flag is a keyword rather than
    something inferred from the leaf shapes -- inference would have to guess
    which of the two a kernel's leading axis is on exactly that market.
    """
    _n_agents, act_dim, act_shape, low, high = action_layout(spec, bounds)
    net = SharedActorCritic(act_dim=act_dim, hidden=tuple(cfg.hidden),
                            init_scale=cfg.init_scale)
    _apply = (partial(_apply_per_agent, net) if per_agent_params
              else net.apply)

    def greedy(params, obs):
        """The policy mean for `obs`, in the market's own action shape.

        No sample is drawn and `log_std` is discarded, so two calls on the same
        `params` and `obs` return the same action.  `obs` is standardised with
        the same frozen statistics the training run used; passing raw
        observations here would evaluate the policy at inputs it never saw.
        """
        mean, _log_std, _value = _apply(params, (obs - obs_mean) / obs_std)
        return to_action(mean, low, high).reshape(act_shape)

    return greedy


def make_ippo(env: Tuple[Callable, Callable, Callable, dict],
              bounds: Tuple[jnp.ndarray, jnp.ndarray],
              cfg: IPPOConfig,
              obs_mean: jnp.ndarray,
              obs_std: jnp.ndarray,
              extra_info_keys: Sequence[str] = (),
              per_agent_params: bool = False,
              convergence_key: Optional[str] = "converged",
              valid_key: Optional[str] = None,
              learner_dtype: Optional[str] = None,
              env_chunks: int = 1):
    """Build `(init, iterate)` for one market.

    `iterate` is one PPO iteration: collect `horizon` steps in `n_envs`
    environments, compute GAE, and run `epochs` passes of `minibatches` updates.

    It is a pure function of `(params, opt_state, env_state, key, env_params)`
    and returns the same carry plus a metrics dict, so the caller can `jit` it
    once and drive it from a Python loop over iterations.

    `extra_info_keys` names `info` entries to carry out of the rollout as
    per-step series under `step_<key>`, since what the agent actually saw while
    training cannot be reconstructed afterwards by replaying the final policy --
    the policy changed throughout training.  The keys are named by the caller
    rather than listed here, because a market-specific name appearing in this
    shared module is itself the warning sign: a name that also exists in
    another market would silently pick up that market's quantity.

    `convergence_key` names the `info` entry that says whether the market's
    solve converged, and `None` says this market has no such quantity.  Naming
    it is the same arrangement `extra_info_keys` uses and for the same reason:
    the name is market-specific, and this module reading it unconditionally is
    what made market 04 -- whose clearing is two sorted orders and a comparison
    reduction, with no solver, no tolerance and no iteration bound -- fail with
    a bare `KeyError` when it was wired in (measured).  Under
    `None` neither `step_converged` nor `unconverged_frac` is produced, because
    the alternative on offer was to have the adapter feed a constant `True`,
    which would report `unconverged_frac = 0.0` for a market where no solver
    ever ran -- a number that reads as a measurement and is a property of the
    adapter.  **The default is the name the four solver-backed markets use**,
    so every driver already on disk keeps the two metrics it records; a market
    without the quantity passes `None` explicitly.

    `valid_key` names a per-sample boolean in `info` that the market sets when
    that sample's own clearing is usable.  Under `None`, the default and the
    path every archive was produced on, nothing is masked and every expression
    below is the one it was before this keyword existed.  Under a name, a sample
    the market marked unusable is excluded from the LOSS -- its reward keeps the
    value the market produced, because the reward is the
    settlement profit and a substituted number would be a fabrication -- and
    `metrics["masked_samples"]` counts what was excluded, so the exclusion is
    visible rather than silent.

    **The mask runs backwards along the horizon, not on the single sample.**
    `_gae` accumulates from the end (`adv[t] = delta[t] + gamma*lambda*adv[t+1]`),
    so an unusable reward at step `t` is already inside the advantage of every
    earlier step of that environment.  Masking only step `t` would leave those
    earlier samples carrying it.  A reverse cumulative AND therefore masks steps
    `0..t` of that environment and leaves `t+1..` alone, which is exactly the
    set `_gae` contaminated.

    **Bad values are replaced before they enter the graph, not weighted to
    zero.**  ``nan * 0.0 == nan``, so a zero weight spreads a bad sample instead
    of removing it, and the same holds for the gradient through a `where` whose
    unselected branch is non-finite.  `_update` therefore substitutes zeros for
    the masked samples' `adv` and `ret` and applies the weight on top.

    `learner_dtype` is the precision the gradient step runs in; `None`, the
    default, is the path every archive in the repository was produced on and is
    bit-identical to the module before this keyword existed.  What it casts and
    what it deliberately leaves alone are in the module docstring.

    `per_agent_params` gives every agent its own copy of the architecture
    instead of one shared copy; `init` then returns a pytree whose every leaf
    carries a leading `n_agents` axis, and the rollout, the update and
    `make_greedy_action` all line that axis up against the agent axis of the
    observation.  What it costs and what else it changes are in the module
    docstring; the default is the shared path, unchanged in every expression.

    `env_chunks` is the number of sequential pieces the environment step of
    the rollout is taken in; `1`, the default, is the single `vmap` every
    archive was produced on.  It is a keyword and not a field of `IPPOConfig`
    for the reason `per_agent_params` is: the drivers write `vars(cfg)` into
    every product, and this is not a hyperparameter -- it is how the same
    batch is laid onto the device, and the drivers stamp it separately.  Must
    divide `cfg.n_envs`; refused at construction otherwise (module docstring).
    """
    reset, _step, step_auto_reset, spec = env
    # Read at construction, so a market whose boundary rule this module cannot
    # act on is refused when it is wired rather than on its first iteration.
    # A missing key is an error and NOT a default: the default would be the
    # bootstrap branch, so a market that settles a stock at the boundary would
    # be wired in silently, which is the one failure this field exists to
    # prevent.
    if "termination" not in spec:
        raise KeyError(
            "spec has no 'termination' field, so nothing says whether this "
            "market's episode boundary is settled or merely cut, and `_gae` "
            "needs that to decide whether to bootstrap; the keys present are "
            f"{sorted(spec)}")
    if spec["termination"] not in _BOOTSTRAP_AT_DONE:
        raise ValueError(
            f"spec['termination']={spec['termination']!r} is not one of "
            f"{sorted(_BOOTSTRAP_AT_DONE)}; `_gae` branches on this field and "
            f"has no defensible default for a boundary rule it has not been "
            f"told about")
    bootstrap_at_done = _BOOTSTRAP_AT_DONE[spec["termination"]]
    n_agents, act_dim, act_shape, low, high = action_layout(spec, bounds)
    # The environment step of the rollout, bound here at construction: the
    # bare `vmap` under the default, `env_chunks` sequential pieces of it
    # otherwise (module docstring, `chunked_env_step`).
    _step_envs = chunked_env_step(step_auto_reset, cfg.n_envs, env_chunks)
    net = SharedActorCritic(act_dim=act_dim, hidden=tuple(cfg.hidden),
                            init_scale=cfg.init_scale)

    # The three places the two parameter layouts differ, each bound HERE, in
    # Python, at construction.  Bound this way the shared arm is `net.apply`,
    # `net.init` and `entropy` themselves rather than wrappers that happen to
    # be the identity, so the shared path is bit-identical across the arrival
    # of this option by construction and not by a belief about how XLA folds a
    # `moveaxis` of zero or a division by one.
    if per_agent_params:
        _apply = partial(_apply_per_agent, net)

        def _init_params(key, z):
            """One independent network per agent, from `n_agents` split keys.

            `z` is ``(n_agents, obs_dim)`` and is mapped alongside the keys, so
            each agent is shaped on its own observation row.  Which row is used
            does not change the result: `Dense` reads only the trailing
            dimension and the initialisers read only the key.
            """
            return jax.vmap(net.init)(jax.random.split(key, n_agents), z)

        def _entropy(log_std, low, high):
            """One agent's entropy, so the metric means the same in both paths.

            `entropy` sums over every axis of `log_std`, which is ``(act_dim,)``
            when the policy is shared and ``(n_agents, act_dim)`` when it is
            not.  Left alone, the per-agent path would report `n_agents` times
            the number the shared path reports, and `ent_coef` would silently
            weigh that much more; dividing recovers the per-agent quantity both
            the metric and the loss term are named for.
            """
            return entropy(log_std, low, high) / n_agents
    else:
        _apply = net.apply
        _init_params = net.init
        _entropy = entropy

    def _norm(obs):
        """Standardise an observation against the frozen reference statistics."""
        return (obs - obs_mean) / obs_std

    # The learner's side of the four float64 quantities that reach `_loss`.
    # Bound here, in Python, so that the default arm below IS `_norm`, `low` and
    # `high` themselves and not a demoted copy of them (module docstring).
    _cast = learner_cast(learner_dtype)
    if _cast is None:
        _norm_learner, _low_learner, _high_learner = _norm, low, high
    else:
        _mean_l, _std_l = _cast((obs_mean, obs_std))
        _low_learner, _high_learner = _cast((low, high))

        def _norm_learner(obs):
            """`_norm` at the learner's precision, on an already-cast batch."""
            return (obs - _mean_l) / _std_l

    def init(key, env_params):
        """Build the starting carry: ``(params, tx, opt_state, env_state, env_obs)``.

        `tx` is the optimiser itself and is returned rather than closed over, so
        that `iterate` stays a pure function of what it is handed.  The network
        is shaped on the observation of a single un-batched `reset`, which is
        why `params` carry no environment axis; the `n_envs` environments of the
        carry are then reset separately.
        """
        key_net, key_reset = jax.random.split(key)
        obs, _ = reset(key_reset, env_params)
        params = _init_params(key_net, _norm(obs))
        if _cast is not None:
            # flax's `Dense` stores kernels and biases in float32 already, but
            # `SharedActorCritic.log_std` comes from `nn.initializers.zeros`
            # with no dtype and is therefore float64 under x64 (measured).  One
            # float64 leaf in `params` is enough to promote `log_prob` and the
            # entropy term, so the cast is over the whole tree and not over a
            # list of names that would go stale when the network gains a
            # parameter.
            params = _cast(params)
        # `clip_by_global_norm` bounds one step; it does not bound the walk.
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(cfg.lr) if cfg.weight_decay == 0.0
            else optax.adamw(cfg.lr, weight_decay=cfg.weight_decay))
        keys = jax.random.split(key_reset, cfg.n_envs)
        env_obs, env_state = jax.vmap(reset, in_axes=(0, None))(keys, env_params)
        return params, tx, tx.init(params), env_state, env_obs

    def _act(params, obs, key):
        """One agent-batch of actions: ``(action, pre, logp, value)``.

        `action` is what the environment consumes, in its own `action_shape`;
        `pre` is the same draw before squashing, which is what the update
        re-scores, and `logp` is that draw's log-density under the current
        policy, summed over an agent's action coordinates.  `value` is the
        critic at `obs`.  Called under `vmap` over the environment axis, so
        `obs` here is one environment's ``(n_agents, obs_dim)``.
        """
        mean, log_std, value = _apply(params, _norm(obs))
        pre = sample_pre(key, mean, log_std)
        # back to the env's own action shape: a 1-D market gets its trailing
        # unit axis dropped, a 2-D market's reshape is the identity.  The
        # reshape only moves the axis boundary -- it never clips or rescales,
        # because the offer maps document that clipping an out-of-range action
        # is correcting the agent's action on its behalf.
        return (to_action(pre, low, high).reshape(act_shape), pre,
                log_prob(pre, mean, log_std, low, high), value)

    def _rollout(params, env_state, env_obs, key, env_params):
        """Collect `horizon` steps in `n_envs` environments under one `lax.scan`.

        The scan carries ``(env_state, obs, key)`` and emits one dictionary per
        step, so every leaf of `traj` has leading axes ``(horizon, n_envs, ...)``
        and the agent axis, where there is one, sits behind them.  `obs` in the
        carry is the observation the action was chosen from, and it is the one
        emitted -- the successor goes into the carry for the next step.

        Stepping goes through `step_auto_reset`, so an episode that ends inside
        the horizon is reset in place and the scan keeps its fixed length.  The
        carried key is split three ways per step -- one
        to carry on with, one for the policy and one for the environment -- and
        the last two are split again per environment, so no two environments and
        no two steps draw from the same key.

        Returns:
            ``(env_state, obs, key, traj, last_value)``.  The first three
            continue the rollout on the next iteration; `last_value` is the
            critic at the state the scan stopped in, which `_gae` bootstraps
            from.
        """
        def one(carry, _):
            state, obs, k = carry
            k, k_act, k_env = jax.random.split(k, 3)
            act_keys = jax.random.split(k_act, cfg.n_envs)
            action, pre, logp, value = jax.vmap(_act, in_axes=(None, 0, 0))(
                params, obs, act_keys)
            env_keys = jax.random.split(k_env, cfg.n_envs)
            nxt_obs, nxt_state, reward, costs, done, info = _step_envs(
                env_keys, state, action, env_params)
            out = dict(obs=obs, pre=pre, logp=logp, value=value, reward=reward,
                       done=done, costs=costs,
                       # absent, not defaulted, where the market declares no
                       # convergence flag; see `convergence_key` above
                       **({} if convergence_key is None else
                          {"converged": _require(info, convergence_key,
                                                 "convergence_key")}),
                       #: absent under `valid_key=None`, so `_update` below can
                       #: branch on the key rather than on a defaulted array
                       **({} if valid_key is None else
                          {"valid": _require(info, valid_key, "valid_key")}),
                       # a missing key is an error, not an empty series: a
                       # caller that misspells one would otherwise get a run
                       # with no record of the thing it asked to record.  The
                       # message lists what IS there, because the usual cause is
                       # a name that belongs to a different market's `info`
                       **{f"extra_{k}": _require(info, k) for k in extra_info_keys},
                       # carried for the per-step record rather than the update:
                       # a per-iteration reduction would miss transient behaviour
                       # a time series can catch.  `cursor` comes along because
                       # the provider set is the committed set at that period,
                       # and separation taken over all units instead conflates
                       # periods with different committed sets.
                       action=action, cursor=state.cursor)
            return (nxt_state, nxt_obs, k), out

        (state, obs, key), traj = jax.lax.scan(
            one, (env_state, env_obs, key), None, length=cfg.horizon)
        _, _, last_value = _apply(params, _norm(obs))
        return state, obs, key, traj, last_value

    def _gae(traj, last_value):
        """Per-agent GAE, bootstrapping at the boundary iff the market says to.

        **The criterion, not the convention:** bootstrap at the boundary if and
        only if the environment leaves a state quantity there unpriced.  Pricing
        it in the reward *and* bootstrapping counts it twice; doing neither wipes
        it.  `spec["termination"]` is the market's own answer, and this function
        branches on it in **Python**, at construction: the bootstrap arm is the
        expression it always was rather than that expression multiplied by a mask
        that happens to be one.  Markets 01, 02, 03 and 05 are therefore
        bit-identical across the arrival of this branch by construction, and not
        by a belief about how XLA folds a multiplication by 1.0.

        **For the four markets that declare `"truncation"` the answer is
        bootstrap**, because there is no discarded state quantity at the
        boundary: none of them carries storage, and the exogenous series, the
        commitment and the previous output all continue past the episode end.
        **Market 04 declares `"terminal"`**: `envs/p2p/env.py` settles the energy
        left in the battery with a terminal leg inside `reward`, so its boundary
        is masked here instead.  `delta` loses its continuation value and the
        recursion is cut, which makes the value target at that step the reward of
        that step alone -- ``ret = (r - V) + V = r``, for every `gamma` and every
        `gae_lambda`.  That identity is what
        `tests/learning/test_ippo_gae_terminal_l1.py` asserts, and it needs no
        GAE formula of its own to state.

        **Two things the measurement behind this branch corrected**, because the
        mechanism had been written down before it was measured
        (2026-08-27, CPU, on 04 at `episode_len=4`, `horizon=12`, untrained
        critic):

        * What used to be bootstrapped at a settled boundary is **not**
          `info["terminal_obs"]`.  `_rollout` emits the observation `step`
          returns, and that one is already past the auto-reset merge, so what
          entered `delta` was the first observation of the *next* episode -- a
          state holding `initial_soc`, not the stock just sold.  It was not the
          same energy priced twice; it was another episode's state value added on
          top of a boundary that had already been settled.  (`envs/p2p/env.py`
          guards the `terminal_obs` route against exactly this, and `_gae` never
          took that route.)
        * At a boundary inside the horizon the bootstrap is the **smaller** of
          the two errors.  Measured: `gamma * V` contributes mean|.| 4.84e-01
          against 1.40e+01 for the recursion carrying on across the boundary,
          i.e. 3.4% of the total; the bootstrap term is visible alone only at the
          last step of a rollout, where the recursion term is structurally zero
          (there, +4.96e-01 against a terminal leg of 7.72e+00, 6.4%).  The mask
          therefore multiplies `gae` as well as `next_value`; masking only the
          bootstrap would leave 96.6% of the defect in place.

        Returns:
            ``(adv, ret)``, both shaped like ``traj["value"]``: the advantage
            estimates and the value targets they imply, ``adv + value``.  The
            recursion is elementwise over the environment and agent axes, so an
            agent's advantage is built from its own reward and value series
            alone.
        """
        if bootstrap_at_done:
            def one(carry, x):
                """One backward step, carrying ``(gae, next_value)``.

                The scan runs in reverse, so `next_value` is the critic one step
                *later* in time; it enters as `last_value` and is then replaced
                by the current step's own `value` on the way back.
                """
                gae, next_value = carry
                reward, value = x["reward"], x["value"]
                delta = reward + cfg.gamma * next_value - value
                gae = delta + cfg.gamma * cfg.gae_lambda * gae
                return (gae, value), gae
        else:
            def one(carry, x):
                """The same backward step, with the settled boundary cut out.

                `cont` is zero on the step where `done` fired, which removes the
                continuation value from `delta` and stops the recursion from
                carrying the next episode's advantage back across the boundary.
                Both are needed; the second is the larger of the two (docstring
                above).
                """
                gae, next_value = carry
                reward, value, done = x["reward"], x["value"], x["done"]
                # `done` is one flag per environment and `value` is one number
                # per agent, so the trailing axis is added rather than left to
                # broadcast by accident: a market whose `done` already carried
                # the agent axis would otherwise silently produce an
                # (agents x agents) outer product here instead of failing.
                if done.ndim != value.ndim - 1:
                    raise ValueError(
                        f"`done` has shape {done.shape} and `value` has shape "
                        f"{value.shape}; the mask assumes one flag per "
                        f"environment against one value per agent, so `done` "
                        f"must have exactly one axis fewer")
                cont = 1.0 - done[..., None].astype(value.dtype)
                delta = reward + cfg.gamma * next_value * cont - value
                gae = delta + cfg.gamma * cfg.gae_lambda * cont * gae
                return (gae, value), gae

        (_, _), adv = jax.lax.scan(
            one, (jnp.zeros_like(last_value), last_value), traj,
            reverse=True)
        return adv, adv + traj["value"]

    def _loss(params, batch):
        """PPO's clipped surrogate plus the value and entropy terms.

        The stored `pre` is re-scored under the current parameters, so `ratio`
        is the importance weight against the behaviour policy that collected the
        batch.  Advantages are standardised within the minibatch, over the agent
        axis as well as the sample axis, so one shared scale serves every agent.
        The entropy term enters with a minus sign, so a positive `ent_coef`
        rewards a wider policy rather than penalising one.

        Returns:
            ``(total, aux)``, where `aux` carries the three terms separately
            plus `approx_kl` and `clip_frac` as diagnostics of how far the
            update moved.
        """
        mean, log_std, value = _apply(params, _norm_learner(batch["obs"]))
        logp = log_prob(batch["pre"], mean, log_std, _low_learner, _high_learner)
        ratio = jnp.exp(logp - batch["logp"])
        adv = batch["adv"]
        #: `w` is absent under `valid_key=None`, and then every line below is the
        #: unweighted one it was before.  Under a mask it is 1.0/0.0 broadcast
        #: over the agent axis; `wsum` is floored at one so an all-masked
        #: minibatch gives a zero gradient rather than a division by zero.
        w = batch.get("w")
        if w is None:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            unclipped = ratio * adv
            clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            pg = -jnp.minimum(unclipped, clipped).mean()
            vf = 0.5 * ((value - batch["ret"]) ** 2).mean()
        else:
            wsum = jnp.maximum(w.sum(), 1.0)
            #: **the standardisation is masked too.** A masked sample left in
            #: `adv.mean()` / `adv.std()` would move the scale every surviving
            #: sample is divided by, so excluding it from the loss and not from
            #: the moments would leak it back in.
            a_mean = (adv * w).sum() / wsum
            a_var = (((adv - a_mean) ** 2) * w).sum() / wsum
            adv = (adv - a_mean) / (jnp.sqrt(a_var) + 1e-8)
            unclipped = ratio * adv
            clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            pg = -(jnp.minimum(unclipped, clipped) * w).sum() / wsum
            vf = 0.5 * (((value - batch["ret"]) ** 2) * w).sum() / wsum
        ent = _entropy(log_std, _low_learner, _high_learner)
        total = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
        return total, dict(pg_loss=pg, vf_loss=vf, entropy=ent,
                           approx_kl=jnp.mean(batch["logp"] - logp),
                           clip_frac=jnp.mean(
                               jnp.abs(ratio - 1.0) > cfg.clip_eps))

    def _update(params, tx, opt_state, traj, adv, ret, key):
        """Run `epochs` passes of `minibatches` gradient steps over one rollout.

        Both loops are `lax.scan`, carrying ``(params, opt_state)``: the outer
        one over a fresh permutation key per epoch, the inner one over the
        minibatches that permutation produced.

        The sample axis is ``horizon * n_envs`` only.  The agent axis is left
        inside each sample by the reshape, so a minibatch draws whole
        environment-steps and an agent is never separated from the others it
        was cleared against.  `minibatches` must divide that product exactly:
        the floor division sizes a minibatch, and the reshape that follows it
        fails at trace time on any remainder.

        Returns:
            ``(params, opt_state, aux)``, with `aux` the `_loss` diagnostics
            averaged over every epoch and minibatch.
        """
        if "valid" in traj:
            #: reverse cumulative AND along the horizon: step `t` of an
            #: environment is usable only if every step from `t` to the end was.
            #: See `valid_key` in the docstring for why it is not the single step.
            ok = jnp.asarray(traj["valid"], jnp.bool_)               # (H, n_envs)
            ok = jnp.flip(jnp.cumprod(jnp.flip(ok.astype(jnp.int32), axis=0),
                                      axis=0), axis=0).astype(jnp.bool_)
            wts = ok.astype(adv.dtype)[..., None]                    # agent axis
            #: **replace, then weight.** `nan * 0.0` is `nan`, and the gradient
            #: through a `where` whose unselected branch is non-finite is also
            #: `nan`, so the bad numbers must not enter the graph at all.
            adv = jnp.where(wts > 0, adv, jnp.zeros_like(adv))
            ret = jnp.where(wts > 0, ret, jnp.zeros_like(ret))
        else:
            wts = None
        flat = dict(obs=traj["obs"], pre=traj["pre"], logp=traj["logp"],
                    adv=adv, ret=ret,
                    **({} if wts is None else
                       {"w": jnp.broadcast_to(wts, adv.shape)}))
        n = cfg.horizon * cfg.n_envs
        flat = jax.tree.map(lambda x: x.reshape((n,) + x.shape[2:]), flat)
        if _cast is not None:
            # the one place the rollout's precision meets the learner's; `traj`,
            # `adv` and `ret` keep theirs, so the recorded series and GAE are
            # what they were
            flat = _cast(flat)
        size = n // cfg.minibatches

        def epoch(carry, k):
            params, opt_state = carry
            order = jax.random.permutation(k, n)
            mb = jax.tree.map(
                lambda x: x[order].reshape((cfg.minibatches, size)
                                           + x.shape[1:]), flat)

            def one(carry, batch):
                params, opt_state = carry
                (_, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
                    params, batch)
                updates, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), aux

            return jax.lax.scan(one, (params, opt_state), mb)

        (params, opt_state), aux = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(key, cfg.epochs))
        aux = jax.tree.map(jnp.mean, aux)
        #: counted on the (step, env) grid, not on the agent axis: a sample is
        #: one environment-step, which is the unit the mask acts on
        aux["masked_samples"] = (jnp.asarray(0.0) if wts is None else
                                 (1.0 - wts[..., 0]).sum())
        return params, opt_state, aux

    def iterate(params, tx, opt_state, env_state, env_obs, key, env_params):
        """One PPO iteration: rollout, GAE, then the epoch and minibatch passes.

        Takes the carry `init` produced, plus a `key` and `env_params`, and
        returns that carry again without `tx`, which is handed back in unchanged
        on the next call.  The `key` that comes back is the remainder of this
        call's split, so a Python loop over iterations advances the stream by
        feeding the returned carry straight back in.

        Returns:
            ``(params, opt_state, env_state, env_obs, key, metrics)``.
            `metrics` mixes two kinds of entry and the prefix says which: plain
            names are reductions over the whole iteration, while ``step_*``
            names are per-step series shaped like the rollout, left unreduced
            for the caller to record.
        """
        key, k_roll, k_upd = jax.random.split(key, 3)
        env_state, env_obs, _, traj, last_value = _rollout(
            params, env_state, env_obs, k_roll, env_params)
        adv, ret = _gae(traj, last_value)
        params, opt_state, aux = _update(params, tx, opt_state, traj, adv, ret,
                                         k_upd)
        metrics = dict(aux)
        metrics.update(
            reward_mean=jnp.mean(traj["reward"]),
            reward_per_agent=jnp.mean(traj["reward"], axis=(0, 1)),
            costs_mean=jnp.mean(traj["costs"]),
            # per-step series, not reductions: the caller records them
            step_reward=traj["reward"],
            step_action=traj["action"],
            step_cursor=traj["cursor"],
            **{f"step_{k}": traj[f"extra_{k}"] for k in extra_info_keys},
            #: `unconverged_frac` is the share of cleared periods the market
            #: itself flags as not converged.  It is a metric and not an
            #: assertion: recorded rather than assumed to be rare.  Both entries
            #: are absent for a market that declares no convergence flag
            #: (`convergence_key=None`), so a caller that records them fails by
            #: name instead of recording a constant nobody measured.
            **({} if convergence_key is None else dict(
                step_converged=traj["converged"],
                unconverged_frac=1.0 - jnp.mean(traj["converged"].astype(
                    jnp.float64)))),
        )
        return params, opt_state, env_state, env_obs, key, metrics

    return init, iterate


def observation_statistics(env, env_params, key, n_envs: int, horizon: int,
                           env_chunks: int = 1):
    """Mean and standard deviation of the observation under the truthful action.

    Frozen for the whole run, for the reason the module docstring gives.  The
    standard deviation is floored so that a constant observation coordinate,
    which the static block of every market has several of, does not divide by
    zero and does not become an arbitrarily large input.

    **The returned statistics depend on the order in which this function
    derives its keys, so they equal an archived ``obs_mean``/``obs_std`` only
    while that order is unchanged.**  Two callers under `tools/ancillary/`
    take them from a checkpoint when one carries them and recompute them here
    when it does not, and those two branches are the same quantity only under
    that condition.  The comment beside the `lax.scan` says what the order
    currently is and why it stands.

    `env_chunks` is the same keyword `make_ippo` takes and does the same thing
    to the same step: this function is `n_envs` environments times a full
    horizon of clearings, taken before the first iteration, so it is the first
    place a case that does not fit the card at `n_envs` fails.  The keys are
    split exactly as before and the statistics are reductions over the joined
    ``(horizon, n_envs, ...)`` sample, so the key-order caveat above is
    untouched by the value of this keyword.
    """
    reset, _step, step_auto_reset, spec = env
    baseline = spec["baseline_action"]
    _step_envs = chunked_env_step(step_auto_reset, n_envs, env_chunks,
                                  action_batched=False)
    keys = jax.random.split(key, n_envs)
    obs, state = jax.vmap(reset, in_axes=(0, None))(keys, env_params)

    def one(carry, _):
        state, obs, k = carry
        k, k_env = jax.random.split(k)
        env_keys = jax.random.split(k_env, n_envs)
        nxt_obs, nxt_state, *_ = _step_envs(env_keys, state, baseline,
                                            env_params)
        # the pre-step observation, so the reset state is the first sample and
        # the state after the last step is not sampled at all
        return (nxt_state, nxt_obs, k), obs

    # `keys[0]` again: the scan's root key is the one environment 0 already
    # drew its start period from.  **This does not meet the explicit-PRNG rule
    # of this package** -- a key is used for two roles -- and it is left
    # standing deliberately rather than overlooked.  Splitting by role first
    # changes what this function returns: measured 2026-08-26 on the real-time
    # market at `n_envs=4, horizon=6`, `sum(obs_mean)` moves 3796.97 -> 3640.86
    # (`episode_len=48`) and 3499.57 -> 3485.46 (`episode_len=2`).  The
    # statistics standardise every observation a policy ever sees, and
    # two ancillary-market calibration drivers each take them from a
    # checkpoint when one carries them and recompute them here when it does
    # not; changing only the recomputed branch gives one quantity two values
    # inside one script, which is worse than leaving both old.  An input that
    # has already produced a constant in force is "change the value **and**
    # re-run", so the value is not changed without the re-run being decided.
    # The archived `obs_mean`/`obs_std` of the ancillary pilot were stored
    # against exactly this: one extra split before `k_stat` silently loses the
    # former, and no product turns red.
    _, seen = jax.lax.scan(one, (state, obs, keys[0]), None, length=horizon)
    # `seen` is (horizon, n_envs, n_agents, obs_dim) and the first three axes
    # are all pooled: one mean and one deviation per observation coordinate,
    # shared by every agent, which is the shape a shared policy standardises by
    mean = jnp.mean(seen, axis=(0, 1, 2))
    std = jnp.std(seen, axis=(0, 1, 2))
    return mean, jnp.where(std > 1e-8, std, 1.0)
