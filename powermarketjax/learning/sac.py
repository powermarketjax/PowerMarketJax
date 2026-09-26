"""SAC: off-policy actor-critic with a replay buffer that lives inside the `jit`.

**Why a second learner exists.**  The baseline set gained
two SAC columns, shared parameters and one network per agent, on the
condition "test it thoroughly first, wire it in only if it works".  This module is
the apparatus for that test.  It mirrors `ippo.py` where the two algorithms make
the same decision and departs from it only where SAC itself does:

* **Independent learners, no centralised critic.**  Every
  agent's Q function reads its own observation and its own action; the agent
  axis is a batch axis on the shared path and a `vmap` axis on the per-agent
  path, exactly as `ippo._apply_per_agent` arranges it.
* **`per_agent_params` is a keyword of `make_sac`, not a config field**, for the
  reason `ippo.py` gives: the drivers write `vars(cfg)` into every product.
  The automatic temperature is one scalar on the shared path and one per agent
  on the per-agent path, because each independent learner has its own entropy
  target to meet.
* **The whole iteration runs inside one `jit`.**  The rollout is a `lax.scan`
  over the horizon with `vmap` over environments; the replay buffer is a
  fixed-size pytree carried between iterations (FIFO cursor, uniform sampling
  over the filled prefix); the gradient steps are a `lax.scan`.  A host-side
  buffer would put a Python loop on the rollout path, which the all-in-`jit`
  design forbids.
* **The episode boundary branches on `spec["termination"]`** through the same
  table `ippo._gae` uses: bootstrap at a truncated boundary, mask a settled one.
* **Rewards enter the critic divided by a frozen scale.**  SAC regresses Q on
  the raw reward, and on market 01 one step's profit is of order 1e5 to 1e6
  dollars, where IPPO's standardised advantages made it insensitive.  The scale
  is the pooled standard deviation of the per-agent reward under the truthful
  action (`reward_statistics`), fitted once and frozen, like the observation
  statistics -- a quantity derived by the apparatus, not a knob.

**Hyperparameter provenance.**  CleanRL `sac_continuous_action.py`, read from
GitHub master on 2026-09-03: `gamma` 0.99, `tau` 0.005, `batch_size` 256,
`policy_lr` 3e-4, `q_lr` 1e-3 (also used for the temperature), initial `alpha`
0.2 with autotuning to a target entropy of minus the action dimension, two
hidden layers of 256 ReLU units in both actor and critic, `log_std` squashed
through `tanh` into `[-5, 2]`.  What is NOT carried over, each stated in
`SACConfig` and in the driver's provenance stamp: a 1e6-transition buffer
(sized here in env-steps against device memory), 5e3 random warm-up steps
(one market step is one clearing; the buffer is not warmed with random bids),
and `policy_frequency = 2` (actor and critic step at the same frequency here).
Initialisers are flax's defaults rather than PyTorch's, which is a difference
in the initial weights and is stated rather than hidden.

**`learner_dtype` demotes the gradient step and nothing else**, exactly as
`ippo.make_ippo` does and through the same `ippo.learner_cast`; that docstring
carries the reason.  SAC's own float64 leaf is `log_alpha`
(`jnp.log(cfg.init_alpha)` is float64 under x64), and its own float64 batch is
the replay sample on a market that rolls out in float64 -- the buffer keeps the
rollout's precision, leaf by leaf (`obs` at the observation's dtype, `pre` and
`reward` at the dtype the actor and the market actually produce), and the cast
happens where a sample leaves it.  Until 2026-09-18 `pre` and `reward` were
sized on the action box instead, which is float64 under x64 whatever the
market's precision; on the float32 markets that widened the replayed `pre` and
so ran the two critics' replay pass in float64.

**The entropy terms of PPO and SAC are not the same quantity.**  PPO's
`ent_coef` weighs a closed-form surrogate of the policy entropy; SAC's `alpha`
weighs `-log pi` of the sampled action and moves under autotuning.  Aligning
one coefficient does not align the two learners (the lesson of market 04), so
SAC is compared with IPPO on market outcomes and never on training diagnostics.
"""
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from .ippo import (_BOOTSTRAP_AT_DONE, _require, action_layout,
                   learner_cast)
from .policy import _squashable, log_prob, to_action


@dataclass(frozen=True)
class SACConfig:
    """Every field is required; a default carried across markets is a calibration
    nobody performed (the same rule as `IPPOConfig`)."""

    n_envs: int
    horizon: int
    #: replay capacity in env-steps; one env-step holds every agent's transition
    buffer_size: int
    #: env-steps per gradient step; the agent axis rides inside each sample
    batch_size: int
    #: gradient steps per env-step collected, so `utd_ratio * n_envs * horizon`
    #: updates per iteration.  CleanRL takes one update per env-step (1.0).
    utd_ratio: float
    gamma: float
    #: Polyak coefficient of the target networks
    tau: float
    policy_lr: float
    q_lr: float
    alpha_lr: float
    #: initial temperature; it moves under autotuning from the first update
    init_alpha: float
    hidden: Sequence[int]
    log_std_min: float
    log_std_max: float
    #: the critic sees `reward / reward_scale`; `reward_statistics` gives it
    reward_scale: float


class SACActor(nn.Module):
    """Gaussian policy with a state-dependent, squashed `log_std`.

    Unlike `SharedActorCritic`, whose `log_std` is a free parameter, SAC's
    exploration scale is a head of the network (CleanRL's `Actor`), and it is
    squashed through `tanh` into `[log_std_min, log_std_max]` so no state can
    ask for a zero or an unbounded standard deviation.  `mean` is the pre-squash
    location; `policy.to_action` maps it into the market's box.
    """

    act_dim: int
    hidden: Sequence[int]
    log_std_min: float
    log_std_max: float

    @nn.compact
    def __call__(self, obs):
        x = obs
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        mean = nn.Dense(self.act_dim)(x)
        log_std = jnp.tanh(nn.Dense(self.act_dim)(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1.0)
        return mean, log_std


class SoftQ(nn.Module):
    """One Q head over ``concat(obs, action)``; the trailing unit axis is dropped."""

    hidden: Sequence[int]

    @nn.compact
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(1)(x)[..., 0]


def _q_action(pre, low, high):
    """What the critic sees of an action: `tanh(pre)` on bounded coordinates,
    `pre` itself on unbounded ones.

    The critic is fed the squashed coordinate in `(-1, 1)` rather than the
    market's own units, so its input scale does not depend on `markup_max`; an
    unbounded coordinate has no squash and is passed through, as `to_action`
    passes it through.  The ancillary reserve columns used to be the one such
    coordinate and no longer are: that market publishes their box, so on the five markets as they stand this branch is unreached.
    It is kept because which coordinates a market bounds is the market's to
    say, and a critic that assumed all of them were would be reading `tanh` of
    a number that never went through one.
    """
    return jnp.where(_squashable(low, high), jnp.tanh(pre), pre)


def _actor_per_agent(net, params, z):
    """`SACActor.apply` with the agent axis of `z` mapped against the parameters."""
    zz = jnp.moveaxis(z, -2, 0)
    mean, log_std = jax.vmap(net.apply)(params, zz)
    return jnp.moveaxis(mean, 0, -2), jnp.moveaxis(log_std, 0, -2)


def _q_per_agent(net, params, z, a):
    """`SoftQ.apply` with the agent axis of `z` and `a` mapped against the parameters."""
    q = jax.vmap(net.apply)(params, jnp.moveaxis(z, -2, 0), jnp.moveaxis(a, -2, 0))
    return jnp.moveaxis(q, 0, -1)


def _nets(spec, bounds, cfg):
    n_agents, act_dim, act_shape, low, high = action_layout(spec, bounds)
    actor = SACActor(act_dim=act_dim, hidden=tuple(cfg.hidden),
                     log_std_min=cfg.log_std_min, log_std_max=cfg.log_std_max)
    q = SoftQ(hidden=tuple(cfg.hidden))
    return n_agents, act_dim, act_shape, low, high, actor, q


def make_sac_greedy_action(spec: dict,
                           bounds: Tuple[jnp.ndarray, jnp.ndarray],
                           cfg: SACConfig,
                           obs_mean: jnp.ndarray,
                           obs_std: jnp.ndarray,
                           per_agent_params: bool = False) -> Callable:
    """`(params, obs) -> action`: the actor's mean, in the market's action shape.

    `params` is the full learner pytree `make_sac` returns; only `params["actor"]`
    is read.  Deterministic, as `ippo.make_greedy_action` is, so evaluation
    reports the policy and not the policy plus its exploration noise.
    """
    _n, _d, act_shape, low, high, actor, _q = _nets(spec, bounds, cfg)
    _apply = (partial(_actor_per_agent, actor) if per_agent_params
              else actor.apply)

    def greedy(params, obs):
        mean, _log_std = _apply(params["actor"], (obs - obs_mean) / obs_std)
        return to_action(mean, low, high).reshape(act_shape)

    return greedy


def make_sac(env: Tuple[Callable, Callable, Callable, dict],
             bounds: Tuple[jnp.ndarray, jnp.ndarray],
             cfg: SACConfig,
             obs_mean: jnp.ndarray,
             obs_std: jnp.ndarray,
             extra_info_keys: Sequence[str] = (),
             per_agent_params: bool = False,
             convergence_key: Optional[str] = "converged",
             learner_dtype: Optional[str] = None):
    """Build `(init, iterate)` for one market, with the carry `make_ippo` uses.

    `init(key, env_params)` returns ``(params, tx, learner, env_state, env_obs)``
    and `iterate(params, tx, learner, env_state, env_obs, key, env_params)`
    returns ``(params, learner, env_state, env_obs, key, metrics)``, so a driver
    written against `make_ippo` drives this one through the same positions.
    `params` holds the actor, both critics, both targets and `log_alpha`;
    `learner` holds the three optimiser states and the replay buffer; `tx` is
    the static triple of optimisers.

    `extra_info_keys`, `convergence_key` and `learner_dtype` mean what they mean
    in `make_ippo`, and for the same reasons; `None` is the path every archive
    in the repository was produced on.
    """
    reset, _step, step_auto_reset, spec = env
    if "termination" not in spec:
        raise KeyError(
            "spec has no 'termination' field, so nothing says whether this "
            "market's episode boundary is settled or merely cut; the keys "
            f"present are {sorted(spec)}")
    if spec["termination"] not in _BOOTSTRAP_AT_DONE:
        raise ValueError(
            f"spec['termination']={spec['termination']!r} is not one of "
            f"{sorted(_BOOTSTRAP_AT_DONE)}")
    bootstrap_at_done = _BOOTSTRAP_AT_DONE[spec["termination"]]
    if not (cfg.reward_scale > 0.0) or not jnp.isfinite(cfg.reward_scale):
        # `nan` is the sentinel `hyperparams.SAC_SHARED` ships with: the scale
        # is fitted per run, and a run that forgot to fit it must not train
        # on `reward / nan`
        raise ValueError(f"reward_scale={cfg.reward_scale!r} must be a finite "
                         f"positive number; fit it with `reward_statistics`")
    n_agents, act_dim, act_shape, low, high, actor, qnet = _nets(spec, bounds, cfg)
    per_iter = cfg.n_envs * cfg.horizon
    if per_iter > cfg.buffer_size:
        raise ValueError(f"one iteration collects {per_iter} env-steps but the "
                         f"buffer holds {cfg.buffer_size}; the FIFO write would "
                         f"overwrite this iteration's own transitions")
    n_updates = int(round(cfg.utd_ratio * per_iter))
    if n_updates < 1:
        raise ValueError(f"utd_ratio {cfg.utd_ratio} gives {n_updates} updates "
                         f"per iteration")
    #: CleanRL: minus the action dimension, per agent
    target_entropy = -float(act_dim)

    if per_agent_params:
        _actor = partial(_actor_per_agent, actor)
        _q = partial(_q_per_agent, qnet)

        def _init_nets(key, z):
            """One actor and two critics per agent, from split keys; `z` is
            ``(n_agents, obs_dim)`` and shapes each agent on its own row."""
            ka, k1, k2 = jax.random.split(key, 3)
            a0 = jnp.zeros((n_agents, act_dim), z.dtype)
            return dict(
                actor=jax.vmap(actor.init)(jax.random.split(ka, n_agents), z),
                q1=jax.vmap(qnet.init)(jax.random.split(k1, n_agents), z, a0),
                q2=jax.vmap(qnet.init)(jax.random.split(k2, n_agents), z, a0),
                log_alpha=jnp.full((n_agents,), jnp.log(cfg.init_alpha)))

        def _alpha_loss(log_alpha, logp):
            """Each agent's temperature answers to its own entropy: the batch
            mean is taken per agent and the agents are summed, so the gradient
            on `log_alpha[j]` sees agent `j`'s samples only."""
            per_agent = jnp.mean(logp + target_entropy, axis=0)
            return jnp.sum(-jnp.exp(log_alpha) * per_agent)
    else:
        _actor = actor.apply
        _q = qnet.apply

        def _init_nets(key, z):
            ka, k1, k2 = jax.random.split(key, 3)
            a0 = jnp.zeros(z.shape[:-1] + (act_dim,), z.dtype)
            return dict(actor=actor.init(ka, z), q1=qnet.init(k1, z, a0),
                        q2=qnet.init(k2, z, a0),
                        log_alpha=jnp.asarray(jnp.log(cfg.init_alpha)))

        def _alpha_loss(log_alpha, logp):
            return -jnp.exp(log_alpha) * jnp.mean(logp + target_entropy)

    def _norm(obs):
        return (obs - obs_mean) / obs_std

    # The learner's side of the float64 quantities that reach the two losses:
    # the observation statistics, the action box (`_q_action` and `log_prob`
    # both read it) and the replay sample.  Bound here, in Python, so the
    # default arm below IS `_norm`, `low` and `high` themselves.
    _cast = learner_cast(learner_dtype)
    if _cast is None:
        _norm_learner, _low_learner, _high_learner = _norm, low, high
    else:
        _mean_l, _std_l = _cast((obs_mean, obs_std))
        _low_learner, _high_learner = _cast((low, high))

        def _norm_learner(obs):
            """`_norm` at the learner's precision, on an already-cast batch."""
            return (obs - _mean_l) / _std_l

    def _alpha_of(params):
        """`alpha` broadcast against a ``(..., n_agents)`` `logp`."""
        return jnp.exp(params["log_alpha"])

    def init(key, env_params):
        """The starting carry: ``(params, tx, learner, env_state, env_obs)``."""
        key_net, key_reset = jax.random.split(key)
        obs, _ = reset(key_reset, env_params)
        z = _norm(obs)
        nets = _init_nets(key_net, z)
        params = dict(nets, q1_target=nets["q1"], q2_target=nets["q2"])
        if _cast is not None:
            # Before the optimisers are built, so their moments follow.  The
            # networks are already float32 (flax's default `param_dtype`), but
            # `log_alpha` is float64 -- `jnp.log(cfg.init_alpha)` of a Python
            # float under x64 -- and `alpha` multiplies `logp` inside both
            # losses, which is enough to promote the whole update back.
            params = _cast(params)
        tx = (optax.adam(cfg.policy_lr), optax.adam(cfg.q_lr),
              optax.adam(cfg.alpha_lr))
        keys = jax.random.split(key_reset, cfg.n_envs)
        env_obs, env_state = jax.vmap(reset, in_axes=(0, None))(keys, env_params)
        # the buffer's leaves are shaped on one env-step, i.e. on `obs` of a
        # single un-batched reset; the leading axis is the capacity.
        #
        # `pre` and `reward` take the dtype the ROLLOUT writes into them, read
        # off an abstract evaluation of one `_act` + one `step_auto_reset` on
        # this market, not `jnp.result_type(low)`.  The box is float64 under
        # x64 while a market whose observation is float32 (day-ahead,
        # real-time) samples a float32 `pre` and pays a float32 reward; sizing
        # the leaves on the box widened both exactly on the FIFO write and, at
        # replay, promoted the critic's forward and backward pass on
        # `batch["pre"]` to float64 (`_q_action` of a float64 `pre`,
        # concatenated with a float32 `z`).  Measured 2026-09-18 (RTX
        # 4500 Ada, 813nem market 02): that promotion was 37% of the gradient
        # step's FLOP and about 80% of its time, 96.3 -> 17.2 ms per step once
        # removed.  A market that rolls out in float64 (ancillary) gets the
        # same float64 buffer from this probe as before, bit for bit.
        _, state0 = reset(key_reset, env_params)

        def _written(p, o, s, k):
            action, pre = _act(p, o, k)
            _o, _s, reward, *_rest = step_auto_reset(k, s, action, env_params)
            return pre, reward

        pre_s, reward_s = jax.eval_shape(_written, params, obs, state0, key_reset)
        B = cfg.buffer_size
        buffer = dict(
            obs=jnp.zeros((B,) + obs.shape, obs.dtype),
            pre=jnp.zeros((B, n_agents, act_dim), pre_s.dtype),
            reward=jnp.zeros((B, n_agents), reward_s.dtype),
            next_obs=jnp.zeros((B,) + obs.shape, obs.dtype),
            done=jnp.zeros((B,), jnp.bool_),
            cursor=jnp.asarray(0, jnp.int32),
            filled=jnp.asarray(0, jnp.int32))
        learner = dict(opt_actor=tx[0].init(params["actor"]),
                       opt_q=tx[1].init((params["q1"], params["q2"])),
                       opt_alpha=tx[2].init(params["log_alpha"]),
                       buffer=buffer)
        return params, tx, learner, env_state, env_obs

    def _act(params, obs, key):
        """One environment's sampled action: ``(action, pre)``."""
        mean, log_std = _actor(params["actor"], _norm(obs))
        pre = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape,
                                                          mean.dtype)
        return to_action(pre, low, high).reshape(act_shape), pre

    def _rollout(params, env_state, env_obs, key, env_params):
        """`horizon` steps in `n_envs` environments; emits one transition per step.

        Unlike `ippo._rollout`, the successor observation is emitted alongside
        the observation the action was chosen from, because the critic's target
        needs it and a replay sample has no neighbour in time to read it from.
        """
        def one(carry, _):
            state, obs, k = carry
            k, k_act, k_env = jax.random.split(k, 3)
            act_keys = jax.random.split(k_act, cfg.n_envs)
            action, pre = jax.vmap(_act, in_axes=(None, 0, 0))(params, obs,
                                                               act_keys)
            env_keys = jax.random.split(k_env, cfg.n_envs)
            nxt_obs, nxt_state, reward, costs, done, info = jax.vmap(
                step_auto_reset, in_axes=(0, 0, 0, None))(
                    env_keys, state, action, env_params)
            out = dict(obs=obs, pre=pre, reward=reward, next_obs=nxt_obs,
                       done=done, costs=costs, action=action,
                       cursor=state.cursor,
                       **({} if convergence_key is None else
                          {"converged": _require(info, convergence_key,
                                                 "convergence_key")}),
                       **{f"extra_{k}": _require(info, k) for k in extra_info_keys})
            return (nxt_state, nxt_obs, k), out

        (state, obs, key), traj = jax.lax.scan(
            one, (env_state, env_obs, key), None, length=cfg.horizon)
        return state, obs, key, traj

    def _push(buffer, traj):
        """FIFO write of one rollout's ``horizon * n_envs`` env-steps."""
        n = per_iter
        idx = (buffer["cursor"] + jnp.arange(n)) % cfg.buffer_size
        flat = lambda x: x.reshape((n,) + x.shape[2:])
        out = dict(buffer)
        for k in ("obs", "pre", "reward", "next_obs", "done"):
            out[k] = buffer[k].at[idx].set(flat(traj[k]))
        out["cursor"] = (buffer["cursor"] + n) % cfg.buffer_size
        out["filled"] = jnp.minimum(buffer["filled"] + n, cfg.buffer_size)
        return out

    def _sample(buffer, key):
        idx = jax.random.randint(key, (cfg.batch_size,), 0, buffer["filled"])
        batch = {k: buffer[k][idx] for k in ("obs", "pre", "reward", "next_obs",
                                             "done")}
        # the one place the rollout's precision meets the learner's; the buffer
        # itself keeps the environment's, because the rollout writes it
        return batch if _cast is None else _cast(batch)

    def _cont(done, like):
        """The continuation mask, branched in Python on the boundary rule."""
        if bootstrap_at_done:
            return jnp.ones_like(like)
        if done.ndim != like.ndim - 1:
            raise ValueError(f"`done` has shape {done.shape} against a "
                             f"per-agent value of shape {like.shape}")
        return 1.0 - done[..., None].astype(like.dtype)

    def _q_loss(q_params, params, batch, key):
        """Twin-Q regression on the soft Bellman target."""
        q1, q2 = q_params
        z, z_next = _norm_learner(batch["obs"]), _norm_learner(batch["next_obs"])
        mean, log_std = _actor(params["actor"], z_next)
        pre_next = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape,
                                                               mean.dtype)
        logp_next = log_prob(pre_next, mean, log_std, _low_learner, _high_learner)
        a_next = _q_action(pre_next, _low_learner, _high_learner)
        q_next = jnp.minimum(_q(params["q1_target"], z_next, a_next),
                             _q(params["q2_target"], z_next, a_next))
        alpha = _alpha_of(params)
        target = (batch["reward"] / cfg.reward_scale
                  + cfg.gamma * _cont(batch["done"], q_next)
                  * (q_next - alpha * logp_next))
        target = jax.lax.stop_gradient(target)
        a = _q_action(batch["pre"], _low_learner, _high_learner)
        q1_pred, q2_pred = _q(q1, z, a), _q(q2, z, a)
        loss = jnp.mean((q1_pred - target) ** 2) + jnp.mean((q2_pred - target) ** 2)
        return loss, dict(q_loss=loss, q_mean=jnp.mean(q1_pred),
                          target_mean=jnp.mean(target))

    def _actor_loss(actor_params, params, batch, key):
        z = _norm_learner(batch["obs"])
        mean, log_std = _actor(actor_params, z)
        pre = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape,
                                                          mean.dtype)
        logp = log_prob(pre, mean, log_std, _low_learner, _high_learner)
        a = _q_action(pre, _low_learner, _high_learner)
        q = jnp.minimum(_q(params["q1"], z, a), _q(params["q2"], z, a))
        alpha = jax.lax.stop_gradient(_alpha_of(params))
        loss = jnp.mean(alpha * logp - q)
        return loss, dict(actor_loss=loss, logp=jax.lax.stop_gradient(logp))

    def _update(params, tx, learner, key):
        tx_actor, tx_q, tx_alpha = tx

        def one(carry, k):
            params, opt_actor, opt_q, opt_alpha = carry
            k_batch, k_q, k_pi = jax.random.split(k, 3)
            batch = _sample(learner["buffer"], k_batch)
            # critic
            (_, q_aux), g_q = jax.value_and_grad(_q_loss, has_aux=True)(
                (params["q1"], params["q2"]), params, batch, k_q)
            upd, opt_q = tx_q.update(g_q, opt_q, (params["q1"], params["q2"]))
            q1, q2 = optax.apply_updates((params["q1"], params["q2"]), upd)
            params = dict(params, q1=q1, q2=q2)
            # actor, against the freshly updated critics
            (_, pi_aux), g_pi = jax.value_and_grad(_actor_loss, has_aux=True)(
                params["actor"], params, batch, k_pi)
            upd, opt_actor = tx_actor.update(g_pi, opt_actor, params["actor"])
            params = dict(params, actor=optax.apply_updates(params["actor"], upd))
            # temperature, on the log-probabilities the actor step sampled
            a_loss, g_a = jax.value_and_grad(_alpha_loss)(params["log_alpha"],
                                                          pi_aux["logp"])
            upd, opt_alpha = tx_alpha.update(g_a, opt_alpha, params["log_alpha"])
            params = dict(params, log_alpha=optax.apply_updates(
                params["log_alpha"], upd))
            # targets
            polyak = lambda t, s: (1.0 - cfg.tau) * t + cfg.tau * s
            params = dict(params,
                          q1_target=jax.tree.map(polyak, params["q1_target"],
                                                 params["q1"]),
                          q2_target=jax.tree.map(polyak, params["q2_target"],
                                                 params["q2"]))
            aux = dict(q_loss=q_aux["q_loss"], q_mean=q_aux["q_mean"],
                       target_mean=q_aux["target_mean"],
                       actor_loss=pi_aux["actor_loss"], alpha_loss=a_loss,
                       # `-logp` is the sampled entropy, per agent
                       entropy=-jnp.mean(pi_aux["logp"]),
                       alpha=jnp.mean(jnp.exp(params["log_alpha"])))
            return (params, opt_actor, opt_q, opt_alpha), aux

        (params, opt_actor, opt_q, opt_alpha), aux = jax.lax.scan(
            one, (params, learner["opt_actor"], learner["opt_q"],
                  learner["opt_alpha"]), jax.random.split(key, n_updates))
        learner = dict(learner, opt_actor=opt_actor, opt_q=opt_q,
                       opt_alpha=opt_alpha)
        return params, learner, jax.tree.map(jnp.mean, aux)

    def iterate(params, tx, learner, env_state, env_obs, key, env_params):
        """One SAC iteration: rollout, push to replay, `n_updates` gradient steps.

        Returns ``(params, learner, env_state, env_obs, key, metrics)``; the
        `step_*` entries are per-step series left unreduced for the driver, the
        rest are reductions, as in `ippo.iterate`.
        """
        key, k_roll, k_upd = jax.random.split(key, 3)
        env_state, env_obs, _, traj = _rollout(params, env_state, env_obs,
                                               k_roll, env_params)
        learner = dict(learner, buffer=_push(learner["buffer"], traj))
        params, learner, aux = _update(params, tx, learner, k_upd)
        metrics = dict(aux)
        metrics.update(
            reward_mean=jnp.mean(traj["reward"]),
            reward_per_agent=jnp.mean(traj["reward"], axis=(0, 1)),
            costs_mean=jnp.mean(traj["costs"]),
            buffer_filled=learner["buffer"]["filled"],
            step_reward=traj["reward"],
            step_action=traj["action"],
            step_cursor=traj["cursor"],
            **{f"step_{k}": traj[f"extra_{k}"] for k in extra_info_keys},
            **({} if convergence_key is None else dict(
                step_converged=traj["converged"],
                unconverged_frac=1.0 - jnp.mean(traj["converged"].astype(
                    jnp.float64)))),
        )
        return params, learner, env_state, env_obs, key, metrics

    return init, iterate


def reward_statistics(env, env_params, key, n_envs: int, horizon: int):
    """Pooled standard deviation of the per-agent reward under the truthful action.

    One scalar for every agent, so dividing by it preserves the ranking of
    profits across agents; floored at one so a market whose truthful reward is
    constant does not divide by zero.  The key discipline is the one
    `ippo.observation_statistics` documents, and the same rollout structure is
    used so the two statistics come from the same kind of sample.
    """
    reset, _step, step_auto_reset, spec = env
    baseline = spec["baseline_action"]
    keys = jax.random.split(key, n_envs)
    obs, state = jax.vmap(reset, in_axes=(0, None))(keys, env_params)

    def one(carry, _):
        state, k = carry
        k, k_env = jax.random.split(k)
        env_keys = jax.random.split(k_env, n_envs)
        _o, nxt_state, reward, *_ = jax.vmap(
            step_auto_reset, in_axes=(0, 0, None, None))(
                env_keys, state, baseline, env_params)
        return (nxt_state, k), reward

    _, seen = jax.lax.scan(one, (state, keys[0]), None, length=horizon)
    std = jnp.std(seen)
    return jnp.where(std > 1e-8, std, 1.0)
