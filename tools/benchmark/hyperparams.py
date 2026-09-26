"""The one shared IPPO configuration, and where each number came from.

`IPPOConfig` requires all fourteen fields and refuses defaults on
purpose (its own docstring: "a default carried across markets is a calibration
nobody performed"), so the configuration a run adopts has to be written down
somewhere.  This is that place, shared by markets 01, 02 and 03.

**The claim this file exists to support is "not tuned per market."**  That claim
rests entirely on the provenance below being specific: "commonly used values"
would not be checkable, and an unfalsifiable provenance is the same as none.

**Source for eleven of the fourteen fields**: CleanRL `ppo_continuous_action.py`,
the `Args` dataclass and `Agent.__init__`, read from
<https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_continuous_action.py>
on 2026-08-18.  Read rather than recalled -- this repository has already paid
once for numbers written down from memory and presented as sourced.

    learning_rate 3e-4 | gamma 0.99 | gae_lambda 0.95 | clip_coef 0.2
    ent_coef 0.0 | vf_coef 0.5 | max_grad_norm 0.5 | update_epochs 10
    num_minibatches 32 | hidden two layers of 64, tanh
    orthogonal init: hidden sqrt(2), actor output 0.01, critic output 1.0

**Two fields are not CleanRL's, and one of those overrides it.**

`n_envs` and `horizon` -- CleanRL runs one environment for 2048 steps, which does
not map onto a `vmap`ped setting at all.  What is kept is the shape of the
decision, not the numbers: `n_envs = 64` is chosen for the device, and their
product 3072 is the batch, divisible by `minibatches = 32` into 96 --
`ippo._update` flattens `horizon * n_envs` and keeps the agent axis inside each
sample, so the batch is over (time x environment) and every agent of a sampled
step travels together.

**`horizon` is declared per market, because the thing it counts is per market.**
A step is a different object in each of the three, so:

  real-time    48 half-hourly steps  = exactly one episode; a rollout never
                                       straddles a day boundary mid-update
  ancillary    48 half-hourly steps  = exactly one episode, same reason
  day-ahead    D = 1, so one episode is ONE step (a whole 24-hour clearing),
                                       and one step costs three interior-point
                                       solves at T = 24 where real-time solves
                                       one period.  `horizon = 4` is FOUR
                                       episodes; the market is a contextual
                                       bandit, so "not straddling a boundary"
                                       has nothing to bound here.

    HORIZON = {"01": 4, "02": 48, "03": 48}

**This is a market definition, not a calibration.**  `horizon` is not one of the
fields the "not tuned per market" claim ranges over: that claim is about the
eleven CleanRL values and `init_scale`, every one of which is the same number in
all three markets.  `horizon` is a rollout shape, and a rollout of one 24-hour
clearing is not the same quantity as a rollout of one half-hour period -- naming
one number for all three would be declaring an equality that does not hold.
Timings likewise do not extrapolate between markets.  (The per-market reading of
this field was reported by [01 day-ahead], 2026-08-18; it became the definition
on 2026-09-07 by the author's ruling, the same move as writing the ancillary
reserve box into market 03's action space rather than carrying it as a flag.)

`init_scale` -- **CleanRL's 0.01 is deliberately not used.**  It is the actor
output layer's orthogonal scale, and `policy.SharedActorCritic` records a
measurement against it: in the ancillary market, with the observation
standardised, a scale of order one keeps every draw of the initial policy off
the unreliable side of the dual-separation boundary, **while both smaller and
larger scales land on it**.  0.01 is smaller.  So this is the one field where an
in-repository measurement beats the cited default.

    **And it is measured for market 03 only.**  Markets 01 and 02 have not been
    measured at it, so for them 1.0 is an assumption inherited from another
    market -- exactly the shape this repository calls a calibration nobody
    performed.  It is recorded here rather than quietly adopted.

The other two orthogonal scales (hidden `sqrt(2)`, critic output `1.0`) are
hardcoded in `SharedActorCritic` and already agree with CleanRL; they are not
configurable and are listed here only so the comparison is complete.

**Not carried over**: CleanRL anneals the learning rate (`anneal_lr=True`).
`IPPOConfig` has no such field, so these runs use a constant learning rate.  That
is a difference from the cited source and is stated rather than left implicit.
"""
import dataclasses as _dc

from powermarketjax.learning.ippo import IPPOConfig
from powermarketjax.learning.sac import SACConfig

#: Batch arithmetic, stated so it can be checked rather than trusted.
N_ENVS = 64
#: Per market, see the docstring: 02/03 one episode of 48 half-hours, 01 four
#: episodes of one 24-hour clearing each.
HORIZON = {"01": 4, "02": 48, "03": 48}
MINIBATCHES = 32
for _mkt, _h in HORIZON.items():
    assert (N_ENVS * _h) % MINIBATCHES == 0, (
        f"market {_mkt}: batch {N_ENVS * _h} is not divisible into "
        f"{MINIBATCHES} minibatches; `ippo._update` reshapes horizon*n_envs "
        f"and would drop the remainder")

SHARED = IPPOConfig(
    n_envs=N_ENVS,                # ours (device), not CleanRL's 1
    horizon=HORIZON["02"],        # ours, per market (see the docstring and
                                  # `shared_for`), not CleanRL's 2048
    epochs=10,                    # CleanRL update_epochs
    minibatches=MINIBATCHES,      # CleanRL num_minibatches
    lr=3e-4,                      # CleanRL learning_rate (no annealing here)
    clip_eps=0.2,                 # CleanRL clip_coef
    gamma=0.99,                   # CleanRL gamma
    gae_lambda=0.95,              # CleanRL gae_lambda
    vf_coef=0.5,                  # CleanRL vf_coef
    ent_coef=0.0,                 # CleanRL ent_coef
    max_grad_norm=0.5,            # CleanRL max_grad_norm
    hidden=(64, 64),              # CleanRL Agent: two tanh layers of 64
    # zero reproduces `optax.adam` exactly, so the three markets keep the
    # behaviour they were measured with until a run overrides it explicitly
    weight_decay=0.0,
    init_scale=1.0,               # NOT CleanRL's 0.01 -- see the docstring
)

def shared_for(market):
    """`SHARED` with `market`'s horizon.  The three drivers call this.

    `SHARED` and `SAC_SHARED` above carry `HORIZON["02"]`, which is also 03's,
    because every tool that imports them directly is a 02, an 03 or a
    market-neutral one; the day-ahead probes rebuild their configuration from
    the archive they read instead.  A driver must not read `SHARED.horizon`.
    """
    return _dc.replace(SHARED, horizon=HORIZON[market])


def sac_shared_for(market):
    """`SAC_SHARED` with `market`'s horizon; same rule as `shared_for`."""
    return _dc.replace(SAC_SHARED, horizon=HORIZON[market])


#: What a product must say about the configuration it ran under, so that "not
#: tuned per market" is checkable from the artifact alone.
PROVENANCE = {
    "source": ("CleanRL ppo_continuous_action.py, Args dataclass and Agent, "
               "fetched 2026-08-18"),
    "url": ("https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/"
            "cleanrl/ppo_continuous_action.py"),
    "from_source": ["epochs", "minibatches", "lr", "clip_eps", "gamma",
                    "gae_lambda", "vf_coef", "ent_coef", "max_grad_norm",
                    "hidden"],
    "ours": {"n_envs": "device choice; CleanRL runs 1 environment",
             "horizon": ("per market, {'01': 4, '02': 48, '03': 48}: one "
                         "episode of 48 half-hours in real-time and ancillary, "
                         "four one-clearing episodes in day-ahead (D=1). This "
                         "is a MARKET DEFINITION, not tuning -- a rollout of "
                         "one 24-hour clearing is not the same quantity as a "
                         "rollout of one half-hour period, so one number for "
                         "all three would declare an equality that does not "
                         "hold; `tuned_per_market` below stays False and "
                         "ranges over the eleven source values and init_scale, "
                         "each identical in all three. CleanRL uses 2048 "
                         "steps"),
             "weight_decay": ("0.0, not sourced from CleanRL's PPO args (it has "
                              "no such field). Added `fbd6189` (2026-08-19) as "
                              "a required field of `IPPOConfig`; 0.0 selects "
                              "`optax.adam` exactly, reproducing the behaviour "
                              "the three markets were measured with. A run "
                              "overrides it explicitly via "
                              "`--weight-decay`")},
    "overrides_source": {
        "init_scale": ("1.0, not CleanRL's 0.01: policy.SharedActorCritic "
                       "records that order-one keeps the initial policy off the "
                       "dual-separation boundary, and both smaller and larger "
                       "scales land on it. MEASURED FOR MARKET 03 ONLY -- for "
                       "01 and 02 this is inherited, not calibrated")},
    "not_carried_over": ("CleanRL anneals the learning rate; IPPOConfig has no "
                         "such field, so lr is constant here"),
    "tuned_per_market": False,
}


# ---------------------------------------------------------------------------
# SAC.  Same rule: one configuration for the three
# markets, every number with a source, differences from the source stated.
#
# **Source for eleven of the fifteen fields**: CleanRL `sac_continuous_action.py`,
# the `Args` dataclass, `Actor` and `SoftQNetwork`, read from
# <https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/sac_continuous_action.py>
# on 2026-09-03.  The count is the eleven named in `SAC_PROVENANCE`, which is
# also the eleven this line enumerates; it read "nine" until 2026-09-06, when
# a review counted them.
#
#     gamma 0.99 | tau 0.005 | batch_size 256 | policy_lr 3e-4 | q_lr 1e-3
#     alpha 0.2 with autotune (alpha's optimiser uses q_lr) | hidden 256, 256 ReLU
#     LOG_STD_MIN -5 | LOG_STD_MAX 2 | one gradient step per env-step
#
# **One of those eleven carries a different unit, and it stays in the source
# list because the number really is CleanRL's -- what differs is what one
# sample is.**  CleanRL draws `batch_size` transitions; `sac.py` draws
# `batch_size` env-steps and every agent's transition rides inside each one, so
# 256 here is 256 x 66 = 16 896 transitions per gradient step on the wholesale
# markets.  Moving the field to `ours` would misattribute the value 256, so the
# unit is stated beside it in `SAC_PROVENANCE["from_source_units"]` instead.
#
# **Ours**: `n_envs` and `horizon` are the IPPO batch (`N_ENVS`, `HORIZON`
# above, with the same per-market caveat about what a horizon means);
# `buffer_size` 32 768 env-steps is a device-memory choice (CleanRL's 1e6
# transitions of a 17-dimensional Hopper observation is 66 x 112 floats per
# env-step here, about 60 KiB with the successor; 32 768 of them is about
# 2 GiB); `reward_scale` is NOT a number in this file -- the driver fits it
# from the truthful rollout (`sac.reward_statistics`) and stamps the value
# into the products, and `make_sac` refuses the sentinel below.
#
# **Not carried over**: `learning_starts` 5e3 random steps (one market step is
# one clearing; the buffer is not warmed with random bids); `policy_frequency`
# 2 (actor and critic step at the same frequency); PyTorch's default
# initialiser (flax's is used).
# ---------------------------------------------------------------------------
SAC_SHARED = SACConfig(
    n_envs=N_ENVS,                # ours, as for IPPO
    horizon=HORIZON["02"],        # ours, per market, as for IPPO
    buffer_size=32_768,           # ours (device memory); CleanRL 1e6
    batch_size=256,               # CleanRL batch_size
    utd_ratio=1.0,                # CleanRL: one update per env-step
    gamma=0.99,                   # CleanRL gamma
    tau=0.005,                    # CleanRL tau
    policy_lr=3e-4,               # CleanRL policy_lr
    q_lr=1e-3,                    # CleanRL q_lr
    alpha_lr=1e-3,                # CleanRL: the temperature uses q_lr
    init_alpha=0.2,               # CleanRL alpha (then autotuned)
    hidden=(256, 256),            # CleanRL Actor / SoftQNetwork
    log_std_min=-5.0,             # CleanRL LOG_STD_MIN
    log_std_max=2.0,              # CleanRL LOG_STD_MAX
    reward_scale=float("nan"),    # sentinel: the driver MUST replace it
)

SAC_PROVENANCE = {
    "source": ("CleanRL sac_continuous_action.py, Args dataclass, Actor and "
               "SoftQNetwork, fetched 2026-09-03"),
    "url": ("https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/"
            "cleanrl/sac_continuous_action.py"),
    "from_source": ["batch_size", "utd_ratio", "gamma", "tau", "policy_lr",
                    "q_lr", "alpha_lr", "init_alpha", "hidden", "log_std_min",
                    "log_std_max"],
    "ours": {"n_envs": "the IPPO batch; CleanRL runs 1 environment",
             "horizon": ("the IPPO batch, per market as in PROVENANCE: a "
                         "market definition, not tuning"),
             "buffer_size": "32768 env-steps, device memory; CleanRL 1e6",
             "reward_scale": ("fitted by the driver from the truthful rollout "
                              "(sac.reward_statistics) and stamped into the "
                              "products; not a number in hyperparams.py")},
    "from_source_units": {
        "batch_size": ("256 env-steps, each carrying every agent's transition "
                       "(66 on the wholesale markets, i.e. 16 896 transitions "
                       "per gradient step); CleanRL draws 256 transitions"),
    },
    "not_carried_over": ("learning_starts 5e3 random steps (none here); "
                         "policy_frequency 2 (actor and critic at the same "
                         "frequency); PyTorch default initialiser (flax's "
                         "default is used); the temperature's log-probability "
                         "is the one the actor step sampled, before that step "
                         "was applied, where CleanRL recomputes it under the "
                         "updated actor (a half-step lag, not an accumulating "
                         "one)"),
    "tuned_per_market": False,
}
