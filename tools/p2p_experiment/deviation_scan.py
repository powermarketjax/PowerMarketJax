"""Excess profit of a single deviator against a truthful population, by size.

Section 11 of the specification states the asymptotic property the uniform-price
double auction rests on: in a market with $m$ buyers and $m$ sellers, the
misreporting of any equilibrium is of order $1/m$ and the inefficiency it causes
of order $1/m^2$.  Section 19 asks for the measurement that decides how much the
price dimension carries at a given population, namely the gain a single
participant obtains by deviating from the truthful submission of section 4 while
every other participant submits truthfully.  This file produces that gain over a
grid of population sizes, so that the measured decay can be read against the
$1/m$ reference rather than assumed to follow it.

The deviation is a one-parameter family that contains the truthful submission.
A participant's truthful price is the export price when its position is a
surplus and the retail tariff when it is a deficit, which the affine price map
of the action module reaches at ``alpha_price`` equal to -1 and +1.  Writing the
side as $\\sigma \\in \\{-1, +1\\}$, the family submits
``alpha_price`` $= \\sigma (1 - 2\\theta)$, so $\\theta = 0$ reproduces the
truthful submission exactly and $\\theta = 1$ submits the opposite end of the
tariff bracket.  A seller therefore asks above its opportunity cost and a buyer
bids below its own by the same fraction of the bracket, which is what makes one
scalar comparable across the two sides.

The battery command is held at zero throughout and the initial state of charge
is the floor.  The premium has to be measured with the battery still, since a
policy that moves it earns arbitrage that has nothing to do with the price
dimension, and starting at the floor makes the terminal settlement of section 8
zero, so no part of the return is a stock rather than a flow.  With a zero
battery command the net position of every participant is the metered
difference, so aggregate supply and aggregate demand do not depend on any
submission, and the volume that a truthful market would trade can be computed
from the panel alone.  That is what makes the efficiency loss below a difference
between two volumes rather than a second optimisation.

Every participant other than the deviator submits truthfully through the
``learner_mask`` of the environment, which replaces the action of a
non-learner with ``baseline_action``.  The mask is a parameter array and not a
static shape, so the deviator is moved by rebuilding the parameters and not by
recompiling the rollout.

Episodes are drawn with the same keys at every point of the grid, so a
difference between two points is a difference in submissions and not in the days
they were scored on.

    JAX_PLATFORMS=cpu PYTHONPATH=tools/p2p_experiment python -m deviation_scan \\
        --sizes 4 8 --episodes 8 --thetas 6 --deviators 2 --seeds 1

``--attribution-from CSV`` switches this file into the second measurement
section 4 of that note named and left undone: the per-period attribution of the
premium.  Instead of scanning 51 points of the deviation axis it evaluates two
-- the truthful submission and the best response the named per-record product
already found -- and returns, for every period of every episode, the clearing
price, the deviator's own signed award and its own submitted quantity.  Those
three are what the identity

    excess = sum_t (lambda*_t - lambda0_t) * award*_t
             + sum_t (lambda0_t - pi_out_t) * (award*_t - award0_t)

is checked with; the first sum is the one that note wrote down and the second is
the channel it asked about, ``pi_out`` being the outside option of the side the
deviator is on in that period (the export price for a surplus, the retail tariff
for a deficit) and therefore the price at which an award is worth nothing to it.
Reading ``best_theta`` off the product rather than rescanning is what makes this
cost 2/51 of a scan; the two are tied together by a self-check, since the excess
recomputed here has to reproduce the ``gain_eur`` of the row it came from.

``--pricing-rule`` selects one of `envs/p2p/clearing.PRICING_RULES`.  The
default is the market's own and every number already reported by this file was
produced under it; the other two are the diagnostics that module documents, and
they exist for one question this file raised and did not answer.  An earlier
scan measured that the price sits on an end of
the tariff bracket in 95.65% of periods at 1200 households, and left
the premium's reversal above 256 households unexplained with that structure
named as a candidate.  Scanning the same grid under a rule whose truthful price
is the midpoint of the bracket in every period is the controlled contrast that
decides it.  A run under either diagnostic is not a run of this market, and the
rule in force is written into the product from the environment's own ``spec``.
"""
import argparse
import csv
import dataclasses
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.envs.p2p import (load_fluvius_households, make_p2p_env,
                                     make_p2p_params)
from powermarketjax.envs.p2p.clearing import PRICING_RULES
from powermarketjax.resources.battery import make_battery_bundle

#: The scenario as fixed in section 15 of the specification.
PI_EXP, PI_RET = 73.0, 333.4
DELTA = 0.25
KAPPA = 13.88
EPISODE_LEN = 96
#: The floor of the state-of-charge window of section 3.2.
SOC_FLOOR = 0.15


def population(series, n_agents, seed):
    """Column indices of `n_agents` households drawn evenly from the groups.

    The publisher divides the panel into four groups of equal size, and a draw
    that ignored them would let a small population fall into one kind of
    premises.  The draw below takes the same count from each group under its own
    generator, so two seeds differ in which premises are present and not in how
    the kinds are mixed.
    """
    groups = np.asarray(series.groups)
    names = sorted(set(groups.tolist()))
    per_group, remainder = divmod(n_agents, len(names))
    rng = np.random.default_rng(seed)
    picked = []
    for i, name in enumerate(names):
        members = np.flatnonzero(groups == name)
        take = per_group + (1 if i < remainder else 0)
        picked.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(picked))


def build(series, columns, pricing_rule="award-consistent-midpoint"):
    """`(params, env)` for the population named by `columns`.

    `pricing_rule` reaches `make_p2p_env` unchanged; the caller reads what took
    effect back out of ``env[3]["pricing_rule"]`` rather than out of the value
    it passed.
    """
    n_agents = len(columns)
    one_way = math.sqrt(0.85)
    battery = make_battery_bundle(
        n_devices=n_agents, capacity_mwh=0.011, power_mw=0.011 / 2.1,
        eta_charge=one_way, eta_discharge=one_way, soc_min=SOC_FLOOR,
        soc_max=1.0, initial_soc=SOC_FLOOR, dt_hours=DELTA,
        cycle_cost_per_mwh=0.0)
    params = make_p2p_params(
        p_pv=np.ascontiguousarray(series.injection[:, columns]),
        load=np.ascontiguousarray(series.offtake[:, columns]),
        battery=battery, kappa=np.full(n_agents, KAPPA, np.float32),
        learner_mask=np.ones(n_agents, bool), episode_len=EPISODE_LEN)
    return params, make_p2p_env(n_agents, PI_EXP, PI_RET, DELTA,
                                pricing_rule=pricing_rule)


def potential_volume(series, columns):
    """Volume a truthful market would trade in each period, in MWh.

    With the battery still the net position is the metered difference, so this
    is a property of the panel and the population, and every submission leaves
    it unchanged.  It is the ceiling that the traded volume of the clearing is
    measured against.
    """
    surplus = series.injection[:, columns] - series.offtake[:, columns]
    supply = np.maximum(surplus, 0.0).sum(axis=1) * DELTA
    demand = np.maximum(-surplus, 0.0).sum(axis=1) * DELTA
    return np.minimum(supply, demand)


def make_scan(env, ceiling):
    """`(thetas, keys) -> (deviator return, traded volume, forgone volume)`.

    The deviator is the single participant the parameter mask selects, so the
    action below is submitted by all and kept for one.
    """
    reset, _, step_auto, _ = env

    def episode(params, theta, key):
        key, sub = jax.random.split(key)
        _, state = reset(sub, params)

        def body(carry, _):
            state, key = carry
            key, step_key = jax.random.split(key)
            surplus = params.p_pv[state.cursor] - params.load[state.cursor]
            side = jnp.where(surplus >= 0.0, -1.0, 1.0)
            price = side * (1.0 - 2.0 * theta)
            action = jnp.stack([jnp.zeros_like(price), price], axis=1)
            _, nxt, reward, _, _, info = step_auto(
                step_key, state, action, params)
            forgone = ceiling[state.cursor] - info["traded_volume"]
            # The position of the deviator, which is what its gain has to be
            # read per unit of: the panel is heterogeneous by a factor of
            # several, so a gain in euro per episode is not comparable between
            # two premises, let alone between two populations drawn from them.
            own = jnp.sum(jnp.abs(surplus) * params.learner_mask) * DELTA
            # The structure section 3 of the note measured, carried per point
            # so that a run under a diagnostic pricing rule shows the collapse
            # it was built to remove rather than being believed to have removed
            # it.  `at_end` is 1 where the price sits on an end of the tariff
            # bracket and `at_mid` where it sits at its midpoint; the tolerance
            # is a float32 rounding allowance on numbers of order 300, not a
            # band wide enough to catch a genuinely interior price.
            price = info["clearing_price"]
            tol = jnp.float32(1e-3)
            at_end = ((jnp.abs(price - PI_EXP) <= tol)
                      | (jnp.abs(price - PI_RET) <= tol)).astype(jnp.float32)
            at_mid = (jnp.abs(price - 0.5 * (PI_EXP + PI_RET))
                      <= tol).astype(jnp.float32)
            return (nxt, key), jnp.stack(
                [jnp.sum(reward * params.learner_mask),
                 info["traded_volume"], forgone, own,
                 info["clearing_price"] * info["traded_volume"],
                 at_end, at_mid])

        _, per_period = jax.lax.scan(body, (state, key), None,
                                     length=EPISODE_LEN)
        return jnp.sum(per_period, axis=0)

    grid = jax.vmap(jax.vmap(episode, in_axes=(None, None, 0)),
                    in_axes=(None, 0, None))
    return jax.jit(grid)


#: Columns of the observation `env._get_obs` builds, by position.  They are
#: read here rather than recomputed: ``info["terminal_obs"]`` is the observation
#: of the state the period actually produced, so its "previous own" block is
#: this period's award, profit and net position, and its "public" block is this
#: period's clearing price and traded volume.  That last coincidence is what
#: makes the layout checkable at run time instead of asserted from the source:
#: `CH_PRICE_FROM_OBS` below must equal `CH_PRICE`, which comes out of ``info``
#: by name, and a shifted or renumbered observation breaks that equality.
OBS_NET_PREV, OBS_AWARD_PREV, OBS_PROFIT_PREV = 8, 9, 10
OBS_PRICE_PREV, OBS_VOLUME_PREV = 11, 12

#: Channels of the per-period record.  Everything the attribution needs is a
#: scalar per period, so the product is (n_theta, n_episodes, episode_len, 8)
#: and no per-agent array leaves the device.
(CH_PRICE, CH_AWARD, CH_PROFIT, CH_NET, CH_QUOTE, CH_PRICE_FROM_OBS,
 CH_REWARD, CH_VOLUME, CH_ANY_PARTLY, CH_N_PARTLY, CH_N_PARTLY_MID,
 CH_N_PARTLY_LOOSE) = range(12)
N_CHANNELS = 12

#: The relative tolerances the "clears partly" test is evaluated at.  It is a
#: strict-inequality test on an award against the quantity submitted, and the
#: award comes out of a float32 ``cumsum`` over the whole population, so its
#: rounding grows with the population while the tolerance does not.  Three
#: tolerances are carried rather than one because the market-wide count has an
#: external check the per-deviator count does not -- two participants clearing
#: partly is exactly the case that prices at the midpoint of the bracket -- so
#: the tolerance can be calibrated instead of asserted.
PARTLY_RTOLS = (1e-5, 1e-4, 1e-3)


def make_period_scan(env):
    """`(params, thetas, keys) -> (n_theta, n_episodes, episode_len, 8)`.

    The same rollout `make_scan` drives, kept per period instead of summed.
    The deviator is the one participant ``learner_mask`` selects, so a masked
    sum over the population is that participant's own value and not a market
    aggregate.

    ``CH_QUOTE`` is the only channel recomputed here rather than read back: the
    action map's price map is affine and this repeats it, which is checked at
    ``theta = 0``, where a seller's quote must be ``PI_EXP`` and a buyer's
    ``PI_RET`` exactly.  Everything else comes out of ``info`` or out of the
    observation of the state the period produced.
    """
    reset, _, step_auto, _ = env

    def episode(params, theta, key):
        key, sub_key = jax.random.split(key)
        _, state = reset(sub_key, params)

        def body(carry, _):
            state, key = carry
            key, step_key = jax.random.split(key)
            surplus = params.p_pv[state.cursor] - params.load[state.cursor]
            side = jnp.where(surplus >= 0.0, -1.0, 1.0)
            alpha = side * (1.0 - 2.0 * theta)
            action = jnp.stack([jnp.zeros_like(alpha), alpha], axis=1)
            _, nxt, reward, _, _, info = step_auto(
                step_key, state, action, params)
            mask = params.learner_mask.astype(jnp.float32)
            # exactly one entry is set, so a masked sum is a selection
            own = lambda column: jnp.sum(column * mask)
            obs = info["terminal_obs"]
            quote = (0.5 * (1.0 - alpha) * PI_EXP
                     + 0.5 * (1.0 + alpha) * PI_RET)
            # The same test the attribution applies to the deviator, but
            # over the whole population and not through the mask: section 3 of
            # the note reads the collapse of the price interval as the trace of
            # a participant clearing partly, and that reading needs the count of
            # periods in which anyone does, which no per-deviator quantity can
            # give.  `n_partly` is carried beside it so that "at most one in the
            # market" is checked here rather than taken from the specification.
            aw_all = obs[:, OBS_AWARD_PREV]
            q_all = DELTA * jnp.abs(obs[:, OBS_NET_PREV])
            counts = []
            for rtol in PARTLY_RTOLS:
                partly_all = ((jnp.abs(aw_all) > rtol * q_all)
                              & (jnp.abs(aw_all) < (1.0 - rtol) * q_all))
                counts.append(jnp.sum(partly_all.astype(jnp.float32)))
            n_partly = counts[0]
            return (nxt, key), jnp.stack([
                info["clearing_price"],
                own(obs[:, OBS_AWARD_PREV]),
                own(obs[:, OBS_PROFIT_PREV]),
                own(obs[:, OBS_NET_PREV]),
                own(quote),
                own(obs[:, OBS_PRICE_PREV]),
                own(reward),
                info["traded_volume"],
                jnp.minimum(n_partly, 1.0),
                n_partly, counts[1], counts[2]])

        _, per_period = jax.lax.scan(body, (state, key), None,
                                     length=EPISODE_LEN)
        return per_period

    grid = jax.vmap(jax.vmap(episode, in_axes=(None, None, 0)),
                    in_axes=(None, 0, None))
    return jax.jit(grid)


#: Relative tolerance on an award compared against the quantity submitted, and
#: absolute tolerance on a price in EUR/MWh.  The awards are the one inexact
#: output of the clearing (`clearing.py` module doc: ``cumsum`` is a parallel
#: scan), and the quantities here are of order 1e-4 MWh, so the comparison that
#: decides "partly cleared" is made relative to the submission and not against
#: an absolute floor picked for a different order of magnitude.
AWARD_RTOL = 1e-5
PRICE_ATOL = 1e-3


def attribute(out, theta_star):
    """Reduce one record's per-period product to the attribution scalars.

    `out` is ``(2, n_episodes, episode_len, 8)`` with the truthful submission
    first.  Every returned quantity is a sum over the periods of an episode and
    then a mean over episodes, so it is per episode and comparable with the
    ``gain_eur`` of the per-record product; the counts are periods per episode
    out of ``episode_len`` and not fractions.
    """
    per_episode = lambda x: float(x.sum(axis=1).mean())
    lam0, lam1 = out[0, :, :, CH_PRICE], out[1, :, :, CH_PRICE]
    a0, a1 = out[0, :, :, CH_AWARD], out[1, :, :, CH_AWARD]
    net = out[0, :, :, CH_NET]
    quantity = DELTA * np.abs(net)
    # The side is read off the net position, which the battery being still
    # makes independent of theta; the outside option is that side's, and it is
    # the price at which an award earns the deviator nothing.
    pi_out = np.where(net >= 0.0, PI_EXP, PI_RET)

    excess_t = out[1, :, :, CH_PROFIT] - out[0, :, :, CH_PROFIT]
    price_channel_t = (lam1 - lam0) * a1
    award_channel_t = (lam0 - pi_out) * (a1 - a0)

    partly = lambda a: ((np.abs(a) > AWARD_RTOL * quantity)
                        & (np.abs(a) < (1.0 - AWARD_RTOL) * quantity))
    moved = np.abs(lam1 - lam0) > PRICE_ATOL
    return dict(
        theta_star=float(theta_star),
        excess_eur=per_episode(excess_t),
        price_channel_eur=per_episode(price_channel_t),
        award_channel_eur=per_episode(award_channel_t),
        # the identity as section 4 of the note wrote it: everything the price
        # channel alone does not account for
        identity_residual_eur=per_episode(excess_t - price_channel_t),
        # what is left once the award channel is taken out too; this one is a
        # float32 rounding residual and nothing else, and it is reported so
        # that a non-zero identity residual can be told apart from arithmetic
        numeric_residual_eur=per_episode(
            excess_t - price_channel_t - award_channel_t),
        own_position_mwh=per_episode(quantity),
        truthful_profit_eur=per_episode(out[0, :, :, CH_PROFIT]),
        # the diagnostic section 4 asked for, in three readings of the same
        # period set: the deviator is the participant clearing partly, the
        # price moved at all, and the price sits on the deviator's own quote
        periods_deviator_partly_cleared=per_episode(partly(a1)),
        periods_deviator_partly_cleared_truthful=per_episode(partly(a0)),
        periods_deviator_awarded=per_episode(
            np.abs(a1) > AWARD_RTOL * quantity),
        periods_price_moved=per_episode(moved),
        periods_price_at_own_quote=per_episode(
            (np.abs(lam1 - out[1, :, :, CH_QUOTE]) <= PRICE_ATOL)
            & (quantity > 0.0)),
        periods_award_channel_live=per_episode(
            (np.abs(lam0 - pi_out) > PRICE_ATOL)
            & (np.abs(a1 - a0) > AWARD_RTOL * np.maximum(quantity, 1e-12))),
        # the two halves of that conjunction, reported apart: an award channel
        # of zero because no award moved is a different fact from an award
        # channel of zero because what moved was priced at the outside option,
        # and the conjunction alone cannot tell them apart
        periods_award_changed=per_episode(
            np.abs(a1 - a0) > AWARD_RTOL * np.maximum(quantity, 1e-12)),
        # the other half of what makes the award channel live, on its own and
        # not conditioned on an award having moved: how often the truthful
        # price is *not* the deviator's own outside option.  Where it is, an
        # award is worth exactly nothing to it and losing one costs nothing,
        # which is why the channel can be dead while awards move
        periods_price_off_outside_option=per_episode(
            np.abs(lam0 - pi_out) > PRICE_ATOL),
        # market-wide, at the truthful submission: how often *anyone* clears
        # partly.  Section 3 of the note reads the price interval collapsing as
        # that happening; this is the count that decides whether the two rates
        # are the same rate.  `max_participants_partly_cleared` is the check on
        # "at most one in the market", which the estimator below relies on.
        periods_any_partly_cleared_truthful=per_episode(
            out[0, :, :, CH_ANY_PARTLY]),
        max_participants_partly_cleared=float(
            np.max(out[:, :, :, CH_N_PARTLY])),
        # The breakdown, because the three cases price differently and the
        # count alone cannot be read against section 3's collapse rate.  One
        # straddler pins both ends of the interval on its own quote, which at
        # the truthful submission is an end of the tariff bracket; two -- one
        # per side -- pin the two ends on the two quotes, the interval comes
        # out inverted and the midpoint rule returns the middle of the bracket.
        periods_no_partly_truthful=per_episode(
            out[0, :, :, CH_N_PARTLY] < 0.5),
        periods_one_partly_truthful=per_episode(
            np.abs(out[0, :, :, CH_N_PARTLY] - 1.0) < 0.5),
        periods_two_partly_truthful=per_episode(
            out[0, :, :, CH_N_PARTLY] > 1.5),
        periods_two_partly_truthful_rtol1e4=per_episode(
            out[0, :, :, CH_N_PARTLY_MID] > 1.5),
        periods_two_partly_truthful_rtol1e3=per_episode(
            out[0, :, :, CH_N_PARTLY_LOOSE] > 1.5),
        periods_any_partly_truthful_rtol1e4=per_episode(
            out[0, :, :, CH_N_PARTLY_MID] > 0.5),
        periods_any_partly_truthful_rtol1e3=per_episode(
            out[0, :, :, CH_N_PARTLY_LOOSE] > 0.5),
        # the same tolerance sweep on the deviator's own count, which is the
        # one the note's question two is answered with
        periods_deviator_partly_cleared_truthful_rtol1e4=per_episode(
            (np.abs(a0) > 1e-4 * quantity)
            & (np.abs(a0) < (1.0 - 1e-4) * quantity)),
        periods_deviator_partly_cleared_truthful_rtol1e3=per_episode(
            (np.abs(a0) > 1e-3 * quantity)
            & (np.abs(a0) < (1.0 - 1e-3) * quantity)),
        # section 3's two counts, recomputed here at the truthful submission so
        # that they and the straddler breakdown come out of one product
        periods_price_at_bracket_end_truthful=per_episode(
            (np.abs(lam0 - PI_EXP) <= PRICE_ATOL)
            | (np.abs(lam0 - PI_RET) <= PRICE_ATOL)),
        periods_price_at_bracket_mid_truthful=per_episode(
            np.abs(lam0 - 0.5 * (PI_EXP + PI_RET)) <= PRICE_ATOL),
        periods_deviator_on_sell_side=per_episode(net >= 0.0),
        periods_deviator_awarded_truthful=per_episode(
            np.abs(a0) > AWARD_RTOL * quantity),
        awarded_mwh_truthful=per_episode(np.abs(a0)),
        # the deviator's own energy that the price move was earned on, which is
        # what a count of periods cannot say on its own
        awarded_mwh_while_price_moved=per_episode(np.abs(a1) * moved),
        awarded_mwh=per_episode(np.abs(a1)),
        # the layout and the terminal leg, checked rather than assumed
        obs_layout_max_abs_diff=float(np.max(np.abs(
            out[:, :, :, CH_PRICE_FROM_OBS] - out[:, :, :, CH_PRICE]))),
        # the price map, checked at the one point where its value is known in
        # advance: at theta = 0 a seller's quote is the export price and a
        # buyer's the retail tariff, exactly, so this is zero unless the quote
        # channel selects the wrong participant, the battery is not still (the
        # net position would then leave the metered surplus and the side could
        # flip) or the affine map repeated here has drifted from `action.py`
        quote_truthful_max_abs_diff=float(np.max(np.abs(
            out[0, :, :, CH_QUOTE] - pi_out.astype(np.float32)))),
        reward_minus_profit_max_abs=float(np.max(np.abs(
            out[:, :, :, CH_REWARD] - out[:, :, :, CH_PROFIT]))),
        net_theta_drift_max_abs=float(np.max(np.abs(
            out[1, :, :, CH_NET] - out[0, :, :, CH_NET]))))


def read_best_theta(path, pricing_rule):
    """`{(n_agents, seed, deviator): best_theta}` for one arm of the product.

    The grid is taken from the file and not from the flags: the point of this
    mode is to revisit the records that product already reported, so a size or
    a seed it does not carry is not a point this mode has anything to say
    about.
    """
    with path.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["pricing_rule"] == pricing_rule]
    if not rows:
        raise SystemExit(
            f"{path} carries no row with pricing_rule={pricing_rule!r}")
    return {(int(r["n_agents"]), int(r["seed"]), int(r["deviator"])):
            (float(r["best_theta"]), float(r["gain_eur"]),
             float(r["own_position_mwh"])) for r in rows}


def run_attribution(args):
    """The two-point rerun and the per-period attribution it produces."""
    series = load_fluvius_households()
    table = read_best_theta(args.attribution_from, args.pricing_rule)
    sizes = sorted({k[0] for k in table})
    seeds = sorted({k[1] for k in table})
    print(f"grid read off {args.attribution_from}: sizes {sizes}, "
          f"seeds {seeds}, {len(table)} records, "
          f"pricing_rule={args.pricing_rule}", flush=True)
    records = []
    for n_agents in sizes:
        for seed in seeds:
            wanted = sorted(d for (n, s, d) in table if n == n_agents
                            and s == seed)
            if not wanted:
                continue
            columns = population(series, n_agents, seed)
            params, env = build(series, columns, args.pricing_rule)
            in_force = env[3]["pricing_rule"]
            if in_force != args.pricing_rule:
                raise SystemExit(
                    f"asked for pricing_rule={args.pricing_rule!r} and the "
                    f"environment reports {in_force!r}")
            # The deviators are not re-drawn: they are read off the product and
            # then checked against the draw the scan would make, so a product
            # produced under a different draw is a failure here and not a
            # silent change of which premises were asked to deviate.
            drawn = sorted(np.random.default_rng(90_000 + seed).choice(
                n_agents, size=min(args.deviators, n_agents),
                replace=False).tolist())
            if drawn != wanted:
                raise SystemExit(
                    f"N={n_agents} seed={seed}: the product names deviators "
                    f"{wanted} and this file's draw gives {drawn}")
            scan = make_period_scan(env)
            keys = jax.random.split(jax.random.PRNGKey(7_000 + seed),
                                    args.episodes)
            for deviator in wanted:
                theta_star, gain, position = table[(n_agents, seed, deviator)]
                mask = np.zeros(n_agents, bool)
                mask[deviator] = True
                pinned = dataclasses.replace(
                    params, learner_mask=jnp.asarray(mask))
                started = time.time()
                out = np.asarray(scan(pinned, jnp.asarray(
                    [0.0, theta_star], np.float32), keys))
                row = attribute(out, theta_star)
                row.update(n_agents=int(n_agents), seed=int(seed),
                           deviator=int(deviator),
                           product_gain_eur=gain,
                           product_own_position_mwh=position,
                           seconds=round(time.time() - started, 1))
                records.append(row)
                share = (abs(row["identity_residual_eur"])
                         / abs(row["excess_eur"])
                         if row["excess_eur"] != 0.0 else float("nan"))
                print(f"N={n_agents:5d} seed={seed} i={deviator:5d} "
                      f"theta*={theta_star:.2f}  "
                      f"excess {row['excess_eur']:+.6f} "
                      f"(product {gain:+.6f})  "
                      f"price {row['price_channel_eur']:+.6f}  "
                      f"award {row['award_channel_eur']:+.6f}  "
                      f"resid/excess {share:8.2e}  "
                      f"partly {row['periods_deviator_partly_cleared']:6.3f}  "
                      f"[{row['seconds']:.1f} s]", flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(dict(
            mode="attribution", episodes=args.episodes,
            episode_len=EPISODE_LEN, pi_exp=PI_EXP, pi_ret=PI_RET,
            delta_hours=DELTA,
            best_theta_from=str(args.attribution_from),
            pricing_rule=in_force,
            is_market_pricing_rule=bool(env[3]["is_market_pricing_rule"]),
            award_rtol=AWARD_RTOL, price_atol=PRICE_ATOL,
            records=records), indent=1))
        print(f"wrote {args.out}")


def run(args):
    series = load_fluvius_households()
    thetas = np.linspace(0.0, 1.0, args.thetas, dtype=np.float32)
    records = []
    for n_agents in args.sizes:
        for seed in range(args.seeds):
            columns = population(series, n_agents, seed)
            params, env = build(series, columns, args.pricing_rule)
            in_force = env[3]["pricing_rule"]
            if in_force != args.pricing_rule:
                raise SystemExit(
                    f"asked for pricing_rule={args.pricing_rule!r} and the "
                    f"environment reports {in_force!r}")
            ceiling = jnp.asarray(potential_volume(series, columns))
            scan = make_scan(env, ceiling)
            keys = jax.random.split(jax.random.PRNGKey(7_000 + seed),
                                    args.episodes)
            # The deviators are drawn rather than spread over the index, so
            # that two populations differ in size and not in which premises
            # were asked to deviate.  At the largest population every draw
            # holds the whole panel, and the seed then varies only this.
            spread = np.sort(np.random.default_rng(90_000 + seed).choice(
                n_agents, size=min(args.deviators, n_agents), replace=False))
            for deviator in spread:
                mask = np.zeros(n_agents, bool)
                mask[deviator] = True
                pinned = dataclasses.replace(
                    params, learner_mask=jnp.asarray(mask))
                started = time.time()
                # The grid is evaluated in slices of the deviation axis.  Every
                # period of every episode of every slice is live at once, and at
                # the largest population the whole grid asks for more device
                # memory than the card has; the slices are concatenated below
                # and the result does not depend on their width.
                out = np.concatenate([
                    np.asarray(scan(pinned, jnp.asarray(chunk), keys))
                    for chunk in np.array_split(
                        thetas, math.ceil(len(thetas) / args.theta_chunk))])
                profit, volume, forgone, own, paid, at_end, at_mid = (
                    out[..., i].mean(axis=1) for i in range(7))
                records.append(dict(
                    n_agents=int(n_agents), seed=int(seed),
                    deviator=int(deviator), theta=thetas.tolist(),
                    profit=profit.tolist(), volume=volume.tolist(),
                    forgone_volume=forgone.tolist(),
                    own_position=own.tolist(),
                    volume_weighted_payment=paid.tolist(),
                    # the two counts are sums over the 96 periods of an episode
                    # and then a mean over episodes, so they are periods per
                    # episode out of 96 and not fractions
                    periods_price_at_bracket_end=at_end.tolist(),
                    periods_price_at_bracket_mid=at_mid.tolist(),
                    excess=(profit - profit[0]).tolist(),
                    seconds=round(time.time() - started, 1)))
                best = int(np.argmax(profit))
                print(f"N={n_agents:5d} seed={seed} i={deviator:5d}  "
                      f"truthful {profit[0]:+9.4f}  best {profit[best]:+9.4f} "
                      f"at theta={thetas[best]:.2f}  "
                      f"excess {profit[best] - profit[0]:+.5f} EUR/episode  "
                      f"[{records[-1]['seconds']:.1f} s]", flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(dict(
            episodes=args.episodes, episode_len=EPISODE_LEN,
            pi_exp=PI_EXP, pi_ret=PI_RET,
            # read back out of the last environment built, not off the flag
            pricing_rule=in_force,
            is_market_pricing_rule=bool(env[3]["is_market_pricing_rule"]),
            records=records), indent=1))
        print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[4, 8, 16, 64, 128, 256, 512, 900, 1200])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--deviators", type=int, default=12)
    parser.add_argument("--thetas", type=int, default=51)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--theta-chunk", type=int, default=4)
    parser.add_argument("--pricing-rule", default="award-consistent-midpoint",
                        choices=list(PRICING_RULES),
                        help="the market's own rule is the default; the other "
                             "two are diagnostics and a run under them is not "
                             "a run of this market")
    parser.add_argument("--attribution-from", type=Path, default=None,
                        help="a per-record CSV in the schema "
                             "`deviation_summary --per-record-csv` writes; "
                             "given one, this file reruns each of its records "
                             "at the truthful submission and at the best "
                             "response the row names, and reports the "
                             "per-period attribution of the premium instead "
                             "of scanning the deviation axis")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    (run_attribution if args.attribution_from is not None else run)(args)


if __name__ == "__main__":
    main()
