"""L0: `horizon` is market 01's own value, and `off_shared` says so.

**What went wrong without this test.** `horizon` was one number, 48, shared by
the three markets, and market 01 ran at 4 because one of its steps is a whole
24-hour clearing rather than a half-hour period. Every 01 product therefore
came out stamped `off_shared: {"horizon": [48, 4]}` -- the stamp whose job is to
say "this run is not the reported configuration" -- for running the only
configuration 01 has. Measured 2026-09-07 on a seed-0 run of market 01, whose day
products carry exactly that stamp while they were reported as the baseline.

The failure is silent in the direction that matters: a reader filtering products
by `off_shared` drops 01's whole year window, and nothing errors.

`horizon` is now declared per market (`hyperparams.HORIZON`) and the driver
compares against its own market's value. The three cases below are the three a
reader has to be able to tell apart, and the third is what keeps the fix from
being "never stamp anything".
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "benchmark"))
from hyperparams import (HORIZON, MINIBATCHES, N_ENVS, SAC_SHARED,  # noqa: E402
                         SHARED, sac_shared_for, shared_for)


def off_shared_for(given_horizon, market="01"):
    """The driver's `off_shared` rule, on the horizon field alone.

    This mirrors `run_rl_01.py`'s block rather than importing it: that block
    runs inside `main()` after a fixture load and a case build, so calling it
    here would need a case on disk. The mirrored expression is one line and the
    test below pins the driver's own source against it.
    """
    base = shared_for(market)
    if given_horizon is None:
        return {}
    if given_horizon == base.horizon:
        return {}
    return {"horizon": [base.horizon, given_horizon]}


def test_no_flag_is_on_shared():
    """01 with no `--horizon` runs at 4 and is not stamped."""
    assert shared_for("01").horizon == 4
    assert off_shared_for(None) == {}


def test_the_market_value_is_on_shared():
    """`--horizon 4` repeats 01's own value, so it is not a deviation."""
    assert off_shared_for(4) == {}


def test_another_markets_value_is_off_shared():
    """`--horizon 48` is 02/03's value, and on 01 it is a real deviation."""
    assert off_shared_for(48) == {"horizon": [4, 48]}


def test_the_driver_compares_against_its_own_market():
    """The driver reads `shared_for(MARKET)`, not the module-level `SHARED`.

    Pinned against the source because the whole defect was a comparison against
    the wrong base, and that comparison is one `getattr` -- a rewrite that
    reintroduces `getattr(SHARED, name)` would pass every assertion above.
    """
    src = (Path(__file__).resolve().parents[2] / "tools" / "benchmark"
           / "run_rl_01.py").read_text(encoding="utf-8")
    assert 'MARKET = "01"' in src
    assert "base = shared_for(MARKET) if args.algo == \"ippo\" else sac_shared_for(MARKET)" in src
    assert "getattr(SHARED, name)" not in src


@pytest.mark.parametrize("market", sorted(HORIZON))
def test_every_market_batch_divides_into_minibatches(market):
    """Each market's own batch, not just the shared one, feeds `_update`.

    `ippo._update` reshapes `horizon * n_envs` and drops the remainder, so a
    horizon that does not divide is a silently smaller batch. 01's 64 x 4 = 256
    is the case the old single-value assert never saw.
    """
    assert (N_ENVS * HORIZON[market]) % MINIBATCHES == 0


@pytest.mark.parametrize("market", sorted(HORIZON))
def test_horizon_is_the_only_field_that_moves(market):
    """Per-market means per-market in one field; everything else is shared.

    This is what carries `tuned_per_market: False`: if a second field started
    varying by market, that claim would be false and this test is where it
    fails rather than in a reviewer's reading.
    """
    for cfg, base in ((shared_for(market), SHARED),
                      (sac_shared_for(market), SAC_SHARED)):
        # `SAC_SHARED.reward_scale` is the nan sentinel the driver must replace,
        # and `nan != nan`, so compare it by repr rather than by value -- a
        # plain `!=` would report it moved in every market and hide a real one.
        moved = [f for f in vars(base)
                 if repr(getattr(cfg, f)) != repr(getattr(base, f))]
        assert moved in ([], ["horizon"]), (market, moved)
