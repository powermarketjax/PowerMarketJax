"""The day-ahead position, read back at real-time resolution.

The real-time market settles against what the day-ahead market cleared.  That
position is produced offline by `tools/commitment/da_position.py` and frozen in a
fixture; this module reads it and maps it onto real-time periods.

**One day-ahead hour covers two real-time periods**, and the map is a repeat
rather than an interpolation: the schedule, commitment and price of one day-ahead
period apply to every real-time period it covers.  Interpolating would invent a
half-hourly day-ahead schedule that no auction ever cleared, and the deviation
`p - q_da` would then be measured against a fiction.

The map lives here, in one place, because the settlement's money-balance identity
cannot detect an error in it: the identity holds whatever `q_da` is substituted,
so a mis-mapped position satisfies it exactly as well as a correct one.
"""
from pathlib import Path
from typing import Dict, Optional

import json
import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures"

#: Real-time periods per day-ahead period.  Not a tunable: it is the ratio
#: of the two series' native resolutions, and `load_da_position` checks the
#: fixture agrees rather than assuming it.
PERIODS_PER_HOUR = 2

#: Real-time periods in a market day.
T_RT = 48


def hour_of_period(t_rt, periods_per_hour: int = PERIODS_PER_HOUR):
    """Day-ahead period index covering real-time period ``t_rt``.

    Integer division, and the direction matters: real-time periods 0 and 1 both
    fall in day-ahead hour 0.  Accepts a Python int, a numpy array **or a traced
    JAX value** -- it does not coerce, because the map has to be usable from
    inside `jit` as well as from the offline fixture code.
    """
    return t_rt // periods_per_hour


def to_real_time(x, axis: int = 0, periods_per_hour: int = PERIODS_PER_HOUR):
    """Repeat a day-ahead quantity along ``axis`` onto real-time periods.

    `np.repeat` rather than `np.tile`: the two hours of a day-ahead period are
    adjacent, so hour 0 must produce periods 0 and 1, not periods 0 and 24.  The
    two agree only when `periods_per_hour == 1`.
    """
    return np.repeat(np.asarray(x), periods_per_hour, axis=axis)


def load_da_position(case: str = "29gb", chain: str = "step1prime",
                     path: Optional[Path] = None) -> Dict:
    """Load one day-ahead position fixture and its metadata.

    Returns the five position quantities at **day-ahead** resolution, plus
    ``meta``.  They are not expanded here, since `d_da` and `s_da` are per bus;
    use `to_real_time` on whichever of them a caller needs per real-time period.

    Read ``meta`` before using the arrays.  The position depends on a declared
    day-ahead policy (truthful, markup 1.0) and on the commitment chain, and a
    position built under a different one describes a different day-ahead market.
    """
    if path is None:
        path = FIXTURE_DIR / f"day_ahead_position_{case}_T24_{chain}.npz"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no day-ahead position fixture at {path}; build one with "
            f"`python tools/commitment/da_position.py`")
    z = np.load(path)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    got = out["meta"].get("periods_per_hour")
    if got != PERIODS_PER_HOUR:
        raise ValueError(
            f"fixture at {path} was built at {got} real-time periods per "
            f"day-ahead period, this module maps at {PERIODS_PER_HOUR}")
    return out
