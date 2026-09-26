"""The exogenous commitment fixture: the day window and initial boundary for the
day-ahead environment.

`tools/commitment/precommit.py` builds it offline, where a CPU solver is allowed,
and freezes it as a fixture; this module reads it back.  The environment itself
solves the commitment from the agents' offers -- this fixture only supplies the
day window and the initial day boundary each episode starts from.

    fixture = load_commitment()          # the relaxed-then-rounded schedule
    fixture["commitment"]                # (n_days, n_units, n_periods) int8

Each day of the fixture also carries the boundary that day was solved against:
`p_init`, `commitment_prev`, `up_time` and `down_time`.  The offline sweep is
chained, so for every day but the first that boundary is the one the preceding
day ended at, and the first comes from clearing the first period on its own.
The environment reads it at `reset`, where there is no previous day to take it
from, and it must not reconstruct one instead: a commitment met by a boundary
it was not computed against sheds load and prices at the value of lost load
that a chained boundary would not.

The fixture lives in `tests/fixtures/`, which is where `.gitignore` re-includes
binaries for frozen offline results.  That location is a consequence of the
fixture being an offline artefact rather than case data; a wheel install would
not carry it.
"""
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures"

#: The case the derived path names when the caller names neither a case nor a
#: path.  It is the only case this repository has a committed commitment product
#: for, and thirteen call sites in the tree pass neither argument.  It is **not**
#: an expectation about a fixture handed over by ``path``: see ``case`` below.
DEFAULT_CASE = "29gb"

#: What `p_min_scale` means on a fixture that does not record it.  The field did
#: not exist before 2026-09-05 and every fixture written until then ran the
#: registered case, so absent reads as 1.0.  `precommit.IMPLIED_PRIOR` and
#: `da_position.P_MIN_SCALE_IMPLIED_PRIOR` already record that reading; this
#: mirrors them rather than inventing a third.
P_MIN_SCALE_IMPLIED_PRIOR = 1.0


def load_commitment(mode: str = "relax", case: Optional[str] = None,
                    n_periods: int = 24, path: Optional[Path] = None,
                    p_min_scale: Optional[float] = None) -> Dict:
    """Load one commitment fixture and its metadata.

    ``mode`` is ``"relax"`` for the relaxed commitment followed by rounding,
    which is what the environment runs on, or ``"milp"`` for the exact mixed
    integer commitment, which exists to measure what the rounding costs and
    covers only a few days.

    ``case`` names which fixture is wanted, and ``path`` names one directly; a
    caller does not have to do both.  Given a ``path`` and no ``case`` the
    fixture's own ``meta["case"]`` stands, because the caller has already said
    which file they mean and the case is a property of that file rather than of
    the call.  Given a ``case`` -- passed, or defaulted to `DEFAULT_CASE` while
    deriving a path -- it is checked against the file, so naming one is still an
    assertion and a mis-stamped fixture at a derived path still raises.

    **``case`` defaulted to ``"29gb"`` until 2026-09-05 and that default was a
    barrier, not a design**.
    Sixteen drivers write ``load_commitment(path=..., n_periods=T)``, so a
    `73rts` product could not be read by any of them.  Relaxing it is safe
    because those drivers no longer assume the case either: thirteen take theirs
    from ``load_case(meta["case"])`` and pair the demand through
    `demand_from_meta`, one (`run_withholding.py`) reads only ``meta["dates"]``,
    and the two in `tests/` name a literal `29gb` path -- 13 + 1 + 2 = 16.
    ``n_periods`` keeps its default and its check: the horizon is what the
    caller's own arrays are shaped for, so it is a property of the call and not
    of the file.

    Returns the fixture's arrays plus ``meta``, a dict recording the mode, the
    scenario parameters, the offer basis, the boundary construction and the dates
    covered.  Read ``meta`` before using the arrays: a schedule built at a
    different ``cap_scale`` describes a different market.

    ``p_min_scale`` is the one scenario factor this loader refuses on rather than
    merely records, because it is the one a caller cannot notice: it is applied to
    the *case* rather than passed to the operators, so `make_env` never sees it and
    the thirteen drivers that take their case from ``load_case(meta["case"])`` drop
    it silently.  Measured 2026-09-13: eleven products in the tree carry a value
    other than 1.0, and seven scripts that consume a MILP reference build their own
    case without reading the field.  The failure is not an exception but a wrong
    market -- a commitment solved at 0.80x the minimum output, cleared against the
    registered one.  On `case73rts` that showed up as 34 of 36 days not converging,
    negative shedding (over-generation) and ``profit -inf``.

    So a fixture that declares a value other than 1.0 is only readable by a caller
    that names the same value.  Every fixture written before 2026-09-05 records no
    field at all and reads as 1.0 (`P_MIN_SCALE_IMPLIED_PRIOR`), so this refuses
    nothing that worked before: measured, of the twelve commitment fixtures in the
    tree the two `73rts` ones declare 0.80 and every other declares 1.0 or nothing.
    """
    if path is None:
        case = DEFAULT_CASE if case is None else case
        path = FIXTURE_DIR / f"day_ahead_commitment_{case}_T{n_periods}_{mode}.npz"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no commitment fixture at {path}; build one with "
            f"`python tools/commitment/precommit.py --mode {mode}`")
    z = np.load(path)
    out = {k: z[k] for k in z.files if k != "meta"}
    # `meta` was stored as a JSON string and comes back as a 0-d numpy array
    out["meta"] = json.loads(str(z["meta"]))
    if case is not None and out["meta"]["case"] != case:
        raise ValueError(f"fixture at {path} is for case "
                         f"{out['meta']['case']!r}, not {case!r}")
    if out["meta"]["n_periods"] != n_periods:
        raise ValueError(f"fixture at {path} covers T={out['meta']['n_periods']}, "
                         f"not the T={n_periods} n_periods asks for")
    declared = float(out["meta"].get("p_min_scale", P_MIN_SCALE_IMPLIED_PRIOR))
    if p_min_scale is None:
        if declared != P_MIN_SCALE_IMPLIED_PRIOR:
            raise ValueError(
                f"fixture at {path} was built at p_min_scale={declared} and this "
                f"call did not name one.  That factor is applied to the case, not "
                f"to the operators, so nothing downstream would have raised: pass "
                f"p_min_scale={declared} here and give the same value to "
                f"`scale_min_output(load_case(meta['case']), ...)`.")
    elif float(p_min_scale) != declared:
        raise ValueError(
            f"fixture at {path} was built at p_min_scale={declared} and this call "
            f"asks for {float(p_min_scale)}; the commitment would be cleared "
            f"against a different case than it was solved for")
    return out
