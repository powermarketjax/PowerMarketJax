"""Market 01's economic-withholding arm -- the `--market 01` entry point.

Everything this file used to hold is now in `run_withholding.py`, which serves
markets 01, 02 and 03 from one implementation.  The generalisation
is its own proof: market 01's product rebuilt through the general path is
identical to the one this file produced, number for number, everything outside
`run_point`'s timestamps and blob.

The names below are re-exported rather than removed because they are the arm's
two pure decisions and the L0 test names this module
(`tests/tools/test_withholding_01_l0.py`).  What the arm *is*, what it is not,
and how the two modes are run: `run_withholding.py`'s module docstring.

    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \\
    conda run -n powermarketjax --no-capture-output python \\
        tools/benchmark/run_withholding_01.py sweep --days train36 --shard 0/3 \\
        --fixture day_ahead_commitment_29gb_T24_relax_y1_365d_c0.6_r1.00.npz \\
        --out-dir unilateral_br_01/train36
    ... python tools/benchmark/run_withholding_01.py assemble \\
        --dir unilateral_br_01/train36 \\
        --out 01-y1-train36.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_withholding import (EXIT_REL, TIE_REL, assemble, best_index,  # noqa: E402,F401
                             classify, main as _main, sweep)


def main():
    _main(market="01")


if __name__ == "__main__":
    main()
