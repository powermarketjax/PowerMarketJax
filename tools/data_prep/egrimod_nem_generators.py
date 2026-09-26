"""Download the egrimod-nem generator table used by ``case813nem``.

Source: https://github.com/akxen/egrimod-nem-dataset, file
``generators/generators.csv``, documented in Xenophon, A. K. & Hill, D. J.,
"Open grid model of Australia's National Electricity Market allowing
backtesting against historic data", *Scientific Data* 5, 180203 (2018),
https://doi.org/10.1038/sdata.2018.203.  The repository states CC BY 4.0.

Licence: not yet reviewed for redistribution; see the Data section of the
README.  The table is compiled from AEMO data, so its redistribution status
follows AEMO's terms rather than only the repository's.  This script only
downloads; the network files next to it in
``powermarketjax/case/raw_cases/egrimod_nem/`` are shipped, this one is not.

Output: ``generators.csv``, byte for byte as published at the pinned commit.
Copy it into ``powermarketjax/case/raw_cases/egrimod_nem/`` to build
``case813nem``.

Processing decisions:

* **The commit is pinned** (``COMMIT``, the repository head, 2018-08-03; the
  file last changed at 130356b, 2018-08-01).  A branch name would make the
  download depend on the day it is run.
* **No processing at all.**  The bytes are checked against the SHA-256 of the
  copy ``case813nem`` was built from, and a mismatch exits non-zero after
  writing the file, so the difference can be inspected.

    python tools/data_prep/egrimod_nem_generators.py --out-dir /tmp/pmj-data
"""
from __future__ import annotations

import hashlib
import sys

import _fetch

NAME = "egrimod_nem_generators"
REPO_URL = "https://github.com/akxen/egrimod-nem-dataset"
COMMIT = "4806603ca56dcc3b5d7118eaac1b3e33b69802db"
URL = f"https://raw.githubusercontent.com/akxen/egrimod-nem-dataset/{COMMIT}/generators/generators.csv"
SHA256 = "295fd3dcc3e992f76db6904f6fa9e94da4813afeb4ecef0fbd2cafd507ef9da2"
BYTES = 40138
VENDORED_DIR = _fetch.REPO / "powermarketjax/case/raw_cases/egrimod_nem"


def main() -> None:
    args = _fetch.parser(__doc__).parse_args()
    out = _fetch.out_dir(args, VENDORED_DIR)
    blob = _fetch.get(URL)
    path = out / "generators.csv"
    path.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    print(f"wrote {path} ({len(blob):,} bytes, sha256 {digest})")
    if digest != SHA256:
        sys.exit(f"sha256 differs from the copy case813nem was built from "
                 f"({SHA256}, {BYTES} bytes): compare before using it")


if __name__ == "__main__":
    main()
