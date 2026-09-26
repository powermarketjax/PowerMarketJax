# egrimod-nem source data — licence, and one question it does not settle

The six CSV files in this directory are copied verbatim from
<https://github.com/akxen/egrimod-nem-dataset> (`network/` and `generators/`),
retrieved 2026-08-21.  The dataset is released under **CC BY 4.0**
(`license.md` in that repository), and is documented in

> Xenophon, A. K. & Hill, D. J. Open grid model of Australia's National
> Electricity Market allowing backtesting against historic data.
> *Scientific Data* **5**, 180203 (2018).  <https://doi.org/10.1038/sdata.2018.203>

Attribution is required and is carried in the case docstring as well as here.

## The provenance question this licence does not answer

The two halves of the dataset do not have the same provenance, and only one of
them is clean.

* **The network** (`network_*.csv`) is built from Geoscience Australia's
  transmission line, substation and power station datasets, themselves CC BY 4.0
  © Commonwealth of Australia (Geoscience Australia) 2017, with nodal demand
  shares derived from Australian Bureau of Statistics population data, also
  CC BY 4.0.  Redistribution here is exactly what those licences grant.

* **The generator table** (`generators.csv`) is compiled from AEMO's Market
  Management System Data Model and its National Transmission Network Development
  Plan database.  Those are AEMO bytes, redistributed by a third party that
  applies CC BY 4.0 to the result.

`tools/data_prep/aemo_nsw1_price_and_rooftop_pv.py` states this repository's
position on precisely that construction, and it is not a position this file may
quietly overturn: a second-hand distributor that self-reports CC BY satisfies
the *form* of "licence read from the response" while sourcing the same AEMO
bytes, so if AEMO's terms do not permit redistribution then a downstream CC BY
claim launders the licence rather than resolving it.  AEMO's own notice grants
permission to **use** AEMO Material with attribution and says nothing about
reproduction or republication, and `aemo.com.au` returns 403 to every automated
client, so the notice cannot be re-verified programmatically.

Four datasets already in this tree (`aemo_5min_demand`, `aemo_forecast`,
`aemo_nsw1_dispatch_price`, `ausgrid_zone_substation_fy25_imputed`) wait on one
ruling on that question.  **`generators.csv`, and therefore the unit table of
`case813nem`, is a fifth.**  One decision settles all five; nothing here
narrows it, and no part of this directory should be read as having settled it.
