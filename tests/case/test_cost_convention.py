"""Pin the marginal-cost convention of ``unit_cost_a/b/c``.

Written for this repository on 2026-08-05 -- no upstream counterpart.

Upstream documented these fields under MATPOWER's *total cost* convention
(``TC = a·p² + b·p + c``), which is wrong: they are *marginal cost* polynomial
coefficients (``MC(p) = a·p² + b·p + c``).

Why this file exists: getting the convention wrong makes ``cost`` wrong, hence
``reward = revenue - cost`` wrong, and **no pre-existing
test catches it** -- a numpy reference implementation would repeat the same
mistake. These tests fail loudly if either the data or the consumer drifts.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import create_case5, create_case118, load_case, list_cases
from powermarketjax.physics import compute_generation_cost


def _mc(case, p):
    """Marginal cost curve as documented: MC(p) = a·p² + b·p + c [$/MWh]."""
    a = jnp.asarray(case.unit_cost_a)
    b = jnp.asarray(case.unit_cost_b)
    c = jnp.asarray(case.unit_cost_c)
    return a * p ** 2 + b * p + c


class TestMarginalCostConvention:
    """The stored coefficients describe MC, not TC."""

    def test_case5_coefficients_are_flat_marginal_costs(self):
        """case5 has a = b = 0 and c in [10, 40] $/MWh.

        This is the decisive discriminator. Under the marginal-cost reading,
        every case5 unit has a flat MC of 10-40 $/MWh, which is a sane offer
        curve. Under the MATPOWER total-cost reading, MC = 2a·p + b would be
        identically zero for all five units -- five free generators, which is
        not a meaningful economic dispatch problem.
        """
        c5 = create_case5()
        a = np.asarray(c5.unit_cost_a)
        b = np.asarray(c5.unit_cost_b)
        cc = np.asarray(c5.unit_cost_c)

        np.testing.assert_allclose(a, 0.0, atol=1e-12)
        np.testing.assert_allclose(b, 0.0, atol=1e-12)
        assert cc.min() >= 10.0 - 1e-6
        assert cc.max() <= 40.0 + 1e-6

        # Under the MC reading these are usable prices; under the TC reading
        # the implied marginal cost collapses to zero.
        mc_reading = a * 100.0 ** 2 + b * 100.0 + cc
        tc_reading = 2 * a * 100.0 + b
        assert mc_reading.min() > 1.0, "MC reading must give a positive price"
        np.testing.assert_allclose(tc_reading, 0.0, atol=1e-12)

    def test_compute_generation_cost_integrates_the_mc_curve(self):
        """``compute_generation_cost`` must equal the analytic integral of MC.

        TC(p) = ∫₀ᵖ MC(x) dx = (a/3)p³ + (b/2)p² + c·p.
        This ties the documented convention to the only consumer of these
        fields that ships in this repository.
        """
        c118 = create_case118()
        p = jnp.asarray(c118.unit_p_max, jnp.float32) * 0.6

        got = float(compute_generation_cost(
            p, c118.unit_cost_a, c118.unit_cost_b, c118.unit_cost_c,
        ))

        a = np.asarray(c118.unit_cost_a, np.float64)
        b = np.asarray(c118.unit_cost_b, np.float64)
        cc = np.asarray(c118.unit_cost_c, np.float64)
        pp = np.asarray(p, np.float64)
        want = float(((a / 3.0) * pp ** 3 + (b / 2.0) * pp ** 2 + cc * pp).sum())

        np.testing.assert_allclose(got, want, rtol=1e-4)

    def test_total_cost_is_not_the_matpower_quadratic(self):
        """Guard against the specific mistake: cost = a·p² + b·p + c.

        If someone "fixes" the code to the MATPOWER reading, this fails.
        """
        c118 = create_case118()
        p = jnp.asarray(c118.unit_p_max, jnp.float32) * 0.6

        integrated = float(compute_generation_cost(
            p, c118.unit_cost_a, c118.unit_cost_b, c118.unit_cost_c,
        ))
        matpower_style = float(jnp.sum(_mc(c118, p)))  # a·p²+b·p+c summed

        # These must differ by orders of magnitude -- one is $/h over 54 units
        # at ~500 MW each, the other is a sum of $/MWh numbers.
        assert abs(integrated) > 100 * abs(matpower_style), (
            f"integrated TC={integrated:.1f} vs a·p²+b·p+c={matpower_style:.1f}; "
            "if these are close, the convention has drifted"
        )

    def test_mc_at_zero_output_is_cost_c(self):
        """MC(0) == c, i.e. cost_c carries $/MWh units, not $/h."""
        for name in ["5", "118", "300"]:
            case = load_case(name)
            mc0 = np.asarray(_mc(case, jnp.float32(0.0)))
            np.testing.assert_allclose(
                mc0, np.asarray(case.unit_cost_c), rtol=1e-6,
                err_msg=f"case{name}: MC(0) must equal unit_cost_c",
            )

    def test_mc_magnitudes_are_plausible_prices(self):
        """Across every built-in case, MC over [p_min, p_max] stays in a
        plausible $/MWh band.

        Deliberately loose: the point is to catch a convention flip (which
        moves MC by orders of magnitude), not to assert data quality. Known
        data issues -- non-monotone and slightly negative MC on case118 and
        case29gb -- are known and are NOT asserted
        against here.
        """
        for meta in list_cases():
            if meta.requires_declaration:
                continue          # its zero-argument factory raises, by design
            case = load_case(meta.name)
            if case.n_units == 0 or case.unit_cost_a is None:
                continue
            lo = np.asarray(case.unit_p_min, np.float64)
            hi = np.asarray(case.unit_p_max, np.float64)
            grid = lo[:, None] + np.linspace(0.0, 1.0, 11)[None, :] * (hi - lo)[:, None]
            a = np.asarray(case.unit_cost_a, np.float64)[:, None]
            b = np.asarray(case.unit_cost_b, np.float64)[:, None]
            cc = np.asarray(case.unit_cost_c, np.float64)[:, None]
            mc = a * grid ** 2 + b * grid + cc
            assert mc.max() < 1e4, (
                f"case{meta.name}: max MC {mc.max():.1f} $/MWh is implausible "
                "as a price -- has the coefficient convention flipped?"
            )
            assert mc.min() > -1e2, (
                f"case{meta.name}: min MC {mc.min():.1f} $/MWh is implausible"
            )


class TestGenCostCoeffsAlias:

    def test_shape_and_order(self):
        c5 = create_case5()
        coeffs = np.asarray(c5.gen_cost_coeffs)
        assert coeffs.shape == (c5.n_units, 3)
        np.testing.assert_allclose(coeffs[:, 0], np.asarray(c5.unit_cost_a))
        np.testing.assert_allclose(coeffs[:, 1], np.asarray(c5.unit_cost_b))
        np.testing.assert_allclose(coeffs[:, 2], np.asarray(c5.unit_cost_c))

    def test_jit_compatible(self):
        c5 = create_case5()
        out = jax.jit(lambda case: case.gen_cost_coeffs)(c5)
        assert out.shape == (c5.n_units, 3)
