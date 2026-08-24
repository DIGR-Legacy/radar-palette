"""Engines must agree with the reference; solvers must beat it.

Two claims carry this module, and each has a test that fails if the claim is
false rather than a test that merely exercises the code.

**Equivalence.** Every engine is a different way to compute the same operator, so
every engine is checked against the reference implementation directly --- and,
for the engines accurate enough to justify it, against exact trig-polynomial
arithmetic computed here in closed form. A test that only checked an engine
against itself (its own adjoint identity, say) would pass for an engine that is
consistently wrong; ``torchkbnufft`` initially was exactly that, off by 69% from
a phase-convention mismatch while passing its adjoint test perfectly.

**Improvement.** The direct solver is claimed to be more accurate than the
conjugate-gradient default. That is tested against a field which is *exactly*
band-limited on the recovery lattice, so the correct lattice is known analytically
and "more accurate" is measured against truth rather than against the other
method's output.
"""

from __future__ import annotations

import numpy as np
import pytest

# test_evaluator is a sibling module, not a package member -- the suite has no
# tests/__init__.py -- so this is a plain top-level import that relies on pytest
# putting the test directory on sys.path (rootdir-relative insertion, the default
# "prepend" import mode). Reusing its fixtures rather than restating them keeps
# the two NUFFT test modules measuring the same sweep.
from test_evaluator import (
    RANGE_FIRST_M,
    RANGE_SPACING_M,
    band_limited_field,
    make_radar,
    uniform_azimuths,
)

from radar_palette.gridding import nufft_engines as engines
from radar_palette.gridding.census import SweepClass, census_sweep
from radar_palette.gridding.evaluator import SweepSpectralEvaluator
from radar_palette.gridding.nufft import _AzimuthNufftOperator

# Engines installable as optional extras; the test is skipped rather than failed
# when the extra is absent, so a minimal install still collects a full suite.
OPTIONAL_ENGINES = ("finufft", "ducc0", "torch")
ALL_ENGINES = engines.ENGINES

# The reference's own Kaiser-Bessel kernel is accurate to ~2.5e-4 against exact
# arithmetic at the default kb_width=4, oversamp=2. Engines that share that
# kernel must match it much more closely than that; engines with a different
# kernel can only be expected to agree to the kernel error itself.
KB_KERNEL_ERROR = 2.5e-4


def engine_or_skip(name):
    """Skip rather than fail when an optional engine is not installed."""
    if name not in engines.available_engines():
        pytest.skip(f"engine {name!r} requires an optional extra that is absent")
    return name


@pytest.fixture
def jittered():
    """A sweep classified NON_UNIFORM: 120 rays, +/-0.3-spacing azimuth jitter."""
    ngates = 34
    nrays = 120
    spacing = 360.0 / nrays
    rng = np.random.default_rng(2)
    azimuths = np.sort(
        (uniform_azimuths(nrays) + rng.uniform(-0.3, 0.3, nrays) * spacing) % 360.0
    )
    geometry = census_sweep(make_radar(azimuths, ngates=ngates), 0)
    assert geometry.sweep_class is SweepClass.NON_UNIFORM
    values = band_limited_field(
        azimuths, ngates, azimuth_modes=(0, 1, 3), range_modes=(0, 1, 2)
    )
    return geometry, azimuths, values, ngates


def exact_lattice_case(nrays=120, ngates=8, jitter=0.3, seed=3, modes=(0, 1, 3, 7)):
    """A field exactly band-limited on the lattice, so truth is known in closed form.

    Every accuracy claim about the solvers rests on this: the field is a finite
    sum of azimuthal harmonics with ``|m| <= 7``, evaluated both at the jittered
    measured azimuths and at the uniform lattice the solvers are trying to
    recover. The lattice values are therefore the exact answer, not a proxy for
    it, and an error can be reported as an absolute figure rather than as a
    difference between two approximations.
    """
    spacing = 360.0 / nrays
    rng = np.random.default_rng(seed)
    azimuths = np.sort(
        (np.arange(nrays) * spacing + rng.uniform(-jitter, jitter, nrays) * spacing)
        % 360.0
    )
    coefficients = np.random.default_rng(seed + 100)
    amplitude = coefficients.normal(size=(len(modes), ngates))
    phase = coefficients.uniform(0.0, 2.0 * np.pi, (len(modes), ngates))

    def evaluate(sample_azimuths):
        out = np.zeros((len(sample_azimuths), ngates))
        for index, mode in enumerate(modes):
            out += amplitude[index][None, :] * np.cos(
                np.deg2rad(mode * sample_azimuths)[:, None] + phase[index][None, :]
            )
        return out

    geometry = census_sweep(make_radar(azimuths, ngates=ngates), 0)
    lattice_azimuths = np.arange(nrays) * 360.0 / nrays
    return geometry, azimuths, evaluate(azimuths), evaluate(lattice_azimuths)


def exact_forward(operator, lattice_values):
    """Evaluate the band-limited interpolant in closed form, no kernel involved.

    The mathematical definition of what every engine's ``forward`` approximates:
    the trig polynomial whose coefficients are the lattice DFT, with the Nyquist
    mode split (a cosine) rather than assigned to one side. Deliberately written
    as an explicit sum over modes so it shares no machinery with any engine.
    """
    n_lattice = operator.n_lattice
    coefficients = np.fft.fft(lattice_values, axis=0) / n_lattice
    modes = np.fft.fftfreq(n_lattice, d=1.0 / n_lattice).astype(int)
    radians = np.deg2rad(operator.azimuths)
    out = np.zeros((operator.n_rays, lattice_values.shape[1]))
    for index, mode in enumerate(modes):
        if n_lattice % 2 == 0 and mode == -n_lattice // 2:
            out += np.real(np.cos(mode * radians)[:, None] * coefficients[index])
        else:
            out += np.real(np.exp(1j * mode * radians)[:, None] * coefficients[index])
    return out


class TestEngineRegistry:
    def test_the_dependency_free_engines_are_always_available(self):
        """A minimal install must still have a working NUFFT path."""
        available = engines.available_engines()
        assert "reference" in available
        assert "scipy" in available

    def test_the_default_engine_needs_no_optional_dependency(self):
        assert engines.DEFAULT_ENGINE in ("reference", "scipy")

    def test_available_engines_is_a_subset_of_the_declared_set(self):
        assert set(engines.available_engines()) <= set(ALL_ENGINES)

    def test_an_unknown_engine_is_a_value_error(self, jittered):
        geometry, azimuths, _, _ = jittered
        with pytest.raises(ValueError, match="engine must be one of"):
            engines.make_operator(azimuths, geometry, engine="nope")

    def test_auto_resolves_to_the_default_not_to_the_fastest_installed(self, jittered):
        """``auto`` must not depend on what happens to be installed.

        An engine chosen from the environment would make a result irreproducible
        between two machines with the same code, which matters more here than the
        speed difference, because the library engines are not bit-identical to
        the reference.
        """
        geometry, azimuths, _, _ = jittered
        assert (
            engines.make_operator(azimuths, geometry, engine="auto").name
            == engines.DEFAULT_ENGINE
        )

    def test_an_absent_engine_names_the_extra_to_install(self, jittered):
        """The error must be actionable, not an ImportError from deep inside."""
        geometry, azimuths, _, _ = jittered
        absent = [
            name for name in OPTIONAL_ENGINES if name not in engines.available_engines()
        ]
        if not absent:
            pytest.skip("every optional engine is installed")
        with pytest.raises(engines.EngineUnavailableError, match="pip install"):
            engines.make_operator(azimuths, geometry, engine=absent[0])

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_every_engine_reports_its_identity_and_lattice(self, jittered, engine):
        geometry, azimuths, _, _ = jittered
        described = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        ).describe()
        assert described["engine"] == engine
        assert described["n_nominal"] == geometry.nrays


class TestEngineEquivalence:
    """Different machinery, same operator."""

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_the_adjoint_is_the_transpose_of_the_forward(self, jittered, engine):
        """Per-engine, because a broken transpose invalidates both solvers.

        This is the reference module's load-bearing test, and it is not inherited:
        an engine could pair a correct forward with a mis-scaled adjoint and every
        smoothness check would still pass.
        """
        geometry, azimuths, _, _ = jittered
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        assert operator.adjoint_test() < 1e-10

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_the_lattice_size_is_identical_across_engines(self, jittered, engine):
        """Engines may differ in how they transform, never in what they solve."""
        geometry, azimuths, _, _ = jittered
        reference = _AzimuthNufftOperator(azimuths, geometry)
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        assert operator.n_lattice == reference.n_lattice
        np.testing.assert_allclose(operator.weights, reference.weights)
        np.testing.assert_array_equal(operator.order, reference.order)

    def test_the_reference_engine_is_the_reference_bit_for_bit(self, jittered):
        """Not merely close: the same code, so the same bits.

        If this ever fails, the engine wrapper has started reimplementing the
        reference instead of delegating to it.
        """
        geometry, azimuths, values, _ = jittered
        expected, _ = _AzimuthNufftOperator(azimuths, geometry).solve(values, n_cg=12)
        got, _ = engines.make_operator(azimuths, geometry, engine="reference").solve(
            values, n_cg=12, solver="cg"
        )
        np.testing.assert_array_equal(got, expected)

    def test_the_scipy_engine_matches_the_reference_to_round_off(self, jittered):
        """The default engine's substitutions are exact, not approximations.

        CSR spreading replaces ``np.add.at`` and ``rfft`` replaces ``fft`` on real
        input; neither changes the arithmetic, so the tolerance here is round-off
        (~1e-13) and not a kernel error (~1e-4). A regression that made the
        default engine merely *approximately* right would land between the two.
        """
        geometry, azimuths, values, _ = jittered
        expected, _ = _AzimuthNufftOperator(azimuths, geometry).solve(values, n_cg=12)
        got, _ = engines.make_operator(azimuths, geometry, engine="scipy").solve(
            values, n_cg=12, solver="cg"
        )
        assert np.max(np.abs(got - expected)) < 1e-13 * np.ptp(expected)

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_every_engine_agrees_with_the_reference_to_the_kernel_error(
        self, jittered, engine
    ):
        """The floor on agreement is the reference's own approximation error.

        No engine can be expected to match the reference more closely than the
        reference matches exact arithmetic, so 10x the Kaiser-Bessel kernel error
        is the honest bound for an engine with a different kernel --- and it is
        still tight enough to catch a convention or scaling mistake, which shows
        up at tens of percent.
        """
        geometry, azimuths, values, _ = jittered
        expected, _ = _AzimuthNufftOperator(azimuths, geometry).solve(values, n_cg=12)
        got, _ = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        ).solve(values, n_cg=12, solver="cg")
        assert np.max(np.abs(got - expected)) < 10 * KB_KERNEL_ERROR * np.ptp(expected)

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_a_one_dimensional_input_matches_a_single_column(self, jittered, engine):
        """1-D and 2-D calls must not take different code paths to different answers."""
        geometry, azimuths, _, _ = jittered
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        rng = np.random.default_rng(4)
        lattice = rng.normal(size=operator.n_lattice)
        rays = rng.normal(size=operator.n_rays)
        np.testing.assert_allclose(
            operator.forward(lattice), operator.forward(lattice[:, None]).ravel()
        )
        np.testing.assert_allclose(
            operator.adjoint(rays), operator.adjoint(rays[:, None]).ravel()
        )

    def test_the_nyquist_split_does_not_change_a_real_valued_result(self, jittered):
        """Documents a property that had to be measured, not assumed.

        The reference module splits the Nyquist coefficient half to ``+N0/2`` and
        half to ``-N0/2`` because assigning it wholly to one side "produces a
        complex interpolant". That is true of the *interpolant*, and it is worth
        keeping for that reason --- but for a real-valued field it makes no
        difference to what this operator returns, because ``forward`` takes the
        real part and ``Re exp(-i m phi) == cos(m phi)`` when the coefficient is
        real. Measured here: the forward differs by round-off and the adjoint by
        exactly zero.

        Recorded as a test because the alternative is worse in both directions. A
        contributor who assumes the split is load-bearing will not touch a
        pathological-looking branch that is in fact inert; one who assumes it is
        dead code will delete a line that *is* load-bearing for complex input and
        for fidelity to the reference. Neither would learn this from the code, and
        a mutation that removes the split passes every other test in this file.
        """
        import scipy.sparse as sparse

        geometry, azimuths, _, _ = jittered
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        assert operator.n_lattice % 2 == 0, "fixture must have a Nyquist bin"

        modes = np.fft.fftfreq(operator.n_lattice, d=1.0 / operator.n_lattice).astype(
            int
        )
        nyquist_column = int(np.argmin(modes))
        unsplit = operator._pad.toarray()
        unsplit[:, nyquist_column] = 0.0
        unsplit[0, nyquist_column] = 1.0

        rng = np.random.default_rng(3)
        lattice = rng.normal(size=(operator.n_lattice, 3))
        rays = rng.normal(size=(operator.n_rays, 3))
        split_pad = operator._pad

        def forward_with(pad):
            operator._pad = pad
            return operator.forward(lattice)

        def adjoint_with(pad):
            operator._pad = pad
            return operator.adjoint(rays)

        try:
            forward_split = forward_with(split_pad)
            forward_unsplit = forward_with(sparse.csr_matrix(unsplit))
            adjoint_split = adjoint_with(split_pad)
            adjoint_unsplit = adjoint_with(sparse.csr_matrix(unsplit))
        finally:
            operator._pad = split_pad

        assert np.max(np.abs(forward_split - forward_unsplit)) < 1e-12 * np.ptp(
            forward_split
        )
        np.testing.assert_array_equal(adjoint_split, adjoint_unsplit)

    def test_the_dense_engine_is_exact_to_round_off(self, jittered):
        """It forms the DFT matrix, so there is no kernel error to measure.

        The strongest statement available about any engine here, and the reason a
        dependency-free engine can reach the accuracy the NUFFT libraries do: on
        the azimuth axis the exact operator is small enough to write down.
        """
        geometry, azimuths, _, _ = jittered
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        lattice = np.random.default_rng(9).normal(size=(operator.n_lattice, 3))
        truth = exact_forward(operator, lattice)
        assert np.max(np.abs(operator.forward(lattice) - truth)) < 1e-12 * np.ptp(truth)

    @pytest.mark.parametrize("engine", ("finufft", "ducc0"))
    def test_the_library_engines_beat_the_reference_kernel_on_exact_arithmetic(
        self, jittered, engine
    ):
        """These engines are *more* accurate than the code they replace.

        Which is why they differ from the reference by ~5e-4: that gap is the
        reference's Kaiser-Bessel error, not theirs. Asserting this keeps the
        difference from being read as an engine defect.
        """
        geometry, azimuths, _, _ = jittered
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine), eps=1e-12
        )
        reference = _AzimuthNufftOperator(azimuths, geometry)
        lattice = np.random.default_rng(6).normal(size=(operator.n_lattice, 3))
        truth = exact_forward(operator, lattice)
        engine_error = np.max(np.abs(operator.forward(lattice) - truth))
        reference_error = np.max(np.abs(reference.forward(lattice) - truth))
        assert engine_error < 1e-8 * np.ptp(truth)
        assert engine_error < reference_error


class TestDirectSolver:
    """The claim is accuracy, and it is measured against known truth."""

    @pytest.mark.parametrize("engine", ("dense", "finufft", "ducc0"))
    def test_on_an_exact_engine_it_is_orders_more_accurate(self, engine):
        """The headline claim, and it needs an engine without a kernel error.

        Twelve CG iterations recover this field to ~3.6e-5; the direct solve on
        the same transforms reaches ~1e-10. The margin asserted is deliberately
        loose (100x against a measured 3e5) so the test tracks the claim rather
        than the figures.
        """
        geometry, azimuths, values, truth = exact_lattice_case()
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        iterative, _ = operator.solve(values, n_cg=12, solver="cg")
        direct, _ = operator.solve(values, solver="direct")
        scale = np.ptp(truth)
        iterative_error = np.max(np.abs(iterative - truth)) / scale
        direct_error = np.max(np.abs(direct - truth)) / scale
        assert direct_error < 1e-8
        assert direct_error < iterative_error / 100.0

    def test_on_a_kaiser_bessel_engine_it_converges_to_the_kernel_error(self):
        """The honest limit of the direct solve on an approximate transform.

        A direct solve inverts the operator it is given, so on the default
        Kaiser-Bessel engine it lands on that engine's own ~3e-4 kernel error and
        buys no accuracy at this jitter --- the iteration was never the binding
        constraint. Asserting it keeps the accuracy claim attached to the
        engine/solver *pair*, which is where it belongs: a reader who took the
        headline figure to be a property of ``solver='direct'`` alone would be
        wrong, and this test is what says so.
        """
        geometry, azimuths, values, truth = exact_lattice_case()
        operator = engines.make_operator(azimuths, geometry, engine="scipy")
        direct, _ = operator.solve(values, solver="direct")
        error = np.max(np.abs(direct - truth)) / np.ptp(truth)
        assert 1e-5 < error < 1e-2

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_it_reaches_the_limit_conjugate_gradients_converge_towards(self, engine):
        """Same solution, one factorisation instead of many iterations.

        Establishes that ``direct`` is not a *different* estimator that happens to
        score better --- it is the least-squares solution CG is approaching, so
        running CG far past its default lands on the same answer. True on every
        engine, including the approximate ones: each converges to the solution of
        *its own* normal equations, which is exactly the claim.
        """
        geometry, azimuths, values, _ = exact_lattice_case()
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        converged, info = operator.solve(values, n_cg=500, cg_tol=1e-14, solver="cg")
        direct, _ = operator.solve(values, solver="direct")
        assert info["cg_resid_end"] < 1e-10
        assert np.max(np.abs(direct - converged)) < 1e-7 * np.ptp(converged)

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_the_advantage_widens_as_the_sampling_degrades(self, engine):
        """CG's convergence rate degrades with jitter; a factorisation does not.

        This is the one accuracy gain that holds on *every* engine, including the
        Kaiser-Bessel ones, because what degrades is the conditioning of the
        system rather than the fidelity of the transform. At +/-0.45-spacing
        jitter twelve iterations reach only ~1.7e-2 while the direct solve holds
        4.5e-4 on the same transforms --- so the fixed iteration count, not the
        kernel, is what limits the default here.
        """
        geometry, azimuths, values, truth = exact_lattice_case(jitter=0.45)
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        iterative, _ = operator.solve(values, n_cg=12, solver="cg")
        direct, _ = operator.solve(values, solver="direct")
        scale = np.ptp(truth)
        iterative_error = np.max(np.abs(iterative - truth)) / scale
        direct_error = np.max(np.abs(direct - truth)) / scale
        assert iterative_error > 1e-3
        assert direct_error < iterative_error / 10.0

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_it_works_on_every_engine(self, jittered, engine):
        """Solver and engine are independent axes, so all pairs must compose."""
        geometry, azimuths, values, _ = jittered
        lattice, info = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        ).solve(values, solver="direct")
        assert info["solver"] == "direct"
        assert info["engine"] == engine
        assert np.all(np.isfinite(lattice))

    def test_it_reports_the_conditioning_it_solved_at(self, jittered):
        """``normal_cond`` is the diagnostic for a suspicious sector result."""
        geometry, azimuths, values, _ = jittered
        _, info = engines.make_operator(azimuths, geometry).solve(
            values, solver="direct"
        )
        assert info["normal_cond"] > 1.0
        assert info["ridge_rel"] == engines.DEFAULT_RIDGE

    def test_the_factorisation_is_cached_across_fields(self, jittered):
        """One geometry, many fields: the factorisation must be paid for once."""
        geometry, azimuths, values, _ = jittered
        operator = engines.make_operator(azimuths, geometry)
        operator.solve(values, solver="direct")
        cached = operator._normal_cache
        operator.solve(values * 2.0 + 1.0, solver="direct")
        assert operator._normal_cache is cached

    def test_a_changed_ridge_is_refactorised_rather_than_reused(self, jittered):
        """The cache key must include the ridge, or it would silently be ignored."""
        geometry, azimuths, values, _ = jittered
        operator = engines.make_operator(azimuths, geometry)
        _, loose = operator.solve(values, solver="direct", ridge=1e-4)
        _, tight = operator.solve(values, solver="direct", ridge=1e-10)
        assert loose["normal_cond"] < tight["normal_cond"]

    def test_an_unknown_solver_is_a_value_error(self, jittered):
        geometry, azimuths, values, _ = jittered
        with pytest.raises(ValueError, match="solver must be one of"):
            engines.make_operator(azimuths, geometry).solve(values, solver="lstsq")


class TestSectorConditioning:
    """A sector's normal matrix is singular by construction, not by accident."""

    @pytest.fixture
    def sector(self):
        """A jittered 30-120 deg sector: 91 rays onto a 360-point lattice."""
        spacing = 1.0
        rng = np.random.default_rng(7)
        count = 91
        azimuths = np.sort(
            30.0 + np.arange(count) * spacing + rng.uniform(-0.3, 0.3, count) * spacing
        )
        geometry = census_sweep(make_radar(azimuths, ngates=20), 0)
        assert geometry.sweep_class is SweepClass.NON_UNIFORM
        values = band_limited_field(
            azimuths, 20, azimuth_modes=(0, 1, 3), range_modes=(0, 1, 2)
        )
        return geometry, azimuths, values

    def test_the_lattice_exceeds_the_ray_count(self, sector):
        """Which is what makes the system underdetermined -- the premise."""
        geometry, azimuths, _ = sector
        operator = engines.make_operator(azimuths, geometry)
        assert operator.n_lattice > operator.n_rays

    def test_the_default_ridge_keeps_a_rank_deficient_sector_solvable(self, sector):
        """Without regularisation the Cholesky of a singular matrix fails outright."""
        geometry, azimuths, values = sector
        lattice, info = engines.make_operator(azimuths, geometry).solve(
            values, solver="direct"
        )
        assert np.all(np.isfinite(lattice))
        assert info["normal_cond"] > 1e6

    def test_it_fits_the_measured_rays_better_than_the_iterative_default(self, sector):
        """The right yardstick when the lattice is not unique.

        With more unknowns than measurements there is no single correct lattice,
        so accuracy is judged by how well the recovered lattice reproduces the
        rays that *were* measured. The direct solve fits them to 8.9e-11; twelve
        CG iterations reach 3.7e-4. Unlike the full-circle case this gain needs no
        high-accuracy engine, because the deficiency being repaired is the rank of
        the system rather than the fidelity of the kernel.
        """
        geometry, azimuths, values = sector
        operator = engines.make_operator(azimuths, geometry)
        measured = values[operator.order]
        scale = np.ptp(values)
        iterative, _ = operator.solve(values, n_cg=12, solver="cg")
        direct, _ = operator.solve(values, solver="direct")
        iterative_residual = np.max(np.abs(operator.forward(iterative) - measured))
        direct_residual = np.max(np.abs(operator.forward(direct) - measured))
        assert direct_residual / scale < 1e-6
        assert direct_residual < iterative_residual

    def test_a_larger_ridge_trades_data_fit_for_conditioning(self, sector):
        """The knob does what it claims, in the direction it claims."""
        geometry, azimuths, values = sector
        operator = engines.make_operator(azimuths, geometry)
        measured = values[operator.order]
        loose, loose_info = operator.solve(values, solver="direct", ridge=1e-4)
        tight, tight_info = operator.solve(values, solver="direct", ridge=1e-10)
        assert loose_info["normal_cond"] < tight_info["normal_cond"]
        assert np.max(np.abs(operator.forward(loose) - measured)) > np.max(
            np.abs(operator.forward(tight) - measured)
        )


class TestEvaluatorWiring:
    """The evaluator must pass the choice through and report what it used."""

    def test_the_default_path_is_unchanged_in_behaviour(self, jittered):
        """Swapping the default engine must not move the evaluator's output.

        The whole justification for defaulting to ``scipy`` is that it is the
        reference algorithm with exact substitutions. Measured end to end, on the
        evaluated field rather than on the lattice.
        """
        geometry, azimuths, values, ngates = jittered
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        default = SweepSpectralEvaluator(values, geometry, azimuths).evaluate(
            range_mesh, azimuth_mesh
        )
        reference = SweepSpectralEvaluator(
            values, geometry, azimuths, az_engine="reference"
        ).evaluate(range_mesh, azimuth_mesh)
        assert np.max(np.abs(default - reference)) < 1e-10 * np.ptp(reference)

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_the_engine_is_recorded_in_the_report(self, jittered, engine):
        geometry, azimuths, values, _ = jittered
        report = SweepSpectralEvaluator(
            values, geometry, azimuths, az_engine=engine_or_skip(engine)
        ).report
        assert report.az_path == "nufft_kb"
        assert report.extras["engine"] == engine
        assert report.extras["adjoint_test_rel"] < 1e-10

    def test_the_solver_is_recorded_in_the_report(self, jittered):
        geometry, azimuths, values, _ = jittered
        for solver in engines.SOLVERS:
            report = SweepSpectralEvaluator(
                values, geometry, azimuths, az_solver=solver
            ).report
            assert report.extras["solver"] == solver

    def test_the_direct_solver_reports_no_conjugate_gradient_residuals(self, jittered):
        """It does not iterate, so it must not claim a residual history."""
        geometry, azimuths, values, _ = jittered
        extras = SweepSpectralEvaluator(
            values, geometry, azimuths, az_solver="direct"
        ).report.extras
        assert "cg_resid_end" not in extras
        assert extras["normal_cond"] > 1.0

    def test_the_direct_solver_recovers_a_smooth_field(self, jittered):
        geometry, azimuths, values, ngates = jittered
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        recovered = SweepSpectralEvaluator(
            values, geometry, azimuths, az_solver="direct"
        ).evaluate(range_mesh, azimuth_mesh)
        assert np.max(np.abs(recovered - values)) < 0.01 * np.ptp(values)

    def test_a_constant_field_survives_the_direct_solver(self, jittered):
        """The sharpest test of the NUFFT path, on the alternative solver.

        The reference module records 0.07% deviation here with its CG default and
        60% without. A direct solve must be at least as good as the default, or it
        has no business being offered.
        """
        geometry, azimuths, _, ngates = jittered
        constant = np.full((azimuths.size, ngates), 12.5)
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        recovered = SweepSpectralEvaluator(
            constant, geometry, azimuths, az_solver="direct", field_units="other"
        ).evaluate(range_mesh, azimuth_mesh)
        assert np.max(np.abs(recovered - 12.5)) / 12.5 < 1e-3

    def test_an_unknown_solver_is_rejected_at_construction(self, jittered):
        """Not deferred to the one path that happens to consult it."""
        geometry, azimuths, values, _ = jittered
        with pytest.raises(ValueError, match="az_solver"):
            SweepSpectralEvaluator(values, geometry, azimuths, az_solver="lstsq")

    def test_an_unknown_solver_is_rejected_even_on_a_uniform_sweep(self):
        """The path that ignores the argument must still validate it.

        A uniform sweep never reaches the NUFFT path, so a typo here would
        otherwise pass silently and only fail later on a jittered sweep.
        """
        azimuths = uniform_azimuths(120)
        geometry = census_sweep(make_radar(azimuths, ngates=20), 0)
        assert geometry.sweep_class is SweepClass.EXACT_UNIFORM_PERIODIC
        values = band_limited_field(azimuths, 20)
        with pytest.raises(ValueError, match="az_solver"):
            SweepSpectralEvaluator(values, geometry, azimuths, az_solver="lstsq")
