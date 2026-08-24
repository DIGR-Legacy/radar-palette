"""Interchangeable backends for the azimuth NUFFT, and for the solve on top of it.

:mod:`radar_palette.gridding.nufft` hand-rolls a Kaiser-Bessel spreading kernel,
its Fourier transform by quadrature, and a conjugate-gradient solve on the
density-weighted normal equations. That implementation is correct and is kept as
the reference. This module factors the same operator into two independent choices
so that faster machinery --- some of it already in SciPy, some in specialised
NUFFT libraries --- can be used without changing the mathematics.

Two axes, chosen separately
---------------------------
An **engine** supplies ``forward`` (type-2: lattice to measured rays) and
``adjoint`` (type-1: rays to lattice). A **solver** turns those two products into
a recovered lattice. Every engine is validated against the reference operator and
every engine passes the same adjoint-consistency test, so the two axes compose
freely: any solver may be run on any engine.

Engines
~~~~~~~
``reference``
    :class:`~radar_palette.gridding.nufft._AzimuthNufftOperator`, unchanged. The
    definition of correct; the baseline every other engine is measured against.
``dense``
    No interpolation kernel at all: the DFT matrix, formed explicitly. Sounds
    wasteful and is not, because on the azimuth axis the matrix is
    ``n_rays x n_modes`` and both are ray counts --- 8 MB at 720 rays. Exact to
    round-off, and the fastest engine here below ~720 rays. **No new
    dependencies.**
``scipy``
    The reference algorithm with two mechanical substitutions: the spreading
    stencil becomes one CSR sparse matrix (replacing ``numpy.add.at``, which is an
    unbuffered scatter and was 38% of solve time), and the spread-to-spectrum
    transform becomes :func:`scipy.fft.rfft` (the spread field is real, so the
    reference was computing a conjugate-symmetric half it then discarded). Both
    changes are exact: this engine agrees with ``reference`` to ~1e-15, which is
    round-off, not approximation. **No new dependencies**, so this is the default.
``finufft``
    Type-1/type-2 transforms from the Flatiron ``finufft`` library, with a reused
    ``Plan`` and ``n_trans`` batching over range gates. Uses an
    exponential-of-semicircle kernel rather than Kaiser-Bessel, so ``kb_width``
    and ``oversamp`` do not apply; accuracy is requested directly as ``eps``.
``ducc0``
    The same transforms from ``ducc0.nufft``, batched over gates and
    multithreaded internally. Included for accuracy parity with ``finufft``, not
    for speed: measured on these sweep shapes it is the **slowest** engine here
    (2-4x the reference on ``cg``), because a radar azimuth axis is short enough
    that its per-call setup dominates the transform it is setting up for. Reach
    for it if you want ``finufft`` accuracy without the ``finufft`` dependency.
``torch``
    ``torchkbnufft``: genuinely Kaiser-Bessel (table-interpolated), batching range
    gates on the coil axis, and able to run on a GPU with ``device='cuda'``. The
    CPU timings here are around the reference's; the reason to select it is an
    already-GPU-resident pipeline.

Engine availability is checked at construction, never at import, so a minimal
install keeps working and asking for an absent engine raises a message naming the
extra to install. :func:`available_engines` reports what the environment supports.

Solvers
~~~~~~~
``cg``
    Conjugate gradients on ``(E^H W E) c = E^H W f``, iteration for iteration the
    same as the reference. Preserves the existing ``cg_resid_*`` diagnostics.
``direct``
    Cholesky factorisation of the same normal matrix, which is the limit CG is
    iterating towards --- reached in one factorisation instead of approached.

Why a direct solve suits *this* problem
---------------------------------------
The normal matrix is ``n_lattice x n_lattice``, and for a radar sweep
``n_lattice`` is a **ray count** --- 120 to 1440, not an image dimension. A dense
Cholesky at that size is microseconds, and the per-gate cost drops to a BLAS-3
triangular solve shared across all gates. The iterative machinery CG exists to
avoid is not expensive enough here to be worth avoiding. It replaces 25
transforms (12 CG iterations, each a forward and an adjoint) with one
factorisation, which is where the speed comes from --- and it is the larger
effect of the two axes. On a 720 x 1200 sweep, relative to ``nufft.py``:

===========  ==========  ==============
engine       ``cg``      ``direct``
===========  ==========  ==============
scipy        2.3x        13.5x
dense        1.8x        **32.7x**
finufft      1.2x        29.2x
===========  ==========  ==============

Choosing a faster engine buys about 2x; changing the solver buys 10-30x. Past
~1440 rays ``finufft`` overtakes ``dense`` (33x against 22x), which is the
``O(n log n)`` versus ``O(n^2)`` crossover; radar azimuth normally sits below it.

**The accuracy it buys depends on the engine, and the distinction is the whole
point.** Measured on a field *exactly* band-limited on the lattice, so the
correct answer is known in closed form (120 rays, +/-0.3-spacing jitter,
relative error against truth):

===========  ==============  ==============
engine       ``cg`` (12)     ``direct``
===========  ==============  ==============
scipy        3.0e-4          3.2e-4
torch        3.0e-5          3.0e-5
dense        3.6e-5          **1.1e-10**
finufft      3.6e-5          **2.5e-10**
ducc0        3.6e-5          **3.1e-10**
===========  ==============  ==============

A direct solve is only as accurate as the operator it inverts. On a
Kaiser-Bessel engine it converges to that engine's own kernel error (~3e-4) and
so gains nothing on accuracy at low jitter --- the iteration was never the
binding constraint there. On an engine whose transform is exact (``dense``) or
near-exact (``finufft``, ``ducc0``), the kernel error is gone and the direct
solve reaches 1e-10: six orders of magnitude, from pairing the right solver with
the right engine rather than from either alone.

Where the direct solver helps on *every* engine is degraded sampling, because
CG's convergence rate degrades with the conditioning while a factorisation does
not. At +/-0.45-spacing jitter, 12 iterations reach only 1.7e-2 on the ``scipy``
engine against 4.5e-4 for the direct solve on the same transforms --- a 37x gain
with no change of engine, and 4.3e7 with ``dense``.

The ridge term is load-bearing, not cosmetic
--------------------------------------------
For a **sector**, ``n_lattice`` exceeds the ray count --- a 30-120 deg sector at
1 deg spacing fits 91 rays onto a 366-point lattice --- so the normal matrix is
singular by construction (rank-deficient by 275 there, smallest eigenvalue at
round-off). A bare Cholesky fails outright. ``ridge`` is scaled by
``trace(B) / n_lattice`` so it means the same thing at any sweep size, and it
selects the minimum-norm solution among the infinitely many that fit the data
equally well. On that sector the direct solve reproduces the measured rays to
8.9e-11 where 12 CG iterations reach 3.7e-4, a factor of 4e6 --- and unlike the
full-circle case this gain does not need a high-accuracy engine, because what is
being fixed is the rank deficiency rather than the kernel.

Accuracy of the engines themselves
----------------------------------
Against exact trig-polynomial arithmetic, the reference's own Kaiser-Bessel
kernel at its default ``kb_width=4, oversamp=2`` is accurate to 2.5e-4. The
``dense`` engine is exact to round-off, and ``finufft``/``ducc0`` reach ~2e-13.
So at default settings the approximation error of this operator is the *kernel*,
not the transform: substituting a more accurate transform changes the answer by
~5e-4 rather than converging to the reference. That is a move towards exact
arithmetic, but it is still a change, which is why the default engine is the one
that reproduces the reference bit-for-bit.

Every figure quoted above is produced by ``benchmarks/bench_nufft_engines.py``;
the timings are from an arm64 laptop, so re-run it rather than trusting them on
other hardware.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.fft import fft, ifft, next_fast_len, rfft

from radar_palette.gridding.nufft import (
    _AzimuthNufftOperator,
    kaiser_bessel_beta,
    kaiser_bessel_kernel,
)

__all__ = [
    "DEFAULT_ENGINE",
    "DEFAULT_RIDGE",
    "DEFAULT_SOLVER",
    "ENGINES",
    "SOLVERS",
    "EngineUnavailableError",
    "available_engines",
    "make_operator",
]

# The engine that reproduces the reference to round-off and needs no dependency
# beyond the ones this package already requires.
DEFAULT_ENGINE = "scipy"

# ``cg`` preserves the reference's behaviour and its ``cg_resid_*`` diagnostics
# exactly. ``direct`` is faster on every engine, and far more accurate on the ones
# whose transform is exact (see the module docstring), but changing the default
# solve is a decision for the evaluator's own default, not for this module to make
# silently.
DEFAULT_SOLVER = "cg"

# Relative to trace(B) / n_lattice, so it is scale- and size-independent. Large
# enough to make a rank-deficient sector factorable, small enough that the data
# residual it costs (~1e-10 on a sector, measured) stays far below the
# Kaiser-Bessel engines' own kernel error (~3e-4).
DEFAULT_RIDGE = 1e-10

ENGINES = ("reference", "scipy", "dense", "finufft", "ducc0", "torch")
SOLVERS = ("cg", "direct")

# Import name -> the extra that provides it, for the error message.
_ENGINE_REQUIREMENTS = {
    "reference": (),
    "scipy": (),
    "dense": (),
    "finufft": (("finufft", "spectral"),),
    "ducc0": (("ducc0", "spectral-ducc"),),
    "torch": (("torch", "spectral-torch"), ("torchkbnufft", "spectral-torch")),
}


class EngineUnavailableError(RuntimeError):
    """An engine was requested whose optional dependency is not installed."""


def available_engines():
    """Engine names usable in this environment, in preference order.

    Returns
    -------
    tuple of str
        Always contains ``'reference'`` and ``'scipy'``, which need nothing
        beyond this package's required dependencies.
    """
    usable = []
    for name in ENGINES:
        if all(
            importlib.util.find_spec(module) is not None
            for module, _ in _ENGINE_REQUIREMENTS[name]
        ):
            usable.append(name)
    return tuple(usable)


def _require(engine):
    """Raise a message naming the extra to install, rather than an ImportError."""
    missing = [
        (module, extra)
        for module, extra in _ENGINE_REQUIREMENTS[engine]
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        modules = ", ".join(module for module, _ in missing)
        extras = sorted({extra for _, extra in missing})
        raise EngineUnavailableError(
            f"engine {engine!r} needs {modules}, which is not installed "
            f"(pip install radar-palette[{','.join(extras)}]). "
            f"Available engines: {', '.join(available_engines())}"
        )


# --------------------------------------------------------------------------- #
# Solvers. Expressed in terms of forward/adjoint only, so every solver works on
# every engine -- including the reference operator, which is how the equivalence
# tests compare a solver against itself across engines.
# --------------------------------------------------------------------------- #


def solve_cg(operator, values, n_cg=0, cg_tol=1e-10):
    """Conjugate gradients on the density-weighted normal equations.

    Iteration for iteration identical to
    :meth:`~radar_palette.gridding.nufft._AzimuthNufftOperator.solve`, including
    the analytic ``n_lattice / 360`` scale on the initial gridded estimate and the
    explicit iteration count (a loop variable would be left at its initial value
    if the body never ran).

    Parameters
    ----------
    operator : object
        Anything supplying ``forward``, ``adjoint``, ``order``, ``n_lattice``.
    values : array_like
        Ray-major sweep values, shape ``(n_rays, n_gates)``.
    n_cg : int
        Iterations. ``<= 0`` returns the unrefined density-compensated gridding.
    cg_tol : float
        Relative residual at which to stop early.

    Returns
    -------
    lattice : numpy.ndarray
    info : dict
    """
    values = np.asarray(values, dtype=np.float64)
    rays = values[operator.order]
    right_hand_side = operator.adjoint(rays)
    lattice = right_hand_side * (operator.n_lattice / 360.0)
    info = {"solver": "cg", "n_cg": int(n_cg)}
    if n_cg <= 0:
        return lattice, info

    residual = right_hand_side - operator.adjoint(operator.forward(lattice))
    reference_norm = max(float(np.linalg.norm(right_hand_side)), 1e-30)
    info["cg_resid_start"] = float(np.linalg.norm(residual) / reference_norm)

    direction = residual.copy()
    residual_sq = float(np.sum(residual * residual))
    iterations_run = 0
    for _ in range(int(n_cg)):
        applied = operator.adjoint(operator.forward(direction))
        denominator = float(np.sum(direction * applied))
        if denominator == 0.0:
            break
        iterations_run += 1
        step = residual_sq / denominator
        lattice = lattice + step * direction
        residual = residual - step * applied
        residual_sq_new = float(np.sum(residual * residual))
        if np.sqrt(residual_sq_new) / reference_norm < cg_tol:
            residual_sq = residual_sq_new
            break
        direction = residual + (residual_sq_new / residual_sq) * direction
        residual_sq = residual_sq_new

    info["cg_iters_run"] = iterations_run
    info["cg_resid_end"] = float(np.linalg.norm(residual) / reference_norm)
    return lattice, info


def solve_direct(operator, values, ridge=DEFAULT_RIDGE, **_ignored):
    """Cholesky solve of the same normal equations CG iterates towards.

    The normal matrix ``B = A^T W A`` is built one column at a time through the
    engine's own ``forward``/``adjoint``, so it is exactly the operator CG would
    have applied --- no separate derivation to drift out of step --- and the
    factorisation is cached on the operator, amortised over every field gridded
    with the same geometry.

    ``ridge`` is relative to ``trace(B) / n_lattice``. It is required, not
    optional: a sector's normal matrix is singular by construction (see the module
    docstring) and a bare Cholesky fails on it.

    Note that this solver is only as accurate as the engine it inverts. On a
    Kaiser-Bessel engine it converges to that engine's kernel error (~3e-4); pair
    it with ``dense``, ``finufft`` or ``ducc0`` to get the ~1e-10 the module
    docstring tabulates.

    Returns
    -------
    lattice : numpy.ndarray
    info : dict
        Includes ``normal_cond``, the condition number of the regularised matrix,
        which is the diagnostic to read when a sector's answer looks wrong.
    """
    values = np.asarray(values, dtype=np.float64)
    rays = values[operator.order]
    right_hand_side = operator.adjoint(rays)
    was_flat = right_hand_side.ndim == 1
    rhs_2d = right_hand_side[:, None] if was_flat else right_hand_side

    factor, condition = _normal_cholesky(operator, float(ridge))
    lattice = sla.cho_solve(factor, rhs_2d)
    info = {
        "solver": "direct",
        "n_cg": 0,
        "ridge_rel": float(ridge),
        "normal_cond": condition,
    }
    return (lattice.ravel() if was_flat else lattice), info


def _normal_cholesky(operator, ridge):
    """Factorise ``A^T W A + ridge * scale * I``, caching on the operator."""
    cached = getattr(operator, "_normal_cache", None)
    if cached is not None and cached[0] == ridge:
        return cached[1], cached[2]

    n_lattice = operator.n_lattice
    normal = operator.adjoint(operator.forward(np.eye(n_lattice)))
    # Symmetrise. Defensive rather than load-bearing, and worth being precise
    # about: the measured asymmetry the transforms leave is ~3e-16 relative on
    # every engine, and ``cho_factor(lower=True)`` reads only the lower triangle
    # anyway, so this changes neither the factorisation nor its success. It is
    # kept because the matrix is symmetric in exact arithmetic and this makes the
    # precondition true rather than nearly true, and because ``normal_cond``
    # below *does* read the whole array.
    normal = 0.5 * (normal + normal.T)
    scale = float(np.trace(normal)) / n_lattice
    regularised = normal + ridge * scale * np.eye(n_lattice)
    factor = sla.cho_factor(regularised, lower=True)
    condition = float(np.linalg.cond(regularised))
    operator._normal_cache = (ridge, factor, condition)
    return factor, condition


_SOLVERS = {"cg": solve_cg, "direct": solve_direct}


# --------------------------------------------------------------------------- #
# Engines.
# --------------------------------------------------------------------------- #


class _EngineBase:
    """Shared geometry: lattice sizing, Voronoi weights, mode map, solve dispatch.

    Subclasses supply ``forward``, ``adjoint`` and ``_engine_extras``. Everything
    that decides *what problem is being solved* --- the lattice size, the density
    weights, the split-Nyquist mode map --- lives here, so engines can only differ
    in how they compute the transforms, never in what they compute.
    """

    name = "base"

    def __init__(self, azimuths_deg, geometry, kb_width=4.0, oversamp=2.0):
        azimuths_deg = np.asarray(azimuths_deg, dtype=np.float64) % 360.0
        self.order = np.argsort(azimuths_deg)
        self.azimuths = azimuths_deg[self.order]
        self.n_rays = self.azimuths.size
        self.kb_width = float(kb_width)
        self.oversamp = float(oversamp)

        # Lattice size is measured, never assumed -- identical rule to the
        # reference operator, so every engine grids onto the same lattice.
        self.n_lattice = (
            int(geometry.nrays)
            if geometry.is_full_360
            else int(round(360.0 / geometry.az_spacing_median_deg))
        )

        # Voronoi arc-length weights in degrees, summing to 360.
        self.weights = (
            (np.roll(self.azimuths, -1) - np.roll(self.azimuths, 1)) % 360.0
        ) / 2.0

        self.lattice_modes = np.fft.fftfreq(
            self.n_lattice, d=1.0 / self.n_lattice
        ).astype(int)
        self._build_mode_map()
        self._normal_cache = None

    def _build_mode_map(self):
        """Map the ``N0`` FFT bins onto a contiguous mode list, Nyquist split.

        The reference maps modes onto an oversampled lattice; the library engines
        want a contiguous ``-N0/2 .. +N0/2`` block instead. Both carry the same
        rule: for even ``N0`` the Nyquist coefficient is **split** half to each
        end, which is what keeps the off-lattice interpolant real and symmetric.
        Assigning it wholly to one side produces a complex interpolant.
        """
        n_lattice = self.n_lattice
        self.n_modes = n_lattice + 1 if n_lattice % 2 == 0 else n_lattice
        self.mode_low = -(self.n_modes // 2)
        rows, columns, weights = [], [], []
        for column, mode in enumerate(self.lattice_modes):
            if n_lattice % 2 == 0 and mode == -n_lattice // 2:
                rows += [0, self.n_modes - 1]
                columns += [column, column]
                weights += [0.5, 0.5]
            else:
                rows.append(mode - self.mode_low)
                columns.append(column)
                weights.append(1.0)
        self._pad = sp.csr_matrix(
            (weights, (rows, columns)),
            shape=(self.n_modes, n_lattice),
            dtype=np.float64,
        )

    @staticmethod
    def _as_columns(values):
        """Accept a 1-D vector or a 2-D (axis, gates) block; remember which."""
        return (values[:, None], True) if values.ndim == 1 else (values, False)

    def _modes_from_lattice(self, values_2d):
        """Lattice values to contiguous mode coefficients, transposed for engines."""
        coefficients = fft(values_2d, axis=0) / self.n_lattice
        return np.ascontiguousarray((self._pad @ coefficients).T)

    def _lattice_from_modes(self, modes_by_column):
        """Contiguous mode coefficients (batch-major) back to lattice values."""
        truncated = self._pad.T @ np.ascontiguousarray(modes_by_column.T)
        return np.real(ifft(truncated, axis=0))

    def adjoint_test(self, seed=0):
        """Relative mismatch of ``<A u, f>_W`` against ``<u, A^H f>``.

        Should be at machine-precision level. A value near 1 means a scale factor
        is missing from the transpose, and conjugate gradients on the normal
        equations would be solving the wrong problem. Every engine is held to
        this, because an engine whose adjoint is not the transpose of its forward
        would break both solvers in ways a smoothness check would not catch.
        """
        rng = np.random.default_rng(seed)
        lattice = rng.normal(size=(self.n_lattice, 1))
        rays = rng.normal(size=(self.n_rays, 1))
        forward_inner = float(
            np.sum(self.forward(lattice) * rays * self.weights[:, None])
        )
        adjoint_inner = float(np.sum(lattice * self.adjoint(rays)))
        return float(
            abs(forward_inner - adjoint_inner)
            / max(abs(forward_inner), abs(adjoint_inner), 1e-30)
        )

    def solve(self, values, n_cg=0, cg_tol=1e-10, solver=DEFAULT_SOLVER, ridge=None):
        """Recover the uniform lattice from the measured rays.

        Parameters
        ----------
        values : array_like
            Ray-major sweep values, ``(n_rays, n_gates)``.
        n_cg, cg_tol : int, float
            Passed to the ``cg`` solver; ignored by ``direct``.
        solver : {'cg', 'direct'}
        ridge : float, optional
            ``direct`` only; defaults to :data:`DEFAULT_RIDGE`.

        Returns
        -------
        lattice : numpy.ndarray
        info : dict
        """
        if solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        if solver == "direct":
            lattice, info = solve_direct(
                self, values, ridge=DEFAULT_RIDGE if ridge is None else ridge
            )
        else:
            lattice, info = solve_cg(self, values, n_cg=n_cg, cg_tol=cg_tol)
        info["engine"] = self.name
        return lattice, info

    def describe(self):
        """Engine parameters, for the evaluator's report.

        ``kb_width``/``oversamp`` are reported as *requested* on every engine, but
        only the Kaiser-Bessel engines act on them; ``finufft`` and ``ducc0``
        select their own grid from ``eps``. ``_engine_extras`` says what each
        engine actually used.
        """
        described = {
            "engine": self.name,
            "kb_width": self.kb_width,
            "oversamp": self.oversamp,
            "n_nominal": self.n_lattice,
            "n_modes": self.n_modes,
        }
        described.update(self._engine_extras())
        return described

    def _engine_extras(self):
        return {}


class _ReferenceEngine(_AzimuthNufftOperator):
    """The unmodified reference operator, exposed through the engine interface.

    Subclasses rather than wraps, so ``forward`` and ``adjoint`` are literally the
    reference implementations and cannot drift from them. The added ``solve``
    accepts a ``solver`` argument so the direct solver can be run on the reference
    transforms --- which is how the tests establish that the speedup comes from
    the solver and the engine independently.
    """

    name = "reference"

    def __init__(self, azimuths_deg, geometry, kb_width=4.0, oversamp=2.0):
        super().__init__(azimuths_deg, geometry, kb_width=kb_width, oversamp=oversamp)
        self.n_modes = self.n_lattice
        self._normal_cache = None

    def solve(self, values, n_cg=0, cg_tol=1e-10, solver=DEFAULT_SOLVER, ridge=None):
        """As the reference, plus the ``direct`` solver on the same transforms."""
        if solver == "cg":
            lattice, info = super().solve(values, n_cg=n_cg, cg_tol=cg_tol)
            info["solver"] = "cg"
        elif solver == "direct":
            lattice, info = solve_direct(
                self, values, ridge=DEFAULT_RIDGE if ridge is None else ridge
            )
        else:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        info["engine"] = self.name
        return lattice, info

    def describe(self):
        """Engine parameters, matching the reference operator's reported set."""
        return {
            "engine": self.name,
            "kb_width": self.kb_width,
            "oversamp": self.oversamp,
            "kb_beta": self.beta,
            "n_nominal": self.n_lattice,
            "M_oversampled": self.n_oversampled,
            "kernel": "kaiser_bessel",
        }


class _ScipyEngine(_EngineBase):
    """Reference algorithm, SciPy machinery: CSR spreading and a real FFT.

    Two exact substitutions, both of which the reference left on the table:

    ``numpy.add.at`` to a CSR matrix product
        The spreading stencil is a fixed sparse pattern, so the scatter is a
        sparse matrix-vector product. ``np.add.at`` is an unbuffered ufunc method
        that cannot use the BLAS and was 38% of the reference's solve time; the
        same operation as ``S @ f`` is a single indexed reduction, and the
        interpolation is ``S.T @ g`` on the transpose of that one matrix, which
        makes the exactness of the adjoint structural rather than a property to
        re-derive.

    ``fft`` to ``rfft``
        The spread field is real. The reference computed the full complex spectrum
        and used the conjugate-symmetric half it had just computed for nothing.
    """

    name = "scipy"

    def __init__(self, azimuths_deg, geometry, kb_width=4.0, oversamp=2.0):
        super().__init__(azimuths_deg, geometry, kb_width, oversamp)
        self.n_oversampled = next_fast_len(int(np.ceil(self.oversamp * self.n_lattice)))
        self.beta = kaiser_bessel_beta(self.kb_width, self.oversamp)
        self.cell_deg = 360.0 / self.n_oversampled

        position = self.azimuths / self.cell_deg
        self.n_stencil = int(np.floor(self.kb_width)) + 1
        first_cell = np.ceil(position - self.kb_width / 2.0).astype(int)
        stencil_offsets = np.arange(self.n_stencil)[None, :]
        cells = (first_cell[:, None] + stencil_offsets) % self.n_oversampled
        kernel_values = kaiser_bessel_kernel(
            position[:, None] - (first_cell[:, None] + stencil_offsets),
            self.kb_width,
            self.beta,
        )
        ray_index = np.repeat(np.arange(self.n_rays), self.n_stencil)
        # One matrix serves both directions: ``spread = S @ f`` and
        # ``interpolate = S.T @ g``, so forward and adjoint cannot disagree.
        self._spread = sp.csr_matrix(
            (kernel_values.ravel(), (cells.ravel(), ray_index)),
            shape=(self.n_oversampled, self.n_rays),
        )
        self._interpolate = self._spread.T.tocsr()

        # Deapodisation by quadrature from the same kernel used for spreading, so
        # the two are consistent for any (width, beta) -- the reference's rule.
        self.n_half = self.n_oversampled // 2 + 1
        self._deapodisation_half = self._kernel_fourier_transform(
            np.arange(self.n_half)
        )
        contiguous_modes = np.arange(self.n_modes) + self.mode_low
        self._deapodisation_modes = self._kernel_fourier_transform(contiguous_modes)
        self._mode_bins = contiguous_modes % self.n_oversampled

    def _kernel_fourier_transform(self, mode_index, n_quadrature=4096):
        offsets = np.linspace(-self.kb_width / 2.0, self.kb_width / 2.0, n_quadrature)
        kernel = kaiser_bessel_kernel(offsets, self.kb_width, self.beta)
        cell_step = offsets[1] - offsets[0]
        phase = np.exp(
            -2j * np.pi * np.outer(np.asarray(mode_index) / self.n_oversampled, offsets)
        )
        return np.real(phase @ kernel) * cell_step

    def forward(self, lattice_values):
        """Type-2 transform: uniform lattice to measured azimuths."""
        values_2d, was_flat = self._as_columns(lattice_values)
        coefficients = fft(values_2d, axis=0) / self.n_lattice
        modes = self._pad @ coefficients
        padded = np.zeros((self.n_oversampled, values_2d.shape[1]), dtype=complex)
        padded[self._mode_bins] = modes / self._deapodisation_modes[:, None]
        oversampled = np.real(ifft(padded, axis=0)) * self.n_oversampled
        out = self._interpolate @ oversampled
        return out.ravel() if was_flat else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: measured azimuths to uniform lattice.

        The exact transpose of :meth:`forward` under the density weights, by
        construction: it uses the transpose of the same sparse matrix.
        """
        values_2d, was_flat = self._as_columns(ray_values)
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d
        spread = self._spread @ weighted_values
        # rfft, then read the negative modes off the conjugate-symmetric half.
        half_spectrum = rfft(spread, axis=0) / self._deapodisation_half[:, None]
        bins = self._mode_bins
        negative = bins >= self.n_half
        modes = np.empty((self.n_modes, values_2d.shape[1]), dtype=complex)
        modes[~negative] = half_spectrum[bins[~negative]]
        modes[negative] = np.conj(half_spectrum[self.n_oversampled - bins[negative]])
        truncated = self._pad.T @ modes
        out = np.real(ifft(truncated, axis=0))
        return out.ravel() if was_flat else out

    def _engine_extras(self):
        return {
            "kb_beta": self.beta,
            "M_oversampled": self.n_oversampled,
            "kernel": "kaiser_bessel",
        }


class _DenseEngine(_EngineBase):
    """The exact transform: a dense DFT matrix, no interpolation kernel at all.

    Every other engine approximates ``exp(i m phi_j)`` by spreading onto an
    oversampled lattice with a kernel, because for a large problem forming the
    matrix is unthinkable. On the azimuth axis of a radar sweep it is not: the
    matrix is ``n_rays x n_modes``, and both are ray counts. At 720 rays that is
    an 8 MB array and a BLAS-3 ``GEMM`` per transform.

    So this engine has no kernel error, no oversampling factor and no accuracy
    parameter --- ``forward`` is the trig polynomial evaluated exactly, to
    round-off. Combined with the ``direct`` solver it recovers a band-limited
    field to ~1e-10 (the ridge, not the arithmetic), against ~3e-4 for the
    Kaiser-Bessel engines, and it is the fastest engine here up to about 720 rays.
    Beyond ~1440 rays the ``O(n_rays * n_modes)`` product loses to ``finufft``'s
    ``O(n log n)``, which is the crossover the NUFFT literature is written about;
    radar azimuth sits well below it.

    Memory is the reason this is not the default: the matrix is
    ``16 * n_rays * n_modes`` bytes (33 MB at 1440 rays, 133 MB at 2880), where
    the Kaiser-Bessel engines are linear in the ray count.
    """

    name = "dense"

    def __init__(self, azimuths_deg, geometry, kb_width=4.0, oversamp=2.0):
        super().__init__(azimuths_deg, geometry, kb_width, oversamp)
        modes = np.arange(self.n_modes) + self.mode_low
        self._matrix = np.exp(1j * np.outer(np.deg2rad(self.azimuths), modes))

    def forward(self, lattice_values):
        """Type-2 transform, exactly: evaluate the trig polynomial at the rays."""
        values_2d, was_flat = self._as_columns(lattice_values)
        coefficients = fft(values_2d, axis=0) / self.n_lattice
        out = np.real(self._matrix @ (self._pad @ coefficients))
        return out.ravel() if was_flat else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: the conjugate transpose of the same matrix."""
        values_2d, was_flat = self._as_columns(ray_values)
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d
        modes = self._matrix.conj().T @ weighted_values.astype(complex)
        out = np.real(ifft(self._pad.T @ modes, axis=0))
        return out.ravel() if was_flat else out

    def _engine_extras(self):
        return {
            "kernel": "none_exact",
            "matrix_bytes": int(self._matrix.nbytes),
        }


class _FinufftEngine(_EngineBase):
    """Flatiron ``finufft`` type-1/type-2, planned once and batched over gates.

    Accuracy is requested as ``eps`` rather than derived from a kernel width; the
    library picks its own oversampling and its exponential-of-semicircle kernel
    accordingly, so ``kb_width`` and ``oversamp`` are recorded but not used.
    """

    name = "finufft"

    def __init__(
        self,
        azimuths_deg,
        geometry,
        kb_width=4.0,
        oversamp=2.0,
        eps=1e-9,
    ):
        _require("finufft")
        super().__init__(azimuths_deg, geometry, kb_width, oversamp)
        import finufft

        self._finufft = finufft
        self.eps = float(eps)
        self._points = np.ascontiguousarray(np.deg2rad(self.azimuths))
        self._plans = {}

    def _plan(self, nufft_type, n_columns):
        """Plans are cached per (type, batch width): setpts is the expensive part."""
        key = (nufft_type, n_columns)
        if key not in self._plans:
            plan = self._finufft.Plan(
                nufft_type,
                (self.n_modes,),
                n_trans=n_columns,
                eps=self.eps,
                isign=1 if nufft_type == 2 else -1,
            )
            plan.setpts(self._points)
            self._plans[key] = plan
        return self._plans[key]

    def forward(self, lattice_values):
        """Type-2 transform: uniform lattice to measured azimuths."""
        values_2d, was_flat = self._as_columns(lattice_values)
        modes = self._modes_from_lattice(values_2d)
        rays = self._plan(2, values_2d.shape[1]).execute(modes)
        out = np.ascontiguousarray(np.real(rays).T)
        return out.ravel() if was_flat else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: measured azimuths to uniform lattice."""
        values_2d, was_flat = self._as_columns(ray_values)
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d
        source = np.ascontiguousarray(weighted_values.T.astype(complex))
        modes = self._plan(1, values_2d.shape[1]).execute(source)
        out = self._lattice_from_modes(modes)
        return out.ravel() if was_flat else out

    def _engine_extras(self):
        return {"eps": self.eps, "kernel": "exponential_semicircle"}


class _Ducc0Engine(_EngineBase):
    """``ducc0.nufft``: batched over gates and internally multithreaded.

    Same transform contract as the ``finufft`` engine and the same
    ``eps``-driven accuracy, with no PyTorch dependency and thread control
    through ``nthreads`` (``0`` uses every hardware thread).
    """

    name = "ducc0"

    def __init__(
        self,
        azimuths_deg,
        geometry,
        kb_width=4.0,
        oversamp=2.0,
        eps=1e-9,
        nthreads=0,
    ):
        _require("ducc0")
        super().__init__(azimuths_deg, geometry, kb_width, oversamp)
        import ducc0.nufft

        self._nufft = ducc0.nufft
        self.eps = float(eps)
        self.nthreads = int(nthreads)
        self._coordinates = np.ascontiguousarray(
            np.deg2rad(self.azimuths).reshape(-1, 1)
        )

    def forward(self, lattice_values):
        """Type-2 transform: uniform lattice to measured azimuths."""
        values_2d, was_flat = self._as_columns(lattice_values)
        modes = self._modes_from_lattice(values_2d)
        rays = self._nufft.u2nu(
            grid=modes,
            coord=self._coordinates,
            forward=False,
            epsilon=self.eps,
            nthreads=self.nthreads,
            fft_order=False,
        )
        out = np.ascontiguousarray(np.real(rays).T)
        return out.ravel() if was_flat else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: measured azimuths to uniform lattice."""
        values_2d, was_flat = self._as_columns(ray_values)
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d
        source = np.ascontiguousarray(weighted_values.T.astype(complex))
        modes = self._nufft.nu2u(
            points=source,
            coord=self._coordinates,
            forward=True,
            epsilon=self.eps,
            nthreads=self.nthreads,
            out=np.empty((source.shape[0], self.n_modes), dtype=complex),
            fft_order=False,
        )
        out = self._lattice_from_modes(modes)
        return out.ravel() if was_flat else out

    def _engine_extras(self):
        return {
            "eps": self.eps,
            "nthreads": self.nthreads,
            "kernel": "exponential_semicircle",
        }


class _TorchKbNufftEngine(_EngineBase):
    """``torchkbnufft`` table interpolation; range gates ride the coil axis.

    Genuinely Kaiser-Bessel, like the reference, and the only engine here that
    runs on a GPU (``device='cuda'``). Note the phase convention: this library's
    forward transform is ``exp(-i * omega * (k - N // 2))`` for image index ``k``.
    With ``k = m - mode_low`` and ``N = n_modes``, ``k - N // 2`` is exactly the
    signed mode ``m``, so passing ``omega = -phi`` yields the ``exp(+i m phi)``
    this operator is defined by, and no separate fftshift correction is needed.
    Getting that wrong is not a small error --- it produced a 69% discrepancy
    against exact arithmetic, which the adjoint test still passed, because a
    consistently wrong pair of operators is still a consistent pair.
    """

    name = "torch"

    def __init__(
        self,
        azimuths_deg,
        geometry,
        kb_width=4.0,
        oversamp=2.0,
        numpoints=6,
        device="cpu",
    ):
        _require("torch")
        super().__init__(azimuths_deg, geometry, kb_width, oversamp)
        import torch
        import torchkbnufft

        self._torch = torch
        self.numpoints = int(numpoints)
        self.device = torch.device(device)
        self.dtype = torch.complex128
        self.grid_size = int(np.ceil(self.oversamp * self.n_modes))
        self._forward_module = torchkbnufft.KbNufft(
            im_size=(self.n_modes,),
            grid_size=(self.grid_size,),
            numpoints=self.numpoints,
            dtype=self.dtype,
            device=self.device,
        )
        self._adjoint_module = torchkbnufft.KbNufftAdjoint(
            im_size=(self.n_modes,),
            grid_size=(self.grid_size,),
            numpoints=self.numpoints,
            dtype=self.dtype,
            device=self.device,
        )
        omega = -np.deg2rad(self.azimuths)
        omega = (omega + np.pi) % (2.0 * np.pi) - np.pi
        self._omega = torch.tensor(
            omega.reshape(1, -1), dtype=torch.float64, device=self.device
        )

    def forward(self, lattice_values):
        """Type-2 transform: uniform lattice to measured azimuths."""
        values_2d, was_flat = self._as_columns(lattice_values)
        modes = self._modes_from_lattice(values_2d)[None]
        image = self._torch.tensor(modes, dtype=self.dtype, device=self.device)
        rays = self._forward_module(image, self._omega).cpu().numpy()[0]
        out = np.ascontiguousarray(np.real(rays).T)
        return out.ravel() if was_flat else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: measured azimuths to uniform lattice."""
        values_2d, was_flat = self._as_columns(ray_values)
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d
        source = np.ascontiguousarray(weighted_values.T.astype(complex))[None]
        k_space = self._torch.tensor(source, dtype=self.dtype, device=self.device)
        modes = self._adjoint_module(k_space, self._omega).cpu().numpy()[0]
        out = self._lattice_from_modes(modes)
        return out.ravel() if was_flat else out

    def _engine_extras(self):
        return {
            "numpoints": self.numpoints,
            "device": str(self.device),
            "M_oversampled": self.grid_size,
            "kernel": "kaiser_bessel",
        }


_ENGINE_CLASSES = {
    "reference": _ReferenceEngine,
    "scipy": _ScipyEngine,
    "dense": _DenseEngine,
    "finufft": _FinufftEngine,
    "ducc0": _Ducc0Engine,
    "torch": _TorchKbNufftEngine,
}


def make_operator(
    azimuths_deg,
    geometry,
    engine=DEFAULT_ENGINE,
    kb_width=4.0,
    oversamp=2.0,
    **engine_options,
):
    """Build an azimuth NUFFT operator on the named engine.

    Parameters
    ----------
    azimuths_deg : array_like
        Measured azimuths in degrees; need not be sorted or in ``[0, 360)``.
    geometry : SweepGeometry
        Supplies ``nrays``, ``is_full_360`` and ``az_spacing_median_deg``, which
        between them fix the lattice size. Every engine uses the same rule.
    engine : str
        One of :data:`ENGINES`, or ``'auto'`` for the default engine. ``'auto'``
        resolves to :data:`DEFAULT_ENGINE` rather than to the fastest installed
        engine on purpose: a result that silently changes with the contents of the
        environment is not reproducible, and the faster library engines are not
        bit-identical to the reference (see the module docstring).
    kb_width, oversamp : float
        Kaiser-Bessel kernel width and lattice oversampling. Used by the
        ``reference``, ``scipy`` and ``torch`` engines; recorded but unused by
        ``finufft`` and ``ducc0``, which take ``eps`` instead.
    **engine_options
        Engine-specific: ``eps`` (``finufft``, ``ducc0``), ``nthreads``
        (``ducc0``), ``numpoints`` and ``device`` (``torch``).

    Returns
    -------
    object
        An operator with ``forward``, ``adjoint``, ``adjoint_test``, ``solve`` and
        ``describe``.

    Raises
    ------
    ValueError
        Unknown engine name.
    EngineUnavailableError
        Known engine whose optional dependency is not installed.
    """
    if engine == "auto":
        engine = DEFAULT_ENGINE
    if engine not in _ENGINE_CLASSES:
        raise ValueError(f"engine must be one of {ENGINES} or 'auto', got {engine!r}")
    return _ENGINE_CLASSES[engine](
        azimuths_deg,
        geometry,
        kb_width=kb_width,
        oversamp=oversamp,
        **engine_options,
    )
