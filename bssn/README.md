# BSSN Solver

1D spherically-symmetric BSSN evolution of a Schwarzschild black hole puncture,
based on Brown (2008).  The solver uses RK4 time integration, explicit or compact
finite differences, and Kreiss-Oliger or compact filters.

---

## Prerequisites

The solver lives inside the `pynwave2d` repo and imports the `nwave` library from
the parent directory via `sys.path`; no installation step is needed.

Required Python packages:

```
numpy  scipy  numba  matplotlib     # plus tomllib (stdlib, Python >= 3.11)
```

`pentapy` is **optional**: it is needed only when `DerivSolveMethod = "PENTAPY"`
(or a filter's `solve_method = "PENTAPY"`).  The default `LUSOLVE` does not use it.

---

## Quick start

```bash
cd bssn/
python solver.py params.toml
```

Output is written to `output_bssn_0/` (ignored by git).  Progress is printed every
`print_interval` steps; a NaN in the metric triggers an early exit.

---

## Parameter file reference

Parameter files are [TOML](https://toml.io), passed as the sole argument to
`solver.py`.

### Grid and time stepping

| Key | Default | Description |
|-----|---------|-------------|
| `Nx` | — | Number of grid points (odd values are decremented) |
| `Xmin` | — | Left boundary; negative gives a two-sided (extended) domain |
| `Xmax` | — | Right boundary |
| `NGhost` | — | Ghost-zone width (2–4) |
| `cfl` | — | `dt = cfl * dx`; must satisfy 0 < cfl < 1 |
| `Nt` | — | Number of time steps |
| `level` | 0 | Refinement level: multiplies `Nx`, `Nt` and output intervals by 2^level |

### Derivatives

| Key | Default | Description |
|-----|---------|-------------|
| `D1` | — | First-derivative stencil, or `[background, near-BH]` pair when `BHMask = true` |
| `D2` | — | Second-derivative stencil, same form as `D1` |
| `DerivSolveMethod` | `"LUSOLVE"` | Linear solver for compact stencils: `LUSOLVE`, `SCIPY`, `PENTAPY`, `D_LU`, `D_INV` |

Stencil names: explicit `E4`, `E42`, `E44`, `E6`/`E642`, `E666`; compact `JT4`, `JT6`,
`JP6`, `KP4` (D1 only), `DSQ6A`, `DSQ6B`; mixed `ME4`/`ME44`, `ME6`/`ME642`.

### BH stencil mask *(optional)*

| Key | Default | Description |
|-----|---------|-------------|
| `BHMask` | false | Switch to the second `D1`/`D2` stencil near the puncture |
| `BHMaskPos` | 0.0 | Puncture location |
| `BHMaskWidth` | 0.1 | Transition half-width |

### Filters

Filters are given as an array of tables; omit the block for no filtering.  The old
scalar `Filter = ...` keys are rejected.

```toml
[[filters]]
type = "KO6"            # KO6, KO8, JTT4, JTT6, JTP6, JTT8, JTP8, KP4, F2
apply = "Rhs"           # Rhs, Vars, Derivs
sigma = 0.5             # KO dissipation strength (KO6/KO8)
filter_boundary = false
# compact filters: alpha, beta, solve_method, frequency
```

### Black hole and initial data

| Key | Default | Description |
|-----|---------|-------------|
| `Mass` | — | ADM mass M |
| `initial_data` | — | `"Puncture"` or `"EddingtonFinkelstein"` |
| `collapsed_lapse` | — | Start with the pre-collapsed lapse (recommended for `Puncture`) |

### Gauge choices

The lapse and the Gamma-driver shift damping are chosen with two tables; either
may be omitted to get the default (1+log with k = 2, constant η = 2).

```toml
[lapse]
type = "1+log"
k = 2.0

[shift_damping]
type = "constant"
eta = 2.0
```

Lapse (`[lapse]`), Bona-Massó slicing ∂_t α = β ∂_r α − α² f(α) K:

| `type` | f(α) | Keys (default) |
|--------|------|----------------|
| `"1+log"` | k / α | `k` (2.0) |
| `"N1"` | (2/α)(1 − α/2 + α²/3) | — |
| `"N2"` | (2/α)(1 − α + α² − α³/4) | — |
| `"N3"` | (2/α)(1 − 3α/2 + 2α² − α³ + α⁴/5) | — |
| `"ShockFull"` | 1 + κ/α² | `kappa` (1.0) |

The outer Sommerfeld speed for α is the gauge's asymptotic speed √f(1).

Every Bona-Massó gauge also accepts `eta_lapse` (default 0.0), which adds
Laplacian damping `+ eta_lapse D²α`.  The term vanishes for any constant
lapse and damps short wavelengths fastest (rate `eta_lapse k²/2`), but it is
parabolic, so large values need a smaller time step.  Around `0.01–0.03` is
practical at `dr = 0.02`.

```toml
[lapse]
type = "1+log"
eta_lapse = 0.02
```

The experimental `"telegrapher"` lapse is second order in time.  It evolves
a 10th field, `lapseA` = ∂_t α:

    ∂_t α = A + β ∂_r α
    ∂_t A = c² D²α − 2μ (A + 2αK) + s μ² g (α − α_ref) + β ∂_r A

| Key | Default | Meaning |
|-----|---------|---------|
| `c` | 1.5 | gauge speed; also the Sommerfeld speed of α and A |
| `mu` | 1.0 | damping rate |
| `mu2_sign` | −1.0 | s in [−1, 1]; −1 is distortionless (every k moves at c), 0 drops the mass term |
| `mu2_ref` | `"initial"` | α_ref: `"none"`, `"unity"`, `"chi"` (√χ), `"chi_clamped"` (√min(χ,1)), `"initial"` (α at t = 0) |
| `mu2_gate` | `"alpha2"` | g: `"none"`, `"alpha"`, `"alpha2"`, `"chi1"` (off once χ > 1 + 8 `gate_width`) |
| `gate_width` | 0.05 | width of the `"chi1"` gate |

A starts at the 1+log equilibrium −2αK (zero for puncture data).  See
`gauges/telegrapher.py` for why the mass term needs a reference lapse and a gate.
`lapseA` is written to the `.curve` snapshots after `gB`.

Shift damping (`[shift_damping]`), the η in ∂_t B = … − η B:

| `type` | η | Keys (default) |
|--------|---|----------------|
| `"constant"` | η | `eta` (2.0, must be ≥ 1) |
| `"Schnetter"` | η / (1 + (r/R)²) | `eta` (2.0), `R` (50.0) |
| `"eta_MB"` | R₀ \|∂_r W\| / (√g̃_rr (1 − W)²), W = √χ (Müller & Brügmann 2010) | `R0` (1.31241) |
| `"peaked_lorentz"` | bg + peak / (1 + (r/R)²) | `bg` (2.0), `peak` (2.0), `R` (1.31241) |

`eta_MB` is stiff at an under-resolved puncture; see its docstring.  Unknown
`type`s or keys are rejected at startup.  Older parameter files still run: a
top-level `eta = ...` is read as `[shift_damping] type = "constant"`, and an
absent `[lapse]` table means 1+log with k = 2 (giving both forms of `eta` is
an error).

To add a gauge, write a class in `gauges/lapse.py`, `gauges/shift.py` or a new
module imported in `gauges/__init__.py`, with the `@register_lapse("name")` or
`@register_shift("name")` decorator; its constructor arguments are its
parameter-file keys.  A lapse gauge that needs an extra evolved field sets
`n_aux`, `aux_names`, `initial_aux()` and `rhs_aux()`, as the telegrapher does.

### Boundaries and diagnostics

| Key | Default | Description |
|-----|---------|-------------|
| `bssn_outer_bc` | `"Sommerfeld"` | Outer BC: `"Sommerfeld"` or `"Sponge"` |
| `bhmask_outer_radius` | 75.0 | Outer radius of the region used for constraint norms |

### Output

| Key | Default | Description |
|-----|---------|-------------|
| `output_dir` | — | Base name of the output directory (solver appends `_<level>`) |
| `output_interval` | — | Write `.curve` snapshots every this many steps |
| `print_interval` | — | Print diagnostics and log constraint norms every this many steps |

---

## Output files

`<output_dir>_<level>/` contains:

- **`bssn_NNNN.curve`** — snapshot at step NNNN: a `# TIME t` header, then one
  `# name` block per field with two columns `x  value`.  Fields in order:
  `chi, grr, gtt, Arr, K, Gt, alpha, beta, gB`, then constraints `ham, mom, gamcom`.
- **`bssn_constraints.dat`** — CSV with columns `time, Ham, Mom, radius_bh, mass_bh`,
  written every `print_interval` steps.  Norms exclude the region inside the
  horizon and outside `bhmask_outer_radius`.

---

## To Do

+ Test explicit 642 derivs
+ Test advection derivs
+ remove ghost points for output
+ convergence test RK4
+ What's the accuracy gain?
+ Speed test of matrix solver methods
+ Add option to filter derivs.
+ Add option for multiple filters
+ Build compact advective deriv.
+ Convergence testing all around
+ Check derivs of all BSSN functions in test scripts
+ Comment and label test scripts

## Done

+ Validate the RHS using ingoing-EF data
+ Put in extended domain
+ output log grid
+ Make advective derivs more efficient using NUMBA
+ Sommerfeld outer boundary conditions
+ Add KO dissipation
+ Add option to mask constraints near punctures
