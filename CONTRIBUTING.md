# Contributors' Guide

Thanks for your interest in contributing to ContourDynamics.jl!

Contributions of all kinds are welcome: bug reports, documentation improvements,
new examples, performance tweaks, and core features.
The best way to get started is to open a GitHub [issue](https://github.com/subhk/ContourDynamics.jl/issues).

We follow the [ColPrac guide](https://github.com/SciML/ColPrac) for collaborative practices.
New contributors are encouraged to read it before making their first contribution.

A suggested workflow for making contributions is outlined as follows:

## 1. Set up a development environment

We assume you work from your own fork of the repository.

### Step 1: Fork & clone

On GitHub, fork:

``https://github.com/subhk/ContourDynamics.jl``

Then on your machine:

```bash
git clone https://github.com/<your-username>/ContourDynamics.jl.git
cd ContourDynamics.jl
```

### Step 2: Instantiate the project

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## 2. Run the tests

After the environment is instantiated:

**From the shell:**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The test suite checks conservation invariants (circulation, energy, enstrophy)
and validates surgery operations, so all tests should pass before opening a PR.

## 3. Build the documentation

The documentation has its own environment in ``docs/Project.toml``.

### Step 1: Instantiate the docs environment

**From the shell:**

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

### Step 2: Build the docs

```bash
julia --project=docs docs/make.jl
```

The site files will be written to ``docs/build/``.
Open ``docs/build/index.html`` in a browser to preview.

If you change the public API or examples, please update the docs and make sure they build without errors.

## 4. Pull request checklist

Before opening a PR:

- [ ] You created a feature/topic branch (not working on main).
- [ ] ``Pkg.test()`` passes.
- [ ] Docs are updated if behavior or API changed.
- [ ] New public functions have docstrings.
- [ ] The PR description briefly explains:
  - What you changed.
  - Why it's useful.
  - How you tested it.

## 5. Tips for specific areas

### Kernels & velocity computation

If you add a new kernel type, implement `segment_velocity` for it and verify that a
circular vortex patch produces the expected angular velocity analytically.

### Surgery

Changes to surgery logic should be tested with merger and filamentation examples.
Check that circulation is conserved to machine precision after reconnection.

### Diagnostics

All diagnostics are computed from contour geometry via Green's theorem.
New diagnostics should include a test against an analytical solution (e.g. Rankine vortex).

### Ewald summation

The periodic Green's function uses precomputed Fourier coefficients cached per-domain.
If you modify the cache, ensure thread safety (the caches use `ReentrantLock`).

Thank you for helping improve ContourDynamics.jl!
