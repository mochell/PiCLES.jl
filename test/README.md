# PiCLES test layout

This directory now separates automated tests from interactive diagnostics.

## What goes where

- `runtests.jl`: entrypoint for automated tests.
- `unit/`: small, CI-safe smoke/integration tests.
- `unit/test_ode_solvers_legacy.jl`: legacy ODE solver validation script kept for migration into modern `@testset` style.
- `manual/`: visual inspection scripts (plots/movies, exploratory runs).
- `benchmark/`: performance-focused scripts.
- `_archive/`: outdated or superseded scripts kept for reference.

## Running tests

From the repository root:

```bash
julia --project=PiCLES -e 'using Pkg; Pkg.activate("PiCLES/test"); push!(LOAD_PATH, "PiCLES"); include("PiCLES/test/runtests.jl")'
```

## Notes

- Files in `manual/` are intentionally not included by `runtests.jl`.
- Files in `benchmark/` are intentionally not included by `runtests.jl`.
