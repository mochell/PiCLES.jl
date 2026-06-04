# Model

PiCLES is a *2nd-generation+* surface wave model. It sits between
parameterised fetch relations (2nd-gen) and discretised spectral
action-balance models like WW3, WAM, SWAN (3rd-gen). The model targets the
dominant wind sea — the part of the spectrum most relevant to air–sea
coupling — and reduces the state vector to a small number of scalars per
ocean cell.

## State

Each ocean grid cell carries a tiny state vector

```math
s_n = \big(\varepsilon,\; m_x,\; m_y\big),
```

where ``\varepsilon`` is the total wave energy of the parametric peak and
``(m_x, m_y)`` are the energy-weighted group-velocity components.

Wave dynamics live on Lagrangian particles whose state is

```math
s_p = \big(\ln\varepsilon,\; \bar c_{g,x},\; \bar c_{g,y},\; x,\; y\big).
```

## Governing equations

Particles integrate the Kudryavtsev-type system
([Kudryavtsev2021](@citet); [Hell2025](@citet)) for log-energy,
energy-weighted group velocity, and position:

```math
\begin{aligned}
\frac{\mathrm d}{\mathrm dt} \ln\varepsilon
   &= -\bar{\boldsymbol c}_g \cdot \boldsymbol G_n
      + \frac{r_g}{\omega_p}\, \mathcal S^{cg}
      + \mathcal S^{\varepsilon},\\[4pt]
\frac{\mathrm d}{\mathrm dt}\bar c_{g,i}
   &= -\bar c_{g,2}\,\bar c_{g,1}\,\frac{1}{\omega_p}\,\mathcal S^{\mathrm{dir}}
      - \bar c_{g,i}\,\frac{r_g}{\omega_p}\,\mathcal S^{cg},\\[4pt]
\frac{\mathrm d}{\mathrm dt} x_i &= \bar c_{g,i}.
\end{aligned}
```

The source terms ``\mathcal S^{\varepsilon}``, ``\mathcal S^{cg}`` and
``\mathcal S^{\mathrm{dir}}`` parameterise wind input, dissipation and
directional turning. They are configured through
[`PiCLES.ParticleSystems.particle_waves_v5.ODEParameters`](@ref) and tunable
constants ``\gamma``, ``q``, ``C_\alpha``, ``C_\varphi``, ``C_e``, ``r_g``.

Each particle is solved with its own ODE step, independent of the ocean
timestep. Wave–wave interactions are parameterised along trajectories and on
nodes for cross-term interaction.

## Particle-in-Cell remeshing

Every ocean timestep, particle state is *deposited* onto the grid using a
PIC weighting kernel

```math
\hat m = \sum_n w_n\, m_n, \qquad
\hat \varepsilon = \sum_n w_n\, \varepsilon_n,
```

where ``w_n`` are bilinear (CIC) weights to the surrounding nodes. The
deposit is purely additive, so it conserves energy and momentum and handles
sharp gradients and shocks well. The remeshing step is the source of the
"Particle-in-Cell" name and was originally developed for plasma physics
([Brackbill1986](@citet); [Harlow1988](@citet)).

## Why this is fast

For an ocean grid with ``N_s`` cells, the work is dominated by
``\mathcal{O}(N_s\, \log N_s)`` particle integration with no strict CFL
constraint on the ocean timestep — particles take their own short steps
between deposits. This is what gives PiCLES a 1–4 order of magnitude
speedup over fully spectral models at resolutions relevant to CMIP6-class
Earth System Models.

## Validation in the paper

The companion JAMES paper validates PiCLES against:

- **Static fetch**: comparison to classical fetch-limited growth curves.
- **Dynamical fetch**: reproduces the moving-fetch experiments of
  Hell et al. (2021).
- **2D box geometries**: open ocean, diagonal boundaries, half-domain
  configurations, growing/decaying winds, rotating winds.
- **Sphere and tripolar grids**: aqua-planet and realistic ocean
  configurations.

Most of these cases live as scripts under `test/T03_*.jl` and
`test/T04_*.jl` in the repository.

## References

```@bibliography
Pages = ["model.md"]
Canonical = false
```
