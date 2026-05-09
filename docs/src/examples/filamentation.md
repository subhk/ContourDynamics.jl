# Filamentation

A perturbed 4:1 elliptical vortex patch sheds thin filaments, following the
Fig. 5 setup in Dritschel (1988). This example shows how surgery keeps the
contour manageable once those filaments become too thin to resolve well.

What to look for:

- the ``m=3`` perturbation grows on the elongated vortex
- thin filaments appear as the ellipse destabilizes nonlinearly
- small features are removed by surgery
- the core vortex remains well resolved

The full script in `examples/filamentation.jl` writes snapshots, a final figure,
an MP4 animation, and `filamentation_diagnostics.csv` with the conservation
diagnostics used to track the Fig. 5 calculation.

To run the velocity calculation on an NVIDIA GPU, install/load CUDA.jl and set
`FILAMENTATION_GPU=true` when running the script. This unbounded single-layer
Euler setup uses the CUDA velocity path and the device-side surgery backend.

```@repl example_filamentation
using ContourDynamics, StaticArrays

function dritschel_perturbed_ellipse(lambda, epsilon, mode, N, pv)
    inv_lambda = inv(lambda)
    nodes = [begin
        theta = 2π * i / N
        ctheta, stheta = cos(theta), sin(theta)
        denom = stheta^2 + inv_lambda^2 * ctheta^2
        amp = epsilon * inv_lambda * cos(mode * theta) / denom
        SVector(ctheta + amp * inv_lambda * ctheta,
                inv_lambda * stheta - amp * stheta)
    end for i in 0:(N - 1)]
    PVContour(nodes, pv)
end

N = 64
lambda = 4.0
epsilon = 0.005
mode = 3
pv = 2π

c = dritschel_perturbed_ellipse(lambda, epsilon, mode, N, pv)
prob = Problem(; contours=[c], dt=0.05,
                 surgery=SurgeryParams(1e-4, 0.01, 0.15, 1e-8, 10))

circulation0 = circulation(prob)
energy0 = energy(prob)
evolve!(prob; nsteps=5)

final_contours = materialize_contours(prob)
println("Final: $(length(final_contours)) contour(s), $(total_nodes(prob)) nodes")
println("Relative circulation change: $(round(abs(circulation(prob) - circulation0) / abs(circulation0); digits=8))")
println("Relative energy change: $(round(abs(energy(prob) - energy0) / abs(energy0); digits=8))");
```

**References:**
- Dritschel, D.G. (1988). *Contour surgery.* J. Comput. Phys. **77**(1), 240--266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
