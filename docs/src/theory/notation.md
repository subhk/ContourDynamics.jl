# Notation and Parameter Glossary

This page collects the mathematical symbols and the corresponding Julia names
used throughout the documentation. Symbols are also defined beside the
equations where they first appear; this glossary is the quick reference.

Unless a page states otherwise, bold lowercase letters are two-dimensional
vectors, ``|\mathbf{v}|`` is the Euclidean norm of ``\mathbf{v}``, and contour
indices are cyclic, so ``\mathbf{x}_{N+1}=\mathbf{x}_1`` for a closed contour.
All length-valued inputs must use the same coordinate units. PV jumps have units
of inverse time, so velocities have units of length per time.

## Flow and contour symbols

| Symbol | Meaning | Julia representation |
|---|---|---|
| ``\mathbf{x}=(x,y)`` | Target or contour-node position | `SVector{2,T}` |
| ``\mathbf{x}'`` | Source or integration position | A point on a source contour |
| ``C_j`` | Boundary of PV patch ``j`` | `PVContour` |
| ``j`` | Contour or node index, according to context | Julia array index |
| ``q(\mathbf{x})`` | Potential-vorticity field; vorticity for 2D Euler | Piecewise-constant field represented by contours |
| ``\Delta q_j`` or ``q_j`` | PV jump carried by contour ``j`` | `PVContour.pv` |
| ``\psi`` | Streamfunction | Recovered implicitly through the Green's function |
| ``\mathbf{u}=(-\partial_y\psi,\partial_x\psi)`` | Incompressible velocity | Returned by `velocity` or `velocity!` |
| ``\mathcal{L}`` | PV-inversion operator | Selected by the kernel type |
| ``G(r)`` | Scalar Green's function at separation ``r`` | `EulerKernel`, `QGKernel`, or `SQGKernel` |
| ``r=|\mathbf{x}-\mathbf{x}'|`` | Source-target distance | Computed by velocity kernels |
| ``dA'`` | Source-area element | Used in the continuous area integral |
| ``d\mathbf{x}'`` | Oriented tangent line element along a contour | Segment or cubic-arc differential |
| ``\hat{\mathbf{t}}`` | Unit tangent of a straight segment | ``(\mathbf{b}-\mathbf{a})/|\mathbf{b}-\mathbf{a}|`` |
| ``\mathbf{n}`` | Unit normal pointing left of the oriented chord | Cubic Dritschel arc normal |

## Kernel symbols

| Symbol | Meaning | Julia name |
|---|---|---|
| ``L_d`` | Rossby deformation radius | `Ld` |
| ``\kappa=L_d^{-1}`` | Inverse deformation radius | Derived internally |
| ``K_0`` | Modified Bessel function of the second kind, order zero | Used by `QGKernel` |
| ``\theta`` | Active surface buoyancy in SQG | PV jump stored in `PVContour.pv` |
| ``\delta_{\mathrm{SQG}}`` | SQG kernel regularization length | `δ_sqg` or `SQGKernel.δ` |
| ``\gamma_E`` | Euler--Mascheroni constant | Limit of the regularized QG remainder |

The SQG regularization ``\delta_{\mathrm{SQG}}`` and the surgery proximity
threshold ``\delta`` are independent parameters. They may be chosen similarly,
but the package does not identify them.

## Periodic and Ewald symbols

| Symbol | Meaning | Julia name or definition |
|---|---|---|
| ``L_x,L_y`` | Domain half-widths | `PeriodicDomain.Lx`, `PeriodicDomain.Ly` |
| ``A=4L_xL_y`` | Full periodic-domain area | Derived internally |
| ``\mathbf{r}`` | Displacement from source to target | ``\mathbf{x}-\mathbf{x}'`` |
| ``\mathbf{n}=(n,m)\in\mathbb{Z}^2`` | Periodic-image index | Truncated by `n_images` |
| ``\mathbf{L}_{\mathbf{n}}=(2nL_x,2mL_y)`` | Image-lattice translation | Derived internally |
| ``\alpha=\sqrt{\pi/(L_xL_y)}`` | Ewald splitting parameter used by the package | `EwaldCache.alpha` |
| ``\mathbf{k}=(\pi p/L_x,\pi s/L_y)`` | Fourier wavevector, ``p,s\in\mathbb{Z}`` | `EwaldCache.kx`, `EwaldCache.ky` |
| ``E_1(z)=\int_z^\infty e^{-t}/t\,dt`` | Exponential integral | `ContourDynamics._expint_e1` |
| ``\operatorname{erf}``, ``\operatorname{erfc}`` | Error function and complementary error function | SQG Ewald splitting |
| `n_fourier` | Maximum Fourier index retained in each direction | Ewald-cache keyword |
| `n_images` | Maximum nearby image index retained in each direction | Ewald-cache keyword |

## Surgery and remeshing symbols

| Symbol | Meaning | Julia name |
|---|---|---|
| ``\mathbf{x}_j`` | Contour node ``j`` | `contour.nodes[j]` |
| ``\ell_j=|\mathbf{x}_{j+1}-\mathbf{x}_j|`` | Length of segment ``j`` | `arc_lengths` output |
| ``\kappa_j`` | Signed local curvature at node ``j`` | Computed during remeshing |
| ``K_j`` | Nonlocal vorticity-weighted curvature scale | Computed during remeshing |
| ``\mathbf{m}_i`` | Midpoint of source segment ``i`` | Derived internally |
| ``d_{ij}=|\mathbf{x}_j-\mathbf{m}_i|`` | Node-to-source-midpoint distance | Derived internally |
| ``\delta`` | Reconnection proximity threshold | `SurgeryParams.δ` or `.delta` |
| ``\mu`` | Minimum target segment length | `SurgeryParams.μ` or `.mu` |
| ``\Delta_{\max}`` | Maximum target segment length | `SurgeryParams.Δ_max` or `.Delta_max` |
| ``A_{\min}`` | Minimum retained absolute contour area | `SurgeryParams.area_min` |
| ``n_{\mathrm{surgery}}`` | Number of timesteps between surgery passes | `SurgeryParams.n_surgery` |
| ``\rho(s)`` | Node-density function along arclength | Remeshing density |
| ``M(s)=\int_0^s\rho(\sigma)d\sigma`` | Cumulative weighted arclength | Used to place new nodes |
| ``L_c`` | Total contour perimeter | Sum of segment lengths |
| ``N_{\mathrm{seg}}`` | Number of redistributed segments | Chosen from density and spacing bounds |
| ``p\in[0,1]`` | Local coordinate along one cubic arc | Quadrature/interpolation parameter |
| ``\eta(p)`` | Normal displacement of the cubic arc from its chord | Cubic interpolation polynomial |

## Time-integration symbols

| Symbol | Meaning | Julia name |
|---|---|---|
| ``n`` | Discrete time-level index | Current evolution step |
| ``\mathbf{x}^n`` | Flat vector of all node positions at level ``n`` | Stepper node buffers |
| ``\Delta t`` | Fixed timestep | `dt` |
| ``\mathbf{k}_1,\ldots,\mathbf{k}_4`` | RK4 stage velocities | `RK4Stepper.k1`, ..., `.k4` |
| `nsteps` | Number of timesteps requested from `evolve!` | `nsteps` keyword |
| `step_offset` | Global step number preceding a continued batch | `step_offset` keyword |

## Multi-layer and beta-plane symbols

| Symbol | Meaning | Julia name or representation |
|---|---|---|
| ``N`` | Number of physical QG layers | `nlayers(kernel_or_problem)` |
| ``i,j\in\{1,\ldots,N\}`` | Physical-layer indices | Tuple positions in `layers` |
| ``\mathbf{C}=[C_{ij}]`` | Physical layer-stretching operator; may be nonsymmetric for unequal depths | `coupling` |
| ``\mathbf{W}=\operatorname{diag}(H_i)`` | Layer-thickness weight matrix satisfying ``\mathbf W\mathbf C=\mathbf C^{\mathsf T}\mathbf W`` | `layer_thicknesses` |
| ``\mathbf{P}`` | Matrix whose columns are vertical eigenmodes | `MultiLayerQGKernel.eigenvectors` |
| ``\mathbf{\Lambda}=\operatorname{diag}(\lambda_m)`` | Diagonal matrix of modal eigenvalues | `MultiLayerQGKernel.eigenvalues` |
| ``m`` | Vertical-mode index | Index into modal eigenvalues |
| ``L_d^{(m)}=1/\sqrt{|\lambda_m|}`` | Deformation radius of a nonbarotropic mode | Entry of `Ld` |
| ``\beta`` | Meridional gradient of planetary PV | `beta` |
| ``y`` | Meridional coordinate | Second coordinate of a node |
| ``n_\beta`` | Number of beta-staircase interfaces | `n_beta` argument to `beta_staircase` |
| ``\Delta y=2L_y/n_\beta`` | Staircase step spacing | Derived by `beta_staircase` |
| ``q_{\mathrm{full}}`` | PV represented by all live contours | Current contour state |
| ``q_{\mathrm{ref}}`` | PV represented by the frozen straight staircase | `BetaPlaneQGKernel.reference_contours` |
| ``q_{\mathrm{regular}}`` or ``q_r`` | Regular PV inverted after reference subtraction | Combined contour and analytic correction |

## High-level constructor keywords

The principal [`Problem`](@ref) keywords map to the notation as follows:

| Keyword | Meaning |
|---|---|
| `contours` | Single-layer `Vector{PVContour}`; mutually exclusive with `layers` |
| `layers` | Tuple of per-layer contour vectors for `kernel=:multilayer_qg` |
| `dt` | Positive fixed timestep ``\Delta t`` |
| `kernel` | `:euler`, `:qg`, `:beta_plane_qg`, `:sqg`, or `:multilayer_qg` |
| `Ld` | One deformation radius for QG/beta-plane QG, or ``N-1`` modal radii for multi-layer QG |
| `beta` | Beta-plane gradient ``\beta``; required by `:beta_plane_qg` |
| `δ_sqg` | SQG regularization ``\delta_{\mathrm{SQG}}``; required by `:sqg` |
| `coupling` | Physical ``N\times N`` stretching matrix; required by `:multilayer_qg` |
| `layer_thicknesses` | Positive ``H_i`` values used to symmetrize unequal-depth multi-layer coupling |
| `domain` | `:unbounded` or `:periodic` |
| `Lx`, `Ly` | Positive half-widths of a periodic domain |
| `stepper` | `:RK4` |
| `surgery` | A preset, `:none`, or an explicit `SurgeryParams` |
| `dev` | `CPU()` or `GPU()` storage/execution target |
| `T` | Floating-point type used consistently by geometry, kernels, and buffers |
