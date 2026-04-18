using ContourDynamics
using Documenter, DocumenterVitepress

makedocs(;
    modules = [ContourDynamics],
    authors = "Subhajit Kar",
    repo = "https://github.com/subhk/ContourDynamics.jl/blob/{commit}{path}#{line}",
    sitename = "ContourDynamics.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/subhk/ContourDynamics.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "2D Euler Vortex Patch" => "tutorial_euler.md",
            "Quasi-Geostrophic" => "tutorial_qg.md",
        ],
        "Examples" => "examples.md",
        "Theory & Method" => [
            "Overview" => "theory.md",
            "Contour Dynamics" => "theory/contour_dynamics.md",
            "Ewald Summation" => "theory/ewald_summation.md",
            "Contour Surgery" => "theory/contour_surgery.md",
            "Multi-Layer QG" => "theory/multilayer_qg.md",
            "Time Integration" => "theory/time_integration.md",
            "References" => "theory/references.md",
        ],
        "API Reference" => [
            "Overview" => "api.md",
            "Types" => "api/types.md",
            "Velocity & Acceleration" => "api/velocity.md",
            "Time Integration" => "api/time_integration.md",
            "Surgery" => "api/surgery.md",
            "Diagnostics" => "api/diagnostics.md",
            "Helpers" => "api/helpers.md",
            "Periodic & Ewald" => "api/periodic_ewald.md",
            "Devices" => "api/devices.md",
            "Internals" => "api/internals.md",
        ],
        "Contributing" => "contributing.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo = "github.com/subhk/ContourDynamics.jl",
    push_preview = true,
)
