using ContourDynamics
using Documenter, DocumenterVitepress

makedocs(;
    modules = [ContourDynamics],
    authors = "Subhajit Kar",
    repo = "https://github.com/subhk/ContourDynamics.jl",
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
        "API Reference" => "api.md",
        "Contributing" => "contributing.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo = "github.com/subhk/ContourDynamics.jl",
    push_preview = true,
)
