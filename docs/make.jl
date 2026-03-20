using Documenter
using ContourDynamics

makedocs(
    sitename = "ContourDynamics.jl",
    modules = [ContourDynamics],
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "2D Euler Vortex Patch" => "tutorial_euler.md",
            "Quasi-Geostrophic" => "tutorial_qg.md",
        ],
        "API Reference" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
)
