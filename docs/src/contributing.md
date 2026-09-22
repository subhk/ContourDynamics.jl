# Contributors' Guide

This is a short guide for potential **ContourDynamics.jl** contributors.

Contributions and questions are welcome. To get involved, please
[open an issue](https://github.com/subhk/ContourDynamics.jl/issues) to start a discussion.

For the full guide — including setup instructions, testing, and PR checklist — see our
[CONTRIBUTING.md](https://github.com/subhk/ContourDynamics.jl/blob/main/CONTRIBUTING.md).

We follow the [ColPrac guide](https://github.com/SciML/ColPrac) for collaborative practices.

Build and validate the website locally with:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Every build, including pull-request builds, executes examples and doctests and
fails on documentation errors. Use `@example` blocks for runnable examples;
`@repl` blocks can render exceptions as ordinary output. Named `@example`
blocks share state within a page. Deployment is enabled separately by
`DOCUMENTER_DEPLOY=true` in the documentation workflow.

Thanks for helping improve `ContourDynamics.jl`.
