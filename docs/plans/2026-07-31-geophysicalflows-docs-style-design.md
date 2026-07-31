# GeophysicalFlows-Style Documentation Design

## Goal

Give the ContourDynamics.jl documentation the same clear, research-oriented
layout language as GeophysicalFlows.jl while retaining ContourDynamics content,
navigation, branding, and versioned URLs.

## Reference and approach

GeophysicalFlows.jl uses Documenter's built-in `HTML` renderer with a collapsed
navigation depth of two. ContourDynamics currently uses
DocumenterVitepress. The closest and most maintainable match is therefore to
switch rendering backends rather than imitate the classic theme with custom
VitePress CSS.

The migration will:

- replace `DocumenterVitepress.MarkdownVitepress` with `Documenter.HTML`;
- use `collapselevel = 2`, matching the reference site's navigation density;
- retain CI-dependent pretty URLs and set the canonical URL to the stable site;
- remove the now-unused DocumenterVitepress dependency;
- replace the VitePress-only homepage front matter with ordinary Documenter
  Markdown organized like the reference site's overview page;
- preserve the existing page hierarchy and `deploydocs` behavior.

No GeophysicalFlows branding, text, or assets will be copied.

## Verification

The documentation build must complete without errors. Generated output must use
Documenter's classic HTML assets, contain the configured navigation pages and
canonical URL, render the homepage without literal VitePress front matter, and
contain no VitePress renderer artifacts. Existing package
tests are unaffected because the change is isolated to the documentation
environment.
