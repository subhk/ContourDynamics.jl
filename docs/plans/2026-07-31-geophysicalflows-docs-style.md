# GeophysicalFlows-Style Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Render ContourDynamics.jl documentation with the classic Documenter HTML layout used by GeophysicalFlows.jl.

**Architecture:** Replace the VitePress renderer at the documentation build boundary while leaving content, navigation, and deployment unchanged. Validate the generated static site rather than adding application-level tests for a documentation configuration change.

**Tech Stack:** Julia, Documenter.jl, Markdown, GitHub Pages

### Task 1: Replace the documentation renderer

**Files:**
- Modify: `docs/make.jl`
- Modify: `docs/Project.toml`
- Modify: `docs/src/index.md`

**Step 1: Verify the current renderer is the undesired baseline**

Run: `rg -n 'DocumenterVitepress|MarkdownVitepress' docs/make.jl docs/Project.toml`

Expected: matches in both files, demonstrating that the desired classic renderer is absent.

**Step 2: Implement the renderer migration**

Use `Documenter.HTML` with `collapselevel = 2`, CI-aware `prettyurls`, the stable canonical URL, and explicit repository metadata. Remove DocumenterVitepress from imports and dependencies. Replace VitePress-only homepage front matter with ordinary Documenter Markdown and correct the GPU admonition indentation.

**Step 3: Instantiate and build the documentation**

Run: `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`

Run: `julia --project=docs docs/make.jl`

Expected: both commands exit successfully and create `docs/build/index.html`.

**Step 4: Verify generated style and navigation**

Assert that the generated HTML contains the Documenter classic assets and canonical URL, contains representative navigation entries, contains no literal VitePress front matter, and contains no VitePress assets.

**Step 5: Review the diff**

Run: `git diff --check`

Run: `git status --short`

Expected: only the planned documentation configuration and plan files are changed.
