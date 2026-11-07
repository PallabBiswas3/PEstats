# CLAUDE.md

Project memory lives in **BRAIN.md** — history, decision log, capabilities, architecture, conventions and roadmap. Read it before making changes, and update it (especially the decision log and roadmap) after significant ones.

@BRAIN.md

## Quick rules
- Package is `PBstats` (repo is `PEstats`). Time is axis 0; 2D data is `(n_samples, n_channels)`.
- New `Data` methods: follow "Conventions for adding a method" in BRAIN.md §5.
- Before finishing: `python -m pytest` and `ruff check PBstats tests benchmarks examples --select F,E9,B --ignore B905,B008` must pass.
- Version lives in two places: `pyproject.toml` and `PBstats/__init__.py`.
