# What is on GitHub, and what is not

The public tree is a run-kit plus current tests and the Pages gallery.

- `articlegen/`, `tests/`, `tools/`, `Dockerfile`, `render.yaml`, `pyproject.toml`, `requirements.txt`, and the Pages/tests/health/live-smoke workflows are the product.
- `index.html` is the Pages site. Keep it.
- `drafts/` stays in the repo as the local CLI review surface. It is not
  deployed to GitHub Pages.

Factory (`adws/`, `specs/`, `requests/`, `app_docs/`, `justfile`) stays on this machine and is gitignored. Do not add it back.

`docs/pipeline.*` is the public process diagram (README). Keep it.

Extra docs (project memory, journal-style notes, session handoffs, the old docs-current workflow) live in `archive/` on this machine and are gitignored. Do not put them back on GitHub.
