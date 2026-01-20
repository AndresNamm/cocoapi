# Repository Guidelines

## Project Structure & Module Organization
- `pycocotools/`: Core Python modules (`coco.py`, `cocoeval.py`, `mask.py`) and the Cython wrapper `_mask.pyx`.
- `../common/`: Shared C sources consumed by the extension build (e.g., `maskApi.c`).
- `data/`, `results/`: Local datasets and outputs used by demos.
- `*.ipynb`, `pycocoEvalDemo.py`: Notebooks and demo scripts that exercise evaluation flows.
- `setup.py`: Build configuration for the Cython extension.

## Build, Test, and Development Commands
Use Pixi for the recommended environment and task runner:
- `pixi install`: Create the conda environment for the project.
- `pixi run build`: Build the Cython extension in place (`pycocotools/_mask.*`) and remove `build/`.
- `pixi run test`: Run `pycocoEvalDemo.py` as a smoke test of evaluation behavior.
- `pixi run tclean`: Remove generated `.c` and `.so` artifacts in `pycocotools/`.
Direct build (without Pixi): `python setup.py build_ext --inplace`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8-ish naming (snake_case functions, CapWords classes).
- C/Cython: keep existing naming in `_mask.pyx` and `maskApi.c` as-is; prefer minimal diffs.
- No formatter or linter is configured; keep changes focused and readable.

## Testing Guidelines
- Primary check is the demo script: `python pycocoEvalDemo.py` (or `pixi run test`).
- There is no formal unit test suite in this directory; treat demos as smoke tests.
- When adding behavior, update or add a small demo invocation that exercises it.

## Commit & Pull Request Guidelines
- Recent commits are short, sentence-style messages without a strict convention. Keep messages concise and descriptive (e.g., "Fix mask IoU edge case").
- PRs should include: a summary of changes, how you tested (`pixi run test`), and any dataset assumptions.
- If outputs change, note where to find new artifacts (`results/`) and include screenshots only when they clarify evaluation output.

## Configuration Notes
- The build depends on Cython and NumPy; ensure headers are available via the Pixi environment.
- The extension sources reference `../common`, so keep directory structure intact when building.
