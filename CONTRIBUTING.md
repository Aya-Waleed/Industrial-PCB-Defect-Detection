# Team Workflow (Aya · Heba · Saif)

Simple rules so 3 people can push to the same repo without last-minute conflicts.

## Branches
- `main` — always working. Only merged via Pull Request, never pushed to directly.
- `aya/data-*` — dataset prep, cleaning, split scripts.
- `heba/model-*` — training experiments, evaluation.
- `saif/app-*` — Streamlit app, inference wrapper.

## Before opening a PR
1. Pull latest `main` and rebase your branch on it.
2. Run the integration smoke test: `python tests/test_integration.py` — must be 7/7 pass.
3. If you changed `model/best.pt` or `model/class_names.txt`, re-run the smoke test
   (it checks the two stay in sync) before pushing.

## Merge order for a release
`aya/data-*` → `heba/model-*` (needs the dataset) → `saif/app-*` (needs `best.pt`)
Merge in this order to avoid one branch depending on files that don't exist yet in `main`.

## Final submission checklist
- [ ] `reports/model_evaluation/evaluation_results.md` generated from the final held-out test set
- [ ] `model/model_config.yaml` reviewed against the final checkpoint (the current file is populated from checkpoint metadata)
- [ ] `python tests/test_integration.py` passes
- [ ] `python scripts/benchmark_inference.py` run once on the demo machine, FPS noted in README
- [ ] At least 1-2 out-of-distribution images tested manually (photos not from the training dataset) to sanity-check generalization
- [ ] Screenshot(s) of the running app added to `README.md`
- [ ] Dataset source/citation confirmed and added to `README.md` (see note below — the class set matches the public "PCB Defect Dataset" / its augmented variants; confirm the exact source Aya used and its license before submitting)
