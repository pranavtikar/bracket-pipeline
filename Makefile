.RECIPEPREFIX = >
.PHONY: mesh fea converge dataset audit train cv test

mesh:
> python scripts/make_mesh.py

fea:
> python scripts/run_fea.py

converge:
> python scripts/run_convergence.py
> python -m bracket.report results/convergence.csv

dataset:
> python scripts/run_dataset.py

audit:
> python scripts/audit_labels.py
> python scripts/audit_report.py
> python scripts/build_clean_dataset.py

train:
> python -m bracket.surrogate

cv:
> python scripts/cv_check.py

test:
> python -m pytest -q
.PHONY: figures

figures:
> python scripts/plot_geometry.py
> python scripts/plot_stress.py
