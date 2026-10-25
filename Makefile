.PHONY: install install-locked test validation experiments figures tables summary check metadata manifest reproduce clean

install:
	python -m pip install -e .

install-locked:
	python -m pip install -r requirements-lock.txt
	python -m pip install -e . --no-deps

test:
	python -m pytest -q

validation:
	python scripts/run_all_experiments.py --experiment validation

experiments:
	python scripts/run_all_experiments.py --experiment all

figures:
	python scripts/make_figures.py
	python scripts/make_www_summary_figure.py

tables:
	python scripts/make_tables.py

summary:
	python scripts/summarize_results.py

check:
	python scripts/check_outputs.py

metadata:
	python scripts/write_run_metadata.py

manifest:
	python scripts/write_manifest.py

reproduce:
	python scripts/reproduce_all.py

clean:
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info
	rm -f data/processed/*.csv
	rm -f results/tables/* results/run_metadata.json results/reproduction_check.json
	rm -f results/logs/*.log
	rm -f figures/*.pdf figures/*.png
	rm -f RESULTS_SUMMARY.md REPRODUCIBILITY_MANIFEST.txt
