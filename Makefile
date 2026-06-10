# `make all` regenerates every table and figure from the raw results already in
# results/. It never re-runs an experiment, so a report cannot silently
# disagree with the data it describes. Use `make experiment` for that.

PY := .venv/bin/python
CONFIG := config/default.yaml
RESULTS := results
REPORT := report

.PHONY: all install test test-all experiment permute report clean distclean

all: report

install:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest tests/ -q -m "not slow"

test-all:
	$(PY) -m pytest tests/ -q

# Full study. Takes minutes, not seconds -- see README for the budget.
experiment:
	$(PY) -m mofs.runner --config $(CONFIG) --out $(RESULTS)

# The leak detector. Labels shuffled; every selector must land at chance.
permute:
	$(PY) -m mofs.runner --config $(CONFIG) --out $(RESULTS) --permute

# Smoke test of the entire pipeline in well under a minute.
quick:
	$(PY) -m mofs.runner --config $(CONFIG) --out $(RESULTS)/quick --quick
	$(PY) -m mofs.report.build --results $(RESULTS)/quick --out $(RESULTS)/quick/report

report:
	$(PY) -m mofs.report.build --results $(RESULTS) --out $(REPORT)

# Everything, from nothing. This is the command the README promises.
reproduce: experiment permute report

clean:
	rm -rf $(REPORT) $(RESULTS)/quick

distclean: clean
	rm -rf $(RESULTS)/*.csv $(RESULTS)/*.npz $(RESULTS)/*.json $(RESULTS)/*.log
