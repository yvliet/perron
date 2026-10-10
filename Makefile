.PHONY: verify test compile figures all

compile:
	python scripts/compile.py paper/main.tex

figures:
	python scripts/generate_figures.py

test:
	pytest -q

verify:
	pytest -q
	python scripts/check_facts.py
	python scripts/verify_paper.py

all: figures compile test verify
