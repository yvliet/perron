.PHONY: verify test

verify:
	pytest -q
	python scripts/check_facts.py
	python scripts/verify_paper.py

test:
	pytest -q
