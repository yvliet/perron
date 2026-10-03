.PHONY: verify test

verify:
	pytest -q
	python scripts/check_facts.py

test:
	pytest -q
