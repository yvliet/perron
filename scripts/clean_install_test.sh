#!/usr/bin/env bash
set -euo pipefail

echo "=== Perron-Core Clean-Room Installation Test ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Build wheel
echo "[1/4] Building distribution wheel..."
python3 -m pip install --upgrade build twine
python3 -m build --wheel --outdir "${REPO_ROOT}/dist"

WHEEL_FILE=$(find "${REPO_ROOT}/dist" -name "perron_core-0.3.0*.whl" | head -n 1)
echo "Built wheel: ${WHEEL_FILE}"

# Twine check
echo "[2/4] Checking package metadata with twine..."
twine check "${WHEEL_FILE}"

# Isolated venv
echo "[3/4] Creating isolated virtual environment..."
TMP_VENV=$(mktemp -d)/venv
python3 -m venv "${TMP_VENV}"
source "${TMP_VENV}/bin/activate"

echo "[4/4] Installing wheel and verifying entrypoints..."
pip install "${WHEEL_FILE}"
perron --version
perron --help
perron-mcp --version

deactivate
rm -rf "$(dirname "${TMP_VENV}")"

echo "=== All clean installation tests PASSED successfully ==="
