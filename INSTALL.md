# Installation Guide

## Prerequisites

- **Python 3.12+** (strict requirement)
- **uv** package manager (recommended) or pip

## Quick Start with uv (Recommended)

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone repository and navigate to project

```bash
cd /path/to/time-series-flu
```

### 3. Create virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Install dependencies

```bash
# Install all dependencies (this uses uv.lock for reproducibility)
uv sync --no-install-project

# Or install with dev dependencies for Jupyter
uv sync --no-install-project --all-extras
```

**Note:** `--no-install-project` flag is used because this is a notebook-based analysis project, not a Python package.

### 5. Verify installation

```bash
# Quick check
python -c "import pandas, numpy, torch, chronos, statsforecast, optuna; print('✅ All packages installed')"

# Comprehensive test (includes numpy <2.0 verification for TabPFN)
python test_numpy_compat.py
```

**Critical:** This project requires **numpy <2.0** for TabPFN compatibility. See [NUMPY_VERSION_PROTECTION.md](NUMPY_VERSION_PROTECTION.md) for details.

---

## Alternative: Using pip

If you prefer pip (slower but more familiar):

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Jupyter notebook support
pip install -r requirements-dev.txt
```

---

## Dependency Files

This project uses multiple dependency specifications for different tools:

1. **`pyproject.toml`** - Primary dependency specification (PEP 621 standard)
   - Used by uv for lock file generation
   - Defines all runtime and development dependencies

2. **`uv.lock`** - Lock file for reproducible installs (auto-generated)
   - Contains exact versions of all transitive dependencies
   - Ensures reproducible builds across environments
   - Generated with: `uv lock`

3. **`requirements.txt`** - pip-compatible format
   - For users who prefer pip over uv
   - Manually maintained to match pyproject.toml

4. **`requirements-dev.txt`** - Development dependencies
   - Jupyter, notebook, ipykernel for interactive development

---

## Package Categories

### Core Dependencies (Always Required)

**Data Science Stack:**
- pandas (>=2.1.0) - Data manipulation
- numpy (>=1.26.0) - Numerical computing
- scipy (>=1.16.0) - Scientific computing
- matplotlib (>=3.10.0) - Plotting
- seaborn (>=0.13.0) - Statistical visualization

**Statistical Models:**
- statsmodels (>=0.14.0) - SARIMAX baseline
- statsforecast (>=2.0.0) - AutoARIMA optimization (NEW for nb/02b)

**Machine Learning:**
- lightgbm (>=4.0.0) - Gradient boosting baseline
- optuna (>=4.5.0) - Hyperparameter tuning (NEW for nb/02c)

**Foundation Models:**
- chronos-forecasting (>=1.5.0) - Chronos-Tiny time series foundation model
- tabpfn-client (>=0.1.10) - TabPFN API client
- tabpfn-time-series (>=1.0.0) - TabPFN time series wrapper
- autogluon.timeseries (>=1.4.0) - AutoGluon time series framework

**Deep Learning:**
- torch (>=2.7.0) - PyTorch (CPU-only, no CUDA)

**Utilities:**
- python-dotenv (>=1.0.0) - Environment variable management (.env file)
- tqdm (>=4.67.0) - Progress bars
- ruptures (>=1.1.0) - Change point detection (used in examples)

### Development Dependencies (Optional)

Install only if running Jupyter notebooks locally:
- jupyter (>=1.0.0)
- notebook (>=7.0.0)
- ipykernel (>=6.29.0)
- ipywidgets (>=8.0.0)

---

## Updating Dependencies

### When to update dependencies?

1. **After adding new imports** to notebooks
2. **After installing new packages** with `uv pip install <package>`
3. **Before sharing code** to ensure requirements.txt is up-to-date

### How to update?

```bash
# Regenerate lock file after modifying pyproject.toml
uv lock

# Sync environment with updated dependencies
uv sync --no-install-project

# Export to requirements.txt (for pip users)
uv pip freeze > requirements.txt
```

---

## Common Issues

### Issue: "ModuleNotFoundError" when running notebooks

**Solution:** Make sure virtual environment is activated:
```bash
source .venv/bin/activate
```

### Issue: "uv: command not found"

**Solution:** Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Issue: CUDA/GPU errors with PyTorch

**Solution:** This project is CPU-only. If you see GPU-related warnings:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
```

### Issue: TabPFN API authentication errors

**Solution:** Create `.env` file in project root:
```bash
PRIORLABS_API_KEY=your_api_key_here
```

### Issue: Slow pip installs

**Solution:** Switch to uv (10-100× faster):
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Use uv instead of pip
uv pip install -r requirements.txt
```

---

## Verifying Notebook Dependencies

Before running notebooks, verify all imports work:

```bash
# Test nb/01 dependencies
python -c "import pandas, numpy, matplotlib; print('✅ nb/01 ready')"

# Test nb/02 dependencies (baseline models)
python -c "import statsmodels, lightgbm, chronos, tabpfn_client; print('✅ nb/02 ready')"

# Test nb/02b dependencies (SARIMA optimization)
python -c "import statsforecast; print('✅ nb/02b ready')"

# Test nb/02c dependencies (LightGBM optimization)
python -c "import optuna, lightgbm; print('✅ nb/02c ready')"

# Test nb/03-04 dependencies
python -c "import scipy, seaborn; print('✅ nb/03-04 ready')"
```

---

## Environment Variables

Create a `.env` file in the project root for API keys:

```bash
# TabPFN API Key (required for nb/02 TabPFN-TS model)
PRIORLABS_API_KEY=your_api_key_here
```

**Security Note:** Never commit `.env` file to git. It's already in `.gitignore`.

---

## Reproducing Exact Environment

To reproduce the exact environment used for the benchmark:

```bash
# Clone repo
git clone <repo-url>
cd time-series-flu

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install from lock file (exact versions)
uv sync --no-install-project

# Verify
uv pip list
```

The `uv.lock` file ensures bit-for-bit reproducibility of the dependency tree.

---

## Next Steps

After installation:

1. **Verify environment:** Run `nb/00_env_setup.ipynb`
2. **Prepare data:** Run `nb/01_data_prep.ipynb`
3. **Generate forecasts:** Run `nb/02_roll_loop.ipynb` (baseline models)
4. **Optimize models:** Run `nb/02b_sarima_optimization.ipynb` and `nb/02c_lightgbm_optimization.ipynb`
5. **Evaluate:** Run `nb/03_evaluation.ipynb`
6. **Visualize:** Run `nb/04_report.ipynb`

See `CLAUDE.md` for detailed notebook execution protocols.
