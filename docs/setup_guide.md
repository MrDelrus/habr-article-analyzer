# Project Setup Guide

This document explains how to set up and work with the project environment. It provides instructions for both **developers** (who actively modify code, run tests, and use pre-commit hooks) and **reviewers** (who only need to run or check the project).

## Developer Setup

### 1. Make sure you have Poetry installed (preferably v1.5+):

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
```

### 2. Use the correct Python version:

```bash
python3.11 -m venv env
source env/bin/activate
```

Or let Poetry create its virtual environment automatically:

```bash
poetry env use python3.11
```

### 3. Install dependencies:

#### a) Base installation (without heavy extras):

```bash
poetry install
```

- Installs the core dependencies of the project.
- `ml_training` is installed without optional extras (nlp, gpu).

#### b) Installation with `ml_training` extras:

```bash
poetry install --extras "ml_training/nlp ml_training/gpu"
```

- Installs `ml_training` with optional heavy extras.
- Use this if you need NLP models, deep learning, or GPU-enabled libraries.

**Note:** heavy extras are large and may take time to install.

### 4. Set up pre-commit hooks:

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

- Ensures code is automatically checked/formatted before commits

### 5. Using the environment:

Activate before working:

```bash
poetry shell
```

Or run commands directly:

```bash
poetry run python script.py
```

Deactivate when done:

```bash
exit  # or deactivate if using venv
```

## Reviewer Setup

- Reviewers **do not need** development dependencies or pre-commit hooks
- Install only runtime dependencies:

### 1. Create and activate a Python 3.11 virtual environment:

```bash
python3.11 -m venv env
source env/bin/activate
```

### 2. Ensure Poetry uses this Python version:

```bash
poetry env use python3.11
```

### 3. Install only runtime dependencies:

```bash
poetry install --no-dev
```

### 4. (Optional) If you need ml_training extras for NLP or GPU:

```bash
poetry install --no-dev --extras "ml_training/nlp ml_training/gpu"
```

This allows reviewers to run or test the project without installing linters, testing tools, or heavy development dependencies.
