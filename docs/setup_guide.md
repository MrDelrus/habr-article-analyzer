# Project Setup Guide

This document explains how to set up and work with the project environment. It provides instructions for both **developers** (who actively modify code, run tests, and use pre-commit hooks) and **reviewers** (who only need to run or check the project).

## Developer Setup

1. Create and activate a virtual environment. For Linux/macOS:

```
python -m venv env
source env/bin/activate
```

- Always activate your virtual environment before working with the project
- This prevents version conflicts between developers and ensures consistent behavior

2. Install pip.and pip-tools. Use the following versions:

```
pip install "pip==25.1" "pip-tools==7.5.1"
```

3. Install dependencies:

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

- [requirements.txt](../requirements.txt) - runtime/project dependencies
- [requirements-dev.txt](../requirements-dev.txt) - development dependencies (linters, tests, nbqa, pre-commit hooks)

4. Set up pre-commit hooks:

```
pre-commit install
pre-commit install --hook-type commit-msg
```

- Ensures code is automatically checked/formatted before commits

5. Deactivate the environment:

```
deactivate
```

- Use this when you’re done working in the project
- Safely returns you to your system Python environment
- Helps prevent accidentally installing packages outside the project environment

## Reviewer Setup

- Reviewers **do not need** development dependencies or pre-commit hooks
- Only [requirements.txt](../requirements.txt) is necessary:

```
python -m venv env
source env/bin/activate
pip install "pip==25.1" "pip-tools==7.5.1"
pip install -r requirements.txt
```

- This allows running or reviewing the project without installing linters or testing tools
