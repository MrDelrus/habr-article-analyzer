# Project Structure

The repository follows a modular structure inspired by common best practices in machine learning and data science projects. Each directory has a clearly defined purpose to ensure reproducibility, scalability, and maintainability.

```
habr_article_analyzer/
├── .github/workflows/            # CI/CD workflows for automated testing, linting, and deployment
├── data/
│   ├── raw/                      # Original raw datasets (not tracked in Git)
│   ├── interim/                  # Partially processed datasets (not tracked in Git)
│   ├── processed/                # Fully processed datasets for training (not tracked in Git)
│   └── tiny/                     # Small dataset for CI/tests (tracked in Git)
├── docs/
│   ├── README.md                 # Main documentation index (overview, quick links)
│   ├── setup_guide.md            # Environment setup for devs & reviewers
│   └── project_structure.md      # Detailed project directory tree with explanations
├── env/                          # Python virtual environment (ignored in Git)
├── logs/                         # Training/debug logs (ignored in Git)
├── models/                       # Trained model weights (ignored in Git)
├── notebooks/                    # Jupyter notebooks for exploration, prototyping, and EDA
├── outputs/                      # Generated artifacts: figures, predictions, reports (ignored in Git)
├── src/                          # Source code for the project
│   └── habr_article_analyzer/
│       ├── data.py               # Functions to download and load the Habr dataset
│       └── __init__.py
├── tests/                        # Unit and integration tests
├── .gitignore                    # Git ignore rules
├── .pre-commit-config.yaml       # Pre-commit configuration
├── LICENSE                       # Project license
├── README.md                     # Project general information
├── pyproject.toml                # Project configuration and dependencies
├── requirements.in               # Runtime dependencies
├── requirements-dev.in           # Dev / CI dependencies
├── requirements.txt              # Pinned runtime dependencies
└── requirements-dev.txt          # Pinned dev dependencies
```
