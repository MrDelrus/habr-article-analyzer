# Project Structure

The repository follows a modular structure inspired by common best practices in machine learning and data science projects. Each directory has a clearly defined purpose to ensure reproducibility, scalability, and maintainability.

```graphql
├── data                                    # Data and markup
│   ├── markup
│   ├── targets
│   └── tiny
├── docs                                    # Project documentation
├── notebooks                               # Jupyter notebooks
│   ├── baselines
│   └── exploratory_data_analysis
├── src                                     # Main project source code
│   ├── core                                # Common utilities and configuration
│   │   └── core
│   └── ml_training                         # ML module for model training
│       └── ml_training
│           ├── data
│           ├── metrics
│           ├── models
│           │   ├── baseline
│           │   ├── encoders
│           │   └── predictors
│           └── trainers
└── tests                                   # Unit and integration tests
    ├── core
    └── ml_training
```
