# Article Analyzer for Habr

**Team Project for Master's Program in Artificial Intelligence, Faculty of Computer Science, Higher School of Economics (HSE)**

---

## Project Overview

**Title:** Article Analyzer for Habr.

**Goal:** Predict tag for the article based on its content.

**Description:** This project addresses a text classification problem using various approaches from classical machine learning and modern natural language processing (NLP). The expected outcomes include a functional prototype capable of tag prediction and a research report summarizing the methodology and results.

## Project Structure

The repository follows a modular structure inspired by common best practices in machine learning and data science projects. Each directory has a clearly defined purpose to ensure reproducibility, scalability, and maintainability.

```
├── .github/workflows/        # CI/CD workflows for automated testing, linting, and deployment
├── data/                     # Datasets
├── docs/                     # Project documentation, API references and architecture diagrams
├── models/                   # Trained and serialized models, model checkpoints
├── notebooks/                # Jupyter notebooks for exploration, prototyping, and EDA
├── src/                      # Source code for the project
├── tests/                    # Unit and integration tests
├── .pre-commit-config.yaml   # Configuration for pre-commit hooks (linting, formatting, etc.)
├── .gitignore                # Git ignore rules
├── LICENSE                   # Project license
├── README.md                 # Project documentation (this file)
├── pyproject.toml            # Project configuration and dependencies
├── requirements.txt          # Core dependencies
├── requirements-dev.txt      # Development dependencies (testing, linting, etc.)
```

## Team Members

This project was developed collaboratively by the following team members:

- Nikita Zolin - [Telegram](https://t.me/vozmuvseros)
- Sergiy Iarygin - [Telegram](https://t.me/SergiyYar)

Project's supervisor:

- Sergey Sysoev - [Telegram](https://t.me/yegerless)

## Work Plan

| Phase | Description | Deadline |
|-------|--------------|-----------|
| **Checkpoint 1** | Project setup | End of September |
| **Checkpoint 2** | Exploratory data analysis (EDA) | Mid-October |
| **Checkpoint 3** | Development of linear ML models | End of November |
| **Checkpoint 4** | Development of nonlinear ML models | Mid-January |
| **Interim defense** | Presentation of intermediate results | End of January |
| **Checkpoint 5** | Service development and integration | Mid-February |
| **Checkpoint 6** | Deep Learning implementation | Late March |
| **Checkpoint 7** | Model and service improvement | Mid-May |
| **Final defense** | Final project presentation and defense | June |

## Expected Results

1. Trained ML models for tag prediction.
2. Analytical report describing methods and results.
3. Visualization of results and perfomance metrics.
4. Recommendations for further development.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
