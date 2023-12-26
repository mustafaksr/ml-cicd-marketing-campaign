# ML CI/CD for Marketing Campaign

## Overview

This repository, **ml-cicd-marketing-campaign**, is a side project developed by me as part of the ["CI/CD for Machine Learning (GitOps)"](https://www.wandb.courses/courses/ci-cd-for-machine-learning) course of [W&B](https://wandb.ai/site). The project focuses on implementing MLOps practices and CI/CD workflows using GitHub Actions, with a special emphasis on integrating the W&B platform for experiment tracking and automated report generation.

## Course Objectives

The objectives of the course include:

- Automating ML pipelines with GitHub Actions
- Implementing automated testing for ML code
- Setting up branch protection rules
- Integrating GitHub API in CI/CD workflows
- Incorporating W&B API for experiment tracking
- Generating programmatic reports using W&B
- Promoting models to the registry with W&B

## Workflow

The primary CI/CD workflow for this project is defined in the `.github/workflows/ml-cicd.yml` file. The workflow is triggered on pushes to the `main` branch, pull requests, and manual dispatch events. The steps of the workflow include:

1. **Copy Repository Contents:** Checks out the repository content.
2. **Install Conda Environment:** Creates a Conda environment based on the specifications provided in `test/conda-environment.yaml`.
3. **Test Conda Environment:** Runs tests to ensure the Conda environment is set up correctly.
4. **Generate EDA Report:** Executes a script (`generate_eda_report/report.py`) to auto-generate an Exploratory Data Analysis (EDA) report in W&B platform.
5. **Unit Test:** Executes unit tests (`test/test_add_features.py`).
6. **Smoke Test:** Performs a smoke test on the training process (`smoke_test_train/train.py`).
7. **Hyperparameter Tuning:** Conducts hyperparameter tuning (`hyperparameters_tuning/tune_hyperparameter.py`).
8. **Train:** Trains the machine learning model (`train/train.py`).
9. **Deploy and Test API:** Deploys the model and tests the API (`inference/inference.py`).

## Repository Structure

The project follows a structured layout:

- **common:** Contains utility functions.
- **generate_eda_report:** Includes scripts for generating EDA reports.
- **hyperparameters_tuning:** Holds scripts for hyperparameter tuning.
- **inference:** Contains scripts for model inference and API testing.
- **smoke_test_train:** Includes a script for smoke testing the training process.
- **test:** Contains Conda environment specifications and test scripts.
- **train:** Contains data (`bank.csv`) and scripts for model training.

```
├── client
│   ├── deployment.py
│   └── promote_model.py
├── common
│   ├── __init__.py
│   └── utils.py
├── generate_eda_report
│   └── report.py
├── hyperparameters_tuning
│   └── tune_hyperparameter.py
├── inference
│   └── inference.py
├── smoke_test_train
│   └── train.py
├── test
│   ├── conda-environment.yaml
│   ├── test_add_features.py
│   └── testlib.py
├── train
│   ├── bank.csv
│   └── train.py
├── LICENSE
├── README.md
├── requirements.txt
```


## Secrets

The workflow relies on the following secrets:

- `GITHUB_TOKEN`: GitHub token for workflow authentication.
- `WANDB_API_KEY`: W&B API key for experiment tracking.

## Usage

To use this project, fork the repository and customize it according to your ML model and project requirements. Ensure that you set up the required secrets in your GitHub repository for proper integration with W&B.

Feel free to explore and modify the workflow to fit your specific needs and extend the functionality based on your ML project requirements.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
