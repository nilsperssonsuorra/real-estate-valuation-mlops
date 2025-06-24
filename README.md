[![codecov](https://codecov.io/gh/nilsperssonsuorra/real-estate-valuation-mlops/branch/main/graph/badge.svg)](https://app.codecov.io/gh/nilsperssonsuorra/real-estate-valuation-mlops)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
# 🏠 Automated Housing Valuation MLOps Project
![Application Screenshot](https://github.com/user-attachments/assets/00332f1c-d5a8-43d4-b940-ca3f4cf79820)

This repository contains an end-to-end machine learning project to predict the final sale price of real estate in Sweden. It documents the entire MLOps lifecycle, from data collection and model training to a containerized, cloud-deployed interactive web application.

## ✨ Key Features

*   **Robust Data Scraper**: A Selenium-based scraper to collect real-time housing data from Hemnet.se.
*   **Quantile Regression Model**: An XGBoost model trained with quantile regression to predict a likely price interval (lower, median, and upper bounds), not just a single value.
*   **Interactive Frontend**: A user-friendly Streamlit application for predicting housing prices and understanding the model's decisions via SHAP value explanations.
*   **CI/CD Automation**: Automated testing, linting, and deployment pipelines using GitHub Actions.
*   **Cloud-Native Deployment**: The entire application is containerized with Docker and deployed to Azure App Service, with data and models stored in Azure Blob Storage.
*   **Automated Retraining**: A weekly scheduled workflow scrapes new data, cleans it, and retrains the model to prevent drift, automatically updating the production application.

## 🛠️ Tech Stack

| Category            | Technologies                                                                   |
| ------------------- | ------------------------------------------------------------------------------ |
| **Machine Learning**| Python, XGBoost, Scikit-learn, Pandas, SHAP, pip-tools                         |
| **Frontend & UI**   | Streamlit                                                                      |
| **Cloud & DevOps**  | Azure (App Service, Blob Storage, Container Registry), Docker, GitHub Actions  |
| **Monitoring**      | Azure Application Insights                                                     |
| **Code Quality**    | Pytest, Ruff                                                                   |

## 🏗️ System Architecture

The project follows a modern MLOps architecture, ensuring reproducibility, automation, and scalability.

[Scraper: GitHub Actions] -> [Azure Blob Storage: Raw Data] -> [Processing & Training: GitHub Actions] -> [Azure Blob Storage: Models & Artifacts] -> [Docker Image: GitHub Actions] -> [Azure Container Registry] -> [Azure App Service: Streamlit App]

## 🚀 Getting Started (Local Development)

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

*   Python 3.12
*   Git
*   Docker Desktop

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/nilsperssonsuorra/real-estate-valuation-mlops.git
    cd real-estate-valuation-mlops
    ```

2.  **Create a virtual environment and install dependencies:**

    This project uses `pip-tools` for deterministic dependency management. The `make install` command handles everything for you.
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    make install
    ```
    > If you modify `requirements.in`, run `make requirements` to update `requirements.txt`.

### Running the Pipeline & Application

A `Makefile` provides convenient commands for all common tasks.

| Command             | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| `make all-checks`   | Runs all code quality checks (linting with Ruff and tests with Pytest).|
| `make run-pipeline` | Executes the full local data pipeline: scrape -> clean -> train.       |
| `make docker-build` | Builds the Docker image for the Streamlit application.                 |
| `make docker-run`   | Runs the built Docker container.                                       |
| `make clean`        | Removes temporary Python cache files.                                  |


**Example Workflow:**

1.  **Generate local model artifacts:**
    ```sh
    make run-pipeline
    ```
    This will create the `data/` and `models/` directories with the necessary files.

2.  **Build and run the Streamlit app via Docker:**
    ```sh
    make docker-build
    make docker-run
    ```
    Open your browser and navigate to `http://localhost:8501`.

## 🗺️ Project Roadmap

* ✅ Data Scraping: Robust web scraper for Hemnet.se using Selenium and BeautifulSoup.
* ✅ Model Training: XGBoost model with Quantile Regression to estimate price intervals.
* ✅ Interactive Frontend: Streamlit app for housing price prediction with UI interface.
* ✅ Cloud Automation: Data stored in Azure Blob Storage and weekly retraining workflow is automated.
* ✅ CI/CD & Deployment: Automated testing, training, and deployment via GitHub Actions, containerized with Docker and deployed to Azure App Service.
* ✅ System Monitoring: Integrated Azure Application Insights for logging and performance monitoring.
