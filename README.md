[![codecov](https://codecov.io/gh/nilsperssonsuorra/real-estate-valuation-mlops/branch/main/graph/badge.svg)](https://app.codecov.io/gh/nilsperssonsuorra/real-estate-valuation-mlops)
# 🏠 Automated Housing Valuation

An end-to-end machine learning project to predict the final sale price of real estate in Sweden. This repository documents the entire MLOps lifecycle, from data collection to a deployed, interactive application.

***

## 🗺️ Project Roadmap

- ✅ **Data Scraping:** Robust web scraper for Hemnet.se using `Selenium` and `BeautifulSoup`.
- ✅ **Model Training:** `XGBoost` model with Quantile Regression to estimate price intervals.
- ✅ **Interactive Frontend:** Streamlit app for housing price prediction with UI interface.
- ⏳ **Cloud Automation:** Deploy scraper via `Azure Functions`, store data in `Azure Blob Storage`.
- ⏳ **CI/CD & Deployment:** Automate training + deployment via `GitHub Actions`, containerize with `Docker`, and deploy to `Azure App Service`.
- ⏳ **System Monitoring:** Integrate `ELK Stack` for logging and monitoring.

***

## 🛠️ Tech Stack

| Category            | Technologies                                                                    |
|--------------------|----------------------------------------------------------------------------------|
| **Machine Learning**| `Python`, `XGBoost`, `Scikit-learn`, `Pandas`, `SHAP`                           |
| **Frontend & UI**   | `Streamlit`                                                                     |
| **Cloud & DevOps**  | `Azure (Functions, App Service, Blob Storage)`, `Docker`, `GitHub Actions`      |
| **Monitoring**      | `ELK Stack (Elasticsearch, Logstash, Kibana)`                                   |
