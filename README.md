# 🏠 Automated Housing Valuation

An end-to-end machine learning project to predict the final sale price of real estate in Sweden. This repository documents the entire MLOps lifecycle, from data collection to a deployed, interactive application.

***

## 🗺️ Project Roadmap

-   ✅ **Data Scraping:** Develop a robust web scraper for Hemnet.se using `Selenium` and `BeautifulSoup`.
-   ⏳ **Model Training:** Train an `XGBoost` model with Quantile Regression to predict housing prices.
-   ⏳ **Interactive Frontend:** Build a `Streamlit` app for price estimation with maps and `SHAP` explainability.
-   ⏳ **Cloud Automation:** Deploy the scraper to `Azure Functions` and store data in `Azure Blob Storage`.
-   ⏳ **CI/CD & Deployment:** Create a `GitHub Actions` pipeline, containerize with `Docker`, and deploy to `Azure App Service`.
-   ⏳ **System Monitoring:** Integrate the `ELK Stack` for centralized logging and monitoring.

***

## 🛠️ Tech Stack

| Category            | Technologies                                                                   |
| ------------------- | ------------------------------------------------------------------------------ |
| **Machine Learning**| `Python`, `XGBoost`, `Scikit-learn`, `Pandas`, `SHAP`                            |
| **Frontend & UI**   | `Streamlit`                                                                    |
| **Cloud & DevOps**  | `Azure (Functions, App Service, Blob Storage)`, `Docker`, `GitHub Actions`       |
| **Monitoring**      | `ELK Stack (Elasticsearch, Logstash, Kibana)`                                  |
