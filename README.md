# AI-Classification-for-Marketing-Campaign-data-Kaggle
This repository contains an end-to-end machine learning pipeline designed to predict customer responses to marketing campaigns. The codebase is structured sequentially, following industry-standard data science workflows from raw data ingestion to model evaluation. 

## 1. Data Ingestion & Preprocessing
The pipeline begins by loading the raw tabular data (`marketing_campaign.csv`), this dataset is from KAGGLE, we do not own this data set (for the purpose of completing a course project only). The initial preprocessing script handles data integrity and cleaning:
* **Missing Value Imputation:** Identifies and resolves null values (e.g., handling missing income records using median imputation or row deletion).
* **Data Type Conversion:** Standardizes data types, particularly converting date strings into datetime objects for temporal analysis.
* **Outlier Handling:** Detects and mitigates extreme outliers in financial columns to prevent model skewing.

## 2. Feature Engineering
To maximize the predictive power of the models, the codebase includes a dedicated feature engineering phase that derives new, high-value variables from the existing dataset:
* **Customer Demographics:** Calculates the actual age of the customer and the duration of their enrollment.
* **Behavioral Aggregation:** Consolidates individual spending categories into cumulative metrics (e.g., Total Spend) and aggregates purchase channels (e.g., Total Purchases).
* **Categorical Encoding:** Transforms categorical text variables (such as Education and Marital Status) into numerical formats using techniques like One-Hot Encoding or Label Encoding.
* **Feature Scaling:** Applies Standard Scaler or Min-Max Scaler to ensure numerical features possess uniform distribution, which is critical for distance-based algorithms.

## 3. Exploratory Data Analysis (EDA)
The code includes visual diagnostics and statistical summaries to uncover data distributions and feature correlations. This section generates correlation matrices and distribution plots to identify which features most heavily influence the target variable (campaign acceptance).

## 4. Model Training Pipeline
The core of the codebase is the classification engine, which splits the processed data into training and testing sets (typically an 80/20 or 70/30 split). It trains multiple classification algorithms to benchmark performance:
* **Baseline Models:** Logistic Regression for linear relationships and interpretability.
* **Tree-Based Models:** Decision Trees and Random Forests to capture non-linear patterns and handle complex feature interactions.
* **Hyperparameter Tuning:** Utilizes Grid Search or Random Search cross-validation to optimize the parameters of the best-performing models.

## 5. Model Evaluation & Metrics
The final section of the code evaluates the trained models against the unseen testing data. The evaluation outputs a comprehensive suite of classification metrics:
* **Accuracy:** The overall correctness of the model.
* **Precision & Recall:** Critical metrics for marketing, highlighting the trade-off between false positives (marketing to non-responders) and false negatives (missing potential responders).
* **F1-Score:** The harmonic mean of precision and recall for assessing imbalanced classes.
* **Confusion Matrix:** A visual output detailing the exact counts of True Positives, True Negatives, False Positives, and False Negatives.
