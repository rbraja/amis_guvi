# AMIS Full Project KT Document

## 1. Project Summary
This project is an end-to-end Apartment Maintenance Intelligence System (AMIS) built in Streamlit with data cleaning, validation, integration, EDA, machine learning, NLP, and a RAG assistant.

### Main objectives
- Clean and validate raw apartment maintenance data
- Create an integrated dataset for analytics
- Build predictive ML models for priority and resolution time
- Recommend technicians using historical performance
- Perform sentiment analysis on resident feedback
- Offer a complaint category classifier
- Provide a RAG assistant over rules, service logs, and PDF manuals

## 2. Workspace Structure

- `app.py` — Streamlit entry point
- `pages/` — Streamlit multipage application screens
- `src/` — core data, ML, RAG, and utility modules
- `data/` — raw, cleaned, integrated, and RAG data sources
- `vector_db/` — ChromaDB persistent index
- `README.md` — quick start info
- `requirements.txt` — Python dependencies

## 3. Dependencies
Installed via `pip install -r requirements.txt`.

Key libraries:
- `streamlit` — interactive app UI
- `pandas`, `numpy` — data processing
- `scikit-learn` — ML, TF-IDF, metrics
- `chromadb` — vector store for semantic search
- `sentence-transformers` — embedding model
- `pypdf` — PDF extraction
- `transformers`, `torch`, `datasets`, `sentencepiece`, `accelerate` — transformer model stack
- `joblib` — model serialization

## 4. Project Paths and Configuration

### `src/config.py`
Defines global project paths:
- `BASE_DIR` — repository root
- `DATA_DIR` — `data/`
- `MODEL_DIR` — `models/`
- `REPORT_DIR` — `reports/`
- model file paths:
  - `SENTIMENT_MODEL_PATH`
  - `CATEGORY_MODEL_PATH`

Use this file when path changes must be applied across modules.

## 5. Data Loading and Profiling

### `src/data_loader.py`
Loads all CSV files from a folder into a dictionary of DataFrames.

Key function:
- `load_all_tables(folder_path)`
  - scans CSV files
  - loads each into `pandas.read_csv`
  - returns `{table_name: df}`

### `src/profiler.py`
Profiles datasets for row/column counts, duplicates, invalid values, and nulls.

Key functions:
- `profile_tables(tables)`
- `count_invalid_emails(df)`
- `count_invalid_dates(df)`
- `count_invalid_phones(df)`
- `count_negative_resolution(df)`

### `pages/1_Data_Profile.py`
Data profiling UI workflow:
- loads `data/raw`
- displays dataset summary and preview
- shows column-level missing values
- allows profile download as CSV

## 6. Data Cleaning

### `src/cleaner.py`
Contains data cleaning utility functions.

Key functions:
- `fix_emails(df)`
  - finds any email-like columns
  - replaces invalid values with `invalid@email.com`
- `fix_dates(df)`
  - converts date columns to `datetime`
  - coerces invalid formats
- `fix_resolution_time(df)`
  - replaces negative `resolution_time_hours` values with `0`
- `fix_category_codes(df)`
  - normalizes legacy codes like `ELC` / `ELEC` / `PLMB` / `HVAC_OLD`

### `pages/2_Data_Cleaning.py`
Cleaning workflow:
- loads raw CSVs from `data/raw`
- applies all cleaner functions per table
- saves cleaned data to `data/cleaned_by_you/`
- shows cleaning log in the UI

## 7. Validation

### `src/validator.py`
Provides validation checks.

Key functions:
- `compare_row_counts(raw_tables, cleaned_tables)`
  - compares raw vs cleaned row counts
- `validate_targets(cleaned_tables, expected_rows)`
  - verifies expected row counts for specific tables

### `pages/3_Data_Validation.py`
Validation workflow:
- reads raw data and cleaned data
- compares row counts
- validates target counts for `residents`, `apartments`, and `maintenance_requests`
- saves report to `reports/validation_report.csv`

## 8. Data Integration

### `src/integrator.py`
Joins cleaned tables into a master dataset.

Key functions:
- `remove_dataset_version(df)` — drops `dataset_version` if present
- `integrate_tables(tables)` — merges:
  - `maintenance_requests`
  - `apartments`
  - `residents`
  - `technicians`
  - `building_assets`
  - `service_logs`
  - `resident_feedback`

### `pages/4_Data_Integration.py`
Integration workflow:
- loads cleaned tables from `data/cleaned_by_you`
- creates `master_dataset.csv` in `data/integrated/`
- displays shape and preview

## 9. EDA and Feature Engineering

### `src/feature_engineering.py`
Adds features for modeling and analytics.

Key functions:
- `create_season(month)` — maps month to season
- `add_season_flag(df)` — creates `season_flag`
- `add_complaint_hour(df)` — extracts hour from `request_date`
- `add_repeat_complaint(df)` — flag repeated resident complaints
- `add_asset_failure_count(master_df, asset_failure_df)` — merges failure counts by `asset_id`

### `pages/5_EDA_Feature_Engineering.py`
Feature engineering workflow:
- loads `data/integrated/master_dataset.csv`
- enriches dataset with season, hour, repeat complaints, failure counts
- generates basic charts for tower, floor, season, and category
- saves `master_dataset_featured.csv`

## 10. Machine Learning

### 10.1 Priority Prediction

#### `src/ml_priority.py`
Trains a priority classifier using:
- `category`, `subcategory`, `season_flag`, `complaint_hour`, `complaint_text`
- text vectorization with `TfidfVectorizer`
- one-hot encoding for categorical features
- `RandomForestClassifier`

Outputs:
- trained pipeline model
- accuracy score
- classification report

#### `pages/6_ML_Priority_Prediction.py`
UI workflow:
- trains model on `master_dataset_featured.csv`
- shows accuracy and report
- predicts priority for new input cases

### 10.2 Resolution Time Prediction

#### `src/ml_regression.py`
Trains regression model for `resolution_time_hours`:
- log-transform target with `np.log1p`
- features: `category`, `priority`, `season_flag`, `complaint_hour`, `asset_failure_count`
- dummy encodes categorical features
- `RandomForestRegressor`
- evaluates MAE, RMSE, R2

#### `pages/7_ML_Resolution_Time_Prediction.py`
UI workflow:
- trains regression model
- predicts resolution time from manual inputs
- inverts log-transform for final hours

### 10.3 Technician Recommendation

#### `src/ml_recommendation.py`
Recommends technicians by:
- grouping historical performance by `technician_name` and `category`
- computing average rating, average resolution, completed jobs
- combining into a heuristic score
- ranking top technicians in the selected category

#### `pages/8_ML_Technician_Recommendation.py`
UI workflow:
- selects category
- shows top technician recommendations

### 10.4 Complaint Category Classifier

#### `src/category_classifier.py`
Trains a category classifier using:
- `complaint_text`
- TF-IDF vectorization
- `RandomForestClassifier`
- saves model to `models/category_classifier.pkl`

#### `pages/10_Complaint_Category_Classifier.py`
UI workflow:
- trains category classifier
- predicts category for new complaint text

## 11. Sentiment Analysis

### 11.1 ML Sentiment Baseline

#### `src/sentiment_baseline.py`
Trains an ML sentiment classifier using:
- TF-IDF on `feedback_text`
- `RandomForestClassifier`
- evaluates accuracy and weighted F1
- saves model to `models/sentiment_rf.pkl`

Functions:
- `train_sentiment_model(df)`
- `predict_sentiment(text)`

### 11.2 DistilBERT Sentiment Inference

#### `src/distilbert_sentiment.py`
Uses Hugging Face `transformers` pipeline:
- model: `distilbert-base-uncased-finetuned-sst-2-english`
- function: `predict_distilbert(text)`

### `pages/9_Sentiment_Analysis.py`
UI workflow:
- trains ML baseline sentiment model
- predicts sentiment for input text using both ML and DistilBERT

## 12. RAG Assistant

### 12.1 RAG utilities

#### `src/rag_utils.py`
Loads embedding model:
- `SentenceTransformer("all-MiniLM-L6-v2")`
- `get_embedding(text)` returns vector list

### 12.2 Vector DB setup

#### `src/rag_setup.py`
Builds or updates ChromaDB index:
- uses `chromadb.PersistentClient(path=os.path.join(BASE_DIR, "vector_db"))`
- adds society rules CSV
- adds service logs repair notes
- scans PDF directories:
  - `data/rag`
  - `data/asset_manuals`
  - `data/rag/asset_manuals`
- extracts PDF text with `pypdf` and embeds it

Key functions:
- `add_society_rules(collection, rules_path)`
- `add_service_logs(collection, logs_path)`
- `extract_text_from_pdf(pdf_path)`
- `add_pdf_documents(collection, pdf_dirs)`
- `main()` triggers the full build

### 12.3 Fallback TF-IDF loader

#### `src/rag_loader.py`
Loads knowledge sources for fallback search and TF-IDF retrieval.

Functions:
- `load_rag_data()` loads CSV rows and PDF text
- scans `data/rag`, `data/asset_manuals`, and `data/rag/asset_manuals`

### 12.4 Search pipeline

#### `src/rag_search.py`
Search function:
- `search_documents(question, top_k=3)`
  - embeds query
  - queries ChromaDB
  - if Chroma fails or returns empty, runs `_tfidf_search(question, top_k)`

TF-IDF fallback:
- vectorizes documents from `load_rag_data()`
- computes cosine similarity
- returns top matches with `backend: "tfidf"`

### `pages/12_RAG_Assistant.py`
UI workflow:
- asks user question
- selects `top_k`
- can rebuild vector DB from UI
- displays results with `source`, `score`, and `backend`

## 13. How to Run the Full App

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run Streamlit:
```bash
streamlit run app.py
```
3. Use pages in order:
   1. `Day 1 - Data Profiling`
   2. `Day 2 - Data Cleaning`
   3. `Day 3 - Validation`
   4. `Phase 2 - Data Integration`
   5. `Phase 2 - EDA & Feature Engineering`
   6. `Phase 3 - Priority Prediction`
   7. `Phase 3 - Resolution Time Prediction`
   8. `Technician Recommendation`
   9. `Sentiment Analysis`
   10. `Complaint Category Classifier`
   11. `ML vs DL Comparison`
   12. `AMIS RAG Assistant`

## 14. Recommended End-to-End Demo Sequence

1. Data profile raw tables in `data/raw`
2. Run cleaning and save results to `data/cleaned_by_you`
3. Validate cleaned outputs against expected targets
4. Integrate cleaned tables into `data/integrated/master_dataset.csv`
5. Apply feature engineering and save `master_dataset_featured.csv`
6. Train ML models and evaluate:
   - priority prediction
   - resolution regression
   - sentiment prediction
   - category classification
7. Rebuild the RAG index from the UI or `python -m src.rag_setup`
8. Ask a question in the RAG assistant and confirm results

## 15. Notes and Troubleshooting

- Ensure `data/cleaned_by_you` exists before integration and EDA.
- If PDF results are missing in RAG, rebuild the vector DB; the RAG flow now includes nested `data/rag/asset_manuals`.
- If `models/*.pkl` are missing, train the respective pages first or recreate them manually.
- If `reports/validation_report.csv` is not updating, confirm `reports/` exists and the validation page is executed.

## 16. File Cheat Sheet

- `app.py` — Streamlit app entry
- `pages/1_Data_Profile.py`
- `pages/2_Data_Cleaning.py`
- `pages/3_Data_Validation.py`
- `pages/4_Data_Integration.py`
- `pages/5_EDA_Feature_Engineering.py`
- `pages/6_ML_Priority_Prediction.py`
- `pages/7_ML_Resolution_Time_Prediction.py`
- `pages/8_ML_Technician_Recommendation.py`
- `pages/9_Sentiment_Analysis.py`
- `pages/10_Complaint_Category_Classifier.py`
- `pages/11_ML_vs_DL_Comparison.py`
- `pages/12_RAG_Assistant.py`
- `src/data_loader.py`
- `src/cleaner.py`
- `src/validator.py`
- `src/integrator.py`
- `src/feature_engineering.py`
- `src/profiler.py`
- `src/ml_priority.py`
- `src/ml_regression.py`
- `src/ml_recommendation.py`
- `src/sentiment_baseline.py`
- `src/category_classifier.py`
- `src/distilbert_sentiment.py`
- `src/rag_utils.py`
- `src/rag_setup.py`
- `src/rag_loader.py`
- `src/rag_search.py`
- `src/config.py`

---

This document is a complete KT reference for the AMIS project, covering data pipeline, modeling, and RAG search.