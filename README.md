# E-Commerce Sales Analytics and AI-Based Sales Prediction

**IBM SkillsBuild Data Analytics with AI Academic Internship — Capstone Project**  
**Submitted by:** Vishnu Upendra Borusu

---

## Project Overview

This project delivers a complete, end-to-end data analytics and machine-learning pipeline on a real-world e-commerce dataset. It covers data loading, cleaning, integration, exploratory data analysis (EDA), and supervised machine learning to predict order-level net sales revenue.

**Project Title:** E-Commerce Sales Analytics and AI-Based Sales Prediction  
**ML Task:** Supervised Regression — predict `net_sales` per order  
**Models:** Random Forest Regressor · XGBoost Regressor  
**Evaluation Metrics:** MAE · RMSE · R²  

---

## Dataset Files

All CSV files are located in the project root directory. Do not modify them.

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `ecommerce_sales_customer_analytics_150k.csv` | 138,116 | 46 | Main transactions fact table — orders 2021–2025 |
| `customer_master.csv` | 10,004 | 11 | Customer demographic dimension |
| `order_items.csv` | 100,004 | 12 | Order line-item detail (order–product level) |
| `product_catalog.csv` | 1,175 | 9 | Product dimension (category, brand, pricing) |
| `dataset_statistics.csv` | 1 | 11 | Pre-computed KPI summary (validation reference) |

**Date range:** January 2021 – December 2025  
**Countries:** USA, Germany, UK, UAE  
**Join keys:** `customer_id` (main ↔ customer_master) · `order_id` (main ↔ order_items) · `product_id` (order_items ↔ product_catalog)

---

## Project Structure

```
ECommerce_Sales_Analytics_Dataset/
│
├── Vishnu upendra Borusu_ECommerce_Sales_Analytics.ipynb   # Main Jupyter Notebook (34 sections)
├── requirements.txt                           # Python dependencies
├── README.md                                  # This file
│
├── ecommerce_sales_customer_analytics_150k.csv
├── customer_master.csv
├── order_items.csv
├── product_catalog.csv
├── dataset_statistics.csv
│
├── src/
│   ├── data_loader.py      # Load and clean all 5 CSV files
│   ├── eda.py              # EDA functions — generates and saves all charts
│   └── train_model.py      # ML pipeline — train, evaluate, compare, save model
│
├── models/
│   └── saved/
│       ├── best_model.pkl          # Champion model saved with joblib
│       └── model_metadata.pkl      # Model name, features, MAE, RMSE, R²
│
├── output/
│   ├── figures/            # All EDA and ML evaluation charts (PNG)
│   └── reports/            # Reserved for report outputs
│
└── data/
    ├── raw/                # Reserved for raw data copies
    └── processed/          # Reserved for processed data exports
```

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook (recommended)

```bash
jupyter notebook AshaRani_ECommerce_Sales_Analytics.ipynb
```

Run all cells top-to-bottom. The notebook executes the full pipeline from data loading through model training, evaluation, and final findings.

### 3. Run the source scripts individually (optional)

```bash
# Data loading and validation
python src/data_loader.py

# EDA — generates all charts to output/figures/
python src/eda.py

# ML pipeline — trains both models, evaluates, saves best model
python src/train_model.py
```

> All scripts resolve paths relative to the project root. Run them from the project root directory.

---

## Notebook Structure (34 Sections)

| Section | Content |
|---------|---------|
| 1 | Introduction |
| 2 | Problem Statement |
| 3 | Objectives |
| 4 | Import Libraries |
| 5 | Load the CSV Datasets |
| 6 | Dataset Overview |
| 7 | Data Cleaning |
| 8 | Missing Value Analysis |
| 9 | Duplicate Analysis |
| 10 | Data Type Conversion |
| 11 | Data Integration |
| 12 | Feature Engineering |
| 13 | EDA Overview |
| 14 | Univariate Analysis |
| 15 | Bivariate Analysis |
| 16 | Sales and Revenue Analysis |
| 17 | Product and Category Analysis |
| 18 | Customer Analysis |
| 19 | Time-Based Analysis |
| 20 | Correlation Analysis |
| 21 | Key Business Insights |
| 22 | ML Problem Definition |
| 23 | Feature Selection |
| 24 | Train / Test Split |
| 25 | Data Preprocessing |
| 26 | Model 1 — Random Forest Regressor |
| 27 | Model 2 — XGBoost Regressor |
| 28 | Model Evaluation Summary |
| 29 | Model Comparison |
| 30 | Champion Model Selection |
| 31 | Generate Predictions |
| 32 | Final Findings |
| 33 | Conclusion |
| 34 | Future Scope |

---

## Data Cleaning Steps

Applied to `ecommerce_sales_customer_analytics_150k.csv`:

1. Parse `order_date` to datetime
2. Derive `order_year`, `order_month`, `order_quarter`, `order_dayofweek`
3. Create `coupon_used` binary flag from `coupon_code` (before null-fill)
4. Create `delivery_days_missing` flag; fill null delivery fields with `0`
5. Fill sparse categorical nulls (`return_status`, `return_reason`, `review_sentiment`, `customer_review`, `campaign_name`) with `'None'`
6. Cast `is_repeat_customer` from boolean-string to integer (0/1)
7. Cast low-cardinality string columns to `category` dtype
8. Validate financial columns with `pd.to_numeric(..., errors='coerce')`
9. Drop exact row duplicates

---

## EDA Visualisations

All charts are saved to `output/figures/` as PNG files.

| Chart File | Description |
|-----------|-------------|
| `01_missing_values.png` | Missing value percentage bar chart |
| `02_order_status.png` | Order status distribution (bar + pie) |
| `03_numeric_distributions.png` | Histograms for 8 numeric features |
| `04_categorical_distributions.png` | Count charts for 6 categorical features |
| `05_bivariate_scatter.png` | Discount amount vs net_sales; Quantity vs net_sales |
| `06_bivariate_boxplots.png` | Net sales by segment, shipping method, sales channel |
| `07_annual_revenue.png` | Annual gross revenue, net sales, and profit (2021–2025) |
| `08_monthly_trend.png` | Monthly revenue and order count line chart |
| `09_channel_revenue.png` | Net sales by sales channel and marketing channel |
| `10_geo_revenue.png` | Net sales by country and region |
| `11_product_category.png` | Revenue and profit margin by product category |
| `12_top_subcategories.png` | Top 15 subcategories by gross revenue |
| `13_customer_segment.png` | Revenue, orders, and avg order value by segment |
| `14_customer_loyalty_age.png` | Repeat vs new customers; net sales by age group |
| `15_return_rate_country.png` | Return rate (%) by country |
| `16_quarterly_heatmap.png` | Gross revenue heatmap — Year × Quarter |
| `17_day_of_week.png` | Orders and revenue by day of week |
| `18_seasonality.png` | Average order value and total orders by month |
| `19_correlation_heatmap.png` | Pearson correlation matrix (numeric features) |
| `20_net_sales_correlation.png` | Feature correlation with net_sales (target) |
| `22_model_comparison.png` | MAE / RMSE / R² bar chart — Random Forest vs XGBoost |
| `23_predicted_vs_actual_residuals.png` | Predicted vs Actual and Residual distributions |
| `23_cv_r2.png` | 5-fold cross-validation R² comparison |
| `24_feature_importance.png` | Top 20 feature importances from both models |

---

## Machine Learning

### Target Variable

| Variable | Type | Description |
|----------|------|-------------|
| `net_sales` | Continuous numeric | Order revenue after discounts — predicted at order-placement time |

### Features (26 total)

**15 Numeric features** (all known at or before order time):

| Feature | Description |
|---------|-------------|
| `customer_age` | Customer demographic |
| `quantity` | Units ordered |
| `discount_amount` | Discount applied at checkout |
| `shipping_cost` | Logistics cost quoted at checkout |
| `tax_amount` | Tax computed at checkout |
| `loyalty_points_redeemed` | Points applied at checkout |
| `estimated_delivery_days` | Delivery estimate shown at order time |
| `coupon_used` | Binary flag — derived from `coupon_code` |
| `is_repeat_customer` | CRM flag — known at order time |
| `delivery_days_missing` | Structural missingness flag |
| `order_month` | Calendar month (seasonality) |
| `order_quarter` | Quarter (seasonality) |
| `order_year` | Year-level trend |
| `order_dayofweek` | Day pattern |
| `customer_acquisition_cost` | Historical CAC from `customer_master` |

**11 Categorical features** (label-encoded):
`sales_channel`, `gender`, `customer_segment`, `customer_type`, `customer_country`, `region`, `payment_method`, `currency`, `shipping_method`, `marketing_channel`, `warehouse`

### Leakage Exclusions

The following columns are excluded to prevent data leakage:

| Column | Reason |
|--------|--------|
| `gross_sales` | Arithmetic component of `net_sales` |
| `profit`, `profit_margin_percentage`, `product_cost` | Derived from `net_sales` |
| `return_status`, `return_reason` | Post-transaction |
| `customer_rating`, `review_sentiment`, `customer_review` | Post-delivery |
| `loyalty_points_earned` | Awarded after order completes |
| `delivery_status`, `delivery_days` (actual) | Post-fulfillment |
| `customer_lifetime_value`, `customer_order_count` | Aggregates including this order |
| `payment_status` | Post-payment confirmation |
| `order_id`, `customer_id`, `customer_name` | Identifiers |

### Models

| Model | Key Hyperparameters |
|-------|-------------------|
| `RandomForestRegressor` | `n_estimators=200, max_depth=12, min_samples_leaf=4, random_state=42` |
| `XGBRegressor` | `n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42` |

### Evaluation

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error — average prediction error in $ |
| RMSE | Root Mean Squared Error — penalises large errors |
| R² | Proportion of variance in `net_sales` explained by the model |
| CV R² | 5-fold cross-validation R² (mean ± std) on training set |

**Train/test split:** 80% train / 20% test · `random_state=42`  
**Champion selection:** Model with the highest test-set R² is saved automatically.

### Saved Model Files

| File | Description |
|------|-------------|
| `models/saved/best_model.pkl` | Champion model — loaded with `joblib.load()` |
| `models/saved/model_metadata.pkl` | Dict containing model name, target, features, MAE, RMSE, R², CV R² |

---

## Technologies and Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 2.0.0 | Data loading, cleaning, manipulation |
| `numpy` | ≥ 1.24.0 | Numeric operations, RMSE calculation |
| `matplotlib` | ≥ 3.7.0 | All charts and visualisations |
| `seaborn` | ≥ 0.12.0 | Heatmaps, box plots, styled charts |
| `scikit-learn` | ≥ 1.3.0 | `RandomForestRegressor`, `LabelEncoder`, `train_test_split`, metrics |
| `xgboost` | ≥ 2.0.0 | `XGBRegressor` |
| `joblib` | ≥ 1.3.0 | Save and load trained models |
| `jupyter` / `notebook` / `ipykernel` | ≥ latest stable | Jupyter Notebook runtime |

---

## Future Improvements

1. **Time-Series Forecasting** — SARIMA or Prophet on monthly aggregated `net_sales`
2. **Order Status Classification** — predict Completed / Returned / Cancelled
3. **NLP on Reviews** — TF-IDF and sentiment analysis on `customer_review`
4. **Hyperparameter Tuning** — `GridSearchCV` / `RandomizedSearchCV`
5. **SHAP Explainability** — per-prediction SHAP values for business transparency
6. **Customer Churn Model** — predict repeat purchase likelihood
7. **Interactive Dashboard** — Plotly Dash or Power BI connected to the saved model

---

## Acknowledgements

- **Program:** IBM SkillsBuild Data Analytics with AI Academic Internship
- **Scikit-learn documentation:** https://scikit-learn.org
- **XGBoost documentation:** https://xgboost.readthedocs.io
- **Pandas documentation:** https://pandas.pydata.org
