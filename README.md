# 🏡 House Price Prediction System
### An End-to-End Machine Learning Regression Project

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

The **House Price Prediction System** is an end-to-end Machine Learning project developed to estimate residential property prices based on different property characteristics, including living area, bedrooms, bathrooms, location, building quality, waterfront access, and property age.

Traditional house valuation often depends on manual assessment, which can be time-consuming and subjective. This project explores how Machine Learning can support data-driven property valuation by identifying patterns and relationships between property features and their market prices.

Using Python and Scikit-learn, this project follows a complete Machine Learning workflow, starting from data collection and exploratory data analysis (EDA) to feature engineering, preprocessing, model training, evaluation, hyperparameter tuning, and business insight extraction.

The primary goal is to build and evaluate regression models that can accurately predict house prices while understanding which property characteristics contribute most to their estimated value.

## 🎯 Project Objectives

- Perform Exploratory Data Analysis (EDA) to understand house price distributions and property characteristics.
- Identify relationships between property features and house prices.
- Apply feature engineering to create meaningful features.
- Build a preprocessing pipeline for numerical and categorical variables.
- Prevent data leakage through proper preprocessing and train-test separation.
- Train and compare multiple regression models.
- Evaluate model performance using MAE, RMSE, and R² Score.
- Perform hyperparameter tuning using GridSearchCV.
- Analyze model residuals and identify important house price drivers.

## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning and preprocessing |
| Jupyter Notebook | Development environment |

## 📂 Dataset Description

The project uses a residential house price dataset named `house_price_prediction_data.csv`.

The dataset contains:

- **Total Records:** 2,500
- **Original Features:** 13 columns
- **Target Variable:** `price_usd`
- **Missing Values:** 0 in all original columns

### Dataset Features

| Feature | Description |
|---|---|
| `property_id` | Unique property identifier |
| `sqft_living` | Living area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `lot_size_sqft` | Total lot size in square feet |
| `floors` | Number of floors |
| `waterfront` | Indicates waterfront access |
| `view_score` | Property view rating |
| `condition_rating` | Property condition rating |
| `grade_score` | Building quality grade |
| `yr_built` | Year the property was built |
| `neighborhood` | Property neighborhood |
| `price_usd` | House price in USD (Target) |

## 🔍 Project Workflow

### 1. Data Ingestion & Initial Inspection

The first step involved loading the dataset using Pandas and performing initial data inspection.

Activities:
- Loaded the house price dataset.
- Checked the first five records.
- Examined dataset dimensions and data types.
- Checked missing values in all columns.

**Observation:** The dataset contains 2,500 records and 13 original columns, with no missing values.

### 2. Exploratory Data Analysis (EDA)

EDA was performed to understand the distribution of house prices and investigate how location influences property value.

Visualizations:
- House price distribution using Histogram and KDE.
- House price comparison across neighborhoods using Boxplots.
- Summary statistics and skewness analysis.

**Key Findings:**

- House prices are right-skewed, with a skewness value of **1.69**.
- Median house price: approximately **$502,000**.
- Mean house price: approximately **$589,322**.
- Lakeside has the highest median house price at approximately **$1,201,000**.
- Suburbs and Downtown have comparatively lower median prices.

**Insight:** Property location is an important factor in residential house valuation.

### 3. Feature Engineering

Three new features were created using existing dataset columns to improve the model's ability to understand property characteristics.

**New Features:**

1. `property_age`

   Calculates the age of a property using:

   `property_age = 2024 - yr_built`

2. `bath_to_bed_ratio`

   Calculates the ratio of bathrooms to bedrooms:

   `bath_to_bed_ratio = bathrooms / bedrooms`

3. `is_luxury`

   Creates a binary feature based on luxury property conditions:

   - 1 if waterfront = 1 OR grade_score >= 10
   - 0 otherwise

**Key Observation:**

Approximately **12.8%** of the properties are classified as luxury homes.

Average house prices:

- Non-luxury properties: approximately $497,392.
- Luxury properties: approximately $1,213,360.

### 4. Data Preprocessing & Pipeline Construction

A Scikit-learn preprocessing pipeline was constructed to handle numerical and categorical features appropriately.

**Numerical Features:**
- StandardScaler was applied to standardize numerical variables.

**Categorical Features:**
- OneHotEncoder was applied to encode neighborhood categories.
- The first category was dropped to avoid redundant encoded features.

**Preprocessing Tools:**
- `ColumnTransformer`
- `StandardScaler`
- `OneHotEncoder`
- `Pipeline`

**Train-Test Split:**

- Training Data: 80% (2,000 records)
- Testing Data: 20% (500 records)
- Random State: 42

The preprocessing steps were integrated into Scikit-learn pipelines so that transformations are fitted on training data and applied consistently to test data.

### 5. Model Training & Experimentation

Four regression models were trained and evaluated using Scikit-learn pipelines.

**Models Implemented:**

1. **Dummy Regressor (Mean Baseline)**

   Used as a baseline model that predicts the average house price.

2. **Linear Regression**

   Applied Ordinary Least Squares regression to identify linear relationships between property features and house prices.

3. **Ridge Regression**

   Applied L2 regularization to control model complexity and reduce overfitting.

   Initial alpha: 10.0

4. **Lasso Regression**

   Applied L1 regularization to control model complexity and shrink less important coefficients.

   Initial alpha: 100.0

### 6. Model Evaluation & Comparison

All four models were evaluated using three standard regression metrics:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving more weight to larger errors.
- **R² Score:** Measures how well the model explains variations in house prices.

**Model Performance:**

| Model | MAE (USD) | RMSE (USD) | R² Score |
|---|---:|---:|---:|
| Dummy Regressor | 197,459.064 | 279,889.862 | -0.001 |
| Linear Regression | 21,912.163 | 27,734.481 | 0.990 |
| Ridge Regression | 22,135.191 | 28,044.705 | 0.990 |
| Lasso Regression | 21,991.871 | 27,853.799 | 0.990 |

**Key Findings:**

- The Dummy Regressor performed poorly because it only predicts the mean house price.
- Linear, Ridge, and Lasso Regression significantly outperformed the baseline model.
- All three regression models achieved an R² score of approximately 0.99.
- Linear Regression achieved the lowest MAE and RMSE among the initial model comparisons.

### 7. Hyperparameter Tuning with GridSearchCV

To improve Ridge Regression performance, GridSearchCV was used to identify the optimal regularization parameter (alpha).

**Configuration:**

- Model: Ridge Regression
- Cross-validation: 5-Fold
- Scoring Metric: Negative Root Mean Squared Error
- Search Space: `[0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]`

**Best Parameters:**

- Best Alpha: **0.1**
- Best Cross-Validation RMSE: **$29,357.27**

**Tuned Ridge Model Performance on Test Data:**

| Metric | Result |
|---|---:|
| MAE | $21,911 |
| RMSE | $27,734 |
| R² Score | 0.9902 |

The tuned Ridge model achieved strong predictive performance on the held-out test dataset.

### 8. Residual Analysis & Business Insights

Residual analysis was performed to understand the difference between actual and predicted house prices.

**Visualizations:**
- Actual vs. Predicted Price Plot.
- Residual Distribution Plot.
- Top 5 Positive and Top 5 Negative Feature Coefficients.

**Key Business Insights:**

- Living area (`sqft_living`) is one of the strongest positive price indicators.
- Waterfront access positively contributes to estimated house prices.
- Better building quality and higher view scores are associated with higher predicted prices.
- Property age has a negative coefficient, indicating that older properties tend to have lower estimated values when other features are held constant.
- Neighborhood coefficients show differences relative to Downtown, the reference category.

**Top Positive Price Drivers:**

- Living Area
- View Score
- Waterfront
- Building Grade
- Bathrooms

**Top Negative Price Drivers:**

- Suburbs (relative to Downtown)
- Northside (relative to Downtown)
- Lakeside (relative to Downtown)
- Property Age
- Number of Floors

These findings help explain how different property characteristics contribute to the model's predictions. Model coefficients indicate learned associations, not necessarily causal effects.

## 📊 Final Project Results

The project successfully demonstrates a complete Machine Learning regression workflow.

| Project Component | Status |
|---|---|
| Data Inspection | Completed |
| Exploratory Data Analysis | Completed |
| Feature Engineering | Completed |
| Data Preprocessing | Completed |
| Model Training | Completed |
| Model Comparison | Completed |
| Hyperparameter Tuning | Completed |
| Residual Analysis | Completed |
| Business Insights | Completed |

**Final Result:** The tuned Ridge Regression model achieved an R² score of **0.9902**, with a test RMSE of approximately **$27,734**.

## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/house-price-prediction-system.git
```

### 2. Navigate to the Project Directory

```bash
cd house-price-prediction-system
```

### 3. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Download the Dataset

Download `house_price_prediction_data.csv` from the original dataset source used in the notebook.

Create a `data` folder and place the CSV file inside it.

### 5. Update the Dataset Path

Update the dataset-loading cell in the notebook:

```python
df = pd.read_csv("data/house_price_prediction_data.csv")
```

Make sure the filename matches your downloaded dataset.

### 6. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `project-01-house-price-prediction-system-sabrin.ipynb` and execute the cells sequentially.

## 📁 Repository Structure

```text
house-price-prediction-system/
│
├── project-01-house-price-prediction-system-sabrin.ipynb
│
├── data/
│   └── house_price_prediction_data.csv
│
├── README.md
│
└── requirements.txt
```

*Note: The dataset is not included in this repository.*

## 📚 Key Learning Outcomes

Through this project, I gained practical experience in:

- Understanding and analyzing real-world-style structured datasets.
- Performing Exploratory Data Analysis.
- Creating meaningful features through feature engineering.
- Handling numerical and categorical data.
- Building Scikit-learn preprocessing pipelines.
- Training and comparing regression algorithms.
- Evaluating models using MAE, RMSE, and R².
- Performing hyperparameter optimization with GridSearchCV.
- Analyzing residuals and interpreting model coefficients.
- Translating Machine Learning results into business insights.

## 👩‍💻 Author

**Sabrin Alam**

Machine Learning & AI BootCamp  
Batch: MBL-2

**Project:** House Price Prediction System  
**Date:** October 31, 2026

---

⭐ If you find this project interesting, feel free to explore the notebook and share your feedback!

**Thank you for visiting my project!**
