🏡 House Price Prediction System
An end-to-end Machine Learning project that predicts residential house prices using property features such as living area, bedrooms, bathrooms, neighborhood, waterfront access, and building quality.

This project was completed as a portfolio milestone for the Machine Learning & AI BootCamp (MBL-2).
📌 Project Overview
House prices depend on several factors, including property size, location, construction quality, and available amenities. The goal of this project is to explore these factors and build regression models that estimate house prices from property data.

The workflow covers data inspection, exploratory data analysis (EDA), feature engineering, preprocessing, model training, evaluation, hyperparameter tuning, and residual analysis.
🎯 Objectives
- Explore and understand residential house-price data.
- Identify patterns and relationships between property features and prices.
- Create meaningful features using domain knowledge.
- Build a preprocessing pipeline for numerical and categorical data.
- Train and compare multiple regression models.
- Tune Ridge Regression using cross-validation.
- Interpret model coefficients and derive business insights.
🧰 Technologies & Libraries
- Python
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for visualization
- Scikit-learn for preprocessing, model training, evaluation, and tuning
- Jupyter Notebook / Kaggle Notebook for development
📂 Dataset
The project uses a house-price dataset containing 2,500 records and 13 original columns.

The target variable is price_usd.

Main features include:
Feature	Description
sqft_living	Living area of the property
bedrooms	Number of bedrooms
bathrooms	Number of bathrooms
lot_size_sqft	Lot size in square feet
floors	Number of floors
waterfront	Waterfront indicator
view_score	Property view score
condition_rating	Property condition rating
grade_score	Building quality grade
yr_built	Year the property was built
neighborhood	Neighborhood category
price_usd	House price in USD (target)


Dataset source: House Prediction Dataset on Kaggle
The dataset is not included in this repository. Download it from Kaggle and place the CSV file in the data/ directory, or update the dataset path in the notebook to match your environment.

🔍 Project Workflow
1. Data Ingestion & Inspection
- Loaded and inspected the dataset.
- Checked data types, dataset dimensions, and missing values.
- The dataset contains 2,500 rows and 13 columns, with no missing values in the inspected columns.
2. Exploratory Data Analysis (EDA)
- Examined descriptive statistics and the distribution of house prices.
- Used a histogram with KDE to understand the price distribution.
- Compared house prices across neighborhoods using a boxplot.
Key observations from the analysis:
- House prices are right-skewed (skewness: 1.69).
- Median house price: approximately $502,000.
- Mean house price: approximately $589,322.
- Lakeside has the highest median house price (approximately $1,201,000).
- Neighborhood is an important factor in house-price variation.
3. Feature Engineering
Created three additional features:
- property_age: 2024 - yr_built
- bath_to_bed_ratio: bathrooms / bedrooms
- is_luxury: 1 when waterfront == 1 or grade_score >= 10; otherwise 0.
In this dataset, luxury homes represent approximately 12.8% of the records.
4. Data Preprocessing
- Defined X as the feature matrix and y as the target (price_usd).
- Removed property_id and price_usd from the input features.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to the categorical neighborhood feature.
- Used an 80/20 train-test split with random_state=42.
Preprocessing was organized with Scikit-learn's ColumnTransformer and Pipeline.
5. Model Training
Four regression models were trained and compared:
1. Dummy Regressor (Mean Baseline)
2. Linear Regression
3. Ridge Regression (alpha=10)
4. Lasso Regression (alpha=100)
6. Model Evaluation
The models were evaluated on the held-out test set using MAE, RMSE, and R².
Model	MAE (USD)	RMSE (USD)	R²
Dummy (Mean Baseline)	197,459.064	279,889.862	-0.001
Linear Regression	21,912.163	27,734.481	0.990
Ridge (alpha=10)	22,135.191	28,044.705	0.990
Lasso (alpha=100)	21,991.871	27,853.799	0.990


Result: Linear Regression achieved the lowest MAE and RMSE among the four initially compared models, with an R² score of approximately 0.990.
7. Hyperparameter Tuning
Used GridSearchCV with 5-fold cross-validation to tune Ridge Regression.

Tested alpha values:

[0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
- Best alpha: 0.1
- Best cross-validation RMSE: $29,357.27
- Tuned Ridge test MAE: $21,911
- Tuned Ridge test RMSE: $27,734
- Tuned Ridge test R²: 0.9902
8. Residual Analysis & Business Insights
- Compared actual and predicted house prices.
- Inspected residuals to understand prediction errors.
- Analyzed the strongest positive and negative model coefficients.
Key insights from the coefficient analysis:
- Living area, view score, waterfront access, and building grade have positive coefficients.
- Property age has a negative coefficient.
- Neighborhood coefficients are interpreted relative to the dropped reference category (Downtown).
- These results describe relationships learned by the model; they do not establish causation.
Open project-01-house-price-prediction-system-sabrin.ipynb and run the cells in order.
👩‍💻 Author
Sabrin Alam
Machine Learning & AI BootCamp | Batch: MBL-2
📚 What I Learned
- Exploratory Data Analysis and data interpretation
- Feature engineering using domain knowledge
- Numerical scaling and categorical encoding
- Scikit-learn preprocessing pipelines
- Regression model training and comparison
- Model evaluation using MAE, RMSE, and R²
- Hyperparameter tuning with GridSearchCV
- Residual analysis and coefficient interpretation
⭐ If you find this project useful, feel free to explore the notebook and share your feedback.
Project Status: Completed
