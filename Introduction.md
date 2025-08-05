Dominos Pizza Sales Forecasting & Ingredient Purchase Automation:

# Project Overview:
This project forecasts pizza sales using historical data and automates the generation of a weekly ingredient purchase order for a Domino’s Pizza outlet. The workflow includes data cleaning, exploratory data analysis (EDA), time series modeling (ARIMA), and purchase order creation based on predicted sales.


# Table of Contents:

- Project Overview
- Data Sources
- Workflow
- Key Results
- How to Run
- Files in This Repository
- Business Implications
- Contact

# Data Sources:

- Pizza_Sale.xlsx: Historical pizza sales data.
- Pizza_ingredients.xlsx: Ingredient requirements for each pizza.

# Workflow:

 = Data Cleaning & Preprocessing
    - Remove missing and inconsistent entries.
    - Handle outliers in sales quantities.
    - Format date columns.
    - Save cleaned datasets.

 = Exploratory Data Analysis (EDA)
    - Visualize sales trends, seasonality, and top-selling pizzas.
    - Analyze sales quantity distribution and correlations.
    - Check for missing values.

 = Sales Prediction
    - Feature engineering (day of week, month, etc.).
    - Train ARIMA time series model.
    - Evaluate model using Mean Absolute Percentage Error (MAPE).

 = Purchase Order Generation
    - Forecast sales for the next week.
    - Estimate ingredient needs based on predicted pizza sales.
    - Generate and save a detailed purchase order.

# Key Results:

- Cleaned Sales Records:48,580  
- Cleaned Ingredient Records: 514  
- ARIMA Model MAPE:~16%  
- Purchase Order:Generated for the next week, listing required quantities of each ingredient.


# How to Run:

(i) Install dependencies:
    ```
    pip install pandas matplotlib seaborn statsmodels scikit-learn openpyxl
    ```

(ii) Place the Excel files (`Pizza_Sale.xlsx`, `Pizza_ingredients.xlsx`) in the project directory.

(iii) Run the script:
    ```
    python datacleaning.py
    ```

(iv) Outputs:
    - Cleaned datasets: `cleaned_Pizza_Sale.xlsx`, `cleaned_Pizza_ingredients.xlsx`
    - EDA plots (displayed and/or saved)
    - Forecast plot: `arima_sales_forecast.png`
    - Purchase order: `purchase_order_next_week.xlsx`, `purchase_order_next_week.csv`

# Business Implications:

- Improved Inventory Planning: Data-driven forecasts help optimize ingredient orders, reducing waste and stockouts.
- Cost Savings: More accurate purchase orders minimize overstock and associated costs.
- Scalability: The workflow can be adapted for other outlets or products.

