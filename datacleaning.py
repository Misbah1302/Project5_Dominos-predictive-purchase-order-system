import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

# Load data
sales_df = pd.read_excel('Pizza_Sale.xlsx')
ingredients_df = pd.read_excel('Pizza_ingredients.xlsx')

# Data Cleaning
sales_df_cleaned = sales_df.dropna(subset=['pizza_name_id'])
ingredients_df_cleaned = ingredients_df.dropna(subset=['Items_Qty_In_Grams'])

sales_df_cleaned.reset_index(drop=True, inplace=True)
ingredients_df_cleaned.reset_index(drop=True, inplace=True)

# Handle outliers in sales quantity (example: remove extreme values)
if 'quantity' in sales_df_cleaned.columns:
    q_low = sales_df_cleaned['quantity'].quantile(0.01)
    q_high = sales_df_cleaned['quantity'].quantile(0.99)
    sales_df_cleaned = sales_df_cleaned[(sales_df_cleaned['quantity'] >= q_low) & (sales_df_cleaned['quantity'] <= q_high)]

# Format date column if present
if 'order_date' in sales_df_cleaned.columns:
    sales_df_cleaned['order_date'] = pd.to_datetime(sales_df_cleaned['order_date'])

# EDA: Sales Trends

if 'order_date' in sales_df_cleaned.columns and 'quantity' in sales_df_cleaned.columns:
    sales_trend = sales_df_cleaned.groupby('order_date')['quantity'].sum()
    plt.figure(figsize=(12,6))
    sales_trend.plot()
    plt.title('Daily Pizza Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Quantity Sold')
    plt.tight_layout()
    plt.show()
    

# Save cleaned data to new files
sales_df_cleaned.to_excel('cleaned_Pizza_Sale.xlsx', index=False)
sales_df_cleaned.to_csv('cleaned_Pizza_Sale.csv', index=False)
ingredients_df_cleaned.to_excel('cleaned_Pizza_ingredients.xlsx', index=False)
ingredients_df_cleaned.to_csv('cleaned_Pizza_ingredients.csv', index=False)
print("Cleaned data saved as 'cleaned_Pizza_Sale.xlsx', 'cleaned_Pizza_Sale.csv', 'cleaned_Pizza_ingredients.xlsx', and 'cleaned_Pizza_ingredients.csv'.")

# EDA: Seasonality (Monthly)
if 'order_date' in sales_df_cleaned.columns and 'quantity' in sales_df_cleaned.columns:
    sales_df_cleaned['month'] = sales_df_cleaned['order_date'].dt.to_period('M')
    monthly_sales = sales_df_cleaned.groupby('month')['quantity'].sum()
    plt.figure(figsize=(10,5))
    monthly_sales.plot(kind='bar')
    plt.title('Monthly Pizza Sales')
    plt.xlabel('Month')
    plt.ylabel('Total Quantity Sold')
    plt.tight_layout()
    plt.show()

# EDA: Top Selling Pizzas
if 'pizza_name_id' in sales_df_cleaned.columns and 'quantity' in sales_df_cleaned.columns:
    top_pizzas = sales_df_cleaned.groupby('pizza_name_id')['quantity'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(x=top_pizzas.values, y=top_pizzas.index, orient='h')
    plt.title('Top 10 Best Selling Pizzas')
    plt.xlabel('Total Quantity Sold')
    plt.ylabel('Pizza Name ID')
    plt.tight_layout()
    plt.show()

print(f"Cleaned Sales Records: {sales_df_cleaned.shape[0]}")
print(f"Cleaned Ingredient Records: {ingredients_df_cleaned.shape[0]}")

# Distribution of Sales Quantity
if 'quantity' in sales_df_cleaned.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(sales_df_cleaned['quantity'], bins=30, kde=True)
    plt.title('Distribution of Sales Quantity')
    plt.xlabel('Quantity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(sales_df_cleaned.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Missing Values Visualization
plt.figure(figsize=(8,2))
sns.heatmap(sales_df_cleaned.isnull(), cbar=False, yticklabels=False)
plt.title('Missing Values in Sales Data')
plt.tight_layout()
plt.show()

# Feature Engineering: Add day of week, month, etc.
if 'order_date' in sales_df_cleaned.columns:
    sales_df_cleaned['day_of_week'] = sales_df_cleaned['order_date'].dt.dayofweek
    sales_df_cleaned['month'] = sales_df_cleaned['order_date'].dt.month
    sales_df_cleaned['year'] = sales_df_cleaned['order_date'].dt.year
    sales_df_cleaned['is_weekend'] = sales_df_cleaned['day_of_week'].isin([5,6]).astype(int)

# Aggregate daily sales for time series
daily_sales = sales_df_cleaned.groupby('order_date')['quantity'].sum().sort_index()

# Train/Test Split (80% train, 20% test)
split_idx = int(len(daily_sales) * 0.8)
train, test = daily_sales.iloc[:split_idx], daily_sales.iloc[split_idx:]

# ARIMA Model Training (simple ARIMA, you can tune order)
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))
forecast = np.maximum(forecast, 0)  # Ensure no negative predictions

# Model Evaluation
mape = mean_absolute_percentage_error(test, forecast)
print(f"ARIMA Model MAPE: {mape:.2%}")

# Plot actual vs predicted
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', color='orange')
plt.plot(test.index, forecast, label='ARIMA Forecast', color='green')
plt.title('ARIMA Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.legend()
plt.tight_layout()
plt.show()

# Forecast next 7 days using ARIMA
future_forecast = model_fit.forecast(steps=7)
future_forecast = np.maximum(future_forecast, 0)  # No negative predictions

# Create a DataFrame for the forecasted dates and quantities
last_date = daily_sales.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
future_sales_df = pd.DataFrame({'order_date': future_dates, 'predicted_quantity': future_forecast})
print(future_sales_df)
# Estimate pizza type proportions from historical data
pizza_proportions = sales_df_cleaned.groupby('pizza_name_id')['quantity'].sum()
pizza_proportions = pizza_proportions / pizza_proportions.sum()

# Predict number of each pizza to be sold in the next week
total_predicted = future_sales_df['predicted_quantity'].sum()
predicted_pizza_sales = (pizza_proportions * total_predicted).round().astype(int)

# Merge with ingredient requirements
purchase_order = pd.merge(
    predicted_pizza_sales.rename('predicted_pizzas').reset_index(),
    ingredients_df,
    left_on='pizza_name_id',
    right_on='pizza_name_id',
    how='left'
)

# Calculate total ingredient needs
purchase_order['total_ingredient_needed'] = purchase_order['predicted_pizzas'] * purchase_order['Items_Qty_In_Grams']

# Group by ingredient to get total needed for the week
ingredient_order = purchase_order.groupby('pizza_ingredients')['total_ingredient_needed'].sum().reset_index()
ingredient_order = ingredient_order.rename(columns={'total_ingredient_needed': 'quantity_needed_grams'})

print("\nPurchase Order for Next Week:")
print(ingredient_order)

# Save purchase order to Excel and CSV
ingredient_order.to_excel('purchase_order_next_week.xlsx', index=False)
ingredient_order.to_csv('purchase_order_next_week.csv', index=False)

print("Purchase order saved as 'purchase_order_next_week.xlsx' and 'purchase_order_next_week.csv'.")
plt.savefig('arima_sales_forecast.png')
