# ==========================================
# SALES & DEMAND FORECASTING PROJECT
# FUTURE INTERNS - TASK 1
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------

print("Loading Dataset...")

df = pd.read_csv("dataset/Walmart.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# ------------------------------------------
# STEP 2: DATA CLEANING
# ------------------------------------------

print("\nChecking Missing Values:")
print(df.isnull().sum())

print("\nChecking Duplicate Rows:")
print(df.duplicated().sum())

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# ------------------------------------------
# STEP 3: FEATURE ENGINEERING
# ------------------------------------------

print("\nCreating Time Features...")

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Day'] = df['Date'].dt.day

# Convert Holiday_Flag to category
df['Holiday_Flag'] = df['Holiday_Flag'].astype('category')

print("\nUpdated Dataset:")
print(df.head())

# ------------------------------------------
# STEP 4: EXPLORATORY DATA ANALYSIS
# ------------------------------------------

print("\nGenerating Visualizations...")

# Sales Trend Over Time
plt.figure(figsize=(14,6))
plt.plot(df['Date'], df['Weekly_Sales'])
plt.title("Weekly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/sales_trend.png")
plt.show()

# Monthly Sales
monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()

plt.figure(figsize=(10,5))
sns.barplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title("Average Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.savefig("output/monthly_sales.png")
plt.show()

# Holiday vs Non-Holiday Sales
plt.figure(figsize=(8,5))
sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
plt.title("Holiday vs Non-Holiday Sales")
plt.savefig("output/holiday_sales.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")
plt.savefig("output/correlation_heatmap.png")
plt.show()

# ------------------------------------------
# STEP 5: FEATURE SELECTION
# ------------------------------------------

print("\nPreparing Features...")

X = df[[
    'Store',
    'Holiday_Flag',
    'Temperature',
    'Fuel_Price',
    'CPI',
    'Unemployment',
    'Year',
    'Month',
    'Week',
    'Day'
]]

# Convert category to numeric
X['Holiday_Flag'] = X['Holiday_Flag'].cat.codes

y = df['Weekly_Sales']

# ------------------------------------------
# STEP 6: TRAIN TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ------------------------------------------
# STEP 7: MODEL TRAINING
# ------------------------------------------

print("\nTraining Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------------------
# STEP 8: PREDICTIONS
# ------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------
# STEP 9: MODEL EVALUATION
# ------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("="*40)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 Score : {r2:.4f}")

# ------------------------------------------
# STEP 10: ACTUAL VS PREDICTED
# ------------------------------------------

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print("\nPrediction Comparison:")
print(results.head())

plt.figure(figsize=(12,6))

plt.plot(
    results['Actual'].values[:50],
    label='Actual Sales'
)

plt.plot(
    results['Predicted'].values[:50],
    label='Predicted Sales'
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()

plt.savefig("output/prediction_comparison.png")

plt.show()

# ------------------------------------------
# STEP 11: FUTURE FORECASTING
# ------------------------------------------

print("\nGenerating Future Forecast...")

future_data = X_test.head(10)

future_predictions = model.predict(future_data)

forecast_df = pd.DataFrame({
    'Forecasted Sales': future_predictions
})

print("\nFuture Forecasts:")
print(forecast_df)

forecast_df.to_csv(
    "output/future_sales_forecast.csv",
    index=False
)

# ------------------------------------------
# STEP 12: FEATURE IMPORTANCE
# ------------------------------------------

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance")

plt.savefig("output/feature_importance.png")

plt.show()

print("\nProject Completed Successfully!")
