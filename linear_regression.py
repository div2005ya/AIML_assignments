
# Import Pandas to read and process the Excel dataset
import pandas as pd

# Import Matplotlib to create graphs
import matplotlib.pyplot as plt

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Import LinearRegression to create the regression model
from sklearn.linear_model import LinearRegression

# Import evaluation metrics for regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 1: Load the actual Excel dataset
# Keep this file in the same folder as linear_regression.py
df = pd.read_excel("Practical_06_Linear_Regression.xlsx")

# Display the first five rows of the dataset
print("First Five Rows:")
print(df.head())

# Display the number of rows and columns
print("\nDataset Shape:", df.shape)

# Display all column names
print("\nDataset Columns:")
print(df.columns.tolist())


# Step 2: Select the independent variables
# These house features are used to predict the price
X = df[
    [
        "Area_sqft",
        "Bedrooms",
        "Age_Years",
        "Distance_to_City_km"
    ]
]

# Select the dependent variable that we want to predict
y = df["Price_INR"]

# Remove records with missing input or target values
valid_rows = X.notna().all(axis=1) & y.notna()

# Keep only complete records
X = X.loc[valid_rows]
y = y.loc[valid_rows]


# Step 3: Split the dataset into training and testing sets
# 80% of the records are used for training and 20% for testing
# random_state=42 ensures the same split on repeated runs
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Create the Linear Regression model
# The model learns the relationship between house features and prices
model = LinearRegression()

# Train the model using the training dataset
model.fit(X_train, y_train)


# Step 5: Predict house prices for the test dataset
y_pred = model.predict(X_test)

# Display actual and predicted prices side by side
comparison = pd.DataFrame({
    "Actual_Price_INR": y_test.values,
    "Predicted_Price_INR": y_pred.round(2)
})

print("\nActual vs Predicted House Prices:")
print(comparison.to_string(index=False))


# Step 6: Evaluate the regression model
# MAE is the average absolute difference between actual and predicted prices
mae = mean_absolute_error(y_test, y_pred)

# MSE is the average squared prediction error
mse = mean_squared_error(y_test, y_pred)

# RMSE is the square root of MSE and uses the target's units
rmse = mse ** 0.5

# R2 measures performance relative to predicting the mean price
r2 = r2_score(y_test, y_pred)

# Display the evaluation results
print("\nModel Evaluation:")
print("MAE (INR):", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE (INR):", round(rmse, 2))
print("R2 Score:", round(r2, 3))


# Step 7: Display the learned regression coefficients
# Each coefficient estimates the change in price per unit of a feature,
# holding the other features constant
print("\nRegression Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", round(coefficient, 2))

# Display the model intercept
print("Intercept:", round(model.intercept_, 2))


# Step 8: Plot actual prices against predicted prices
plt.figure(figsize=(7, 5))

# Plot actual prices on the x-axis and predictions on the y-axis
plt.scatter(y_test, y_pred)

# Find the common price range for the reference line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

# Draw a line representing perfect predictions
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    label="Perfect Prediction"
)

# Label the axes and add a title
plt.xlabel("Actual House Price (INR)")
plt.ylabel("Predicted House Price (INR)")
plt.title("Linear Regression: Actual vs Predicted Prices")

# Display the legend
plt.legend()

# Adjust the graph layout and display it
plt.tight_layout()
plt.show()


# Step 9: Predict the price of a new example house
# The values below are example inputs, not a real property listing
new_house = pd.DataFrame({
    "Area_sqft": [1500],
    "Bedrooms": [3],
    "Age_Years": [5],
    "Distance_to_City_km": [10.0]
})

# Predict the price using the trained model
predicted_price = model.predict(new_house)[0]

# Display the predicted price in INR
print("\nPredicted Price for New House (INR):",
      round(predicted_price, 2))


# Step 10: Print the final conclusion
print("\nConclusion:")
print("Linear Regression was implemented to predict house prices.")
