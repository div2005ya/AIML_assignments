
# Import Pandas to read and process the Excel dataset
import pandas as pd

# Import Matplotlib to create graphs
import matplotlib.pyplot as plt

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Import StandardScaler to standardize the input features
from sklearn.preprocessing import StandardScaler

# Import SVR to create the Support Vector Regression model
from sklearn.svm import SVR

# Import metrics to evaluate regression performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 1: Load the actual Excel dataset
# Keep this file in the same folder as svr.py
df = pd.read_excel("Practical_07_SVR_Regression.xlsx")

# Display the first five records
print("First Five Rows:")
print(df.head())

# Display the number of rows and columns
print("\nDataset Shape:", df.shape)

# Display the column names
print("\nDataset Columns:")
print(df.columns.tolist())


# Step 2: Select the independent variables
# These building features are used to predict energy consumption
X = df[
    [
        "Temperature_C",
        "Occupancy_Percent",
        "Operating_Hours_Per_Day"
    ]
]

# Select the dependent variable to be predicted
y = df["Energy_Consumption_kWh"]

# Remove rows containing missing values in the selected columns
valid_rows = X.notna().all(axis=1) & y.notna()

# Keep only complete records
X = X.loc[valid_rows]
y = y.loc[valid_rows]


# Step 3: Split the dataset into training and testing sets
# Use 80% of the data for training and 20% for testing
# random_state=42 makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Standardize the input features
# SVR is sensitive to differences in feature scales
scaler = StandardScaler()

# Learn scaling parameters from training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same training parameters
X_test_scaled = scaler.transform(X_test)


# Step 5: Create the Support Vector Regression model
# RBF kernel can model nonlinear relationships
# C controls the penalty for prediction errors
# epsilon defines the error-insensitive margin
model = SVR(
    kernel="rbf",
    C=100,
    epsilon=0.1
)

# Train the SVR model using the scaled training data
model.fit(X_train_scaled, y_train)


# Step 6: Predict energy consumption for the test data
y_pred = model.predict(X_test_scaled)

# Display actual and predicted energy consumption
comparison = pd.DataFrame({
    "Actual_Energy_kWh": y_test.values,
    "Predicted_Energy_kWh": y_pred.round(2)
})

print("\nActual vs Predicted Energy Consumption:")
print(comparison.to_string(index=False))


# Step 7: Evaluate the SVR model
# MAE measures the average absolute prediction error
mae = mean_absolute_error(y_test, y_pred)

# MSE measures the average squared prediction error
mse = mean_squared_error(y_test, y_pred)

# RMSE is the square root of MSE
rmse = mse ** 0.5

# R2 measures performance relative to predicting the mean
r2 = r2_score(y_test, y_pred)

# Display all evaluation metrics
print("\nModel Evaluation:")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))


# Step 8: Plot actual values against predicted values
plt.figure(figsize=(7, 5))

# Plot actual energy consumption against predictions
plt.scatter(y_test, y_pred)

# Find the range for the perfect-prediction reference line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

# Draw a line representing perfect predictions
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    label="Perfect Prediction"
)

# Label the axes and title
plt.xlabel("Actual Energy Consumption (kWh)")
plt.ylabel("Predicted Energy Consumption (kWh)")
plt.title("Support Vector Regression (SVR)")

# Display the legend
plt.legend()

# Adjust spacing and show the graph
plt.tight_layout()
plt.show()


# Step 9: Predict energy consumption for a new building
# These are example input values for a hypothetical building
new_building = pd.DataFrame({
    "Temperature_C": [25.0],
    "Occupancy_Percent": [60],
    "Operating_Hours_Per_Day": [10.0]
})

# Scale the new input using the already fitted scaler
new_building_scaled = scaler.transform(new_building)

# Predict energy consumption using the trained model
predicted_energy = model.predict(new_building_scaled)[0]

# Display the prediction in kWh
print("\nPredicted Energy Consumption (kWh):",
      round(predicted_energy, 2))


# Step 10: Print the final conclusion
print("\nConclusion:")
print("SVR was implemented to predict building energy consumption.")
