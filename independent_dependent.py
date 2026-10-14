
# Import Pandas to read and process the Excel dataset
import pandas as pd

# Import Matplotlib to create graphs
import matplotlib.pyplot as plt

# Import Seaborn to create a correlation heatmap
import seaborn as sns

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Import LinearRegression to build a prediction model
from sklearn.linear_model import LinearRegression

# Import metrics to evaluate prediction performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 1: Load the actual Excel dataset
# Keep this Excel file in the same folder as this Python file
df = pd.read_excel("Practical_01_Association.xlsx")

# Display the first five records
print("First Five Rows:")
print(df.head())

# Display the total number of rows and columns
print("\nDataset Shape:", df.shape)

# Display the column names
print("\nDataset Columns:")
print(df.columns.tolist())

# Check whether any values are missing
print("\nMissing Values:")
print(df.isnull().sum())


# Step 2: Select the independent variables
# These features may help explain or predict final marks
X = df[
    [
        "Study_Hours_Per_Day",
        "Attendance_Percent",
        "Previous_Marks"
    ]
]

# Select the dependent variable that we want to predict
y = df["Final_Marks"]


# Step 3: Calculate Pearson correlation coefficients
# Correlation measures the strength and direction of a linear relationship
correlation = df[
    [
        "Study_Hours_Per_Day",
        "Attendance_Percent",
        "Previous_Marks",
        "Final_Marks"
    ]
].corr(method="pearson")

# Display the correlation matrix
print("\nPearson Correlation Matrix:")
print(correlation.round(3))

# Display how strongly each input feature correlates with final marks
print("\nCorrelation with Final Marks:")
print(correlation["Final_Marks"].sort_values(ascending=False))


# Step 4: Plot a correlation heatmap
# The heatmap displays correlation values using colours
plt.figure(figsize=(8, 6))

# Annotate each cell with its correlation value
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

# Add a title to the graph
plt.title("Correlation Heatmap of Student Marks")

# Adjust the layout and display the heatmap
plt.tight_layout()
plt.show()


# Step 5: Draw scatter plots for each independent variable
# Each plot compares one input feature with final marks
features = [
    "Study_Hours_Per_Day",
    "Attendance_Percent",
    "Previous_Marks"
]

# Create one scatter plot for every feature
for feature in features:

    # Plot the feature values against final marks
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feature], df["Final_Marks"])

    # Label both axes
    plt.xlabel(feature)
    plt.ylabel("Final Marks")

    # Add a title to the plot
    plt.title(f"{feature} vs Final Marks")

    # Display the graph
    plt.tight_layout()
    plt.show()


# Step 6: Divide the dataset into training and testing sets
# 80% of the records are used for training and 20% for testing
# random_state=42 makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 7: Create the Linear Regression model
# The model learns the relationship between inputs and final marks
model = LinearRegression()

# Train the model using the training dataset
model.fit(X_train, y_train)


# Step 8: Predict final marks for the testing dataset
y_pred = model.predict(X_test)

# Display actual and predicted marks for comparison
comparison = pd.DataFrame({
    "Actual_Marks": y_test.values,
    "Predicted_Marks": y_pred.round(2)
})

print("\nActual vs Predicted Marks:")
print(comparison.to_string(index=False))


# Step 9: Evaluate the prediction model
# MAE measures the average absolute prediction error
mae = mean_absolute_error(y_test, y_pred)

# MSE measures the average squared prediction error
mse = mean_squared_error(y_test, y_pred)

# RMSE is the square root of MSE
rmse = mse ** 0.5

# R2 measures performance relative to a mean-baseline model
r2 = r2_score(y_test, y_pred)

# Display all evaluation metrics
print("\nModel Evaluation:")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))


# Step 10: Display the learned coefficient for each feature
# A coefficient describes the model's estimated change in marks
# when that feature increases by one unit, holding others constant
print("\nRegression Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", round(coefficient, 3))

# Display the model's intercept
print("Intercept:", round(model.intercept_, 3))


# Step 11: Plot actual marks against predicted marks
plt.figure(figsize=(6, 5))

# Plot actual marks on the x-axis and predictions on the y-axis
plt.scatter(y_test, y_pred)

# Find the common range for the reference line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

# Draw a diagonal line representing perfect predictions
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    label="Perfect Prediction"
)

# Label the axes and add a title
plt.xlabel("Actual Final Marks")
plt.ylabel("Predicted Final Marks")
plt.title("Actual vs Predicted Final Marks")

# Display the legend and graph
plt.legend()
plt.tight_layout()
plt.show()


# Step 12: Print the final conclusion
print("\nConclusion:")
print("Association between student features and final marks was analyzed.")
print("A Linear Regression model was trained and evaluated successfully.")
