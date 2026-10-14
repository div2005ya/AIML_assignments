
# Import Pandas to read and process the Excel dataset
import pandas as pd

# Import Matplotlib to create graphs
import matplotlib.pyplot as plt

# Import train_test_split to divide the dataset
from sklearn.model_selection import train_test_split

# Import StandardScaler to standardize input features
from sklearn.preprocessing import StandardScaler

# Import make_pipeline to combine scaling and model training
from sklearn.pipeline import make_pipeline

# Import Linear Regression, Ridge, and Lasso models
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Import metrics to evaluate regression performance
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Load the actual Excel dataset
# Keep this file in the same folder as regularization.py
df = pd.read_excel("Practical_08_Regularization.xlsx")

# Display the first five records
print("First Five Rows:")
print(df.head())

# Display the dataset dimensions
print("\nDataset Shape:", df.shape)

# Display all column names
print("\nDataset Columns:")
print(df.columns.tolist())


# Step 2: Select the independent variables
# Record_ID is an identifier, so it is not used for prediction
X = df[
    [
        "Ad_Spend_TV",
        "Ad_Spend_Radio",
        "Ad_Spend_Digital",
        "Website_Visits",
        "Discount_Percent"
    ]
]

# Select the dependent variable to be predicted
y = df["Sales_Units"]

# Remove rows with missing input features or target values
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


# Step 4: Create three regression models
models = {
    # Ordinary Linear Regression has no regularization penalty
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    # Ridge Regression uses an L2 penalty
    # Alpha controls the strength of regularization
    "Ridge (L2)": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    # Lasso Regression uses an L1 penalty
    # It can reduce some coefficients exactly to zero
    "Lasso (L1)": make_pipeline(
        StandardScaler(),
        Lasso(alpha=0.1, max_iter=10000)
    )
}


# Step 5: Train and evaluate each model
# Create an empty list to store the results
results = []

# Process each model one at a time
for name, model in models.items():

    # Train the model using the training dataset
    # Scaling is fitted only on training data by the pipeline
    model.fit(X_train, y_train)

    # Predict sales for the training records
    train_pred = model.predict(X_train)

    # Predict sales for the unseen testing records
    test_pred = model.predict(X_test)

    # Calculate training Mean Squared Error
    train_mse = mean_squared_error(y_train, train_pred)

    # Calculate testing Mean Squared Error
    test_mse = mean_squared_error(y_test, test_pred)

    # Calculate training R-squared score
    train_r2 = r2_score(y_train, train_pred)

    # Calculate testing R-squared score
    test_r2 = r2_score(y_test, test_pred)

    # Store the model name and its evaluation results
    results.append({
        "Model": name,
        "Train_MSE": round(train_mse, 2),
        "Test_MSE": round(test_mse, 2),
        "Train_R2": round(train_r2, 3),
        "Test_R2": round(test_r2, 3)
    })


# Step 6: Display the comparison table
# Convert the results list into a DataFrame
results_df = pd.DataFrame(results)

# Print the performance of all three models
print("\nModel Performance Comparison:")
print(results_df.to_string(index=False))


# Step 7: Plot the test MSE for each model
plt.figure(figsize=(8, 5))

# Draw a bar chart comparing test errors
plt.bar(results_df["Model"], results_df["Test_MSE"])

# Label the axes and give the graph a title
plt.xlabel("Regression Model")
plt.ylabel("Test Mean Squared Error")
plt.title("Regularization Model Comparison")

# Adjust spacing and display the graph
plt.tight_layout()
plt.show()


# Step 8: Display the coefficients learned by Ridge and Lasso
# The pipeline stores the trained regression model in its final step
for name in ["Ridge (L2)", "Lasso (L1)"]:

    # Get the trained model from the pipeline
    regressor = models[name].named_steps[
        "ridge" if name == "Ridge (L2)" else "lasso"
    ]

    # Print the coefficients for each input feature
    print("\nCoefficients for", name)

    for feature, coefficient in zip(X.columns, regressor.coef_):
        print(feature, ":", round(coefficient, 3))


# Step 9: Print the final conclusion
print("\nConclusion:")
print("Linear, Ridge, and Lasso regression were compared.")
print("Test performance helps assess generalization.")
