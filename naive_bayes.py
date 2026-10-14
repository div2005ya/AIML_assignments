
# Import Pandas to read and process the Excel dataset
import pandas as pd

# Import Matplotlib to display the confusion matrix
import matplotlib.pyplot as plt

# Import train_test_split to divide the dataset
from sklearn.model_selection import train_test_split

# Import GaussianNB for Naive Bayes classification
from sklearn.naive_bayes import GaussianNB

# Import StandardScaler to standardize numerical features
from sklearn.preprocessing import StandardScaler

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# Step 1: Load the actual Excel dataset
# Keep the Excel file in the same folder as naive_bayes.py
df = pd.read_excel("Practical_04_Naive_Bayes.xlsx")

# Display the first five rows
print("First Five Rows:")
print(df.head())

# Display the dataset dimensions
print("\nDataset Shape:", df.shape)

# Display the column names
print("\nDataset Columns:")
print(df.columns.tolist())


# Step 2: Identify the target column
# This program expects a column named Class, Label, or Target
possible_targets = ["Class", "Label", "Target"]

target_column = next(
    (col for col in possible_targets if col in df.columns),
    None
)

# Stop with a helpful message if no target column is found
if target_column is None:
    raise ValueError(
        "Target column not found. Check the column names above "
        "and set target_column to the correct label column."
    )

# Remove rows with missing target values
df = df.dropna(subset=[target_column]).copy()

# Select all columns except the target as input features
X = df.drop(columns=[target_column])

# Remove identifier columns if present
id_columns = [
    col for col in X.columns
    if col.lower() in ["id", "student_id", "record_id", "email_id"]
]

X = X.drop(columns=id_columns, errors="ignore")

# Convert categorical input columns into numeric dummy variables
X = pd.get_dummies(X, dummy_na=True, dtype=float)

# Replace infinite values with missing values
X = X.replace([float("inf"), float("-inf")], float("nan"))

# Fill missing numerical values with each column's median
X = X.fillna(X.median())

# Store the target labels
y = df[target_column]

# Check that the dataset has enough features and classes
if X.shape[1] == 0 or y.nunique() < 2:
    raise ValueError("The dataset needs input features and at least two classes.")

# Display the selected target column
print("\nTarget Column:", target_column)

# Display the number of records in each class
print("\nClass Distribution:")
print(y.value_counts())


# Step 3: Split the dataset into training and testing sets
# Use stratification to preserve class proportions where possible
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Step 4: Standardize the numerical features
# Fit the scaler on training data only to prevent data leakage
scaler = StandardScaler()

# Learn the training feature scales and transform training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data using the same training scales
X_test_scaled = scaler.transform(X_test)


# Step 5: Create the Gaussian Naive Bayes model
# GaussianNB assumes that features follow a Gaussian distribution
model = GaussianNB()

# Train the classifier using the training data and labels
model.fit(X_train_scaled, y_train)


# Step 6: Predict the classes of the testing records
y_pred = model.predict(X_test_scaled)

# Calculate and display the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Display precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# Step 7: Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Create a visual display of the confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

# Plot the confusion matrix
display.plot(cmap="Blues")

# Add a title to the graph
plt.title("Naive Bayes Confusion Matrix")

# Display the graph
plt.tight_layout()
plt.show()


# Step 8: Print the final conclusion
print("\nConclusion:")
print("Naive Bayes classification was completed successfully.")
