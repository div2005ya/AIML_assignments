
# Import Pandas to read and process the Excel dataset
import pandas as pd

# Import Matplotlib to display the confusion matrix
import matplotlib.pyplot as plt

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Import KNeighborsClassifier to perform KNN classification
from sklearn.neighbors import KNeighborsClassifier

# Import tools to preprocess numerical and categorical features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import metrics to evaluate the model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


# Step 1: Load the actual Excel dataset
# Keep the Excel file in the same folder as this Python file
df = pd.read_excel("Practical_05_KNN_Classification.xlsx")

# Display the first five rows of the dataset
print("First Five Rows:")
print(df.head())

# Display the dataset dimensions
print("\nDataset Shape:", df.shape)

# Display all column names
print("\nDataset Columns:")
print(df.columns.tolist())


# Step 2: Identify the target column
# The target contains the category that the model must predict
possible_targets = ["Class", "Label", "Target", "Species", "Category"]

# Find the first matching target column
target_column = next(
    (col for col in possible_targets if col in df.columns),
    None
)

# Stop if the target column was not identified
if target_column is None:
    raise ValueError(
        "Target column not found. Check the column names printed above "
        "and update possible_targets with the correct column name."
    )

# Remove rows where the target label is missing
df = df.dropna(subset=[target_column]).copy()

# Separate the input features from the target labels
X = df.drop(columns=[target_column])
y = df[target_column]

# Remove identifier columns because they usually do not describe a class
id_columns = [
    col for col in X.columns
    if col.lower() in [
        "id", "student_id", "customer_id", "record_id"
    ]
]

X = X.drop(columns=id_columns, errors="ignore")

# Display the selected target column and class counts
print("\nTarget Column:", target_column)
print("\nClass Distribution:")
print(y.value_counts())

# Check that there are enough features and at least two classes
if X.shape[1] == 0 or y.nunique() < 2:
    raise ValueError(
        "The dataset must contain input features and at least two classes."
    )


# Step 3: Identify numerical and categorical input columns
# Numerical columns contain numbers, such as measurements
numeric_columns = X.select_dtypes(include="number").columns.tolist()

# Categorical columns contain text or category values
categorical_columns = X.select_dtypes(
    exclude="number"
).columns.tolist()


# Step 4: Prepare numerical features
# Fill missing numerical values with the median, then scale them
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Step 5: Prepare categorical features
# Fill missing categories and convert categories into numeric indicators
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Step 6: Combine the preprocessing steps
# Apply the appropriate preprocessing to each column type
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# Step 7: Split the dataset into training and testing sets
# Use stratification to preserve class proportions when possible
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Step 8: Create the KNN classification pipeline
# n_neighbors=5 means the five nearest training records vote
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])


# Step 9: Train the KNN model
# The pipeline learns preprocessing parameters from training data
model.fit(X_train, y_train)


# Step 10: Predict the classes of the test records
y_pred = model.predict(X_test)

# Display actual and predicted labels
comparison = pd.DataFrame({
    "Actual_Class": y_test.to_numpy(),
    "Predicted_Class": y_pred
})

print("\nActual vs Predicted Classes:")
print(comparison.to_string(index=False))


# Step 11: Evaluate the classifier using accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy as a percentage
print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Display precision, recall, and F1-score for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# Step 12: Display the confusion matrix
# The plot compares actual classes with predicted classes
display = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=model.named_steps["classifier"].classes_,
    xticks_rotation="vertical",
    cmap="Blues"
)

# Add a title to the confusion matrix
plt.title("KNN Classification Confusion Matrix")

# Adjust spacing and display the graph
plt.tight_layout()
plt.show()


# Step 13: Print the final conclusion
print("\nConclusion:")
print("KNN classification was completed successfully.")
