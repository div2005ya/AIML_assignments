
# Practical 10: Multimodal Fake News Detection

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Convert text into numerical TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer

# Encode categorical features and combine different feature types
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfTransformer

# Machine learning model and evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Step 1: Load the actual Excel dataset
file_path = "Practical_10_Fake_News_Multimodal.xlsx"
df = pd.read_excel(file_path)

# Display the dataset
print("First five records:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())


# Step 2: Remove records without a target label
df = df.dropna(subset=["Label"]).copy()

# Separate the target label from input features
y = df["Label"].astype(str).str.strip()

# Select text, categorical, and numerical features
text_column = "Headline_Text"

categorical_columns = [
    "Source_Name",
    "Language",
    "Source_Verified"
]

numerical_columns = [
    "Publication_Year",
    "Share_Count"
]

# Keep only columns available in the workbook
categorical_columns = [
    col for col in categorical_columns if col in df.columns
]

numerical_columns = [
    col for col in numerical_columns if col in df.columns
]

# Check whether the expected text column exists
if text_column not in df.columns:
    raise ValueError("Headline_Text column is missing from the dataset.")

# Remove missing text values safely
X = df[
    [text_column] + categorical_columns + numerical_columns
].copy()

X[text_column] = X[text_column].fillna("")

# Convert numerical columns to numeric values
for col in numerical_columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remove image path because the referenced images are not included
# This program does not directly process image pixels
print("\nNote: Image_Path_Example is not used for image analysis.")


# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Step 4: Prepare text features using TF-IDF
text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=3000
    ))
])


# Step 5: Prepare categorical and numerical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Step 6: Combine all feature-processing pipelines
preprocessor = ColumnTransformer([
    ("text", text_pipeline, text_column),
    ("categorical", categorical_pipeline, categorical_columns),
    ("numerical", numerical_pipeline, numerical_columns)
])


# Step 7: Build the fake news classification model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Step 8: Train the model
model.fit(X_train, y_train)

print("\nModel training completed.")


# Step 9: Predict labels for the test dataset
y_pred = model.predict(X_test)


# Step 10: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(cm)


# Step 11: Visualize the confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(cmap="Blues", values_format="d")
plt.title("Fake News Detection - Confusion Matrix")
plt.tight_layout()
plt.show()


# Step 12: Test a sample news headline
sample_news = pd.DataFrame([{
    "Headline_Text": "Example news headline for testing",
    "Source_Name": "Example Source",
    "Language": "English",
    "Publication_Year": 2026,
    "Share_Count": 100,
    "Source_Verified": "Yes"
}])

# Keep exactly the feature columns used during training
sample_news = sample_news[X.columns]

prediction = model.predict(sample_news)[0]

print("\nSample headline prediction:", prediction)
print("Note: This is a model prediction, not verified fact-checking.")
