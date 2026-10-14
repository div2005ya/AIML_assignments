
# Import Pandas to read and process the Excel dataset
import pandas as pd

# Import Matplotlib to create graphs
import matplotlib.pyplot as plt

# Import StandardScaler to standardize numerical features
from sklearn.preprocessing import StandardScaler

# Import PCA to perform dimensionality reduction
from sklearn.decomposition import PCA


# Step 1: Load the actual Excel dataset
# Keep the Excel file in the same folder as pca.py
df = pd.read_excel("Practical_03_PCA.xlsx")

# Display the first five rows
print("First Five Rows:")
print(df.head())

# Display the dataset dimensions
print("\nDataset Shape:", df.shape)

# Display all column names
print("\nDataset Columns:")
print(df.columns.tolist())


# Step 2: Select numerical columns for PCA
# PCA works with numerical features, not text labels or identifiers
numeric_df = df.select_dtypes(include="number").copy()

# Remove identifier columns because they do not represent useful features
id_columns = [
    col for col in numeric_df.columns
    if col.lower() in ["id", "student_id", "customer_id", "record_id"]
]

numeric_df = numeric_df.drop(columns=id_columns, errors="ignore")

# Remove columns that contain only missing values
numeric_df = numeric_df.dropna(axis=1, how="all")

# Fill remaining missing values with their column median
numeric_df = numeric_df.fillna(numeric_df.median())

# Remove constant columns because they contain no variation
numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

# Check that enough numerical features are available
if numeric_df.shape[1] < 2:
    raise ValueError(
        "PCA needs at least two usable numerical features. "
        "Check the dataset columns."
    )

# Display the numerical features selected for PCA
print("\nFeatures Used for PCA:")
print(numeric_df.columns.tolist())


# Step 3: Standardize the selected features
# Scaling prevents features with larger units from dominating PCA
scaler = StandardScaler()

# Transform features to have approximately zero mean and unit variance
X_scaled = scaler.fit_transform(numeric_df)


# Step 4: Apply PCA and reduce the data to two components
# The two components capture directions of maximum variance
pca = PCA(n_components=2)

# Transform the standardized data into two principal components
X_pca = pca.fit_transform(X_scaled)

# Store the transformed data in a DataFrame
pca_df = pd.DataFrame(
    X_pca,
    columns=["Principal_Component_1", "Principal_Component_2"]
)

# Display the first five rows of the reduced dataset
print("\nFirst Five Rows After PCA:")
print(pca_df.head())


# Step 5: Display explained variance
# Each value indicates the proportion of variance captured by a component
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Calculate the total variance explained by the two components
total_variance = pca.explained_variance_ratio_.sum()

# Display the total explained variance as a percentage
print("Total Variance Explained:",
      round(total_variance * 100, 2), "%")


# Step 6: Plot the explained variance of each component
plt.figure(figsize=(7, 5))

# Create a bar chart for the two principal components
plt.bar(
    ["PC1", "PC2"],
    pca.explained_variance_ratio_ * 100
)

# Label the axes and add a title
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance (%)")
plt.title("PCA Explained Variance")

# Display the graph
plt.tight_layout()
plt.show()


# Step 7: Visualize the reduced dataset
plt.figure(figsize=(7, 5))

# Plot each record using the two principal components
plt.scatter(
    pca_df["Principal_Component_1"],
    pca_df["Principal_Component_2"]
)

# Label the axes and add a title
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Two-Dimensional Data Representation")

# Display the graph
plt.tight_layout()
plt.show()


# Step 8: Save the reduced data to a new Excel file
# The original dataset remains unchanged
pca_df.to_excel("pca_results.xlsx", index=False)

# Print the final conclusion
print("\nConclusion: PCA dimensionality reduction completed.")
print("Reduced data saved to pca_results.xlsx")
