
# Import Pandas to read and process the Excel dataset
import pandas as pd

# Import Matplotlib to create graphs
import matplotlib.pyplot as plt

# Import StandardScaler to standardize numerical features
from sklearn.preprocessing import StandardScaler

# Import KMeans to perform clustering
from sklearn.cluster import KMeans

# Import silhouette_score to evaluate cluster separation
from sklearn.metrics import silhouette_score


# Step 1: Load the actual Excel dataset
# Keep this file in the same folder as clustering.py
df = pd.read_excel("Practical_02_Clustering.xlsx")

# Display the first five rows of the dataset
print("First Five Rows:")
print(df.head())

# Display the number of rows and columns
print("\nDataset Shape:", df.shape)

# Display the column names
print("\nDataset Columns:")
print(df.columns.tolist())


# Step 2: Select the numerical features for clustering
# Customer_ID is an identifier, so it is not used for clustering
X = df[
    [
        "Age",
        "Monthly_Spending_INR",
        "Store_Visits_Per_Month"
    ]
]

# Check whether the selected data contains missing values
print("\nMissing Values:")
print(X.isnull().sum())

# Remove records with missing values in selected features
# Reset the index to keep the data aligned
X = X.dropna().reset_index(drop=True)

# Keep the matching customer IDs for displaying results
customer_ids = df.loc[X.index, "Customer_ID"].reset_index(drop=True)


# Step 3: Standardize the numerical features
# Scaling prevents high-value features from dominating distance calculations
scaler = StandardScaler()

# Transform the features to a comparable scale
X_scaled = scaler.fit_transform(X)


# Step 4: Use the Elbow Method to help choose the number of clusters
# Store the within-cluster sum of squares for different cluster counts
inertia_values = []

# Test cluster counts from 1 to 10
for k in range(1, 11):

    # Create a K-Means model with k clusters
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Fit the model to the standardized data
    kmeans.fit(X_scaled)

    # Save the inertia value for this number of clusters
    inertia_values.append(kmeans.inertia_)

# Plot the number of clusters against inertia
plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), inertia_values, marker="o")

# Add graph labels and title
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")

# Display the Elbow Method graph
plt.tight_layout()
plt.show()


# Step 5: Apply K-Means clustering with 3 clusters
# Three is used here as an example; the elbow graph can guide your choice
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Assign each customer to a cluster
cluster_labels = kmeans.fit_predict(X_scaled)

# Add the assigned cluster number to the dataset
results = df.loc[X.index].copy()
results["Cluster"] = cluster_labels

# Display customer IDs and their cluster assignments
print("\nCustomer Cluster Assignments:")
print(results[
    ["Customer_ID", "Age", "Monthly_Spending_INR",
     "Store_Visits_Per_Month", "Cluster"]
].to_string(index=False))


# Step 6: Display the number of customers in each cluster
print("\nCustomers in Each Cluster:")
print(results["Cluster"].value_counts().sort_index())


# Step 7: Calculate the silhouette score
# The score ranges from -1 to 1; higher is generally better separated
score = silhouette_score(X_scaled, cluster_labels)

# Display the clustering evaluation score
print("\nSilhouette Score:", round(score, 3))


# Step 8: Calculate average feature values for each cluster
# This helps understand the typical characteristics of each group
print("\nAverage Features in Each Cluster:")
print(
    results.groupby("Cluster")[
        ["Age", "Monthly_Spending_INR", "Store_Visits_Per_Month"]
    ].mean().round(2)
)


# Step 9: Visualize clusters using spending and store visits
# The graph displays two features while clustering used all three
plt.figure(figsize=(8, 5))

# Draw the customers and colour them according to their cluster
plt.scatter(
    results["Monthly_Spending_INR"],
    results["Store_Visits_Per_Month"],
    c=results["Cluster"],
    cmap="viridis",
    s=50
)

# Label the axes and give the graph a title
plt.xlabel("Monthly Spending (INR)")
plt.ylabel("Store Visits Per Month")
plt.title("Customer Segmentation Using K-Means")

# Display the graph
plt.tight_layout()
plt.show()


# Step 10: Save the cluster assignments to a new Excel file
# This preserves the original dataset
results.to_excel("clustering_results.xlsx", index=False)

# Print the final conclusion
print("\nConclusion: K-Means clustering completed successfully.")
print("Results saved to clustering_results.xlsx")
