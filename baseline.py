import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import math
from matplotlib import pyplot as plt
from tqdm import tqdm

# ... (Assuming all the functions like haversine and other import statements are defined above)

# Load and preprocess the dataset
data = pd.read_csv('dataset/preprocessed_diningcode.csv')  # Update the path to your file
data['category'] = data['category'].str.split('>').str[-1]


# Prepare features and labels for machine learning
X = data.drop(columns=[
    'category', 'Title', 'Latitude', 'Longitude', 'Info Title', 'Title Place',
    'hashes', 'score', 'userScore', 'heart', 'title', 'address', 'roadAddress',
    'mapx', 'mapy', 'categories'
])
y = data['category']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize variables to store scores
n_components = list(range(100, 200, 10))
n_runs = 20
pca_scores = np.zeros((n_runs, len(n_components)))

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Assuming X_train_scaled is your standardized data
# Calculate the explained variance for each component
pca = PCA().fit(X_train_scaled)

# Calculate the cumulative sum of explained variances
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Determine the number of components that explain at least 95% of the variance
components_for_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1

# Plot the elbow graph
plt.figure(figsize=(8, 4))
plt.plot(cumulative_explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')

# Optional: Add a line and marker for the 95% threshold
plt.axvline(components_for_95, color='r', linestyle='--', label='95% Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.legend()

# Show the plot
plt.show()

# Run the training and prediction process 5 times
for i in tqdm(range(n_runs), desc="Overall Progress"):

    # Baseline model evaluation
    baseline_clf = RandomForestClassifier()
    baseline_clf.fit(X_train_scaled, y_train)
    baseline_y_pred_test = baseline_clf.predict(X_test_scaled)

    # Store baseline results once, as they won't change with PCA components
    if i == 0:
        baseline_accuracy = accuracy_score(y_test, baseline_y_pred_test)
        baseline_f1_micro = f1_score(y_test, baseline_y_pred_test, average='micro')
        baseline_f1_macro = f1_score(y_test, baseline_y_pred_test, average='macro')
        baseline_f1_weighted = f1_score(y_test, baseline_y_pred_test, average='weighted')

    # PCA model evaluation for different numbers of components
    for j, n in enumerate(tqdm(n_components, desc="PCA Progress")):
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        pca_clf = RandomForestClassifier()
        pca_clf.fit(X_train_pca, y_train)
        pca_y_pred_test = pca_clf.predict(X_test_pca)

        # Store PCA results
        pca_scores[i, j] = accuracy_score(y_test, pca_y_pred_test)

# Average the PCA scores across runs for each number of components
average_pca_scores = np.mean(pca_scores, axis=0)

# Find the number of components with the best average score
best_n_components = n_components[np.argmax(average_pca_scores)]
print(f"Best number of PCA components: {best_n_components}")

# Plot the average scores for each number of components
plt.plot(n_components, average_pca_scores, label='PCA Accuracy')
plt.axhline(y=baseline_accuracy, color='r', linestyle='-', label='Baseline Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Average Accuracy Score')
plt.title('Average Accuracy Score for Different Numbers of PCA Components')
plt.legend()
plt.show()

# Perform statistical comparison using a paired t-test
best_pca_scores = pca_scores[:, n_components.index(best_n_components)]
p_value = stats.ttest_rel(best_pca_scores, [baseline_accuracy] * n_runs).pvalue

# Print p-values to compare baseline vs best PCA model
print(f"P-value for accuracy comparison: {p_value:.4f}")

# Print baseline scores
print(f"Baseline Accuracy: {baseline_accuracy}")
print(f"Baseline F1 Micro: {baseline_f1_micro}")
print(f"Baseline F1 Macro: {baseline_f1_macro}")
print(f"Baseline F1 Weighted: {baseline_f1_weighted}")
#print the best number of components's accuracy
print(f"Best number of PCA components: {best_n_components}")
print(f"Best PCA Accuracy: {np.max(average_pca_scores)}")


