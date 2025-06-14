# Customer Segmentation Clustering EDA & Analysis

This repository provides a complete exploratory data analysis (EDA) and clustering workflow for customer data. It includes feature engineering, data preprocessing, and application of multiple clustering algorithms, along with visualization of final clusters on a 2D PCA projection.

## Clustering
Clustering is an unsupervised machine learning technique that groups data points into clusters based on similarity in their features. The goal is to ensure that points within the same cluster are more similar to each other than to points in other clusters. Clustering is commonly used for customer segmentation, image analysis, anomaly detection, and more.

---
---

### K-Means

- **Type**: Partitioning, centroid-based  
- **How it works**:  
  1. Choose _k_ initial centroids (randomly or via heuristic).  
  2. Assign each point to the nearest centroid.  
  3. Recompute centroids as the mean of all points assigned to each cluster.  
  4. Repeat assignment and centroid update until convergence (assignments no longer change or max iterations reached).  
- **Key parameters**:  
  - `n_clusters` (the number of clusters _k_)  
  - `init` (method for centroid initialization, e.g., “k-means++”)  
  - `max_iter` (maximum number of iterations)  
- **Strengths**: Fast, scalable to large datasets.  
- **Weaknesses**: Assumes spherical clusters of similar size; sensitive to outliers and initialization.

---

### DBSCAN

- **Type**: Density-based  
- **How it works**:  
  1. For each point, count the number of neighbors within radius `eps`.  
  2. Points with ≥ `min_samples` neighbors are “core” points; those within `eps` of a core point but with fewer neighbors are “border” points; others are “noise.”  
  3. Form clusters by connecting core points and their reachable border points.  
- **Key parameters**:  
  - `eps` (neighborhood radius)  
  - `min_samples` (minimum number of points to form a dense region)  
- **Strengths**: Can find arbitrarily shaped clusters; handles noise/outliers naturally.  
- **Weaknesses**: Performance degrades in high dimensions; sensitive to parameter settings.

---

### Agglomerative Clustering

- **Type**: Hierarchical, bottom-up  
- **How it works**:  
  1. Start with each point as its own cluster.  
  2. At each step, merge the two closest clusters according to a linkage criterion.  
  3. Repeat until all points are in a single cluster or until the desired number of clusters is reached.  
- **Linkage Criteria**:  
  - **Ward**: Minimizes variance within clusters.  
  - **Average**: Uses average distance between all pairs of points in two clusters.  
  - **Complete**: Uses maximum distance between points in two clusters.  
- **Strengths**: Doesn’t require specifying number of clusters upfront (you can cut the dendrogram); produces a hierarchical tree of clusters.  
- **Weaknesses**: Computationally expensive on large datasets; choice of linkage can greatly affect results.

---

### Fuzzy C-Means

- **Type**: Soft (fuzzy) clustering  
- **How it works**:  
  1. Initialize _c_ cluster centers.  
  2. Compute a membership matrix where each point has a degree of belonging (between 0 and 1) to each cluster, based on distance to cluster centers and a fuzziness parameter _m_.  
  3. Update cluster centers as the weighted mean of all points, using membership degrees.  
  4. Repeat membership update and center update until convergence.  
- **Key parameters**:  
  - `c` (number of clusters)  
  - `m` (fuzziness exponent; _m_ > 1)  
  - `error` (tolerance for convergence)  
- **Strengths**: Captures uncertainty in cluster assignments; useful when cluster boundaries are not well-defined.  
- **Weaknesses**: Requires choosing fuzziness parameter; can be slower than hard clustering methods.

---

## Project Overview

End-to-end customer segmentation pipeline using clustering techniques on a sample customer dataset. The workflow includes:

- Loading and parsing the data (including date parsing)  
- Feature engineering (e.g., `Age`, `TenureDays`)  
- One-hot encoding of categorical variables (`Education`, `Marital_Status`)  
- Feature scaling (using `StandardScaler`)  
- Dimensionality reduction for visualization (PCA to 2D)  
- Applying multiple clustering algorithms:  
  - K-Means  
  - DBSCAN  
  - Agglomerative Clustering  
  - Fuzzy C-Means (via `scikit-fuzzy`)  
- Plotting clusters on PCA axes for comparison  

---

## Prerequisites

- Python 3.7+  
- `numpy`  
- `pandas`  
- `scikit-learn`  
- `matplotlib` / `seaborn`  
- `scikit-fuzzy`  

Install dependencies via:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scikit-fuzzy
